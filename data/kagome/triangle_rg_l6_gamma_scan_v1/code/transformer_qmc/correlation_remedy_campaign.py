"""L2 controlled remedy: variational neural-guide corrections and separate replicas."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply
from threadpoolctl import threadpool_limits

from kagome_qea.exact import operator_matrix
from .correlation_lengths import BASE, ROOT, sha
from .extend_correlation_fields import stem, load_inputs
from .independent_replica_projector import project_independent, spectrum_from_moments
from .model import EnumeratedGuide
from .physics import basis
from .production_protocol import exact_spectra, geometry, identity, production_tags
from .projector import project
from .refine_correlation_guide import refine

FIELDS = [.25, .5, .75]
IDS = list(range(50000, 50008))
SETTINGS = {
    'base': dict(walkers=1024, equil=16., forward=16., start='guide'),
    'population': dict(walkers=2048, equil=16., forward=16., start='guide'),
    'time': dict(walkers=1024, equil=32., forward=32., start='guide'),
    'start': dict(walkers=1024, equil=32., forward=16., start='uniform'),
}


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def parameters(setting):
    s = SETTINGS[setting]
    return dict(n_walkers=s['walkers'], tau_block=.1, n_equilibrate=round(s['equil']/.1),
                n_measure=10, tau_forward=s['forward'], n_cycles=16, start=s['start'])


def train_one(task):
    root, signature, gamma, sid = task
    root = Path(root); tag = stem(gamma, sid); path = root/'guides'/(tag+'.json')
    if path.exists():
        record = json.loads(path.read_text())
        assert record['signature'] == signature and sha(path.with_suffix('.npz')) == record['npz_sha256']
        return dict(stage='guides', gamma=gamma, sample_id=sid, reused=True)
    started = time.monotonic()
    sample, original, inputs, checkpoint = load_inputs(gamma, sid)
    spins, lattice = basis(12), geometry(2)
    ham = operator_matrix(sample.hamiltonian(.15), gamma)
    fit = refine(original.logs, spins, ham, degree=4, maxiter=2000)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix('.tmp.npz'), logs=fit['logs'], coefficients=fit['coefficients'])
    path.with_suffix('.tmp.npz').replace(path.with_suffix('.npz'))
    # Only after variational fitting, load exact amplitudes for evaluation.
    ep = BASE/'qualification/exact'/(tag+'.npz')
    with np.load(ep) as archive:
        exact = archive['amplitude'].copy()
    exact_energy = float(exact @ (ham @ exact))
    p = np.exp(2*fit['logs'])
    old = np.exp(original.logs-original.logs.max()); old /= np.linalg.norm(old)
    exact_spectrum = exact_spectra(exact**2, spins, lattice)['connected_spectrum']
    shifted = ham - exact_energy*eye(4096, format='csr'); trace = float(shifted.diagonal().sum())
    screens = []
    with threadpool_limits(limits=1):
        for label, par in [('base', parameters('base')), ('time', parameters('time'))]:
            begin = par['tau_block']*(par['n_equilibrate']+par['n_measure']); forward=par['tau_forward']
            amplitude = np.exp(fit['logs'])
            left = expm_multiply(-begin*shifted, amplitude, traceA=-begin*trace)
            right = expm_multiply(-forward*shifted, amplitude, traceA=-forward*trace)
            moments, pairs = [], []
            for cycle in range(par['n_cycles']):
                probability = left*right; probability /= probability.sum()
                moments.append(probability@spins)
                pairs.append(spins.T@(probability[:, None]*spins))
                if cycle+1 < par['n_cycles']:
                    left = expm_multiply(-forward*shifted, left, traceA=-forward*trace)
            m, q = np.mean(moments,axis=0), np.mean(pairs,axis=0)
            spectrum = spectrum_from_moments(np.tile(m,(4,1)),np.tile(q,(4,1,1)),lattice)
            screens.append(dict(setting=label, spectrum=spectrum.tolist(),
                                spectral_relative_bias=((spectrum-exact_spectrum)/exact_spectrum).tolist()))
    record = dict(signature=signature, gamma=gamma, sample_id=sid, L=2, h=.15,
                  status='completed', input_sha256=inputs, checkpoint_sha256=checkpoint,
                  npz_sha256=sha(path.with_suffix('.npz')), exact_labels_used=False,
                  guide_kind='frozen_neural_times_variational_degree4', degree=4,
                  history=fit['history'], energy=fit['energy'], exact_energy=exact_energy,
                  energy_bias=fit['energy']-exact_energy, local_energy_variance=fit['local_energy_variance'],
                  fidelity=float((exact@np.sqrt(p))**2), original_fidelity=float((exact@old)**2),
                  exact_spectrum=exact_spectrum.tolist(), screens=screens,
                  seconds=time.monotonic()-started)
    write(path, record)
    return dict(stage='guides',gamma=gamma,sample_id=sid,seconds=record['seconds'],
                fidelity=record['fidelity'],variance=record['local_energy_variance'])


def run_one(task):
    root, signature, gamma, sid, setting, repeat = task
    root=Path(root); tag=stem(gamma,sid); path=root/'runs'/(tag+f'_{setting}_r{repeat}.json')
    seed=int(identity(['correlation-remedy-v1',gamma,sid,setting,repeat])[:16],16)
    if path.exists():
        record=json.loads(path.read_text())
        assert record['signature']==signature and record['seed']==seed
        return dict(stage='gfmc',gamma=gamma,sample_id=sid,setting=setting,repeat=repeat,reused=True)
    started=time.monotonic(); sample,original,inputs,checkpoint=load_inputs(gamma,sid)
    gp=root/'guides'/(tag+'.json'); gm=json.loads(gp.read_text())
    assert gm['signature']==signature and sha(gp.with_suffix('.npz'))==gm['npz_sha256']
    with np.load(gp.with_suffix('.npz')) as archive: guide=EnumeratedGuide(archive['logs'].copy(),12)
    par=parameters(setting if setting in SETTINGS else 'base')
    if setting=='original_independent': guide=original
    with threadpool_limits(limits=1):
        if setting=='corrected_joint':
            joint_par={k:v for k,v in par.items() if k!='start'}
            result=project(sample.hamiltonian(.15),gamma,guide,**joint_par,n_replicas=4,seed=seed,
                           tagger=lambda states:production_tags(states,geometry(2)))
            result={k:np.asarray(v).tolist() for k,v in result.items()}
            alarm=bool(min(result['descendant_ess'])<8 or max(result['largest_lineage'])>.5)
        else:
            result=project_independent(sample.hamiltonian(.15),gamma,guide,geometry(2),**par,seed=seed)
            alarm=result['ancestry_alarm']
    record=dict(signature=signature,status='completed',L=2,h=.15,gamma=gamma,sample_id=sid,
                setting=setting,repeat=repeat,seed=seed,parameters=par,input_sha256=inputs,
                guide_sha256=gm['npz_sha256'] if setting!='original_independent' else None,
                checkpoint_sha256=checkpoint,ancestry_alarm=alarm,result=result,
                ground_state_qualified=False,seconds=time.monotonic()-started)
    write(path,record)
    return dict(stage='gfmc',gamma=gamma,sample_id=sid,setting=setting,repeat=repeat,
                ancestry_alarm=alarm,seconds=record['seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--stage',choices=['guides','pilot','gfmc','all'],default='all')
    args=parser.parse_args(); root=args.output; root.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(__file__).with_name('refine_correlation_guide.py'),
             Path(__file__).with_name('independent_replica_projector.py'),
             Path(__file__).with_name('projector.py'),Path(__file__).with_name('model.py'),
             Path(__file__).with_name('physics.py'),Path(__file__).with_name('production_protocol.py'),
             Path(__file__).with_name('extend_correlation_fields.py'),ROOT/'code/qmc_numerics.py']
    manifest=dict(schema=1,gamma=FIELDS,sample_ids=IDS,settings=SETTINGS,repeats=4,cycles=16,
                  branch=.1,degree=4,maxiter=2000,exact_labels_used=False,
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
                  tolerance=dict(xi_over_side=.01,spectra_relative=.05),
                  qualification_scope='L2 development only; larger sizes require separate validation')
    mp=root/'manifest.json'
    if mp.exists(): assert json.loads(mp.read_text())==manifest, 'Changed remedy identity'
    else: write(mp,manifest)
    signature=identity(manifest)
    stages=['guides','pilot','gfmc'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='guides':
            tasks=[(str(root),signature,g,s) for g in FIELDS for s in IDS]; fn=train_one
        elif stage=='pilot':
            tasks=[(str(root),signature,g,s,k,r) for g,s in [(.25,50004),(.5,50003),(.75,50005)]
                   for k in ['original_independent','corrected_joint','base'] for r in range(4)]; fn=run_one
        else:
            tasks=[(str(root),signature,g,s,k,r) for k in SETTINGS for g in FIELDS for s in IDS for r in range(4)]; fn=run_one
        status_path=root/(stage+'_status.json')
        write(status_path,dict(state='running',completed=0,total=len(tasks),updated_unix=time.time()))
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(fn,task) for task in tasks]
                for count,future in enumerate(as_completed(futures),1):
                    row=future.result(); print(json.dumps(row),flush=True)
                    write(status_path,dict(state='running',completed=count,total=len(tasks),updated_unix=time.time()))
            write(status_path,dict(state='completed',completed=len(tasks),total=len(tasks),updated_unix=time.time()))
        except Exception as error:
            write(status_path,dict(state='failed',error=repr(error),updated_unix=time.time()))
            raise


if __name__=='__main__': main()
