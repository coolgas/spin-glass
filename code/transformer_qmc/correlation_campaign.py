"""Resumable AutoDL CPU campaign: exact disorder ensemble and L2 convergence.

Run from an isolated source snapshot. No instance power operations are performed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply
from threadpoolctl import threadpool_limits

from kagome_qea.exact import operator_matrix
from .diagnose_shared import reference
from .frozen_scan import SELECTED_SHA256, model_from, stage_lock
from .model import EnumeratedGuide
from .physics import basis, make_af_sample
from .production_protocol import exact_spectra, geometry, identity, source_hashes
from .projector import project
from .correlation_lengths import length

GAMMAS = [.25, .5, .75, 1., 1.5, 2., 3.]
IDS = list(range(50000, 50128))
DEVELOPMENT = IDS[:16]
SETTINGS = {
    'base': dict(w=2048, equil=16., forward=16., branch=.1),
    'population': dict(w=4096, equil=16., forward=16., branch=.1),
    'equilibration': dict(w=2048, equil=32., forward=16., branch=.1),
    'forward': dict(w=2048, equil=16., forward=32., branch=.1),
    'cadence': dict(w=2048, equil=16., forward=16., branch=.05),
    'combined': dict(w=4096, equil=32., forward=32., branch=.05),
}
MODEL = None


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def name(sid, gamma):
    return f'L2_s{sid}_g{gamma:g}_h0p15'.replace('.', 'p')


def parameters(setting):
    s = SETTINGS[setting]
    return dict(n_walkers=s['w'], n_replicas=4, tau_block=s['branch'],
                n_equilibrate=round(s['equil']/s['branch']),
                n_measure=round(2/s['branch']), tau_forward=s['forward'], n_cycles=16)


def init_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def one(task):
    stage, root, checkpoint, signature, sid, gamma, setting, repeat = task
    root = Path(root)
    label = name(sid, gamma)
    path = root / stage / (label + (f'_{setting}_r{repeat}' if stage == 'gfmc' else '') + '.json')
    if path.exists():
        record = json.loads(path.read_text())
        assert record['signature'] == signature and record['status'] == 'completed'
        if 'npz_sha256' in record:
            assert digest(path.with_suffix('.npz')) == record['npz_sha256']
        return dict(stage=stage, name=path.name, reused=True)
    start = time.monotonic()
    sample = make_af_sample(2, sid)
    ham = sample.hamiltonian(.15)
    lattice = geometry(2)
    spins = basis(12)
    record = dict(signature=signature, status='completed', L=2, sample_id=sid,
                  gamma=gamma, h=.15, ground_state_qualified=False)
    with threadpool_limits(limits=1):
        if stage == 'exact':
            matrix = operator_matrix(ham, gamma)
            energy, psi, _ = reference(matrix, .15)
            residual = float(np.linalg.norm(matrix@psi-energy*psi))
            assert residual < 1e-9
            np.testing.assert_allclose(psi@psi, 1., atol=1e-10)
            spectrum = exact_spectra(psi**2, spins, lattice)['connected_spectrum']
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path.with_suffix('.tmp.npz'), amplitude=psi,
                                bonds=sample.bonds, J=sample.J)
            path.with_suffix('.tmp.npz').replace(path.with_suffix('.npz'))
            record.update(energy=energy, residual=residual, spectrum=spectrum.tolist(),
                          npz_sha256=digest(path.with_suffix('.npz')),
                          ground_state_qualified=True)
        elif stage == 'guides':
            global MODEL
            if MODEL is None:
                MODEL = model_from(checkpoint, 'cpu')
            MODEL.set_hamiltonian(ham, gamma)
            with torch.no_grad():
                tensor = torch.as_tensor(spins, dtype=torch.long)
                logs = np.concatenate([MODEL.log_amplitudes(chunk).numpy()
                                       for chunk in tensor.split(256)])
            assert np.isfinite(logs).all()
            np.testing.assert_allclose(np.exp(2*logs).sum(), 1., atol=1e-10)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path.with_suffix('.tmp.npz'), logs=logs)
            path.with_suffix('.tmp.npz').replace(path.with_suffix('.npz'))
            record.update(npz_sha256=digest(path.with_suffix('.npz')),
                          checkpoint_sha256=SELECTED_SHA256)
        else:
            guide_path = root/'guides'/(label+'.json')
            gm = json.loads(guide_path.read_text())
            assert gm['signature'] == signature
            assert digest(guide_path.with_suffix('.npz')) == gm['npz_sha256']
            with np.load(guide_path.with_suffix('.npz')) as data:
                logs = data['logs'].copy()
            if stage == 'screen':
                em = json.loads((root/'exact'/(label+'.json')).read_text())
                matrix = operator_matrix(ham, gamma)-em['energy']*eye(4096, format='csr')
                trace = float(matrix.diagonal().sum())
                trial = np.exp(logs-logs.max())
                rows = []
                for setting_name in ['base', 'equilibration', 'forward', 'combined']:
                    p = parameters(setting_name)
                    begin = p['tau_block']*(p['n_equilibrate']+p['n_measure'])
                    fw = p['tau_forward']
                    left = expm_multiply(-begin*matrix, trial, traceA=-begin*trace)
                    right = expm_multiply(-fw*matrix, trial, traceA=-fw*trace)
                    spectra = []
                    for cycle in range(p['n_cycles']):
                        spectra.append(exact_spectra(left*right, spins, lattice)['connected_spectrum'])
                        if cycle+1 < p['n_cycles']:
                            left = expm_multiply(-fw*matrix, left, traceA=-fw*trace)
                    rows.append(dict(setting=setting_name, spectrum=np.mean(spectra,axis=0).tolist()))
                record.update(settings=rows, exact_spectrum=em['spectrum'])
            elif stage == 'gfmc':
                from .production_protocol import production_tags
                p = parameters(setting)
                seed = int(identity([signature, sid, gamma, setting, repeat])[:16], 16)
                result = project(ham, gamma, EnumeratedGuide(logs, 12), **p, seed=seed,
                                 tagger=lambda states: production_tags(states, lattice))
                record.update(setting=setting, repeat=repeat, seed=seed, parameters=p,
                              guide_sha256=gm['npz_sha256'],
                              ancestry_alarm=bool(min(result['descendant_ess']) < 8 or
                                                  max(result['largest_lineage']) > .5),
                              result={k: np.asarray(v).tolist() for k,v in result.items()})
            else:
                raise ValueError(stage)
    record['seconds'] = time.monotonic()-start
    write(path, record)
    return dict(stage=stage, name=path.name, seconds=record['seconds'])


def summarize_exact(root):
    rows = []
    rng = np.random.default_rng(20261103)
    ix = rng.integers(128, size=(4000,128))
    for g in GAMMAS:
        raw = np.array([json.loads((root/'exact'/(name(s,g)+'.json')).read_text())['spectrum'] for s in IDS])
        boot = length(raw[ix].mean(1), 2)
        leave = length((raw.sum(0)-raw)/127, 2)
        row = dict(gamma=g, disorders=128, xi=float(length(raw.mean(0),2)),
                   bootstrap_se=float(np.std(boot,ddof=1)),
                   leave_one_out_min=float(leave.min()), leave_one_out_max=float(leave.max()))
        row['prefix_xi'] = {str(n):float(length(raw[:n].mean(0),2)) for n in [8,16,32,64,128]}
        rows.append(row)
    write(root/'exact_summary.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--stage', choices=['all','exact','guides','screen','gfmc'], default='all')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    assert digest(args.checkpoint) == SELECTED_SHA256
    manifest = dict(schema=1, sample_ids=IDS, development_ids=DEVELOPMENT, gamma=GAMMAS,
                    h=.15, checkpoint_sha256=SELECTED_SHA256, settings=SETTINGS,
                    cycles=16, repeats=4, source_sha256=source_hashes(Path(__file__).resolve().parents[1]),
                    convergence_gamma=[.5,.75,1.],
                    acceptance=dict(xi_over_side_absolute=.01, spectral_relative=.05,
                                    rule='95% error bound inside tolerance; no automatic qualification'),
                    policy='Development runs only; larger-size production follows convergence review')
    signature = identity(manifest)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    with stage_lock(root, 'campaign_'+args.stage):
        mp = root/'manifest.json'
        if mp.exists():
            assert json.loads(mp.read_text()) == manifest, 'Campaign identity changed'
        else:
            write(mp, manifest)
        stages = ['exact','guides','screen','gfmc'] if args.stage == 'all' else [args.stage]
        for stage in stages:
            contexts = [(sid,g) for g in (GAMMAS if stage=='exact' else [.5,.75,1.])
                        for sid in (IDS if stage=='exact' else DEVELOPMENT)]
            if args.pilot:
                contexts = [(50000,.5)]
            choices = [(s,r) for s in SETTINGS for r in range(4)] if stage=='gfmc' else [('',0)]
            tasks = [(stage,str(root),str(args.checkpoint),signature,sid,g,s,r)
                     for s,r in choices for sid,g in contexts]
            if args.pilot:
                tasks = tasks[:1]
            write(root/f'{stage}_status.json',dict(state='running',completed=0,total=len(tasks),pid=os.getpid()))
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
                futures = [pool.submit(one,t) for t in tasks]
                for count,future in enumerate(as_completed(futures),1):
                    value = future.result()
                    print(json.dumps(dict(completed=count,total=len(tasks),**value)),flush=True)
                    write(root/f'{stage}_status.json',dict(state='running',completed=count,total=len(tasks),
                                                          pid=os.getpid(),updated_unix=time.time()))
            if stage=='exact' and not args.pilot:
                summarize_exact(root)
            write(root/f'{stage}_status.json',dict(state='completed',completed=len(tasks),total=len(tasks),updated_unix=time.time()))


if __name__ == '__main__':
    main()
