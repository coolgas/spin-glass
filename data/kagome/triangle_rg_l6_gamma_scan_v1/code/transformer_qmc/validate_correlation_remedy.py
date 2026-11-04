"""Held-out L2 disorder validation with a fixed Hamiltonian-only guide recipe."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from kagome_qea.exact import operator_matrix
from .assess_correlation_escalation import bootstrap, comparison
from .correlation_lengths import length, sha
from .correlation_remedy_campaign import parameters, write
from .diagnose_shared import reference
from .frozen_scan import SELECTED_SHA256, model_from
from .independent_replica_projector import project_independent, spectrum_from_moments
from .model import EnumeratedGuide
from .physics import basis, make_af_sample
from .production_protocol import exact_spectra, geometry, identity
from .refine_correlation_guide import refine

IDS=list(range(50016,50024))
FIELDS=[.25,.5,.75]
MODEL=None


def label(g,s): return f'L2_s{s}_g{g:g}_h0p15'.replace('.','p')


def train(task):
    output,checkpoint,signature,g,s=task; output=Path(output);p=output/'guides'/(label(g,s)+'.json')
    if p.exists():
        record=json.loads(p.read_text());assert record['signature']==signature
        assert sha(p.with_suffix('.npz'))==record['npz_sha256'];return
    global MODEL
    started=time.monotonic();torch.set_num_threads(1)
    if MODEL is None: MODEL=model_from(checkpoint,'cpu')
    sample=make_af_sample(2,s);ham=sample.hamiltonian(.15);spins=basis(12);lattice=geometry(2)
    MODEL.set_hamiltonian(ham,g)
    with torch.no_grad():
        tensor=torch.as_tensor(spins,dtype=torch.long)
        logs=np.concatenate([MODEL.log_amplitudes(chunk).numpy() for chunk in tensor.split(256)])
    np.testing.assert_allclose(np.exp(2*logs).sum(),1.,atol=1e-10)
    matrix=operator_matrix(ham,g)
    fit=refine(logs,spins,matrix,degree=4,maxiter=2000)
    # References computed only after the fixed variational optimization.
    with threadpool_limits(limits=1): exact_energy,exact,_=reference(matrix,.15)
    residual=float(np.linalg.norm(matrix@exact-exact_energy*exact));assert residual<1e-9
    spectrum=exact_spectra(exact**2,spins,lattice)['connected_spectrum']
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p.with_suffix('.tmp.npz'),logs=fit['logs'],coefficients=fit['coefficients'],
                        original_logs=logs,exact_amplitude=exact,bonds=sample.bonds,J=sample.J)
    p.with_suffix('.tmp.npz').replace(p.with_suffix('.npz'))
    write(p,dict(signature=signature,gamma=g,sample_id=s,L=2,h=.15,checkpoint_sha256=SELECTED_SHA256,
        npz_sha256=sha(p.with_suffix('.npz')),exact_labels_used=False,energy=fit['energy'],
        energy_bias=fit['energy']-exact_energy,exact_energy=exact_energy,residual=residual,
        fidelity=float((exact@np.exp(fit['logs']))**2),local_energy_variance=fit['local_energy_variance'],
        original_fidelity=float((exact@np.exp(logs))**2),history=fit['history'],
        exact_spectrum=spectrum.tolist(),seconds=time.monotonic()-started))


def run(task):
    output,signature,g,s,setting,repeat=task;output=Path(output);name=label(g,s)
    p=output/'runs'/(name+f'_{setting}_r{repeat}.json')
    seed=int(identity(['heldout-remedy-v1',g,s,setting,repeat])[:16],16)
    gp=output/'guides'/(name+'.json');gm=json.loads(gp.read_text())
    assert gm['signature']==signature and sha(gp.with_suffix('.npz'))==gm['npz_sha256']
    if p.exists():
        record=json.loads(p.read_text());assert record['signature']==signature and record['seed']==seed;return
    with np.load(gp.with_suffix('.npz')) as data: guide=EnumeratedGuide(data['logs'].copy(),12)
    started=time.monotonic();par=parameters(setting)
    with threadpool_limits(limits=1):
        result=project_independent(make_af_sample(2,s).hamiltonian(.15),g,guide,geometry(2),**par,seed=seed)
    write(p,dict(signature=signature,L=2,h=.15,gamma=g,sample_id=s,setting=setting,repeat=repeat,seed=seed,
        guide_sha256=gm['npz_sha256'],parameters=par,result=result,exact_spectrum=gm['exact_spectrum'],
        seconds=time.monotonic()-started,ground_state_qualified=False))


def assess(output):
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text());signature=identity(manifest)
    assert manifest['sample_ids']==IDS and manifest['gamma']==FIELDS
    assert manifest['settings']=={s:parameters(s) for s in ['base','time']}
    assert manifest['exact_labels_used'] is False and manifest['degree']==4 and manifest['maxiter']==2000
    code=Path(__file__).parent
    for name,digest in manifest['source_sha256'].items(): assert sha(code/name)==digest,name
    guides={};hashes={str(output/'manifest.json'):sha(output/'manifest.json')}
    for g in FIELDS:
        for s in IDS:
            p=output/'guides'/(label(g,s)+'.json');gm=json.loads(p.read_text())
            assert gm['signature']==signature and (gm['L'],gm['h'],gm['gamma'],gm['sample_id'])==(2,.15,g,s)
            assert gm['exact_labels_used'] is False and gm['checkpoint_sha256']==SELECTED_SHA256
            assert gm['residual']<1e-9 and sha(p.with_suffix('.npz'))==gm['npz_sha256']
            with np.load(p.with_suffix('.npz')) as a:
                sample=make_af_sample(2,s);matrix=operator_matrix(sample.hamiltonian(.15),g)
                np.testing.assert_array_equal(a['bonds'],sample.bonds);np.testing.assert_array_equal(a['J'],sample.J)
                np.testing.assert_allclose(np.exp(2*a['logs']).sum(),1.,atol=1e-12)
                assert np.linalg.norm(matrix@a['exact_amplitude']-gm['exact_energy']*a['exact_amplitude'])<1e-9
                np.testing.assert_allclose(exact_spectra(a['exact_amplitude']**2,basis(12),geometry(2))['connected_spectrum'],gm['exact_spectrum'],atol=1e-12)
            guides[g,s]=gm;hashes[str(p)]=sha(p);hashes[str(p.with_suffix('.npz'))]=sha(p.with_suffix('.npz'))
    rows=[];data={};contrasts=[];validated=0;diagnostics=[]
    for g in FIELDS:
        for setting in ['base','time']:
            raw=[];alarms=0;minimum_ess=float('inf');halves=[]
            for s in IDS:
                repeats=[];hs=[]
                for repeat in range(4):
                    p=output/'runs'/(label(g,s)+f'_{setting}_r{repeat}.json');r=json.loads(p.read_text())
                    assert r['signature']==signature and (r['gamma'],r['sample_id'],r['setting'],r['repeat'],r['L'],r['h'])==(g,s,setting,repeat,2,.15)
                    assert r['seed']==int(identity(['heldout-remedy-v1',g,s,setting,repeat])[:16],16)
                    assert r['parameters']==parameters(setting) and r['guide_sha256']==guides[g,s]['npz_sha256']
                    assert r['exact_spectrum']==guides[g,s]['exact_spectrum']
                    result=r['result'];m=np.asarray(result['magnetization_cycles']);q=np.asarray(result['pair_cycles'])
                    assert m.shape==(4,16,12) and q.shape==(4,16,12,12) and np.isfinite(m).all() and np.isfinite(q).all()
                    spectrum=spectrum_from_moments(m.mean(1),q.mean(1),geometry(2))
                    np.testing.assert_allclose(spectrum,result['connected_spectrum'],atol=1e-12)
                    ess=np.asarray(result['descendant_ess']);big=np.asarray(result['largest_lineage'])
                    assert ess.shape==big.shape==(4,16)
                    alarm=bool(ess.min()<8 or big.max()>.5);assert result['ancestry_alarm']==alarm
                    alarms+=alarm;minimum_ess=min(minimum_ess,float(ess.min()));repeats.append(spectrum)
                    hs.append([spectrum_from_moments(m[:,part].mean(1),q[:,part].mean(1),geometry(2)) for part in [slice(0,8),slice(8,16)]])
                    diagnostics.append(dict(gamma=g,sample_id=s,setting=setting,repeat=repeat,min_ess=float(ess.min()),max_lineage=float(big.max()),alarm=alarm))
                    validated+=1;hashes[str(p)]=sha(p)
                raw.append(repeats);halves.append(hs)
            raw=np.asarray(raw);mean=raw.mean((0,1));target=np.mean([guides[g,s]['exact_spectrum'] for s in IDS],axis=0)
            boot,ix=bootstrap(raw,np.random.default_rng(int(identity(['heldout_assess',g,setting])[:16],16)))
            check=comparison(mean,boot,target);bounds=(abs(mean-target)+1.96*boot.std(0,ddof=1))/target
            halves=np.asarray(halves);a,b=halves[:,:,0],halves[:,:,1]
            ab=a[np.arange(8)[None,:,None],ix].mean((1,2));bb=b[np.arange(8)[None,:,None],ix].mean((1,2))
            drift=comparison(b.mean((0,1)),bb,a.mean((0,1)),ab)
            db=(abs(b.mean((0,1))-a.mean((0,1)))+1.96*(bb-ab).std(0,ddof=1))/target
            good=lambda c,b:bool(c['xi_over_side_95_bound'] is not None and c['xi_over_side_95_bound']<=.01 and c['invalid_bootstrap_fraction']==0 and np.all(b<=.05))
            value=length(mean,2)
            rows.append(dict(gamma=g,setting=setting,xi=float(value) if np.isfinite(value) else None,exact_xi=float(length(target,2)),
                mc_se=check['xi_difference_mc_se'],spectral_relative_95_bounds=bounds.tolist(),ancestry_alarms=alarms,minimum_ess=minimum_ess,
                exact_passes=good(check,bounds) and alarms==0,half_drift_passes=good(drift,db),half_drift=drift,half_drift_bounds=db.tolist(),**check))
            data[g,setting]=(mean,boot,target)
        a,ab,target=data[g,'base'];b,bb,_=data[g,'time'];check=comparison(b,bb,a,ab)
        bounds=(abs(b-a)+1.96*(bb-ab).std(0,ddof=1))/target
        contrasts.append(dict(gamma=g,passes=good(check,bounds),spectral_relative_95_bounds=bounds.tolist(),**check))
    assert len(list((output/'runs').glob('*.json')))==validated==192
    report=dict(completed_runs=validated,expected_runs=192,sample_ids=IDS,rows=rows,contrasts=contrasts,
        all_checks_pass=all(r['exact_passes'] and r['half_drift_passes'] for r in rows) and all(c['passes'] for c in contrasts),
        input_sha256=hashes,disorder_diagnostics=diagnostics,
        uncertainty='Fixed-disorder whole-run bootstrap; approximate pointwise 95% bounds; four repeats have limited tail resolution.',
        qualification='Held out from remedy selection. Per-context variational fitting uses Hamiltonians, not exact labels. L2 only.')
    write(output/'validated_review.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--workers',type=int,default=16);p.add_argument('--assess-only',action='store_true')
    a=p.parse_args()
    if a.assess_only:
        report=assess(a.output);print(json.dumps(dict(completed_runs=report['completed_runs'],all_checks_pass=report['all_checks_pass'])));return
    assert sha(a.checkpoint)==SELECTED_SHA256
    code=Path(__file__).parent
    names=['validate_correlation_remedy.py','refine_correlation_guide.py','independent_replica_projector.py',
           'correlation_remedy_campaign.py','projector.py','model.py','physics.py','production_protocol.py','frozen_scan.py']
    manifest=dict(sample_ids=IDS,gamma=FIELDS,settings={s:parameters(s) for s in ['base','time']},degree=4,maxiter=2000,
        exact_labels_used=False,checkpoint_sha256=SELECTED_SHA256,selection='Fixed recipe; disorders excluded from eight-disorder development ensemble',
        source_sha256={name:sha(code/name) for name in names})
    mp=a.output/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:write(mp,manifest)
    signature=identity(manifest)
    for stage in ['guides','runs']:
        tasks=([(str(a.output),str(a.checkpoint),signature,g,s) for g in FIELDS for s in IDS] if stage=='guides' else
               [(str(a.output),signature,g,s,k,r) for k in ['base','time'] for g in FIELDS for s in IDS for r in range(4)])
        sp=a.output/(stage+'_status.json');write(sp,dict(state='running',completed=0,total=len(tasks)))
        try:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                futures=[pool.submit(train if stage=='guides' else run,t) for t in tasks]
                for count,future in enumerate(as_completed(futures),1):
                    future.result();write(sp,dict(state='running',completed=count,total=len(tasks)));print(stage,count,len(tasks),flush=True)
            write(sp,dict(state='completed',completed=len(tasks),total=len(tasks)))
        except Exception as error:
            write(sp,dict(state='failed',error=repr(error)));raise
    report=assess(a.output);print(json.dumps(dict(completed_runs=report['completed_runs'],all_checks_pass=report['all_checks_pass'])))


if __name__=='__main__':main()
