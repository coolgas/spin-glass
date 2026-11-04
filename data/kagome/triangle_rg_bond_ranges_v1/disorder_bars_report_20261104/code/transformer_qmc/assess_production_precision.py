"""Compare independent precision ensembles against their finite-forward targets.

Use the actual cached direct rates, including floating-point discrepancies,
in the Feynman-Kac generator. Short-time equivalence is not ground-state
qualification and does not authorize float32 final production.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from kagome_qea.exact import moments_from_probability
from .experiment import write_json
from .frozen_scan import key
from .physics import basis, correlations, diagonal, make_af_sample
from .production_protocol import TOLERANCES


def finite_context(payload):
    root,output,sid,g,h,precision=payload
    root,output=Path(root),Path(output);name=key(2,sid,g,h,1.)
    path=output/(name+'_'+precision+'.json')
    if path.exists():return str(path)
    start=time.monotonic();checks=root/'qualification';table=checks/'direct_checks'/(name+'.json')
    meta=json.loads(table.read_text());npz=table.with_suffix('.npz')
    assert hashlib.sha256(npz.read_bytes()).hexdigest()==meta['npz_sha256']
    exact=json.loads((checks/'exact'/(name+'.json')).read_text())
    suffix='32' if precision=='float32' else '64'
    with np.load(npz) as data:logs=data['logs'+suffix];ratios=data['ratios'+suffix]
    spins=basis(12);ham=make_af_sample(2,sid).hamiltonian(h)
    source=np.repeat(np.arange(4096),12);target=(np.arange(4096)[:,None]^(1<<np.arange(12))).ravel()
    rates=g*np.exp(ratios)
    A=sparse.csr_matrix((rates.ravel(),(target,source)),shape=(4096,4096))
    A.setdiag(-diagonal(ham,spins)+exact['energy'])
    trace=float(A.diagonal().sum());p0=np.exp(2*(logs-logs.max()));p0/=p0.sum()
    # Settings are validated against the immutable task manifest in main().
    left=expm_multiply(.6*A,p0,traceA=.6*trace)
    right=expm_multiply(.2*A.T,np.ones(4096),traceA=.2*trace)
    cycles=[]
    for cycle in range(4):
        p=left*right
        if p.min() < -1e-12:raise ArithmeticError('negative finite-forward probability')
        p=np.maximum(p,0);p/=p.sum()
        m=moments_from_probability(p,12);c=correlations(p,spins)
        cycles.append(dict(q=m['q'],q2=m['q2'],q4=m['q4'],chi_connected_per_site=c['chi_connected']/12))
        if cycle<3:left=expm_multiply(.2*A,left,traceA=.2*trace)
    write_json(path,dict(sample_id=sid,gamma=g,h=h,precision=precision,cycles=cycles,
        means={k:float(np.mean([c[k] for c in cycles])) for k in cycles[0]},
        direct_table_sha256=meta['npz_sha256'],source_signature=meta['signature'],seconds=time.monotonic()-start))
    return str(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=a.root/'precision_assessment';out.mkdir(exist_ok=True)
    tasks=json.loads((a.root/'pipeline/tasks.json').read_text());tasks=[t for t in tasks if t['stage']=='precision_ensemble']
    expected=dict(n_walkers=512,n_replicas=4,tau_block=.05,n_equilibrate=8,n_measure=4,tau_forward=.2,n_cycles=4)
    assert len(tasks)==3136 and all(t['parameters']==expected for t in tasks)
    records={}
    for t in tasks:
        record=json.loads((a.root/'pipeline/results'/(t['id']+'.json')).read_text())
        assert record['state']=='completed' and record['task']==t and not record['final_data']
        records[t['sample_id'],t['gamma'],t['h'],t['precision'],t['repeat']]=record['result']
    contexts=sorted({(t['sample_id'],t['gamma'],t['h'],t['precision']) for t in tasks})
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(finite_context,(str(a.root),str(out),*c)) for c in contexts]
        for count,future in enumerate(as_completed(futures),1):
            future.result()
            if count%25==0 or count==len(futures):
                write_json(out/'status.json',dict(state='computing',completed=count,expected=len(futures),updated_unix=time.time()))
    rows=[]
    for g,h in sorted({(t['gamma'],t['h']) for t in tasks}):
        for observable in ['q','q2','q4','chi_connected_per_site','energy_per_site']:
            values={prec:[] for prec in ('float32','float64')};finite={prec:[] for prec in values};ground=[]
            for sid in range(50000,50008):
                name=key(2,sid,g,h,1.);ex=json.loads((a.root/'qualification/exact'/(name+'.json')).read_text())
                target=ex['energy']/12 if observable=='energy_per_site' else (ex['observables']['chi_connected']/12 if observable=='chi_connected_per_site' else ex['observables'][observable])
                ground.append(target)
                for prec in values:
                    values[prec].append(np.mean([records[sid,g,h,prec,r][observable] for r in range(4)]))
                    if observable!='energy_per_site':
                        ref=json.loads((out/(name+'_'+prec+'.json')).read_text())
                        finite[prec].append(ref['means'][observable])
            difference=np.array(values['float32'])-values['float64']
            se=float(difference.std(ddof=1)/np.sqrt(8));mean=float(difference.mean());tol=TOLERANCES.get(observable)
            rows.append(dict(gamma=g,h=h,observable=observable,mean_precision_difference=mean,
                disorder_paired_se=se,approx_95_abs_bound=abs(mean)+1.96*se,tolerance=tol,
                short_run_equivalence=None if tol is None else abs(mean)+1.96*se<=tol,
                mean_ground_error={prec:float(np.mean(np.array(v)-ground)) for prec,v in values.items()},
                mean_finite_forward_error={prec:float(np.mean(np.array(values[prec])-v)) for prec,v in finite.items() if v},
                per_disorder_precision_difference=difference))
    tested=[r for r in rows if r['tolerance'] is not None]
    write_json(out/'summary.json',dict(state='completed',rows=rows,contexts=49,disorders=8,repeats=4,
        short_run_equivalence_passes=sum(r['short_run_equivalence'] for r in tested),comparisons=len(tested),
        production_precision_qualified=False,ground_state_qualified=False,
        interpretation='Approximate paired-disorder 95% bounds at one short setting; inconclusive bounds require more repeats. No long-time or large-size precision claim.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    write_json(out/'status.json',dict(state='completed',completed=len(contexts),expected=len(contexts),updated_unix=time.time()))
    print('PRECISION_ASSESSMENT',sum(r['short_run_equivalence'] for r in tested),'of',len(tested),'short-setting comparisons',flush=True)


if __name__=='__main__':
    main()
