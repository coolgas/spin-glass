"""Matched L2 overlap distributions and field screening; energy-only guide fits.

Full-basis training is an ansatz/optimization diagnostic, not sampled scaling.
Exact references enter evaluation after all fits of a context are frozen.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .assess_overlap_onset import overlap_distribution
from .collective_campaign import write, label
from .correlation_lengths import sha
from .physics import basis, make_af_sample, exact_reference
from .production_protocol import exact_spectra, geometry
from .simple_guide_benchmark import fit


FIELDS=[.5,.55,.6,.65,.7,.75,.85,1.]
DISORDERS=list(range(50000,50008))+[50043]


def checked_pq(probability):
    p=np.asarray(probability,dtype=float);p=p/p.sum()
    q,mass=overlap_distribution(p)
    spins=basis(int(np.log2(len(p)))).astype(float)
    m=p@spins;pair=spins.T@(p[:,None]*spins)
    np.testing.assert_allclose(mass.sum(),1.,atol=1e-12)
    np.testing.assert_allclose(mass@q,np.mean(m*m),atol=1e-12)
    np.testing.assert_allclose(mass@(q*q),np.mean(pair*pair),atol=1e-12)
    return q,mass


def pq_metrics(q,p):
    mean=float(q@p);variance=float(((q-mean)**2)@p)
    return dict(mean=mean,variance=variance,negative_mass=float(p[q<0].sum()),
        central_mass=float(p[abs(q)<=1/3+1e-10].sum()),
        entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))))


def context(gamma,sid,output,seeds):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        root=Path(output);name=label(gamma,sid);path=root/(name+'.json')
        signature=dict(gamma=gamma,sample_id=sid,seeds=seeds,
            sources={n:sha(Path(__file__).with_name(n)) for n in
                ['simple_overlap_scan.py','simple_guides.py','simple_guide_benchmark.py','assess_overlap_onset.py']})
        if path.exists():
            r=json.loads(path.read_text());assert r['signature']==signature
            assert sha(path.with_suffix('.npz'))==r['npz_sha256'];return r
        begin=time.monotonic();ham=make_af_sample(2,sid).hamiltonian(.15)
        fits=[];arrays={}
        for kind in ['jastrow','rbm_2']:
            for seed in range(seeds):
                model,logs,record=fit(ham,gamma,kind,seed,'cpu',800,200)
                key=f'{kind}_seed{seed}'
                arrays[key+'_logs']=logs
                torch.save(model.state_dict(),root/(name+'_'+key+'.pt'))
                fits.append(dict(kind=kind,seed=seed,**record))
        # Selection precedes reference calculation; exact labels never select seeds.
        selected={k:min([r for r in fits if r['kind']==k],key=lambda r:r['energy'])['seed']
                  for k in ['jastrow','rbm_2']}
        reference=exact_reference(ham,gamma)
        exact=reference['amplitude'];q,pq=checked_pq(exact**2)
        arrays.update(exact_amplitude=exact,q=q,exact_pq=pq)
        truth=exact_spectra(exact**2,basis(12),geometry(2))['connected_spectrum']
        for row in fits:
            key=f"{row['kind']}_seed{row['seed']}";psi=np.exp(arrays[key+'_logs'])
            _,mass=checked_pq(psi**2);arrays[key+'_pq']=mass
            spectrum=exact_spectra(psi**2,basis(12),geometry(2))['connected_spectrum']
            row.update(fidelity=float((exact@psi)**2),energy_error=row['energy']-reference['energy'],
                pq_tv=float(abs(mass-pq).sum()/2),spectrum=spectrum.tolist(),
                max_spectral_error=float(np.max(abs(spectrum-truth)/truth)),
                selected=row['seed']==selected[row['kind']],pq=mass.tolist())
        np.savez_compressed(path.with_suffix('.npz'),**arrays)
        record=dict(signature=signature,gamma=gamma,sample_id=sid,h=.15,L=2,
            exact=dict(energy=reference['energy'],gap=reference['gap'],residual=reference['residual'],
                pq=pq.tolist(),q=q.tolist(),spectrum=truth.tolist(),**pq_metrics(q,pq)),
            fits=fits,seconds=time.monotonic()-begin,npz_sha256=sha(path.with_suffix('.npz')),
            scope='Selected L2 finite-size crossover diagnostic; no thermodynamic phase classification')
        write(path,record);return record


def summarize(rows,root):
    summary=[]
    for gamma in sorted({r['gamma'] for r in rows}):
        ensemble=[r for r in rows if r['gamma']==gamma and r['sample_id']!=50043]
        if not ensemble:continue
        exact=np.array([r['exact']['pq'] for r in ensemble])
        for kind in ['jastrow','rbm_2']:
            selected=[next(f for f in r['fits'] if f['kind']==kind and f['selected']) for r in ensemble]
            allfits=[f for r in ensemble for f in r['fits'] if f['kind']==kind]
            mass=np.array([r['pq'] for r in selected])
            summary.append(dict(gamma=gamma,kind=kind,disorders=len(ensemble),
                exact_pq=exact.mean(0).tolist(),guide_pq=mass.mean(0).tolist(),
                exact_disorder_se=(exact.std(0,ddof=1)/np.sqrt(len(exact))).tolist() if len(exact)>1 else None,
                guide_disorder_se=(mass.std(0,ddof=1)/np.sqrt(len(mass))).tolist() if len(mass)>1 else None,
                mean_paired_tv=float(np.mean([r['pq_tv'] for r in selected])),
                worst_paired_tv=max(r['pq_tv'] for r in selected),
                median_spectral_error=float(np.median([r['max_spectral_error'] for r in selected])),
                worst_spectral_error=max(r['max_spectral_error'] for r in selected),
                seed_fraction_pq_tv_below_005=float(np.mean([r['pq_tv']<=.05 for r in allfits])),
                minimum_gap=min(r['exact']['gap'] for r in ensemble),
                mean_overlap_variance=float(np.mean([r['exact']['variance'] for r in ensemble]))))
    write(Path(root)/'review.json',dict(rows=rows,summary=summary,
        caveat='Full-basis training only; threshold TV<=0.05 is a descriptive screening rule, not phase or convergence qualification.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--seeds',type=int,default=3)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(fields=FIELDS,disorders=DISORDERS,seeds=args.seeds,
        h=.15,L=2,training='800 Adam + 200 L-BFGS; full-basis energy only',
        selection='minimum variational energy, all seeds retained',
        grouping='50000..50007 ensemble; 50043 separate difficult-case diagnostic',
        sources={n:sha(Path(__file__).with_name(n)) for n in
            ['simple_overlap_scan.py','simple_guides.py','simple_guide_benchmark.py','assess_overlap_onset.py','physics.py']})
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():assert json.loads(manifest_path.read_text())==manifest
    else:write(manifest_path,manifest)
    tasks=[(g,s) for g in FIELDS for s in [50043]+DISORDERS[:-1]]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(context,g,s,str(args.output),args.seeds):(g,s) for g,s in tasks}
        for future in as_completed(futures):
            try:row=future.result()
            except Exception as error:
                write(args.output/'failure.json',dict(context=futures[future],error=repr(error)));raise
            rows.append(row);summarize(rows,args.output)
            write(args.output/'status.json',dict(state='running',completed=len(rows),total=len(tasks)))
            print(json.dumps(dict(gamma=row['gamma'],sample_id=row['sample_id'],completed=len(rows),seconds=row['seconds'])),flush=True)
    write(args.output/'status.json',dict(state='completed',completed=len(rows),total=len(tasks)))


if __name__=='__main__':main()
