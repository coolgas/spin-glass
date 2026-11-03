"""Population, time, start and step audits for energy-selected simple L2 guides."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply
from threadpoolctl import threadpool_limits

from kagome_qea.exact import operator_matrix
from . import collective_campaign as base
from .correlation_lengths import length,sha
from .independent_replica_projector import spectrum_from_moments
from .physics import basis,make_af_sample
from .production_protocol import identity,geometry


def finite_reference(kind,gamma,sid,screen,projector):
    """Exact CT forward-walking schedule for the frozen chosen guide."""
    with threadpool_limits(limits=1):
        guide,_=base.load_guide(projector/kind/'inputs',dict(gamma=gamma,sample_id=sid,guide='corrected'))
        amplitude=np.exp(guide.logs-guide.logs.max());amplitude/=np.linalg.norm(amplitude)
        ham=make_af_sample(2,sid).hamiltonian(.15)
        matrix=operator_matrix(ham,gamma)
        # Variational scalar shift is independent of exact eigenvector labels.
        shift=float(amplitude@(matrix@amplitude))
        shifted=matrix-shift*eye(4096,format='csr');trace=float(shifted.diagonal().sum())
        spins=basis(12);rows=[]
        def advance(x,duration):
            x=expm_multiply(-duration*shifted,x,traceA=-duration*trace)
            return x/np.linalg.norm(x)
        for audit,equil,forward,start in [('main',32.,32.,'guide'),('time',64.,64.,'guide'),
                                         ('start',32.,32.,'uniform')]:
            initial=amplitude if start=='guide' else 1/amplitude
            initial=initial/np.linalg.norm(initial)
            left=advance(initial,equil);right=advance(amplitude,forward)
            spectrum=base.finite_moments(left,right,lambda x:advance(x,forward),16,spins,geometry(2))
            rows.append(dict(audit=audit,spectrum=spectrum.tolist()))
        return dict(kind=kind,gamma=gamma,sample_id=sid,finite=rows)


def assess(root,screen,projector):
    fits=json.loads((screen/'review.json').read_text())['rows']
    references=json.loads((root/'finite-references.json').read_text())['rows']
    groups={}
    for parent in [root,projector]:
        for path in sorted(parent.glob('*/runs/*.json')):
            kind=path.parent.parent.name
            if kind not in ['jastrow','rbm_2']:continue
            r=json.loads(path.read_text());t=r['task']
            assert sha(path.with_suffix('.npz'))==r['npz_sha256']
            assert r['provenance']['source_sha256']==base.source_hashes()
            with np.load(path.with_suffix('.npz')) as data:
                spectrum=spectrum_from_moments(data['magnetization_cycles'].mean(1),data['pair_cycles'].mean(1),geometry(2))
            np.testing.assert_allclose(spectrum,r['spectrum'],atol=1e-12)
            groups.setdefault((kind,t['gamma'],t['sample_id'],t['method'],t['delta'],t['audit']),[]).append(r)
    rows=[];rng=np.random.default_rng(918)
    for (kind,gamma,sid,method,delta,audit),records in sorted(groups.items()):
        truth=np.array(next(r['exact_spectrum'] for r in fits if (r['gamma'],r['sample_id'])==(gamma,sid)))
        raw=np.array([r['spectrum'] for r in records]);mean=raw.mean(0)
        boot=raw[rng.integers(len(raw),size=(2000,len(raw)))].mean(1)
        se=boot.std(0,ddof=1);lens=length(boot,2)/2;valid=np.isfinite(lens);xi=length(mean,2)/2
        reference=next(r for r in references if (r['kind'],r['gamma'],r['sample_id'])==(kind,gamma,sid))
        finite=np.array(next(r['spectrum'] for r in reference['finite'] if r['audit']==('main' if audit=='population' else audit))) if method=='ct' else None
        rows.append(dict(kind=kind,gamma=gamma,sample_id=sid,method=method,delta=delta,audit=audit,repeats=len(raw),
            spectrum=mean.tolist(),spectrum_se=se.tolist(),relative_error=((mean-truth)/truth).tolist(),
            max_relative_error=float(np.max(abs(mean-truth)/truth)),
            finite_projection_spectrum=None if finite is None else finite.tolist(),
            relative_sampling_error=None if finite is None else ((mean-finite)/finite).tolist(),
            finite_projection_relative_error=None if finite is None else ((finite-truth)/truth).tolist(),
            xi_over_L=float(xi) if np.isfinite(xi) else None,
            exact_xi_over_L=float(length(truth,2)/2),
            xi_interval=np.quantile(lens[valid],[.025,.975]).tolist() if valid.any() else None,
            invalid_length_fraction=float(1-valid.mean()),
            alarms=sum(r['ancestry_alarm'] for r in records),
            minimum_descendant_ess=min(r['minimum_descendant_ess'] for r in records),
            total_seconds=sum(r['seconds'] for r in records),ground_state_qualified=False))
    report=dict(rows=rows,scope='Selected L2 diagnostics; 2 audit repeats, whole-run bootstrap. No automatic convergence claim.')
    base.write(root/'review.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True);p.add_argument('--projector',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--hours',type=float,default=2.)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(source_sha256=base.source_hashes(),driver_sha256=sha(Path(__file__)),
        selected_manifest_sha256=sha(args.projector/'manifest.json'))
    path=args.output/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:base.write(path,manifest)
    tasks=[]
    for kind in ['jastrow','rbm_2']:
        for gamma,sid in base.CONTEXTS:
            for audit,method,delta,walkers,equil,forward,start in [
                ('population','ct',.1,4096,32.,32.,'guide'),
                ('time','ct',.1,1024,64.,64.,'guide'),
                ('start','ct',.1,1024,32.,32.,'uniform'),
                ('step','triangle',.05,1024,32.,32.,'guide')]:
                for repeat in range(2):
                    task=dict(L=2,h=.15,gamma=gamma,sample_id=sid,guide='corrected',method=method,
                        delta=delta,epsilon=0.,walkers=walkers,equil=equil,forward=forward,cycles=16,
                        start=start,repeat=repeat,audit=audit)
                    tasks.append((kind,task))
    deadline=time.monotonic()+args.hours*3600;done=0;pending=iter(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={}
        def submit(item):
            kind,task=item
            futures[pool.submit(base.run_one,task,args.output/kind,args.projector/kind/'inputs',1800,deadline)]=item
        for _ in range(args.workers):
            item=next(pending,None)
            if item is not None:submit(item)
        while futures:
            ready,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in ready:
                item=futures.pop(future)
                try:r=future.result()
                except Exception as error:
                    base.write(args.output/'failure.json',dict(item=item,error=repr(error)));raise
                if r['status']=='completed':done+=1
                if time.monotonic()<deadline:
                    item=next(pending,None) if r['status']=='completed' else item
                    if item is not None:submit(item)
                print(json.dumps(dict(completed=done,total=len(tasks),status=r['status'])),flush=True)
            base.write(args.output/'status.json',dict(state='running',completed=done,total=len(tasks)))
    if done<len(tasks):
        base.write(args.output/'status.json',dict(state='paused_time_cap',completed=done,total=len(tasks)));return
    if not (args.output/'finite-references.json').exists():
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(finite_reference,kind,g,s,args.screen,args.projector)
                     for kind in ['jastrow','rbm_2'] for g,s in base.CONTEXTS]
            rows=[f.result() for f in futures]
        base.write(args.output/'finite-references.json',dict(rows=rows,scope='Offline exact CT finite-projection targets'))
    assess(args.output,args.screen,args.projector)
    base.write(args.output/'status.json',dict(state='completed',completed=done,total=len(tasks)))


if __name__=='__main__':main()
