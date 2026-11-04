"""CT GFMC field sensitivity on energy-selected frozen L2 simple guides."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path

import numpy as np

from . import collective_campaign as base
from .correlation_lengths import sha,length
from .production_protocol import identity,geometry
from .independent_replica_projector import spectrum_from_moments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert json.loads((args.scan/'status.json').read_text())['state']=='completed'
    rows=json.loads((args.scan/'review.json').read_text())['rows']
    selected=[r for r in rows if r['sample_id']==50043]
    manifest=dict(scan_sha256=sha(args.scan/'review.json'),driver_sha256=sha(Path(__file__)),
        sources=base.source_hashes(),scope='Difficult disorder 50043 only; L2 enumerated guide initialization; two independent runs per setting')
    path=args.output/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:base.write(path,manifest)
    work=[]
    for row in selected:
        parent=args.scan/(base.label(row['gamma'],row['sample_id'])+'.npz')
        assert sha(parent)==row['npz_sha256']
        for fit in row['fits']:
            if not fit['selected']:continue
            kind=fit['kind'];root=args.output/kind;inputs=root/'inputs'
            (inputs/'guides').mkdir(parents=True,exist_ok=True)
            with np.load(parent) as data:logs=data[f"{kind}_seed{fit['seed']}_logs"].copy()
            path=inputs/'guides'/parent.name
            if path.exists():
                with np.load(path) as data:np.testing.assert_array_equal(data['logs'],logs)
            else:np.savez_compressed(path,logs=logs,original_logs=logs)
            for audit,walkers,equil,forward in [('main',1024,32.,32.),('population',4096,32.,32.),('time',1024,64.,64.)]:
                for repeat in range(2):
                    task=dict(L=2,h=.15,gamma=row['gamma'],sample_id=50043,guide='corrected',method='ct',
                        delta=.1,epsilon=0.,walkers=walkers,equil=equil,forward=forward,cycles=16,
                        start='guide',repeat=repeat,audit=audit)
                    work.append((task,root,inputs))
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(base.run_one,t,o,i,3600):(t,o) for t,o,i in work}
        for f in as_completed(futures):
            t,o=futures[f]
            try:r=f.result();assert r['status']=='completed'
            except Exception as error:
                base.write(args.output/'failure.json',dict(task=t,kind=o.name,error=repr(error)));raise
            records.append(dict(kind=o.name,record=r))
            base.write(args.output/'status.json',dict(state='running',completed=len(records),total=len(work)))
            print(json.dumps(dict(completed=len(records),total=len(work),kind=o.name,task=t)),flush=True)
    summary=[]
    for row in selected:
        truth=np.array(row['exact']['spectrum'])
        for kind in ['jastrow','rbm_2']:
            for audit in ['main','population','time']:
                group=[x['record'] for x in records if x['kind']==kind and
                       x['record']['task']['gamma']==row['gamma'] and x['record']['task']['audit']==audit]
                for r in group:
                    path=args.output/kind/'runs'/(identity(r['task'])[:24]+'.npz')
                    assert sha(path)==r['npz_sha256']
                    with np.load(path) as data:
                        measured=spectrum_from_moments(data['magnetization_cycles'].mean(1),data['pair_cycles'].mean(1),geometry(2))
                    np.testing.assert_allclose(measured,r['spectrum'],atol=1e-12)
                raw=np.array([r['spectrum'] for r in group]);mean=raw.mean(0);xi=length(mean,2)/2
                summary.append(dict(gamma=row['gamma'],sample_id=50043,kind=kind,audit=audit,
                    spectrum=mean.tolist(),spectrum_se=(raw.std(0,ddof=1)/len(raw)**.5).tolist(),
                    max_spectral_error=float(np.max(abs(mean-truth)/truth)),
                    run_spectral_errors=[float(np.max(abs(s-truth)/truth)) for s in raw],
                    alarms=sum(r['ancestry_alarm'] for r in group),
                    minimum_descendant_ess=min(r['minimum_descendant_ess'] for r in group),
                    xi_over_L=float(xi) if np.isfinite(xi) else None,
                    exact_xi_over_L=float(length(truth,2)/2),ground_state_qualified=False))
    base.write(args.output/'review.json',dict(rows=summary,records=records,
        scope='Two-run field sensitivity diagnostic, not precision or larger-L qualification'))
    base.write(args.output/'status.json',dict(state='completed',completed=len(records),total=len(work)))


if __name__=='__main__':main()
