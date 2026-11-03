"""L2 population/time escalation using immutable v3 guide and exact references."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .correlation_campaign import digest, init_worker, name, write
from .frozen_scan import stage_lock
from .model import EnumeratedGuide
from .physics import make_af_sample
from .production_protocol import geometry, identity, production_tags, source_hashes
from .projector import project
from .correlation_lengths import length


def tasks():
    return [dict(sample_id=sid,gamma=g,walkers=w,forward=fw,repeat=r)
            for g,fw in [(.5,64.),(.5,128.),(.75,32.)]
            for w in [8192,16384] for r in range(4) for sid in range(50000,50016)]


def one(payload):
    parent,output,signature,t=payload
    parent,output=Path(parent),Path(output)
    taskid=identity(t)[:24]
    path=output/'runs'/(taskid+'.json')
    if path.exists():
        previous=json.loads(path.read_text())
        assert previous['signature']==signature and previous['task']==t
        return dict(id=taskid,reused=True)
    stem=name(t['sample_id'],t['gamma'])
    gp=parent/'guides'/(stem+'.json');ep=parent/'exact'/(stem+'.json')
    gm,em=json.loads(gp.read_text()),json.loads(ep.read_text())
    assert digest(gp.with_suffix('.npz'))==gm['npz_sha256']
    with np.load(gp.with_suffix('.npz')) as d:logs=d['logs'].copy()
    p=dict(n_walkers=t['walkers'],n_replicas=4,tau_block=.05,n_equilibrate=640,
           n_measure=40,tau_forward=t['forward'],n_cycles=16)
    seed=int(identity([signature,t])[:16],16)
    started=time.monotonic()
    with threadpool_limits(limits=1):
        result=project(make_af_sample(2,t['sample_id']).hamiltonian(.15),t['gamma'],
                       EnumeratedGuide(logs,12),**p,seed=seed,
                       tagger=lambda s:production_tags(s,geometry(2)))
    record=dict(signature=signature,task=t,parameters=p,seed=seed,status='completed',
                ground_state_qualified=False,seconds=time.monotonic()-started,
                guide_sha256=gm['npz_sha256'],exact_record_sha256=digest(ep),
                exact_spectrum=em['spectrum'],
                ancestry_alarm=bool(min(result['descendant_ess'])<8 or max(result['largest_lineage'])>.5),
                result={k:np.asarray(v).tolist() for k,v in result.items()})
    write(path,record)
    return dict(id=taskid,seconds=record['seconds'],ancestry_alarm=record['ancestry_alarm'])


def assess(output):
    records=[json.loads(p.read_text()) for p in (output/'runs').glob('*.json')]
    rows=[]
    for g,fw in [(.5,64.),(.5,128.),(.75,32.)]:
        for w in [8192,16384]:
            selected=[r for r in records if (r['task']['gamma'],r['task']['forward'],r['task']['walkers'])==(g,fw,w)]
            if len(selected)!=64:continue
            selected.sort(key=lambda r:(r['task']['sample_id'],r['task']['repeat']))
            raw=np.array([r['result']['connected_spectrum'] for r in selected]).reshape(16,4,4)
            exact=np.array([r['exact_spectrum'] for r in selected]).reshape(16,4,4)[:,0].mean(0)
            rng=np.random.default_rng(20261103)
            ix=rng.integers(4,size=(4000,16,4))
            boot=raw[np.arange(16)[None,:,None],ix].mean(axis=(1,2))
            xb=length(boot,2)/4;finite=xb[np.isfinite(xb)]
            x=float(length(raw.mean(axis=(0,1)),2)/4);target=float(length(exact,2)/4)
            se=float(finite.std(ddof=1)) if len(finite)>1 else None
            bound=abs(x-target)+1.96*se if se is not None and np.isfinite(x) else None
            sb=(abs(raw.mean(axis=(0,1))-exact)+1.96*boot.std(0,ddof=1))/exact
            alarms=sum(r['ancestry_alarm'] for r in selected)
            passes=bound is not None and bound<=.01 and np.all(sb<=.05) and len(finite)==4000 and alarms==0
            rows.append(dict(gamma=g,forward=fw,walkers=w,xi=x*4 if np.isfinite(x) else None,
                             exact_xi=target*4,mc_se=se*4 if se is not None else None,
                             xi_over_side_95_bound=bound,spectral_relative_95_bounds=sb.tolist(),
                             ancestry_alarms=alarms,invalid_bootstrap_fraction=1-len(finite)/4000,
                             status='within_development_tolerances' if passes else 'unresolved'))
    write(output/'review.json',dict(rows=rows,completed_runs=len(records),expected_runs=384,
          qualification='Development only; fixed-disorder repeat bootstrap with approximate pointwise bounds. Held-out and alternative-start checks pending.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(tasks=tasks(),parent_manifest_sha256=digest(a.parent/'manifest.json'),
                  sources=source_hashes(Path(__file__).resolve().parents[1]),
                  equilibration=32.,measurement=2.,cycles=16,branch=.05,
                  purpose='Resolve population and projection bias; no automatic promotion to production',
                  tolerances=dict(xi_over_side=.01,spectra_relative=.05))
    signature=identity(manifest)
    with stage_lock(a.output,'escalation'):
        mp=a.output/'manifest.json'
        if mp.exists():assert json.loads(mp.read_text())==manifest
        else:write(mp,manifest)
        payloads=[(str(a.parent),str(a.output),signature,t) for t in tasks()]
        # Complete one task first to exercise the new population/time before parallel dispatch.
        write(a.output/'status.json',dict(state='pilot',completed=0,total=len(payloads)))
        with ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker) as pool:
            pilot=pool.submit(one,payloads[0]).result()
            print(json.dumps(dict(pilot=pilot)),flush=True)
            futures=[pool.submit(one,t) for t in payloads[1:]]
            for count,f in enumerate(as_completed(futures),2):
                print(json.dumps(dict(completed=count,total=len(payloads),**f.result())),flush=True)
                write(a.output/'status.json',dict(state='running',completed=count,total=len(payloads),updated_unix=time.time()))
        assess(a.output)
        write(a.output/'status.json',dict(state='completed',completed=len(payloads),total=len(payloads),updated_unix=time.time()))


if __name__=='__main__':main()
