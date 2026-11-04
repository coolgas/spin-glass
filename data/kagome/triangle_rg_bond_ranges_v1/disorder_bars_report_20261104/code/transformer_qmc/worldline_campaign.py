"""Bounded worldline validation and larger-size cost pilots; no exact data loaded."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time

import numpy as np
from .physics import make_af_sample
from .production_protocol import identity
from .correlation_lengths import sha
from .correlation_remedy_campaign import write
from .worldline_qmc import run


def tasks():
    result=[]
    for g,sid in [(.25,50025),(.25,50059),(.5,50043),(.75,50004)]:
        for beta,dt in [(16.,.125),(32.,.125),(32.,.0625)]:
            for replica in range(4):
                result.append(dict(kind='validation',L=2,gamma=g,sample_id=sid,beta=beta,dt=dt,
                    temperatures=24,warmup=4000,sweeps=16000,thin=4,blocks=32,replica=replica,
                    start=[0,1,-1,0][replica]))
    for L in [4,6,10]:
        for replica in range(4):
            result.append(dict(kind='cost_only',L=L,gamma=.75,sample_id=50004,beta=16.,dt=.125,
                temperatures=24,warmup=100,sweeps=400,thin=4,blocks=8,replica=replica,
                start=[0,1,-1,0][replica]))
    return result


def run_one(task,output):
    out=Path(output);key=identity(task)[:24];p=out/'runs'/(key+'.json')
    if p.exists():
        r=json.loads(p.read_text());assert r['task']==task and sha(p.with_suffix('.npz'))==r['npz_sha256'];return r
    seed=int(identity(['worldline-v1',task])[:8],16)
    ham=make_af_sample(task['L'],task['sample_id']).hamiltonian(.15)
    params={k:task[k] for k in ['beta','dt','temperatures','warmup','sweeps','thin','blocks','start']}
    start=time.monotonic();data=run(ham,task['gamma'],seed=seed,**params);seconds=time.monotonic()-start
    p.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(p.with_suffix('.npz'),**data)
    record=dict(task=task,seed=seed,h=.15,seconds=seconds,npz_sha256=sha(p.with_suffix('.npz')),
        M=int(data['M']),worldline_bytes=task['temperatures']*ham.n_sites*int(data['M']),
        energy=float(data['energy_blocks'].mean()),round_trips=int(data['round_trips'].sum()),
        minimum_exchange_acceptance=float(np.min(data['exchange_accepted']/np.maximum(1,data['exchange_attempts']))),
        ground_state_qualified=False)
    write(p,record);return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    # Fail instead of accidentally launching the slow Python fallback in production.
    import numba
    ts=tasks();root=Path(__file__).parent
    manifest=dict(tasks=ts,numba_version=numba.__version__,h=.15,
        source_sha256={n:sha(root/n) for n in ['worldline_qmc.py','worldline_campaign.py','physics.py']},
        method='Discrete imaginary time, temporal FK heat bath clusters, beta replica exchange',
        state_enumeration=False,guide=False,scope='Development validation, not a qualified phase scan')
    mp=a.output/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:write(mp,manifest)
    # Compile once before workers fork; compile overhead excluded from their timings.
    run(make_af_sample(2,50004).hamiltonian(.15),.75,beta=1.,dt=.25,temperatures=2,
        warmup=2,sweeps=8,thin=1,blocks=2,seed=0)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_one,t,str(a.output)) for t in ts]
        for count,future in enumerate(as_completed(futures),1):
            r=future.result();print(json.dumps(dict(completed=count,total=len(ts),result=r)),flush=True)
            write(a.output/'status.json',dict(state='running',completed=count,total=len(ts)))
    write(a.output/'status.json',dict(state='completed',completed=len(ts),total=len(ts)))


if __name__=='__main__':main()
