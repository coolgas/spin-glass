"""Uniform seven-field, 64-disorder corrected-guide L2 benchmark and audits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .correlation_lengths import ROOT, sha
from .correlation_remedy_campaign import write
from .frozen_scan import SELECTED_SHA256
from .independent_replica_projector import project_independent
from .model import EnumeratedGuide
from .physics import make_af_sample
from .production_protocol import geometry, identity
from .validate_correlation_remedy import train, label

FIELDS=[.25,.5,.75,1.,1.5,2.,3.]
IDS=list(range(50000,50064))
AUDIT_IDS=list(range(50024,50032))
SETTINGS={
    'production':dict(n_walkers=1024,tau_block=.1,n_equilibrate=320,n_measure=10,
                      tau_forward=32.,n_cycles=16,start='guide'),
    'population':dict(n_walkers=2048,tau_block=.1,n_equilibrate=320,n_measure=10,
                      tau_forward=32.,n_cycles=16,start='guide'),
    'time':dict(n_walkers=1024,tau_block=.1,n_equilibrate=640,n_measure=10,
                tau_forward=64.,n_cycles=16,start='guide'),
}


def seed_for(g,s,k,r):return int(identity(['expanded-corrected-v1',g,s,k,r])[:16],16)


def run_one(task):
    output,signature,g,s,k,r=task;output=Path(output)
    p=output/'runs'/(label(g,s)+f'_{k}_r{r}.json');seed=seed_for(g,s,k,r)
    gp=output/'guides'/(label(g,s)+'.json');gm=json.loads(gp.read_text())
    assert gm['signature']==signature and sha(gp.with_suffix('.npz'))==gm['npz_sha256']
    if p.exists():
        old=json.loads(p.read_text());assert old['signature']==signature and old['seed']==seed
        assert old['parameters']==SETTINGS[k] and old['guide_sha256']==gm['npz_sha256']
        return dict(gamma=g,sample_id=s,setting=k,repeat=r,reused=True)
    with np.load(gp.with_suffix('.npz')) as a:guide=EnumeratedGuide(a['logs'].copy(),12)
    started=time.monotonic()
    with threadpool_limits(limits=1):
        result=project_independent(make_af_sample(2,s).hamiltonian(.15),g,guide,geometry(2),
                                   **SETTINGS[k],seed=seed)
    write(p,dict(signature=signature,L=2,h=.15,gamma=g,sample_id=s,setting=k,repeat=r,
        seed=seed,parameters=SETTINGS[k],guide_sha256=gm['npz_sha256'],result=result,
        exact_spectrum=gm['exact_spectrum'],seconds=time.monotonic()-started,ground_state_qualified=False))
    return dict(gamma=g,sample_id=s,setting=k,repeat=r,seconds=time.monotonic()-started,
                ancestry_alarm=result['ancestry_alarm'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=18);parser.add_argument('--stage',choices=['all','guides','production','audit'],default='all')
    a=parser.parse_args();assert sha(a.checkpoint)==SELECTED_SHA256
    names=['expanded_correlation_campaign.py','validate_correlation_remedy.py','refine_correlation_guide.py',
           'independent_replica_projector.py','correlation_remedy_campaign.py','projector.py','model.py',
           'physics.py','production_protocol.py','frozen_scan.py','diagnose_shared.py','symmetric.py']
    sources=[Path(__file__).with_name(n) for n in names]+[ROOT/'code/qmc_numerics.py']
    manifest=dict(schema=1,L=2,h=.15,gamma=FIELDS,sample_ids=IDS,audit_ids=AUDIT_IDS,repeats=4,
        settings=SETTINGS,degree=4,maxiter=2000,exact_labels_used=False,checkpoint_sha256=SELECTED_SHA256,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        uncertainty='Whole disorders and repeats resampled; numerical tests conditional on fixed disorders',
        tolerances=dict(xi_over_side_absolute=.01,spectra_relative=.05),
        scope='L2 only: guide optimization enumerates all 4096 states; scalability is unproven')
    mp=a.output/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Changed campaign identity'
    else:write(mp,manifest)
    signature=identity(manifest)
    for stage in (['guides','production','audit'] if a.stage=='all' else [a.stage]):
        if stage=='guides':
            tasks=[(str(a.output),str(a.checkpoint),signature,g,s) for g in FIELDS for s in IDS];fn=train
        else:
            settings=['production'] if stage=='production' else ['population','time']
            ids=IDS if stage=='production' else AUDIT_IDS
            # Interleave fields so progress covers the whole grid.
            tasks=[(str(a.output),signature,g,s,k,r) for k in settings for s in ids for g in FIELDS for r in range(4)];fn=run_one
        sp=a.output/(stage+'_status.json');write(sp,dict(state='running',completed=0,total=len(tasks),updated_unix=time.time()))
        try:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                futures=[pool.submit(fn,t) for t in tasks]
                for count,future in enumerate(as_completed(futures),1):
                    row=future.result()
                    print(json.dumps(dict(stage=stage,completed=count,total=len(tasks),result=row)),flush=True)
                    write(sp,dict(state='running',completed=count,total=len(tasks),updated_unix=time.time()))
            write(sp,dict(state='completed',completed=len(tasks),total=len(tasks),updated_unix=time.time()))
        except Exception as error:
            write(sp,dict(state='failed',error=repr(error),updated_unix=time.time()));raise


if __name__=='__main__':main()
