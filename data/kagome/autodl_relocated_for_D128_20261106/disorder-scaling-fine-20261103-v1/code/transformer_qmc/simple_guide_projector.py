"""Select simple guides by energy alone, then run matched independent GFMC.

All optimization seeds remain in the screen archive. The selected guide is
frozen before projector measurements. This is an L2 diagnostic, with exact
guide-probability initialization available through enumerated lookup.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from . import collective_campaign as campaign
from .correlation_lengths import sha
from .independent_replica_projector import spectrum_from_moments
from .production_protocol import geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--repeats',type=int,default=4)
    args=p.parse_args()
    screen=json.loads((args.screen/'review.json').read_text())['rows']
    status=json.loads((args.screen/'status.json').read_text())
    assert status['state']=='completed', 'Freeze completed training before selection'
    selections=[]; work=[]
    for kind in sorted({r['kind'] for r in screen}):
        output=args.output/kind
        inputs=output/'inputs'
        (inputs/'guides').mkdir(parents=True,exist_ok=True)
        for gamma,sid in campaign.CONTEXTS:
            candidates=[r for r in screen if (r['kind'],r['gamma'],r['sample_id'])==(kind,gamma,sid)]
            selected=min(candidates,key=lambda r:r['energy'])
            name=f'{campaign.label(gamma,sid)}_{kind}_seed{selected["seed"]}'
            parent=args.screen/(name+'.npz')
            assert sha(parent)==selected['npz_sha256']
            with np.load(parent) as data:
                logs=data['logs'].copy()
            path=inputs/'guides'/(campaign.label(gamma,sid)+'.npz')
            if path.exists():
                with np.load(path) as data:np.testing.assert_array_equal(data['logs'],logs)
            else:
                np.savez_compressed(path,logs=logs,original_logs=logs)
            selections.append(dict(kind=kind,gamma=gamma,sample_id=sid,selected_seed=selected['seed'],
                energy=selected['energy'],parent_sha256=sha(parent),guide_sha256=sha(path),
                criterion='minimum variational energy, no exact labels',all_seeds=[r['seed'] for r in candidates]))
            for method in ['ct','triangle']:
                for repeat in range(args.repeats):
                    task=dict(L=2,h=.15,gamma=gamma,sample_id=sid,guide='corrected',method=method,
                        delta=.1,epsilon=0.,walkers=1024,equil=32.,forward=32.,cycles=16,
                        start='guide',repeat=repeat,audit='main')
                    work.append((task,output,inputs))
    manifest=dict(selections=selections,source_sha256=campaign.source_hashes(),
        driver_sha256=sha(Path(__file__)),training_manifest_sha256=sha(args.screen/'manifest.json'),
        settings=dict(repeats=args.repeats,workers=args.workers),
        scope='Full-basis-trained L2 guides; enumerated initialization; four independent GFMC populations')
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():assert json.loads(manifest_path.read_text())==manifest
    else:campaign.write(manifest_path,manifest)
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=iter(work);futures={}
        def submit(item):
            task,output,inputs=item
            futures[pool.submit(campaign.run_one,task,output,inputs,3600)]=item
        for _ in range(args.workers):
            item=next(pending,None)
            if item is not None:submit(item)
        from concurrent.futures import wait,FIRST_COMPLETED
        while futures:
            ready,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in ready:
                task,output,inputs=futures.pop(future)
                record=future.result()
                assert record['status']=='completed'
                records.append((output.name,record))
                campaign.write(args.output/'status.json',dict(state='running',completed=len(records),total=len(work)))
                print(json.dumps(dict(kind=output.name,task=task,seconds=record['seconds'],completed=len(records))),flush=True)
                item=next(pending,None)
                if item is not None:submit(item)
    rows=[];rng=np.random.default_rng(781)
    for selection in selections:
        kind,gamma,sid=(selection[k] for k in ['kind','gamma','sample_id'])
        truth=np.array(next(r['exact_spectrum'] for r in screen if (r['gamma'],r['sample_id'])==(gamma,sid)))
        for method in ['ct','triangle']:
            rs=[r for k,r in records if k==kind and (r['task']['gamma'],r['task']['sample_id'],r['task']['method'])==(gamma,sid,method)]
            raw=np.array([r['spectrum'] for r in rs]);mean=raw.mean(0)
            # Validate both raw hashes and independently reconstructed moments.
            for r in rs:
                from .production_protocol import identity
                path=args.output/kind/'runs'/(identity(r['task'])[:24]+'.npz')
                assert sha(path)==r['npz_sha256']
                with np.load(path) as data:
                    measured=spectrum_from_moments(data['magnetization_cycles'].mean(1),data['pair_cycles'].mean(1),geometry(2))
                np.testing.assert_allclose(measured,r['spectrum'],atol=1e-12)
            boot=raw[rng.integers(len(raw),size=(2000,len(raw)))].mean(1)
            rows.append(dict(kind=kind,gamma=gamma,sample_id=sid,method=method,repeats=len(rs),
                spectrum=mean.tolist(),relative_error=((mean-truth)/truth).tolist(),
                max_relative_error=float(np.max(abs(mean-truth)/truth)),
                spectrum_se=boot.std(0,ddof=1).tolist(),
                alarms=sum(r['ancestry_alarm'] for r in rs),
                minimum_descendant_ess=min(r['minimum_descendant_ess'] for r in rs),
                total_seconds=sum(r['seconds'] for r in rs),ground_state_qualified=False))
    campaign.write(args.output/'review.json',dict(rows=rows,scope='Selected L2 guides; whole-run bootstrap; no disorder-ensemble claim'))
    campaign.write(args.output/'status.json',dict(state='completed',completed=len(records)))


if __name__=='__main__':main()
