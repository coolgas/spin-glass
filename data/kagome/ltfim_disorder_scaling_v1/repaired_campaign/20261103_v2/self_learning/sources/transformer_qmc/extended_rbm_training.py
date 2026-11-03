"""Resume Adam and tempered chains for additional fixed-target RBM training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path

import torch

from .collective_campaign import write
from .correlation_lengths import sha
from .parallel_disorder_scaling import configure_training_worker, fit_name
from . import sampled_simple_guides as sampled
from .projected_overlap_histograms import configure, run_one


def extend(parent, output, additional, device='cuda'):
    """Keep parameters, Adam moments, chains and RNG; reset only stage counters."""
    parent=Path(parent);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    original=saved['signature'];task=dict(original['task'],mode='direct')
    signature=dict(original,task=task,steps=additional)
    for name,digest in signature['sources'].items():assert sha(Path(sampled.__file__).with_name(name))==digest
    assert original['learning_rate']==.001 or device=='cpu'
    provenance=dict(parent=str(parent),parent_sha256=sha(parent),original_signature=original,
        additional_target_updates=additional,continuation='Preserve model, Adam moments, sampler, RNG; target Gamma only',
        source_sha256=sha(Path(__file__)))
    metadata=output/'continuation.json';checkpoint=output/(fit_name(task)+'.pt')
    if metadata.exists():assert json.loads(metadata.read_text())==provenance
    else:
        assert not checkpoint.exists()
        write(metadata,provenance)
    if not checkpoint.exists():
        saved.update(signature=signature,stage=0,step=0,history=[],seconds=0.)
        temporary=checkpoint.with_suffix('.tmp.pt');torch.save(saved,temporary);temporary.replace(checkpoint)
    args=argparse.Namespace(device=device,chains=signature['chains'],temperatures=signature['temperatures'],
        steps=additional,sweeps_per_step=signature['sweeps_per_step'],diagnostic_sweeps=signature['diagnostic_sweeps'],
        learning_rate=signature['learning_rate'],inputs=output/'inputs')
    return sampled.train(task,args,output,float('inf')),checkpoint


def train_seed(parent, output):
    output=Path(output);rows=[]
    for cumulative,increment in [(1200,1200),(3600,2400)]:
        record,parent=extend(parent,output/f'extra_{cumulative}',increment)
        rows.append(dict(extra_updates=cumulative,model=str(parent),record=record))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    parents=[]
    for seed in [0,1]:
        parent=a.campaign/'results/training'/f'L6_s60000_g0p55_rbm_2_anneal_seed{seed}.pt'
        record=json.loads(parent.with_suffix('.json').read_text());assert record['checkpoint_sha256']==sha(parent)
        assert record['status']=='completed'
        parents.append(dict(seed=seed,model=str(parent),record=record,extra_updates=0))
    manifest=dict(parents=parents,extra_updates=[1200,3600],training_workers=2,projector_workers=2,
        source_sha256=sha(Path(__file__)),deadline=None,ground_state_qualified=False,
        rationale='Prespecified disorder60000, both seeds; matched GFMC original vs extended guides, no selection by outcome')
    path=a.output/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:write(path,manifest)
    rows=list(parents);failures=[]
    def status(state,active):
        write(a.output/'status.json',dict(state=state,updated_utc=datetime.now(timezone.utc).isoformat(),active=active,
            completed_extensions=len(rows)-2,failures=failures,ground_state_qualified=False))
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=configure_training_worker) as pool:
        futures={pool.submit(train_seed,r['model'],str(a.output/f"seed{r['seed']}")):r['seed'] for r in parents}
        status('training',list(futures.values()))
        for future in as_completed(futures):
            seed=futures[future]
            try:rows.extend(dict(r,seed=seed) for r in future.result())
            except Exception as error:failures.append(dict(seed=seed,error=repr(error),phase='training'))
            status('training',[s for f,s in futures.items() if not f.done()])
    write(a.output/'training_summary.json',dict(rows=rows,failures=failures))
    specs=[]
    for row in rows:
        if row['extra_updates'] and row['record']['sampling_alarms']:
            failures.append(dict(seed=row['seed'],extra_updates=row['extra_updates'],phase='selection',alarms=row['record']['sampling_alarms']));continue
        for repeat in [0,1]:
            task=dict(L=6,gamma=.55,sample_id=60000,kind='rbm_2',training_seed=row['seed'],
                audit='training_extension',walkers=512,equil=16.,forward=16.,start='guide',delta=.1,cycles=8,repeat=repeat)
            # Separate output per checkpoint preserves identical trajectory RNG
            # seeds across budgets while recording each model hash.
            specs.append((dict(task=task,model=row['model']),str(a.output/'projector'/f"seed{row['seed']}_extra{row['extra_updates']}")))
    status('projecting',len(specs));results=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=configure) as pool:
        futures={pool.submit(run_one,spec,dest):(spec,dest) for spec,dest in specs}
        for future in as_completed(futures):
            spec,dest=futures[future]
            try:results.append(dict(output=dest,result=future.result()))
            except Exception as error:failures.append(dict(spec=spec,output=dest,error=repr(error),phase='projection'))
            status('projecting',sum(not f.done() for f in futures))
    write(a.output/'summary.json',dict(training=rows,projections=results,failures=failures,ground_state_qualified=False))
    status('completed_with_failures' if failures else 'completed',0)


if __name__=='__main__':main()
