"""Finite matched field/forward-time/population controls for L6 ancestry loss."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path

from .collective_campaign import write
from .correlation_lengths import sha
from .projected_overlap_histograms import configure, run_one


def specifications(campaign):
    training=campaign/'results/training'; specs=[]
    for gamma in [.55,.60]:
        name=f'L6_s60000_g{gamma:g}_rbm_2_anneal_seed0'.replace('.','p')
        record=json.loads((training/(name+'.json')).read_text());model=training/(name+'.pt')
        assert record['status']=='completed' and not record['sampling_alarms']
        assert record['checkpoint_sha256']==sha(model)
        assert record['signature']['learning_rate']==.001
        configs=[(512,8.),(512,16.)]
        if gamma==.55:configs.append((2048,16.))
        for walkers,forward in configs:
            for repeat in [0,1]:
                task=dict(L=6,gamma=gamma,sample_id=60000,kind='rbm_2',training_seed=0,
                    audit='ancestry_diagnostic',walkers=walkers,equil=16.,forward=forward,
                    start='guide',delta=.1,cycles=8,repeat=repeat)
                specs.append(dict(task=task,model=str(model),model_sha256=sha(model)))
    return specs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();assert args.workers>0
    specs=specifications(args.campaign)
    assert len(specs)==10 and len({json.dumps(s['task'],sort_keys=True) for s in specs})==10
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),specs=specs,workers=args.workers,
        hypothesis='Increasing Gamma may improve ancestry; current alarms alone do not diagnose a spin-glass phase.',
        fixed='L6, N108, periodic kagome, h=.15, disorder60000, RBM alpha2, seed0, LR .001, equil16, cycles8, two repeats',
        interpretation='Field-specific trained guides differ: this tests operational improvement with field, not a pure Hamiltonian-only causal effect. One disorder is a diagnostic, not a disorder average. Shorter forward time can reduce ancestry loss while increasing projection bias.',
        selection='Disorder60000 and seed0 chosen in advance; seed0 is the completed repaired checkpoint available at both fields. No selection by P(q).',
        acceptance='No automatic production promotion. Compare raw P(q), energy, connected spectrum, repeat spread and per-cycle ancestry across forward times/populations. This limited grid cannot itself certify equilibrium or locate a phase boundary.',
        numerics_sha256=sha(Path(__file__).parent.parent/'qmc_numerics.py'),
        sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},ground_state_qualified=False)
    if args.validate_only:
        print(json.dumps(dict(validated_tasks=len(specs),fields=[.55,.60],workers=args.workers),indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        assert previous['specs']==specs and previous['sources']==manifest['sources']
    else:write(manifest_path,manifest)
    pending=iter(specs);active={};completed=[];failures=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=configure) as pool:
        def fill():
            while len(active)<args.workers:
                spec=next(pending,None)
                if spec is None:break
                assert sha(Path(spec['model']))==spec['model_sha256']
                active[pool.submit(run_one,spec,str(args.output/'projector'))]=spec['task']
        def status(state):
            write(args.output/'status.json',dict(state=state,updated_utc=datetime.now(timezone.utc).isoformat(),
                planned=len(specs),completed=len(completed),active=list(active.values()),failures=failures,
                ground_state_qualified=False))
        fill();status('running')
        while active:
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                task=active.pop(future)
                try:completed.append(future.result())
                except Exception as error:
                    failure=dict(task=task,error=repr(error));failures.append(failure);print(json.dumps(failure),flush=True)
            fill();status('running' if active else ('completed_with_failures' if failures else 'completed'))
    write(args.output/'summary.json',dict(rows=completed,failures=failures,ground_state_qualified=False))


if __name__=='__main__':main()
