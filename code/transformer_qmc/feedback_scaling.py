"""User-prioritized L6/L8/L10 iterative feedback, one bounded worker per size.

L2 is limited to implementation regressions. Each size has a matched baseline
and a strengthened feedback arm, both disorders and both seeds, twelve stints.
Two-stint round-robin dispatch exposes all four contexts early. Complete frozen
measurements at six and twelve stints; no adaptive tags or averages are reused.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import write
from .correlation_lengths import sha
from .feedback_frontier import audit_training,verify_measurement
from .parallel_disorder_scaling import fit_name
from .physics import make_af_sample
from .projected_overlap_histograms import run_one
from .sampled_simple_guides import train,freeze_diagnostics,sampling_alarms
from .self_learning_l6_pilot import copy_verified,measurement_task
from .self_learning_rbm import run as learn
from .simple_guides import RBM


def contexts(L):
    if L not in [6,8,10]:raise ValueError('finite scope: L6/L8/L10')
    return [dict(L=L,gamma=.6,sample_id=sid,kind='rbm_2',seed=seed,mode='anneal')
            for sid in [60000,60001] for seed in [0,1]]


def settings(arm,task,initial,output):
    if arm not in ['baseline','robust']:raise ValueError('unknown feedback arm')
    return argparse.Namespace(output=output,initial=initial,L=task['L'],gamma=.6,
        sample_id=task['sample_id'],seed=task['seed'],stints=12,walkers=1024,
        islands=4 if arm=='robust' else 1,negative_temperatures=16 if arm=='robust' else 1,
        step_rms=.025 if arm=='robust' else 0.,step_max=.15 if arm=='robust' else 0.,
        equil=8.,collect=12.,delta=.1,sample_walkers=64,updates=1000,batch=256,
        gibbs=10,initial_sweeps=400,lr=.001,device='cuda',projection_device='cuda')


def task_for_measurement(task,repeat):
    if task['L']==6:return measurement_task(task,repeat)
    return dict(L=task['L'],gamma=.6,sample_id=task['sample_id'],kind='rbm_2',
        training_seed=task['seed'],audit='self_learning_scaling',walkers=512,equil=16.,
        forward=16.,start='guide',delta=.1,cycles=8,repeat=repeat)


def run(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    regression=json.loads(args.regressions.read_text())
    if not regression['passed'] or regression['skipped'] or regression['tests_run']<18:
        raise ValueError('implementation validation must pass including CUDA')
    for name,digest in regression['source_hashes'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('validation source mismatch: '+name)
    items=contexts(args.L);arms=['robust','baseline']
    manifest=dict(format=1,L=args.L,contexts=items,arms=arms,stints_per_context=12,
        planned_training_contexts=8,planned_stints=96,measurement_milestones=[6,12],
        planned_final_guide_measurements=32,planned_fixed_guide_measurements=8,
        regression_sha256=sha(args.regressions),
        sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        numerics_sha256=sha(Path(__file__).parent.parent/'qmc_numerics.py'),
        scientific_settings={arm:{k:v for k,v in vars(settings(arm,items[0],None,None)).items()
                                  if k not in ['initial','output']} for arm in arms},
        initialization='L6 archived VMC400 controls; L8/L10 repaired VMC400 per annealing stage (1.5,.75,.6), lr=.001.',
        runtime='One sequential worker for this size, no deadline, no automatic retry; three size workers total.',
        scheduling='Two-stint round-robin over contexts and arms; all training contexts finish before frozen measurements.',
        user_priority='L6/L8/L10 receive research compute. L2 limited to short implementation regressions.',
        comparison='Same starting checkpoint, sample count and likelihood updates; robust arm costs more compute and uses 4x walkers.',
        inference='Finite feasibility/reliability campaign; completion, energy or genealogy alone cannot establish convergence.',
        ground_state_qualified=False)
    path=root/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError('scaling manifest mismatch')
    else:write(path,manifest)
    failure_file=root/'failures.json'
    failures=json.loads(failure_file.read_text()) if failure_file.exists() else []
    failed={r['key'] for r in failures};progress={};completed=[]
    def status(state,current=None):
        write(root/'status.json',dict(state=state,current=current,updated_utc=datetime.now(timezone.utc).isoformat(),
            L=args.L,progress=progress,completed_measurement_groups=len(completed),failures=failures,
            deadline=None,ground_state_qualified=False))
    def failure(key,error):
        failures.append(dict(key=key,error=repr(error),traceback=traceback.format_exc(),
            utc=datetime.now(timezone.utc).isoformat()));failed.add(key);write(failure_file,failures)
        print(json.dumps(failures[-1]),flush=True)
    def initial(task):
        name=fit_name(task);folder=root/'controls'/name;folder.mkdir(parents=True,exist_ok=True)
        dest=folder/(name+'.pt')
        if args.L==6:
            source=args.l6/'controls'/name/(name+'.pt')
            for suffix in ['.pt','.json','.npz']:
                copy_verified(source.with_suffix(suffix),dest.with_suffix(suffix))
            for path in sorted((source.parent/'runs').glob('*')):
                copy_verified(path,folder/'runs'/path.name)
        else:
            cfg=argparse.Namespace(chains=256,temperatures=8,steps=400,sweeps_per_step=2,
                diagnostic_sweeps=200,learning_rate=.001,device='cuda')
            status('initial_vmc',task)
            result=train(task,cfg,folder,float('inf'))
            if result['status']!='completed':raise RuntimeError('initial guide did not finish')
        record=json.loads(dest.with_suffix('.json').read_text())
        assert record['task']==task and record['checkpoint_sha256']==sha(dest)
        assert record['npz_sha256']==sha(dest.with_suffix('.npz'))
        return dest
    # Each context gets feedback immediately after its warm start; no L2 campaign.
    for target in range(2,13,2):
        for task in items:
            name=fit_name(task)
            if 'initial/'+name in failed:continue
            try:model_path=initial(task)
            except Exception as error:
                failure('initial/'+name,error);continue
            for arm in arms:
                key=arm+'/'+name;folder=root/'training'/arm/name
                if key in failed:continue
                try:
                    cfg=settings(arm,task,model_path,folder)
                    state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
                    done=state.get('completed_stints',0)
                    if done<target:
                        status('adapting',dict(arm=arm,target_stints=target,**task))
                        learn(cfg,stop_after_stints=target-done)
                    state=json.loads((folder/'status.json').read_text())
                    progress[key]=state['completed_stints']
                    assert progress[key]>=target
                    if target in [6,12]:audit_training(folder,args.L,task['sample_id'])
                    status('adaptation_checkpoint',dict(arm=arm,**task))
                except Exception as error:failure(key,error)
    # Training is prioritized; evaluate retained milestone guides after adaptation.
    for task in items:
        name=fit_name(task)
        if 'initial/'+name in failed:continue
        controls=root/'controls'/name
        measurement_items=[('fixed',0,controls/(name+'.pt'))]
        measurement_items += [(arm,stints,root/'training'/arm/name/f'stint_{stints-1:03d}/model.pt')
                              for arm in arms for stints in [6,12]]
        for arm,stints,model_path in measurement_items:
            key=f'measure/{arm}/{name}/{stints}'
            if key in failed:continue
            if not model_path.exists():
                failure(key,FileNotFoundError(str(model_path)));continue
            folder=root/'measurements'/arm/name/f'stints_{stints:02d}';folder.mkdir(parents=True,exist_ok=True)
            try:
                status('frozen_checks',dict(arm=arm,stints=stints,**task))
                frozen=folder/'frozen.json'
                if frozen.exists():
                    diagnostic=json.loads(frozen.read_text())
                    assert diagnostic['model_sha256']==sha(model_path) and diagnostic['raw_sha256']==sha(frozen.with_suffix('.npz'))
                else:
                    model=RBM(3*args.L**2,2).to('cuda')
                    model.load_state_dict(torch.load(model_path,map_location='cuda',weights_only=False)['model'])
                    checks,raw=freeze_diagnostics(model,make_af_sample(args.L,task['sample_id']).hamiltonian(.15),
                        .6,256,8,200,20261104+task['seed'])
                    np.savez_compressed(frozen.with_suffix('.npz'),**raw)
                    diagnostic=dict(rows=checks,sampling_alarms=sampling_alarms(checks,256),
                        model_sha256=sha(model_path),raw_sha256=sha(frozen.with_suffix('.npz')),ground_state_qualified=False)
                    write(frozen,diagnostic)
                measured=[]
                for repeat in [0,1]:
                    status('projecting',dict(arm=arm,stints=stints,repeat=repeat,**task))
                    if args.L==6 and arm=='fixed':
                        # Exact source-matched control measurements already exist; reuse, hash and reconstruct.
                        from .production_protocol import identity
                        destination=controls/'runs'/(identity(task_for_measurement(task,repeat))[:24]+'.json')
                        record=json.loads(destination.read_text());verification_folder=controls
                    else:
                        record=run_one(dict(model=str(model_path),task=task_for_measurement(task,repeat),
                                            projection_device='cuda'),folder/'projector')
                        verification_folder=folder/'projector'
                    assert record['provenance']['model_sha256']==sha(model_path)
                    record['verification']=verify_measurement(record,verification_folder);measured.append(record)
                result=dict(arm=arm,stints=stints,task=task,model_sha256=sha(model_path),frozen=diagnostic,
                            measurements=measured,ground_state_qualified=False)
                write(folder/'result.json',result);completed.append(result)
                write(root/'measurement_results.json',dict(rows=completed,ground_state_qualified=False))
            except Exception as error:failure(key,error)
    status('completed_with_failures' if failures else 'completed_unqualified')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--L',type=int,choices=[6,8,10],required=True)
    for name in ['output','l6','regressions']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:
        with threadpool_limits(limits=1):run(args)
    except Exception as error:
        write(args.output/'fatal_failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
