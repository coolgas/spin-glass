"""Concurrent sampled training with the original pilot estimators and provenance.

Each task owns its model, optimizer, RNG, and checkpoint. Only the live training
status path is redirected per worker. The scientific source snapshot is reused.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch

from .collective_campaign import write
from .correlation_lengths import sha
from .disorder_scaling import project_task,reduce_results
from .production_protocol import identity
from . import sampled_simple_guides


def configure_training_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def training_worker(task,settings,root,deadline):
    root=Path(root)
    def task_write(path,record):
        path=Path(path)
        if path==root/'status.json':path=root/'worker_status'/f'{os.getpid()}.json'
        write(path,record)
    sampled_simple_guides.write=task_write
    return sampled_simple_guides.train(task,argparse.Namespace(**settings),root,deadline)


def fit_name(task):
    return 'L{L}_s{sample_id}_g{gamma:g}_{kind}_{mode}_seed{seed}'.format(**task).replace('.','p')


def context(task):
    return tuple(task[k] for k in ['L','gamma','sample_id','kind'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trainers',type=int,default=4)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--deadline-unix',type=float,required=True)
    a=p.parse_args()
    if a.trainers<1 or a.workers<1:raise ValueError('Positive worker counts required')
    manifest=json.loads((a.output/'manifest.json').read_text())
    for name,digest in manifest['sources'].items():
        assert sha(Path(__file__).with_name(name))==digest, name
    if manifest['sizes']!=[4,6]:raise ValueError('This continuation is restricted to the authorized L4/L6 pilot')
    deadline=time.monotonic()+max(0,a.deadline_unix-time.time())
    fields_by_size=manifest.get('fields_by_size',{})
    tasks=[dict(L=L,gamma=g,sample_id=sid,kind=kind,seed=seed,mode='anneal')
           for L in manifest['sizes'] for g in fields_by_size.get(str(L),manifest['fields'])
           for sid in manifest['disorders'] for kind in manifest['kinds'] for seed in manifest['seeds']]
    completed_fits={};pending=[];submitted_contexts=set();projectors={};failures=[]
    training=a.output/'training';ctx=mp.get_context('spawn')
    training.mkdir(parents=True,exist_ok=True)
    (a.output/'inputs').mkdir(exist_ok=True)
    # CUDA MPS is inherited only by the training pool. Projectors stay on CPU.
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=ctx) as cpu_pool, \
         ProcessPoolExecutor(max_workers=a.trainers,mp_context=ctx,initializer=configure_training_worker) as gpu_pool:
        def launch_context(key):
            if key in submitted_contexts:return
            candidates=[completed_fits.get(key+(seed,)) for seed in manifest['seeds']]
            if any(r is None for r in candidates):return
            best=min(candidates,key=lambda r:np.mean([d['energy'] for d in r['diagnostics']]))
            name=fit_name(best['task']);model_path=training/(name+'.pt')
            assert sha(model_path)==best['checkpoint_sha256']
            write(a.output/'selections'/(name+'.json'),dict(selected=best['task'],
                  candidate_energies=[dict(seed=r['task']['seed'],energy=float(np.mean([d['energy'] for d in r['diagnostics']]))) for r in candidates],
                  checkpoint_sha256=sha(model_path)))
            L,g,sid,kind=key
            configs=[('main',512,16.,16.,'guide')]
            if sid in manifest['audit_ids']:
                configs += [('population',1024,16.,16.,'guide'),('time',512,32.,32.,'guide'),('start',512,16.,16.,'uniform')]
            for audit,w,eq,fw,start in configs:
                for repeat in [0,1]:
                    task=dict(L=L,gamma=g,sample_id=sid,kind=kind,training_seed=best['task']['seed'],
                              audit=audit,walkers=w,equil=eq,forward=fw,start=start,delta=.1,cycles=8,repeat=repeat)
                    projectors[cpu_pool.submit(project_task,task,str(model_path),str(a.output/'projector'))]=task
            submitted_contexts.add(key)
        for task in tasks:
            record=training/(fit_name(task)+'.json')
            if record.exists():
                r=json.loads(record.read_text());assert r['status']=='completed' and r['task']==task
                assert sha(training/(fit_name(task)+'.pt'))==r['checkpoint_sha256']
                assert sha(training/(fit_name(task)+'.npz'))==r['npz_sha256']
                signature=r['signature'];g=task['gamma']
                stages=len([v for v in sorted(set([1.5,.75,g]),reverse=True) if v>=g])
                assert all(signature[k]==v for k,v in dict(chains=256,temperatures=8,steps=1200//stages,
                           sweeps_per_step=2,diagnostic_sweeps=200).items())
                for name,digest in signature['sources'].items():assert sha(Path(__file__).with_name(name))==digest
                completed_fits[context(task)+(task['seed'],)]=r
            else:pending.append(task)
        for task in tasks:launch_context(context(task))
        active={};iterator=iter(pending);exhausted=False;paused=False
        def fill():
            nonlocal exhausted,paused
            while len(active)<a.trainers and not exhausted:
                if time.monotonic()>=deadline:paused=True;return
                task=next(iterator,None)
                if task is None:exhausted=True;return
                g=task['gamma'];stages=len([v for v in sorted(set([1.5,.75,g]),reverse=True) if v>=g])
                settings=dict(device='cuda',chains=256,temperatures=8,steps=1200//stages,
                              sweeps_per_step=2,diagnostic_sweeps=200,inputs=a.output/'inputs')
                active[gpu_pool.submit(training_worker,task,settings,str(training),deadline)]=task
        def status(state):
            write(a.output/'parallel_status.json',dict(state=state,completed_fits=len(completed_fits),
                planned_fits=len(tasks),active_training=list(active.values()),trainers=a.trainers,
                queued_or_running_projectors=len(projectors),submitted_contexts=len(submitted_contexts),
                completed_runs=len(list((a.output/'projector/runs').glob('*.json'))),
                deadline_unix=a.deadline_unix,failures=failures,updated_unix=time.time()))
        fill();status('running')
        while active or projectors:
            done,_=wait(list(active)+list(projectors),timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                is_training=future in active
                task=(active if is_training else projectors).pop(future)
                try:r=future.result()
                except Exception as error:
                    failure=dict(task=task,error=repr(error),training=is_training)
                    failures.append(failure);write(a.output/('failure_parallel_'+identity(task)[:16]+'.json'),failure)
                    # Preserve failures and drain other tasks; never silently replace a failed sample.
                    print(json.dumps(failure),flush=True);continue
                if is_training:
                    if r['status']=='completed':
                        completed_fits[context(task)+(task['seed'],)]=r;launch_context(context(task))
                    else:paused=True
            fill();reduce_results(a.output)
            status('running' if active else 'draining_projectors')
        from .report_disorder_scaling import report
        report(a.output)
        state='failed' if failures else ('paused_training_budget' if paused else 'completed')
        status(state)
        write(a.output/'status.json',dict(state=state,completed_fits=len(completed_fits),
              completed_runs=len(list((a.output/'projector/runs').glob('*.json'))),executor='parallel_disorder_scaling'))
    if failures:raise RuntimeError(f'{len(failures)} tasks failed; see retained failure records')


if __name__=='__main__':main()
