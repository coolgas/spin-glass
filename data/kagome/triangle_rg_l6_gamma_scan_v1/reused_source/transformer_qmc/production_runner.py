"""Persistent staged production queue; qualification results are never final data.

The source and manifest are immutable once initialized. Checkpoints save the
full projector/RNG state at branch boundaries. The soft budget stops dispatch
and checkpoints workers without powering down the host.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from collections import Counter

import numpy as np
import torch

from .compare_pure_gfmc import BatchedConstantGuide
from .direct_gfmc_pilot import CachedDirectGuide
from .experiment import write_json, jsonable
from .frozen_scan import SELECTED_SHA256, key, model_from, stage_lock
from .model import EnumeratedGuide, TransformerGuide
from .physics import make_af_sample
from .production_protocol import (ANCHORS, SIZES, DEVELOPMENT_IDS, GAMMAS, FIELDS,
                                  design, identity, seed_for, source_hashes, geometry, production_tags)
from .projector import project


_MODEL = {}
_STOP = False


class PauseRun(Exception):
    pass


def stop_requested(signum, frame):
    global _STOP
    _STOP = True


def init_worker():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    signal.signal(signal.SIGTERM, stop_requested)


class TimedGuide(CachedDirectGuide):
    def __init__(self, model, chunk=2048):
        super().__init__(model)
        self.chunk=chunk
        self.query_seconds=0.
        self.sample_seconds=0.
        self.events=0

    def all_flip_log_ratios(self, states):
        start=time.monotonic()
        # Base adapter uses the validated 2048 chunk; other chunks are benchmark-only.
        if self.chunk == 2048:
            value=super().all_flip_log_ratios(states)
        else:
            value=self.model.all_flip_log_ratios(self._states(states),chunk=self.chunk).double().cpu().numpy()
            self.requested+=len(states);self.evaluated+=len(states)
        self.query_seconds+=time.monotonic()-start
        return value

    def sample(self,count,seed):
        start=time.monotonic();value=super().sample(count,seed)
        self.sample_seconds+=time.monotonic()-start
        return value


class ProbeTable(EnumeratedGuide):
    """Cache exactly the direct inference log ratios for a given precision."""
    def __init__(self,logs,ratios):
        super().__init__(logs,12);self.ratios=np.asarray(ratios,dtype=np.float64)

    def all_flip_log_ratios(self,states):
        return self.ratios[self.indexes(states)]


def settings(walkers=32,equilibration=.04,measurement=.08,forward=.04,cycles=2,branch=.02):
    for duration in (equilibration,measurement):
        if abs(duration/branch-round(duration/branch))>1e-8:
            raise ValueError('times must be exact multiples of branch interval')
    return dict(n_walkers=walkers,n_replicas=4,tau_block=branch,
                n_equilibrate=round(equilibration/branch),n_measure=round(measurement/branch),
                tau_forward=forward,n_cycles=cycles)


def make_task(stage,L,sid,gamma,h,method,precision,repeat,parameters,initial='guide',**extra):
    t=dict(stage=stage,L=L,sample_id=sid,gamma=gamma,h=h,method=method,precision=precision,
           repeat=repeat,parameters=parameters,initial=initial,**extra)
    t['seed']=seed_for(t);t['id']=identity(t)[:24]
    return t


def build_tasks():
    tasks=[]
    # Full-grid precision comparison on the same eight exact development disorders.
    small=settings(512,.4,.2,.2,4,.05)
    for r in range(4):
        for sid,g,h in itertools.product(range(50000,50008),GAMMAS,FIELDS):
            for precision in ('float64','float32'):
                tasks.append(make_task('precision_ensemble',2,sid,g,h,'lookup_direct',precision,r,small))
    # Uniform seven-size first wave, deliberately short and labelled execution-only.
    for r in range(4):
        for sid in DEVELOPMENT_IDS[:4]:
            for g,h in itertools.product((.5,1.,2.),(0.,.15,.3)):
                for L in SIZES:
                    for precision in ('float32','float64'):
                        tasks.append(make_task('direct_feasibility',L,sid,g,h,'network',precision,r,settings()))
                    tasks.append(make_task('direct_feasibility',L,sid,g,h,'constant','float64',r,settings()))
    # Prespecified one-parameter growth; every size receives identical candidate settings.
    ladder=[('population64',settings(64)),('population128',settings(128)),
            ('time2',settings(64,.08,.16,.08)),('time4',settings(64,.16,.32,.16)),
            ('cadence',settings(64,.08,.16,.08,2,.01)),
            ('combined',settings(128,.16,.32,.16,4,.01)),
            ('alternative_start',settings(64,.08,.16,.08))]
    for label,parameter in ladder:
        for r in range(4):
            for sid in DEVELOPMENT_IDS[:4]:
                for g,h in itertools.product((.5,1.,2.),(0.,.15,.3)):
                    for L in SIZES:
                        for precision in ('float32','float64'):
                            tasks.append(make_task('feasibility_ladder',L,sid,g,h,'network',precision,r,parameter,
                                                   initial='uniform' if label=='alternative_start' else 'guide',ladder=label))
                        tasks.append(make_task('feasibility_ladder',L,sid,g,h,'constant','float64',r,parameter,
                                               initial='uniform' if label=='alternative_start' else 'guide',ladder=label))
    if len({t['id'] for t in tasks})!=len(tasks) or len({t['seed'] for t in tasks})!=len(tasks):
        raise RuntimeError('task or seed collision')
    return tasks


def initialize(root,checkpoint,checks):
    root.mkdir(parents=True,exist_ok=True)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=SELECTED_SHA256:
        raise RuntimeError('checkpoint identity mismatch')
    sources=source_hashes(Path(__file__).resolve().parents[1])
    evidence=[json.loads((checks/('manifest_'+stage+'.json')).read_text()) for stage in ('gpu','exact')]
    if evidence[0]!=evidence[1]:raise RuntimeError('prerequisite source/checkpoint identities disagree')
    for name in ('projector.py','physics.py','production_protocol.py','model.py','symmetric.py','direct_gfmc_pilot.py'):
        keyname='transformer_qmc/'+name
        if sources[keyname]!=evidence[0]['source_sha256'][keyname]:
            raise RuntimeError(f'qualified implementation changed: {keyname}')
    if evidence[0]['checkpoint_sha256']!=SELECTED_SHA256:raise RuntimeError('prerequisite guide mismatch')
    reservation=json.loads((checks/'remote_reservation_audit.json').read_text())
    if reservation['collisions']:raise RuntimeError('disorder reservation collision')
    tasks=build_tasks()
    manifest=dict(design=design(),checkpoint_sha256=SELECTED_SHA256,
                  source_sha256=sources,task_count=len(tasks),tasks_sha256=identity(tasks),
                  checks_root=str(checks.resolve()),checks_signature=evidence[0]['signature'],schema=1,
                  reservation_audit_sha256=identity(reservation),
                  precision_policy='float32 candidate and float64 reference; final settings await qualification',
                  queue_policy='GPU size-interleaved; CPU exact/ensemble and constant propagation concurrent',
                  budget_policy='12h initial process window; checkpoint and pause; instance remains running')
    manifest['signature']=identity(manifest)
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('production manifest identity mismatch; use a new campaign snapshot')
    write_json(path,manifest)
    taskfile=root/'tasks.json'
    if taskfile.exists() and json.loads(taskfile.read_text())!=tasks:
        raise RuntimeError('task manifest mismatch')
    if not taskfile.exists():write_json(taskfile,tasks)
    for name in ('results','checkpoints','heartbeats','failures','benchmark','disorder'):
        (root/name).mkdir(exist_ok=True)
    coverage=[]
    for L in SIZES:
        for g,h in itertools.product(GAMMAS,FIELDS):
            coverage.append(dict(L=L,gamma=g,h=h,scope='full_grid' if L in ANCHORS else 'cut_selection_pending',
                                 state='awaiting_qualification',required_final_disorders=64,required_repeats=4))
    if not (root/'coverage.json').exists():
        write_json(root/'coverage.json',dict(final_data_started=False,contexts=coverage))
    return manifest,tasks


def verified_result(root,task,signature):
    path=root/'results'/(task['id']+'.json')
    if not path.exists():return False
    value=json.loads(path.read_text())
    if value['signature']!=signature or value['task']!=task or value['state']!='completed':
        raise RuntimeError(f'result identity mismatch: {path}')
    return True


def execute_task(payload):
    root,checkpoint,checks,signature,task,deadline=payload
    root=Path(root);checkpoint=Path(checkpoint);checks=Path(checks)
    if verified_result(root,task,signature):return dict(state='completed',id=task['id'],cached=True)
    started=time.monotonic();wall_started=time.time();last_save=0.;prior_seconds=0.
    ckpath=root/'checkpoints'/(task['id']+'.json.gz')
    heartbeat=root/'heartbeats'/(task['id']+'.json')
    record_path=root/'results'/(task['id']+'.json')
    try:
        sample=make_af_sample(task['L'],task['sample_id'],periodic=True)
        ham=sample.hamiltonian(task['h']);lattice=geometry(task['L'])
        if ham.n_sites!=3*task['L']**2 or len(sample.bonds)!=2*ham.n_sites:
            raise RuntimeError('PBC connectivity mismatch')
        degree=np.bincount(sample.bonds.ravel(),minlength=ham.n_sites)
        if not np.all(degree==4):raise RuntimeError('PBC degree mismatch')
        disorder_digest=hashlib.sha256(sample.bonds.tobytes()+sample.J.tobytes()).hexdigest()
        prior=None
        if ckpath.exists():
            with gzip.open(ckpath,'rt') as handle:prior=json.load(handle)
            if prior['signature']!=signature or prior['task']!=task or prior['disorder_sha256']!=disorder_digest:
                raise RuntimeError('resume identity mismatch')
            prior_seconds=prior['seconds']
        guide=None
        if task['method']=='network':
            cachekey=(str(checkpoint),task['precision'])
            if cachekey not in _MODEL:
                m=model_from(checkpoint,'cuda')
                if task['precision']=='float32':m=m.float().eval()
                _MODEL[cachekey]=m
            model=_MODEL[cachekey]
            model.set_hamiltonian(ham,task['gamma'])
            guide=TimedGuide(model)
            torch.cuda.reset_peak_memory_stats()
        elif task['method']=='constant':
            guide=BatchedConstantGuide(ham.n_sites)
        elif task['method']=='lookup_direct':
            name=key(2,task['sample_id'],task['gamma'],task['h'],1.)
            path=checks/'direct_checks'/(name+'.json')
            data=json.loads(path.read_text());npz=path.with_suffix('.npz')
            if data['status']!='completed' or hashlib.sha256(npz.read_bytes()).hexdigest()!=data['npz_sha256']:
                raise RuntimeError('direct check table mismatch')
            with np.load(npz) as table:
                suffix='32' if task['precision']=='float32' else '64'
                guide=ProbeTable(table['logs'+suffix],table['ratios'+suffix])
        else:raise ValueError(task['method'])
        if task['initial']=='uniform':
            guide.sample=BatchedConstantGuide(ham.n_sites).sample

        def save(state):
            nonlocal last_save
            now=time.time()
            paused=now>=deadline or _STOP or (root/'PAUSE.json').exists()
            if now-last_save<60 and not paused:return
            temporary=ckpath.with_suffix('.tmp.gz')
            elapsed=prior_seconds+time.monotonic()-started
            with gzip.open(temporary,'wt',compresslevel=1) as handle:
                json.dump(jsonable(dict(signature=signature,task=task,disorder_sha256=disorder_digest,
                                        seconds=elapsed,projector=state)),handle,allow_nan=False)
            temporary.replace(ckpath);last_save=now
            write_json(heartbeat,dict(id=task['id'],pid=os.getpid(),updated_unix=now,
                                     state='paused' if paused else 'running',seconds=elapsed,
                                     step=state['next_step'],cycle=state['cycle'],forward_elapsed=state['elapsed'],
                                     L=task['L'],precision=task['precision'],stage=task['stage']))
            if paused:raise PauseRun('process budget or termination requested')
        result=project(ham,task['gamma'],guide,**task['parameters'],seed=task['seed'],
                       resume=None if prior is None else prior['projector'],
                       checkpoint=save,tagger=lambda spins:production_tags(spins,lattice))
        for value in result.values():
            if not np.isfinite(np.asarray(value)).all():raise FloatingPointError('nonfinite projector result')
        result['energy_per_site']=result['energy']/ham.n_sites
        result['chi_connected_per_site']=result['chi_connected']/ham.n_sites
        result['chi_sg_equal_time']=ham.n_sites*result['q2']
        result['q_ea_finite']=result['q']
        elapsed=prior_seconds+time.monotonic()-started
        metrics=dict(seconds=elapsed,segment_seconds=time.monotonic()-started,
                     sample_seconds=getattr(guide,'sample_seconds',None),
                     query_seconds=getattr(guide,'query_seconds',None),
                     requested_states=getattr(guide,'requested',None),evaluated_states=getattr(guide,'evaluated',None),
                     peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if task['method']=='network' else 0.)
        record=dict(state='completed',signature=signature,task=task,result=result,metrics=metrics,
                    disorder_sha256=disorder_digest,checkpoint_sha256=SELECTED_SHA256,
                    ground_state_qualified=False,final_data=False,started_unix=wall_started,finished_unix=time.time(),
                    minimum_descendant_ess=float(min(result['descendant_ess'])),
                    max_lineage=float(max(result['largest_lineage'])),
                    ancestry_alarm=bool(min(result['descendant_ess'])<8 or max(result['largest_lineage'])>.5))
        write_json(record_path,record)
        write_json(heartbeat,dict(id=task['id'],state='completed',updated_unix=time.time(),seconds=elapsed))
        # Completed checkpoint is retained for provenance and possible debugging.
        print('TASK_DONE',task['id'],task['stage'],task['L'],task['precision'],round(elapsed,3),flush=True)
        return dict(state='completed',id=task['id'],seconds=elapsed,L=task['L'],precision=task['precision'])
    except PauseRun:
        return dict(state='paused',id=task['id'])
    except BaseException:
        error=traceback.format_exc()
        write_json(root/'failures'/(task['id']+'.json'),dict(signature=signature,task=task,error=error,updated_unix=time.time()))
        print(error,flush=True)
        raise


def payload(args,manifest,task,deadline,root=None):
    return (str(root or args.output),str(args.checkpoint),str(args.checks),manifest['signature'],task,deadline)


def benchmark_workers(args,manifest,deadline):
    path=args.output/'throughput.json'
    if path.exists():
        data=json.loads(path.read_text())
        if data['signature']!=manifest['signature']:raise RuntimeError('throughput identity mismatch')
        return data['selected_workers']
    results=[]
    for workers in (1,2,4):
        root=args.output/'benchmark'/f'workers{workers}'
        for folder in ('results','checkpoints','heartbeats','failures'):(root/folder).mkdir(parents=True,exist_ok=True)
        tasks=[make_task('throughput',10,60000,.5,h,'network','float32',r,settings()) for r in range(4) for h in (.3,0.)]
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
            futures=[pool.submit(execute_task,payload(args,manifest,t,deadline,root)) for t in tasks]
            done=[f.result() for f in futures]
        if any(d['state']!='completed' for d in done):raise PauseRun('benchmark time window expired')
        elapsed=time.monotonic()-start
        results.append(dict(workers=workers,wall_seconds=elapsed,tasks=len(tasks),tasks_per_second=len(tasks)/elapsed))
        write_json(args.output/'throughput_partial.json',dict(signature=manifest['signature'],results=results))
    best=max(results,key=lambda r:r['tasks_per_second'])
    # Prefer fewer processes when throughput is within 5% of the best observed wave.
    selected=min(r['workers'] for r in results if r['tasks_per_second']>=.95*best['tasks_per_second'])
    write_json(path,dict(signature=manifest['signature'],selected_workers=selected,results=results,
                         interpretation='Eight matched L10 tasks per wave, includes process/model setup; precision candidate only'))
    return selected


def prerequisites(args):
    if (args.checks/'gpu_failure.json').exists() or (args.checks/'exact_failure.json').exists():
        raise RuntimeError('qualification prerequisites failed; inspect full logs')
    for name in ('gpu_check_status.json','exact_status.json'):
        path=args.checks/name
        if not path.exists() or json.loads(path.read_text()).get('state')!='completed':return False
    return True


def supervise(args,manifest,tasks):
    global _STOP
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    started=time.time();deadline=started+args.hours*3600
    while not prerequisites(args):
        if time.time()>=deadline or _STOP:raise PauseRun('waiting for prerequisite checks')
        write_json(args.output/'status.json',dict(state='awaiting_direct_and_exact_checks',updated_unix=time.time(),pid=os.getpid()))
        time.sleep(10)
    gpu_workers=benchmark_workers(args,manifest,deadline)
    completed={t['id'] for t in tasks if verified_result(args.output,t,manifest['signature'])}
    failures=[p for p in (args.output/'failures').glob('*.json')]
    if failures:raise RuntimeError(f'existing failures require review: {failures[:3]}')
    cpu_tasks=iter(t for t in tasks if t['method']!='network' and t['id'] not in completed)
    gpu_tasks=iter(t for t in tasks if t['method']=='network' and t['id'] not in completed)
    active={};paused=False;failure=None
    def next_task(pool,queue,lane):
        nonlocal paused
        if time.time()>=deadline or _STOP or paused or (args.output/'PAUSE.json').exists():return
        t=next(queue,None)
        if t is not None:active[pool.submit(execute_task,payload(args,manifest,t,deadline))]=(t,pool,queue,lane)
    telemetry=(args.output/'gpu.csv').open('a')
    recorder=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','10'],stdout=telemetry)
    try:
        with ProcessPoolExecutor(max_workers=gpu_workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as gpu, \
             ProcessPoolExecutor(max_workers=args.cpu_workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as cpu:
            for _ in range(gpu_workers):next_task(gpu,gpu_tasks,'gpu')
            for _ in range(args.cpu_workers):next_task(cpu,cpu_tasks,'cpu')
            while active:
                for future in list(active):
                    if not future.done():continue
                    task,pool,queue,lane=active.pop(future)
                    try:result=future.result()
                    except BaseException:
                        failure=traceback.format_exc();paused=True
                        write_json(args.output/'PAUSE.json',dict(reason='worker_failure',error=failure,updated_unix=time.time()))
                        continue
                    if result['state']=='completed':completed.add(task['id'])
                    else:paused=True
                    next_task(pool,queue,lane)
                if shutil.disk_usage(args.output).free<5*2**30:
                    paused=True
                    write_json(args.output/'PAUSE.json',dict(reason='disk_headroom_below_5GiB',updated_unix=time.time()))
                if _STOP:
                    paused=True
                    write_json(args.output/'PAUSE.json',dict(reason='termination_requested',updated_unix=time.time()))
                state='pausing' if paused or time.time()>=deadline or _STOP else 'running_qualification'
                write_json(args.output/'status.json',dict(state=state,pid=os.getpid(),updated_unix=time.time(),
                    started_unix=started,deadline_unix=deadline,completed=len(completed),expected=len(tasks),
                    gpu_workers=gpu_workers,cpu_workers=args.cpu_workers,
                    active=[dict(id=t['id'],L=t['L'],stage=t['stage'],precision=t['precision'],lane=lane)
                            for t,_,_,lane in active.values()],final_data_started=False))
                time.sleep(3)
        if failure:raise RuntimeError(failure)
        write_json(args.output/'status.json',dict(state='qualification_complete_pending_review' if len(completed)==len(tasks) else 'paused',
                    pid=os.getpid(),updated_unix=time.time(),completed=len(completed),expected=len(tasks),
                    final_data_started=False,gpu_workers=gpu_workers,cpu_workers=args.cpu_workers))
    finally:
        recorder.terminate();recorder.wait(timeout=15);telemetry.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checks',type=Path,required=True)
    p.add_argument('--hours',type=float,default=12.)
    p.add_argument('--cpu-workers',type=int,default=4)
    p.add_argument('--initialize-only',action='store_true')
    args=p.parse_args()
    if args.hours<=0 or not 1<=args.cpu_workers<=4:raise ValueError('invalid resource limits')
    args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve();args.checks=args.checks.resolve()
    with stage_lock(args.output,'supervisor'):
        manifest,tasks=initialize(args.output,args.checkpoint,args.checks)
        if args.initialize_only:
            print('INITIALIZED',len(tasks),'tasks',manifest['signature'],flush=True);return
        try:supervise(args,manifest,tasks)
        except PauseRun as error:
            write_json(args.output/'status.json',dict(state='paused',reason=str(error),updated_unix=time.time(),final_data_started=False))
        except BaseException:
            error=traceback.format_exc();print(error,flush=True)
            write_json(args.output/'status.json',dict(state='failed',error=error,updated_unix=time.time(),final_data_started=False))
            raise


if __name__=='__main__':
    main()
