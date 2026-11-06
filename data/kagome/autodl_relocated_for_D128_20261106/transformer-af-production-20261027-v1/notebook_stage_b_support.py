"""Execute notebook section 5 using the verified, frozen projector implementation.

Run this operational script outside runner_source. Stage B is a feasibility
pilot, not a ground-state qualification or final-disorder collection.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

STOP = False


def terminate(signum, frame):
    global STOP
    STOP = True


def build_tasks():
    from transformer_qmc.production_runner import make_task, settings
    parameters = settings(64, 2., 1., 1., 4, .1)
    tasks = []
    for repeat in range(4):
        for sid in range(60000, 60004):
            for gamma in (.5, 1., 2.):
                for h in (0., .15, .3):
                    for method in ('network', 'constant'):
                        tasks.append(make_task('notebook_stage_b', 4, sid, gamma, h,
                                               method, 'float64', repeat, parameters))
    if len(tasks) != 288 or len({t['id'] for t in tasks}) != 288:
        raise RuntimeError('Notebook Stage B task identity/count mismatch')
    return tasks


def initialize(root):
    from transformer_qmc.experiment import write_json
    from transformer_qmc.production_protocol import source_hashes, identity
    from transformer_qmc.production_runner import verified_result
    original = json.loads((root/'pipeline/manifest.json').read_text())
    sources = source_hashes(root/'runner_source/code')
    if sources != original['source_sha256']:
        raise RuntimeError('Frozen implementation changed')
    launch = json.loads((root/'operational_launch.json').read_text())
    checkpoint = Path(launch['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != original['checkpoint_sha256']:
        raise RuntimeError('Frozen checkpoint changed')
    handoff = json.loads((root/'notebook_handoff_request.json').read_text())
    old_tasks = json.loads((root/'pipeline/tasks.json').read_text())
    task = next(t for t in old_tasks if t['id'] == handoff['completed_L10_task'])
    if task['L'] != 10 or not verified_result(root/'pipeline', task, original['signature']):
        raise RuntimeError('Requested L10 task has not completed')
    if json.loads((root/'pipeline/status.json').read_text())['state'] != 'paused':
        raise RuntimeError('Original probe queue must be paused before handoff')
    for name in ('gpu_check_status.json', 'exact_status.json'):
        status = json.loads((root/'qualification'/name).read_text())
        if status['state'] != 'completed' or status['completed'] != 392:
            raise RuntimeError('Direct/exact prerequisites incomplete')
    output = root/'notebook_stage_b'
    output.mkdir(exist_ok=True)
    notebook = root/'notebook_stage_b_protocol.ipynb'
    document = json.loads(notebook.read_text())
    section = next(c for c in document['cells'] if '## 5. Stage B:' in ''.join(c['source']))
    text = ''.join(section['source'])
    for token in ('W=64', '2$', 'four cycles', '0.1', '1024', '0.05'):
        if token not in text:
            raise RuntimeError('Notebook Stage B specification needs explicit review: '+token)
    tasks = build_tasks()
    manifest = dict(schema=1, stage='notebook_stage_b', notebook_section=5,
        notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=sources, checkpoint_sha256=original['checkpoint_sha256'],
        parent_signature=original['signature'], task_count=len(tasks), tasks_sha256=identity(tasks),
        checks_signature=original['checks_signature'], deadline_unix=launch['deadline_unix'],
        specification=dict(L=4, n_sites=48, periodic=True, development_ids=list(range(60000,60004)),
            gamma=[.5,1.,2.], h=[0.,.15,.3], repeats=4, methods=['network','constant'],
            precision='float64', parameters=tasks[0]['parameters']),
        final_anchor_sizes=[4,6,8,10], final_anchor_contexts=49, final_data_started=False,
        followup_gates=dict(stage_b='Review matched runtime, ancestry, and cost before representative W256/W1024 and longer-time escalations; include h=0.05 and compare h=0 separately.',
            stage_c='Only after cost feasibility: notebook section 6 population/time/cadence ladders, alternative starts, eight-disorder development policy and all-context qualification.',
            stage_d='Only qualified contexts enter fresh final-disorder collection.'))
    manifest['signature'] = identity(manifest)
    for name, value in [('manifest.json', manifest), ('tasks.json', tasks)]:
        path = output/name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise RuntimeError('Existing notebook-stage identity mismatch: '+name)
        else:
            write_json(path, value)
    for folder in ('results','checkpoints','heartbeats','failures'):
        (output/folder).mkdir(exist_ok=True)
    return output, manifest, tasks, launch


def supervise(root, output, manifest, tasks, launch, stage_name='notebook_stage_b'):
    from transformer_qmc import production_runner as runner
    from transformer_qmc.experiment import write_json
    if (output/'PAUSE.json').exists() or list((output/'failures').glob('*.json')):
        raise RuntimeError('Notebook stage has unreviewed pause/failures')
    completed = {t['id'] for t in tasks if runner.verified_result(output,t,manifest['signature'])}
    queues = {lane: iter([t for t in tasks if t['id'] not in completed and
                         (t['method']=='network') == (lane=='gpu')]) for lane in ('gpu','cpu')}
    deadline = launch['deadline_unix']
    active = {}
    failure = None
    last_status = 0.
    started = time.time()
    def submit(pool, lane):
        if STOP or time.time()>=deadline or (output/'PAUSE.json').exists():
            return
        task = next(queues[lane],None)
        if task is None:
            return
        if any(t['id']==task['id'] for t,_,_ in active.values()):
            raise RuntimeError('Duplicate task rejected')
        payload = (str(output), launch['checkpoint'], str(root/'qualification'),
                   manifest['signature'], task, deadline)
        active[pool.submit(runner.execute_task,payload)] = task,pool,lane
    def status(force=False, state=None):
        nonlocal last_status
        now = time.time()
        if not force and now-last_status<10:
            return
        last_status = now
        counts = {method:sum(t['id'] in completed and t['method']==method for t in tasks)
                  for method in ('network','constant')}
        write_json(output/'status.json', dict(state=state or ('pausing' if STOP or now>=deadline or (output/'PAUSE.json').exists() else 'running_'+stage_name),
            pid=os.getpid(), updated_unix=now, started_unix=started, deadline_unix=deadline,
            completed=len(completed), expected=len(tasks), completed_by_method=counts,
            gpu_workers=1,cpu_workers=4, final_data_started=False, ground_state_qualified=False,
            active=[dict(id=t['id'],L=t['L'],gamma=t['gamma'],h=t['h'],method=t['method'],lane=lane)
                    for t,_,lane in active.values()], error=failure))
    telemetry = (output/'gpu.csv').open('a')
    recorder = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                                 '--format=csv,noheader,nounits','-l','10'],stdout=telemetry)
    try:
        with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=runner.init_worker) as gpu, \
             ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=runner.init_worker) as cpu:
            submit(gpu,'gpu')
            for _ in range(4):
                submit(cpu,'cpu')
            status(True)
            while active:
                if STOP or shutil.disk_usage(output).free<5*2**30:
                    write_json(output/'PAUSE.json',dict(reason='termination_requested' if STOP else 'low_disk',updated_unix=time.time()))
                done,_ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    task,pool,lane = active.pop(future)
                    try:
                        result = future.result()
                    except BaseException:
                        failure = traceback.format_exc()
                        write_json(output/'PAUSE.json',dict(reason='worker_failure',error=failure,updated_unix=time.time()))
                        print(failure,flush=True)
                        continue
                    if result['state']=='completed':
                        completed.add(task['id'])
                    submit(pool,lane)
                status()
        status(True, 'failed' if failure else ('awaiting_'+stage_name+'_review' if len(completed)==len(tasks) else 'paused'))
        if failure:
            raise RuntimeError(failure)
    finally:
        recorder.terminate()
        recorder.wait(timeout=15)
        telemetry.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--initialize-only',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0,str(root/'runner_source/code'))
    from transformer_qmc.frozen_scan import stage_lock
    from transformer_qmc.experiment import write_json
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    # Share the original campaign lock: old and new queues cannot overlap.
    with stage_lock(root/'pipeline','supervisor'):
        output,manifest,tasks,launch = initialize(root)
        if args.initialize_only:
            print('INITIALIZED_NOTEBOOK_STAGE_B',len(tasks),manifest['signature'],flush=True)
            return
        try:
            supervise(root,output,manifest,tasks,launch)
        except BaseException:
            error=traceback.format_exc()
            write_json(output/'status.json',dict(state='failed',pid=os.getpid(),error=error,updated_unix=time.time(),final_data_started=False))
            raise


if __name__=='__main__':
    main()
