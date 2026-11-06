"""Completion-driven operational scheduler for an immutable production snapshot.

Run this file by absolute path outside runner_source/code. Scientific task
identities and checkpoint formats remain those of the original snapshot.
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    sys.path.insert(0,str(root/'runner_source/code'))
    from transformer_qmc import production_runner as runner
    from transformer_qmc.experiment import write_json
    from transformer_qmc.production_protocol import source_hashes, identity
    from transformer_qmc.frozen_scan import stage_lock
    output=root/'pipeline'
    manifest=json.loads((output/'manifest.json').read_text())
    tasks=json.loads((output/'tasks.json').read_text())
    if source_hashes(root/'runner_source/code')!=manifest['source_sha256']:
        raise RuntimeError('frozen scientific source mismatch')
    if identity(tasks)!=manifest['tasks_sha256']:raise RuntimeError('task identity mismatch')
    launch=json.loads((root/'operational_launch.json').read_text())
    checkpoint=Path(launch['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise RuntimeError('checkpoint identity mismatch')
    deadline=launch['deadline_unix']
    workers=json.loads((output/'throughput.json').read_text())['selected_workers']
    if (output/'PAUSE.json').exists():raise RuntimeError('unreviewed pause flag')
    if list((output/'failures').glob('*.json')):raise RuntimeError('unreviewed task failures')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    completed={t['id'] for t in tasks if runner.verified_result(output,t,manifest['signature'])}
    queues={lane:iter(t for t in tasks if t['id'] not in completed and
                     ((t['method']=='network')==(lane=='gpu'))) for lane in ('gpu','cpu')}
    active={};error=None;last_status=0.;last_disk=0.
    operational=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     scientific_signature=manifest['signature'],deadline_unix=deadline,
                     change='Completion-driven dispatch; task algorithms, seeds and settings unchanged')
    write_json(root/'operational_scheduler.json',operational)
    def submit(pool,lane):
        if STOP or time.time()>=deadline or (output/'PAUSE.json').exists():return
        task=next(queues[lane],None)
        if task is None:return
        payload=(str(output),str(checkpoint),str(root/'qualification'),manifest['signature'],task,deadline)
        active[pool.submit(runner.execute_task,payload)]=(task,pool,lane)
    def status(force=False):
        nonlocal last_status
        now=time.time()
        if not force and now-last_status<10:return
        last_status=now
        write_json(output/'status.json',dict(state='pausing' if STOP or now>=deadline or (output/'PAUSE.json').exists() else 'running_qualification',
            pid=os.getpid(),updated_unix=now,started_unix=launch['started_unix'],deadline_unix=deadline,
            completed=len(completed),expected=len(tasks),gpu_workers=workers,cpu_workers=4,
            active=[dict(id=t['id'],L=t['L'],stage=t['stage'],precision=t['precision'],lane=lane) for t,_,lane in active.values()],
            final_data_started=False,operational_scheduler='completion_driven_v2'))
    with stage_lock(output,'supervisor'):
        telemetry=(output/'gpu.csv').open('a')
        recorder=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                                    '--format=csv,noheader,nounits','-l','10'],stdout=telemetry)
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=runner.init_worker) as gpu, \
                 ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=runner.init_worker) as cpu:
                for _ in range(workers):submit(gpu,'gpu')
                for _ in range(4):submit(cpu,'cpu')
                status(True)
                while active:
                    done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                    for future in done:
                        task,pool,lane=active.pop(future)
                        try:result=future.result()
                        except BaseException:
                            error=traceback.format_exc();print(error,flush=True)
                            write_json(output/'PAUSE.json',dict(reason='worker_failure',error=error,updated_unix=time.time()))
                            continue
                        if result['state']=='completed':completed.add(task['id'])
                        submit(pool,lane)
                    now=time.time()
                    if STOP and not (output/'PAUSE.json').exists():
                        write_json(output/'PAUSE.json',dict(reason='termination_requested',updated_unix=now))
                    if now-last_disk>=30:
                        last_disk=now
                        if shutil.disk_usage(output).free<5*2**30:
                            write_json(output/'PAUSE.json',dict(reason='disk_headroom_below_5GiB',updated_unix=now))
                    status()
            write_json(output/'status.json',dict(state='failed' if error else ('qualification_complete_pending_review' if len(completed)==len(tasks) else 'paused'),
                error=error,pid=os.getpid(),updated_unix=time.time(),completed=len(completed),expected=len(tasks),
                final_data_started=False,deadline_unix=deadline,gpu_workers=workers,cpu_workers=4))
            if error:raise RuntimeError(error)
        finally:
            recorder.terminate();recorder.wait(timeout=15);telemetry.close()


if __name__=='__main__':
    main()
