"""Operational thread override; never edits the scientific source or task manifest."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def worker(args):
    from rydberg_kagome.queue import process_identity
    root = args.root.resolve()
    task, = [t for t in read(root/'tasks.json') if t['id'] == args.case]
    plan = read(args.plan)
    original = plan['initial_workers'].get(args.case, {})
    started = original.get('started', time.time())
    elapsed = time.time() - started
    hard_remaining = task.get('hard_seconds', task.get('seconds', 72*3600)+3600)-elapsed
    if hard_remaining <= 0:
        raise RuntimeError('Original invocation deadline has expired')
    limit = int(task.get('memory_gib', 64)*2**30)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    signal.alarm(max(1, int(hard_remaining)))
    for name in THREAD_VARS:
        os.environ[name] = str(args.threads)
    os.environ['RYDBERG_CASE_ROOT'] = str(root/'cases'/args.case)
    registry = root.parent/'queue-reservations'
    os.environ['RYDBERG_RESERVATIONS'] = str(registry)
    with (registry/'admission.lock').open('a') as barrier:
        fcntl.flock(barrier, fcntl.LOCK_EX)
    from rydberg_kagome.storage import BoundedLog
    sys.stdout = sys.stderr = BoundedLog(root/'logs'/f'{args.case}.log')
    from rydberg_kagome.campaign import run_case
    import numpy, scipy, ctypes
    numpy.linalg.svd(numpy.eye(2))
    pools = []
    for library in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                           if '/' in line and 'libmkl_rt.so' in line}):
        handle = ctypes.CDLL(library)
        handle.MKL_Get_Max_Threads.restype = ctypes.c_int
        pools.append({'library': library, 'num_threads': handle.MKL_Get_Max_Threads()})
    if not pools or any(p['num_threads'] != args.threads for p in pools):
        raise RuntimeError(f'Numerical thread setting not verified: {pools}')
    folder = root/'cases'/args.case
    write(folder/f'thread-runtime-{os.getpid()}.json', {
        'pid': os.getpid(), 'birth': process_identity(os.getpid()),
        'threads': args.threads, 'started': time.time(), 'original_started': started,
        'hard_deadline': started+task.get('hard_seconds', task.get('seconds', 72*3600)+3600),
        'threadpools': pools, 'runtime_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256((root/'tasks.json').read_bytes()).hexdigest()})
    result = run_case(str(folder), task.get('L', 6), task['sample'], task['model'],
        task['delta'], task['omega'], task['chi'], task['dimensions'], task['pairs'],
        task.get('max_sweeps', 120), False, False, True,
        max(.001, task.get('seconds', 72*3600)-elapsed), None,
        task.get('memory_gib', 64), None, task.get('checkpoint_interval', 30),
        task.get('final_state_retention', 'full'))
    print(json.dumps({k: result.get(k) for k in
        ('state', 'completed', 'solver_pass', 'sampling_moment_check', 'seconds')}), flush=True)
    if not result.get('completed'):
        write(folder/'worker-exit.json', {'updated': time.time(), 'result': result})


def controller(args):
    from rydberg_kagome import queue
    original_popen = queue.subprocess.Popen
    original_capacity = queue.capacity

    def launch(command, *positional, **kwargs):
        if isinstance(command, list) and len(command) > 4 and command[3] == '-c' and 'adaptive_worker' in command[4]:
            manifest, root, case = command[-3:]
            command[:] = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                'worker', '--root', root, '--plan', str(args.plan),
                '--threads', str(args.threads), '--case', case]
            for name in THREAD_VARS:
                kwargs['env'][name] = str(args.threads)
        return original_popen(command, *positional, **kwargs)

    def capacity(root, active, max_workers, **kwargs):
        slots, report = original_capacity(root, active, max_workers, **kwargs)
        reserved = 0
        for rec in active.values():
            try:
                env = dict(e.split(b'=', 1) for e in Path(f"/proc/{rec['pid']}/environ").read_bytes().split(b'\0') if b'=' in e)
                reserved += int(env.get(b'OPENBLAS_NUM_THREADS', b'4'))
            except (OSError, ValueError):
                reserved += args.threads
        from rydberg_kagome.adaptive_memory import visible_cgroups
        cores = float(len(os.sched_getaffinity(0)))
        for cg in visible_cgroups():
            if (cg/'cpu.max').exists():
                quota, period = (cg/'cpu.max').read_text().split()
                if quota != 'max':
                    cores = min(cores, float(quota)/float(period))
        background = max(0., os.getloadavg()[0]-reserved)
        headroom = max(0., cores-background-2.-reserved)
        cpu_slots = min(max_workers-len(active), int(headroom//args.threads))
        report.update(cpu_headroom=headroom, cpu_slots=max(0, cpu_slots),
                      reserved_cpu_threads=reserved, threads_per_new_case=args.threads)
        return min(slots, max(0, cpu_slots)), report

    queue.subprocess.Popen = launch
    queue.capacity = capacity
    queue.run(str(args.root/'tasks.json'), args.root, 4)


def switch(args):
    from rydberg_kagome.queue import process_identity
    root = args.root.resolve()
    plan = read(args.plan)
    report_path = args.plan.parent/'switch-status.json'
    ownership = (args.plan.parent/'switch.lock').open('a')
    fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = {'owner': os.getpid(), 'started': time.time(), 'state': 'awaiting_checkpoints', 'cases': {}}
    pending = dict(plan['initial_workers'])
    deadline = time.monotonic()+10800
    while pending and time.monotonic() < deadline:
        for case, old in list(pending.items()):
            folder = root/'cases'/case
            status = read(root/'status.json')
            current = status['active'].get(case)
            if (folder/'result.json').exists():
                report['cases'][case] = {'state': 'completed_before_switch'}
                del pending[case]
                continue
            if current and current['pid'] != old['pid']:
                receipts = list(folder.glob(f"thread-runtime-{current['pid']}.json"))
                if receipts:
                    receipt = read(receipts[0])
                    if receipt['threads'] != args.threads:
                        raise RuntimeError('Unexpected replacement thread count')
                    old_status = report['cases'].get(case, {})
                    progress = read(folder/'adaptive-status.json') if (folder/'adaptive-status.json').exists() else {}
                    saved = old_status.get('sweeps', [0, 0])
                    if any(a > b for a, b in zip(progress.get('sweeps', []), saved)):
                        report['cases'][case] = dict(old_status, state='verified_running', new_pid=current['pid'],
                            runtime_receipt=str(receipts[0]), progress=progress)
                        del pending[case]
                continue
            if case in report['cases']:
                continue
            if process_identity(old['pid']) != old['birth']:
                continue
            checkpoint = read(folder/'resume.json')
            if checkpoint['generation'] == plan['initial_generations'][case]:
                continue
            # Freeze this identity before validating the atomic paired pointer.
            os.kill(old['pid'], signal.SIGSTOP)
            try:
                if process_identity(old['pid']) != old['birth']:
                    raise RuntimeError('Worker identity changed')
                checkpoint = read(folder/'resume.json')
                for entry in checkpoint['files']:
                    payload = (folder/entry['file']).resolve()
                    if not payload.is_relative_to(folder.resolve()):
                        raise RuntimeError('Checkpoint outside case')
                    if hashlib.sha256(payload.read_bytes()).hexdigest() != entry['sha256']:
                        raise RuntimeError('Checkpoint hash mismatch')
                evidence = dict(old_launch=old, generation=checkpoint['generation'],
                    sweeps=[len(h) for h in checkpoint['state']['history']], D=checkpoint['state']['D'],
                    files=checkpoint['files'], old_threads=4, new_threads=args.threads, updated=time.time(),
                    reason='authorized CPU thread increase at a newly committed paired checkpoint')
                write(args.plan.parent/f'{case}-handover.json', evidence)
                report['cases'][case] = dict(state='restarting', **evidence)
                os.kill(old['pid'], signal.SIGTERM)
            finally:
                if process_identity(old['pid']) == old['birth']:
                    os.kill(old['pid'], signal.SIGCONT)
        report['updated'] = time.time()
        write(report_path, report)
        if pending:
            time.sleep(30)
    report.update(state='complete' if not pending else 'timed_out', pending=list(pending), finished=time.time())
    write(report_path, report)
    print(json.dumps(report), flush=True)
    if pending:
        raise RuntimeError('Bounded switch did not complete; inspect status without duplicating switch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('worker', 'controller', 'switch'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.threads != 6:
        raise ValueError('This reviewed runtime is restricted to six threads per case')
    globals()[args.mode](args)
