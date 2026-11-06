"""Resource-limited subprocess scheduling for the finite L8 retention study."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .triangle_correlation_dmrg import case, digest, write
from .triangle_correlation_retention import (
    FIELDS, MS, CHIS, IDS, import_diagnostic, report, reuse,
)


def key(task):
    return f"L{task['L']}_g{task['gamma']:g}_s{task['sample_id']}_m{task['m']}_c{task['chi']}"


def result(root, task):
    path = root/'cases'/key(task)/'result.json'
    return json.loads(path.read_text()) if path.exists() else None


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().rsplit(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


class AdoptedProcess:
    """Monitor a worker that survived a controller restart."""
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None if alive(self.pid) else 0


def memory_headroom():
    cg = Path('/sys/fs/cgroup')
    stats = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
    # Inactive file pages are reclaimable by the kernel under memory pressure.
    # Keep active file pages, shmem, and all anonymous memory charged as used.
    reclaimable = min(int(stats['inactive_file']), int(stats['file'])-int(stats['shmem']))
    return (int((cg/'memory.max').read_text()) - int((cg/'memory.current').read_text())
            + max(0, reclaimable))/2**30


def peer_workers(roots):
    workers = []
    for root in roots:
        status = root/'status.json'
        if status.exists():
            for worker in json.loads(status.read_text()).get('running', {}).values():
                if alive(worker['pid']):
                    workers.append(worker)
    return workers


def owned_workers(root):
    """Recover workers even if a controller stopped before writing status."""
    workers = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode().split('\0')
            if ('transformer_qmc.triangle_correlation_parallel' not in argv or
                    '--task-json' not in argv or '--output' not in argv or
                    argv[argv.index('--output')+1] != str(root) or not alive(int(proc.name))):
                continue
            task = json.loads(argv[argv.index('--task-json')+1])
            env = dict(v.split('=', 1) for v in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in v)
            workers.append(dict(task, pid=int(proc.name), threads=int(env.get('OMP_NUM_THREADS', '1'))))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return workers


def run(args):
    if args.dispatch_lock:
        with args.dispatch_lock.open('a') as lock:
            return _run(args, lock)
    return _run(args, None)


def _run(args, dispatch_lock):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    (root/'cases').mkdir(exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    policy = dict(workers=args.workers, threads_per_worker=args.threads,
        adaptive_threads=True, worker_cpu_budget=16,
        worker_memory_budget_gib=args.memory_budget_gib,
        memory_reserve_gib=16, disk_reserve_gib=3,
        cpu_quota_cores=20, diagnostic_pid=args.wait_pid,
        diagnostic_runs_concurrently=True, dense_scan='held',
        scope=dict(L=8, gammas=FIELDS, m=MS, chi=CHIS, disorders=IDS),
        sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        peer_roots=list(map(str,args.peer_roots)),dispatch_lock=str(args.dispatch_lock))
    manifest = root/'manifest.json'
    if manifest.exists():
        previous_policy = json.loads(manifest.read_text())
        assert previous_policy['scope'] == policy['scope']
        write(root/'execution_policy.json', policy)
    else:
        write(manifest, policy)
    reuse(root, args.previous)
    validation = [dict(L=4, gamma=.5, sample_id=60000, m=m, chi=c, validate=True)
                  for c in CHIS for m in MS]
    tasks = [dict(L=8, gamma=g, sample_id=s, m=m, chi=c, validate=False)
             for c in CHIS for m in MS for g in FIELDS for s in IDS]
    ledger = root/'worker_failures.json'
    failures = json.loads(ledger.read_text()) if ledger.exists() else {}
    running = {}
    status_path = root/'status.json'
    for task in owned_workers(root):
        task = dict(task)
        pid = task.pop('pid')
        threads = task.pop('threads', args.threads)
        name = key(task)
        assert name not in running, f'Duplicate live owners for {name}'
        running[name] = dict(process=AdoptedProcess(pid), task=task,
            log=(root/'logs'/f'{name}.log').open('ab'),
            threads=threads,
            memory_gib={2:3., 3:6., 4:12.}[task['chi']],
            disk_gib={2:.1, 3:.4, 4:.8}[task['chi']])
    last_report = 0.
    diagnostic_imported = False
    while True:
        if dispatch_lock:
            fcntl.flock(dispatch_lock, fcntl.LOCK_EX)
        peers = peer_workers(args.peer_roots)
        peer_owned = {key(j['task']) for j in peers if j['mode']=='case'}
        changed = False
        for name, job in list(running.items()):
            code = job['process'].poll()
            if code is None:
                continue
            job['log'].close()
            if code or result(root, job['task']) is None:
                failures[name] = dict(task=job['task'], returncode=code,
                                      log=str(root/'logs'/f'{name}.log'))
                write(ledger, failures)
            del running[name]
            changed = True
        diagnostic_alive = bool(args.wait_pid and alive(args.wait_pid))
        if not diagnostic_alive and args.diagnostic and not diagnostic_imported:
            import_diagnostic(root, args.diagnostic)
            diagnostic_imported = True
            changed = True
        eligible = []
        for task in validation + tasks:
            name = key(task)
            destination = root/'cases'/name
            if not destination.exists():
                for peer in args.peer_roots:
                    source = peer/'cases'/name
                    if (source/'result.json').exists():
                        destination.symlink_to(source.resolve(), target_is_directory=True)
                        changed = True
                        break
            if name in running or name in failures or result(root, task) is not None:
                continue
            if name in peer_owned:
                continue
            if task['L'] == 8:
                gate = dict(task, L=4, gamma=.5, sample_id=60000, validate=True)
                checked = result(root, gate)
                if checked is None or not checked['solver_passed']:
                    continue
            eligible.append(task)
        for task in eligible:
            if len(running) >= args.workers:
                break
            # Reserve for possible D512 growth, beyond each process's current RSS.
            estimate = {2: 3., 3: 6., 4: 12.}[task['chi']]
            spare_cores = 16 - sum(j['threads'] for j in running.values()) - sum(j['threads'] for j in peers)
            if spare_cores < args.threads:
                break
            memory_slots = max(1, min(args.workers, int(args.memory_budget_gib//estimate)))
            threads = min(spare_cores, max(args.threads, min(4, 16//memory_slots)))
            reserved = sum(j['memory_gib'] for j in running.values())
            if reserved + estimate > args.memory_budget_gib:
                continue
            if reserved + estimate + sum(j['memory_gib'] for j in peers) > 72:
                continue
            extra = estimate
            for job in running.values():
                status = Path(f"/proc/{job['process'].pid}/status")
                rss = 0.
                if status.exists():
                    for line in status.read_text().splitlines():
                        if line.startswith('VmRSS:'):
                            rss = int(line.split()[1])/2**20
                extra += max(0., job['memory_gib']-rss)
            for job in peers:
                try:
                    lines = Path(f"/proc/{job['pid']}/status").read_text().splitlines()
                    rss = next(int(l.split()[1])/2**20 for l in lines if l.startswith('VmRSS:'))
                except FileNotFoundError:
                    rss = 0.
                extra += max(0., job['memory_gib']-rss)
            if memory_headroom() < 16 + extra:
                continue
            disk_gib = {2:.1, 3:.4, 4:.8}[task['chi']]
            if shutil.disk_usage(root).free/2**30 < 3 + disk_gib + sum(j['disk_gib'] for j in running.values()) + sum(j['disk_gib'] for j in peers):
                continue
            name = key(task)
            env = dict(os.environ, OMP_NUM_THREADS=str(threads),
                OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                NUMEXPR_NUM_THREADS=str(threads))
            log = (root/'logs'/f'{name}.log').open('ab')
            command = [sys.executable, '-u', '-m',
                       'transformer_qmc.triangle_correlation_parallel', '--output', str(root),
                       '--task-json', json.dumps(task)]
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log)
            running[name] = dict(process=process, log=log, task=task,
                                 memory_gib=estimate, disk_gib=disk_gib, threads=threads)
            changed = True
        rows = [json.loads(p.read_text()) for p in (root/'cases').glob('*/result.json')]
        pending = [t for t in validation+tasks if result(root,t) is None and key(t) not in failures]
        blocked_gates = []
        for task in pending:
            if task['L'] == 8:
                gate = dict(task, L=4, gamma=.5, sample_id=60000, validate=True)
                checked = result(root, gate)
                if key(gate) in failures or (checked and not checked['solver_passed']):
                    blocked_gates.append(key(task))
        actionable = [t for t in pending if key(t) not in blocked_gates]
        passed = sum(r['L']==8 and r['solver_passed'] for r in rows)
        done = not running and not actionable and not diagnostic_alive
        state = ('complete' if passed == 54 else 'complete_with_unresolved_cases') if done else 'running'
        if not running and actionable:
            state = 'waiting_for_resources'
        write(root/'status.json', dict(state=state, solver_passing_L8=passed, target_L8=54,
            completed_L8=sum(r['L']==8 for r in rows),
            running=[dict(pid=j['process'].pid, threads=j['threads'], **j['task']) for j in running.values()],
            workers=args.workers, threads_per_worker=args.threads,
            worker_cpu_threads=sum(j['threads'] for j in running.values()),
            diagnostic_active=diagnostic_alive, pending=len(pending),
            blocked_by_validation=blocked_gates, worker_failures=failures,
            dense_scan='held',peer_worker_threads=sum(j['threads'] for j in peers)))
        now = time.monotonic()
        if done or (changed and now-last_report > 30):
            report(root)
            last_report = time.monotonic()
        if done:
            if dispatch_lock:
                fcntl.flock(dispatch_lock, fcntl.LOCK_UN)
            return
        if dispatch_lock:
            fcntl.flock(dispatch_lock, fcntl.LOCK_UN)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--diagnostic', type=Path)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--memory-budget-gib', type=float, default=48.)
    parser.add_argument('--task-json')
    parser.add_argument('--peer-roots',type=Path,nargs='*',default=[])
    parser.add_argument('--dispatch-lock',type=Path)
    args = parser.parse_args()
    if args.task_json:
        case(args.output, **json.loads(args.task_json))
    else:
        assert args.previous and args.workers > 0 and args.threads > 0
        assert args.workers*args.threads <= 16
        args.output.mkdir(parents=True, exist_ok=True)
        try:
            with (args.output/'controller.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                run(args)
        except Exception:
            import traceback
            (args.output/'failure.txt').write_text(traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
