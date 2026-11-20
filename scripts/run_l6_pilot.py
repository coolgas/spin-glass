"""Finite L6 pilot controller; explicit L, resource admission, detached workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def identity(pid):
    try:
        row = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if row[0] == 'Z' else row[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def capacity(root, active):
    mem = {line.split(':')[0]: int(line.split()[1]) * 1024
           for line in Path('/proc/meminfo').read_text().splitlines()}
    cores = len(os.sched_getaffinity(0))
    available = mem['MemAvailable']
    relative = next(line.split('::', 1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if '::' in line)
    cg = Path('/sys/fs/cgroup') / relative.lstrip('/')
    for path in [cg, *cg.parents]:
        if path == Path('/sys/fs'):
            break
        if (path / 'cpu.max').exists():
            quota, period = (path / 'cpu.max').read_text().split()
            if quota != 'max':
                cores = min(cores, int(int(quota) / int(period)))
        if (path / 'memory.max').exists():
            maximum = (path / 'memory.max').read_text().strip()
            if maximum != 'max':
                available = min(available, int(maximum) - int((path / 'memory.current').read_text()))
    own_rss = sum(rss(row['pid']) for row in active.values())
    # Reserve eight GiB per case for nine-block D256 growth and sixteen GiB for others.
    forecast = 8 * 2**30
    memory_slots = max(0, int((available - 16 * 2**30 - max(0, len(active) * forecast - own_rss)) / forecast))
    # Load includes our workers; reserve two logical CPUs and react to unrelated load.
    external_load = max(0., os.getloadavg()[0] - len(active))
    cpu_slots = max(0, int(cores - 2 - external_load) - len(active))
    free = shutil.disk_usage(root).free
    disk_slots = max(0, int((free - 16 * 2**30 - len(active) * 2**30) / 2**30))
    return min(cpu_slots, memory_slots, disk_slots), dict(cores=cores, available_memory=available,
            worker_rss=own_rss, forecast_bytes_per_worker=forecast, memory_slots=memory_slots,
            cpu_slots=cpu_slots, disk_free=free, external_load=external_load)


def worker(root, task):
    from rydberg_kagome.campaign import run_case
    from rydberg_kagome.results import json_write
    folder = root / 'cases' / task['id']
    result = run_case(str(folder), 6, task['sample'], task['model'], task['delta'], task['omega'],
                      task['chi'], task['dimensions'], task['pairs'], 120, False, task['fixed_production'])
    assert result['inputs']['L'] == 6
    json_write(folder / 'pilot-classification.json', dict(campaign='rydberg-L6-pilot-20261120-v1',
               purpose=task['purpose'], production_campaign=False,
               fixed_production_flag_meaning='Solver protocol matched to L4; not an L6 production authorization'))
    print(json.dumps({k: result[k] for k in ['solver_pass', 'sampling_moment_check', 'seconds', 'peak_rss_kib']}), flush=True)


def controller(root, manifest):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = json.loads(manifest.read_text())
    assert len(tasks) == len({t['id'] for t in tasks}) == 24
    stored = root / 'tasks.json'
    if stored.exists():
        assert json.loads(stored.read_text()) == tasks
    else:
        write(stored, tasks)
    active, children, states = {}, {}, {}
    for task in tasks:
        key = task['id']; launch = root / 'launches' / f'{key}.json'
        if (root / 'cases' / key / 'result.json').exists():
            states[key] = 'completed'
        elif launch.exists():
            row = json.loads(launch.read_text())
            if identity(row['pid']) == row['birth']:
                active[key] = row; states[key] = 'running'
            else:
                states[key] = 'interrupted_or_failed'
        else:
            states[key] = 'pending'
    while True:
        for key, row in list(active.items()):
            if key in children:
                children[key].poll()
            if identity(row['pid']) != row['birth']:
                states[key] = 'completed' if (root / 'cases' / key / 'result.json').exists() else 'failed'
                del active[key]
        slots, resources = capacity(root, active)
        for task in tasks:
            key = task['id']
            if slots <= 0:
                break
            if states[key] != 'pending':
                continue
            env = os.environ.copy()
            for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                env[name] = '1'
            env['PYTHONPATH'] = str(REPO / 'code'); env['PYTHONDONTWRITEBYTECODE'] = '1'
            command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--root', str(root), '--task', json.dumps(task)]
            (root / 'logs').mkdir(exist_ok=True)
            with (root / 'logs' / f'{key}.log').open('ab') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
            row = dict(pid=child.pid, birth=identity(child.pid), command=command, started=time.time())
            write(root / 'launches' / f'{key}.json', row)
            active[key] = row; children[key] = child; states[key] = 'running'; slots -= 1
        counts = {s: list(states.values()).count(s) for s in set(states.values())}
        write(root / 'status.json', dict(updated=time.time(), controller=os.getpid(), counts=counts,
              active=active, resources=resources, states=states, deadline=None))
        if not active and 'pending' not in states.values():
            print(json.dumps(counts), flush=True)
            break
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--task')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    assert root.is_absolute() and not root.is_relative_to(REPO)
    if args.task:
        worker(root, json.loads(args.task))
    else:
        controller(root, args.manifest)
