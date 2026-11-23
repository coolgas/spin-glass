"""Live shared-host headroom checks, independent of DMRG acceptance criteria."""
from pathlib import Path
import resource
import os
import json
import fcntl

GIB=2**30


def visible_cgroups():
    """Inspect the process's visible cgroup ancestry without leaving its mount."""
    base=Path('/sys/fs/cgroup')
    relative=next((line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines()
                   if '::' in line),None)
    if relative is None: return []
    cg=base if '..' in Path(relative).parts else base/relative.lstrip('/')
    paths=[]
    while cg.is_relative_to(base):
        paths.append(cg)
        if cg==base: break
        cg=cg.parent
    return paths


def reserve_growth(forecast):
    """Atomically reserve peak memory against other controllers and growing cases."""
    directory=os.environ.get('RYDBERG_RESERVATIONS')
    if not directory: return True
    from .queue import process_identity
    from .results import json_write
    registry=Path(directory)
    with (registry/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        outstanding=0
        for path in registry.glob('*.json'):
            rec=json.loads(path.read_text())
            if rec['pid']==os.getpid() or process_identity(rec['pid'])!=rec['birth']: continue
            try: rss=int(Path(f"/proc/{rec['pid']}/statm").read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            except FileNotFoundError: continue
            outstanding+=max(0,rec.get('memory_reserved_bytes',24*GIB)-rss)
        mem=snapshot()
        own=registry/f'{os.getpid()}.json'
        rec=json.loads(own.read_text())
        target=max(forecast,rec.get('memory_reserved_bytes',24*GIB))
        if mem['available_bytes']-outstanding-max(0,target-mem['rss_bytes'])<16*GIB:
            return False
        rec['memory_reserved_bytes']=target
        json_write(own,rec)
        return True


def growth_bytes(current_rss, current_D, next_D, peak_rss=0):
    # RSS between sweeps can miss transient tensor allocations. Use the larger
    # observed value, with a 50% margin. This remains a forecast, not a bound.
    if current_rss<0 or peak_rss<0 or current_D<=0 or next_D<current_D:
        raise ValueError('Nonnegative memory and nondecreasing positive D required')
    return int(max(GIB, max(current_rss,peak_rss)*1.5*(next_D/current_D)**2))


def permitted(current_rss, forecast, available, cap, reserve=16*GIB):
    return forecast<=cap and available-max(0,forecast-current_rss)>=reserve


def snapshot():
    values={line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}
    rss=next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines()
             if line.startswith('VmRSS:'))
    available=values['MemAvailable']
    # Honor cgroup v2 limits along the visible ancestry as well as host memory.
    for p in visible_cgroups():
        maximum=p/'memory.max'; used=p/'memory.current'
        if maximum.exists() and used.exists():
            limit=maximum.read_text().strip()
            if limit!='max': available=min(available,int(limit)-int(used.read_text()))
    return dict(rss_bytes=rss,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                available_bytes=max(0,available))
