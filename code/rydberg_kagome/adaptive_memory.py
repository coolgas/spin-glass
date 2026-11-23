"""Live shared-host headroom checks, independent of DMRG acceptance criteria."""
from pathlib import Path
import resource

GIB=2**30


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
    relative=next((line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines()
                   if '::' in line),None)
    if relative is not None:
        base=Path('/sys/fs/cgroup'); cg=base/relative.lstrip('/')
        # A cgroup namespace can expose ../ components in /proc/self/cgroup.
        # Never traverse outside the visible mount; inspect its root instead.
        if '..' in Path(relative).parts: cg=base
        paths=[cg,*cg.parents]
        for p in paths:
            if p==base.parent: break
            maximum=p/'memory.max'; used=p/'memory.current'
            if maximum.exists() and used.exists():
                limit=maximum.read_text().strip()
                if limit!='max': available=min(available,int(limit)-int(used.read_text()))
    return dict(rss_bytes=rss,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                available_bytes=max(0,available))
