"""Detached finite CPU queue with cgroup admission and worker adoption on restart."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .results import json_write


def process_identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        if text[0]=='Z': return None
        return text[19]
    except (FileNotFoundError,ProcessLookupError): return None


def capacity(root,active,max_workers,peak_gib=3.,disk_gib=.15):
    cg=Path('/sys/fs/cgroup')
    quota,period=(cg/'cpu.max').read_text().split()
    cores=float(quota)/float(period) if quota!='max' else len(os.sched_getaffinity(0))
    memory_max=(cg/'memory.max').read_text().strip()
    memory_max=float(memory_max) if memory_max!='max' else float('inf')
    used=int((cg/'memory.current').read_text())
    # Account for the other Rydberg controllers; preserve two cores for the
    # measured unrelated GPU workload and system/diagnostic processes.
    own_pids={r['pid'] for r in active.values()}
    external=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in own_pids: continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if b'rydberg_kagome' in args and b'case' in args: external.append(int(p.name))
        except OSError: pass
    cpu_slots=max(0,min(max_workers-len(active),int(cores)-2-len(external)-len(active)))
    # Count current worker RSS so each admitted worker retains its full forecast.
    worker_rss=0
    for rec in active.values():
        try:
            for line in Path(f"/proc/{rec['pid']}/status").read_text().splitlines():
                if line.startswith('VmRSS:'): worker_rss+=int(line.split()[1])*1024
        except FileNotFoundError: pass
    reserve=16*2**30+max(0,len(active)*peak_gib*2**30-worker_rss)+len(external)*peak_gib*2**30
    ram_slots=max(0,int((memory_max-used-reserve)/(peak_gib*2**30)))
    free=shutil.disk_usage(root).free
    disk_slots=max(0,int((free-8*2**30-len(active)*disk_gib*2**30)/(disk_gib*2**30)))
    return min(cpu_slots,ram_slots,disk_slots),dict(cpu_quota=cores,memory_used=used,
                 memory_max=memory_max,disk_free=free,cpu_slots=cpu_slots,ram_slots=ram_slots,disk_slots=disk_slots,
                 external_case_workers=external)


def run(manifest_path,root,max_workers):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=json.loads(Path(manifest_path).read_text())
    if len({t['id'] for t in tasks})!=len(tasks): raise ValueError("Duplicate task IDs")
    stored=root/'tasks.json'
    if stored.exists() and json.loads(stored.read_text())!=tasks: raise ValueError("Manifest changed")
    if not stored.exists(): json_write(stored,tasks)
    active={}; children={}; statuses={}
    for task in tasks:
        key=task['id']; folder=root/'cases'/key
        if (folder/'result.json').exists(): statuses[key]='completed'; continue
        launch=root/'launches'/f'{key}.json'
        if launch.exists():
            rec=json.loads(launch.read_text())
            if process_identity(rec['pid'])==rec['birth']:
                active[key]=rec; statuses[key]='running'
            else: statuses[key]='interrupted_or_failed'
        else: statuses[key]='pending'
    while True:
        for key,rec in list(active.items()):
            child=children.get(key)
            if child is not None: child.poll()
            if process_identity(rec['pid'])!=rec['birth']:
                folder=root/'cases'/key
                statuses[key]='completed' if (folder/'result.json').exists() else 'failed'
                del active[key]
        slots,resources=capacity(root,active,max_workers)
        for task in tasks:
            if slots<=0: break
            key=task['id']
            if statuses[key]!='pending': continue
            folder=root/'cases'/key
            command=[sys.executable,'-B','-u','-m','rydberg_kagome','case','--output',str(folder),
                '--sample',str(task['sample']),'--model',task['model'],'--delta',str(task['delta']),
                '--omega',repr(task['omega']),'--chi',str(task['chi']),
                '--dimensions',*map(str,task['dimensions']),'--pairs',str(task.get('pairs',20000))]
            if task.get('no_exact',False): command.append('--no-exact')
            if task.get('fixed_production',False): command.append('--fixed-production')
            logs=root/'logs'; logs.mkdir(exist_ok=True)
            env=os.environ.copy()
            for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
                env[name]='1'
            env['PYTHONDONTWRITEBYTECODE']='1'
            with (logs/f'{key}.log').open('ab') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                       env=env,start_new_session=True)
            rec=dict(pid=child.pid,birth=process_identity(child.pid),command=command,started=time.time())
            json_write(root/'launches'/f'{key}.json',rec)
            active[key]=rec; children[key]=child; statuses[key]='running'; slots-=1
        counts={s:list(statuses.values()).count(s) for s in set(statuses.values())}
        json_write(root/'status.json',dict(updated=time.time(),controller=os.getpid(),counts=counts,
                    active=active,resources=resources,deadline=None,max_workers=max_workers))
        if not active and not any(s=='pending' for s in statuses.values()): break
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    run(a.manifest,a.output,a.workers)
