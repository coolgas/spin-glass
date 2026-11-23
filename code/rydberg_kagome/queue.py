"""Detached finite CPU queue with cgroup admission and worker adoption on restart."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import resource
import signal
import subprocess
import sys
import time
from .results import json_write,digest
from .storage import CASE_LIMIT,LOG_LIMIT


def process_identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        if text[0]=='Z': return None
        return text[19]
    except (FileNotFoundError,ProcessLookupError): return None


def completed_status(folder,task):
    try:
        result=json.loads((folder/'result.json').read_text())
        for name,sha in result['hashes'].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder.resolve()) or digest(path)!=sha:
                return 'corrupt'
        if task.get('adaptive'):
            from .adaptive_case import retire_committed_restart
            retire_committed_restart(folder,result)
    except (OSError,ValueError,KeyError):
        return 'corrupt'
    return 'completed' if result.get('sampling_moment_check') else 'sampling_failed'


def restart_status(folder,attempt,task):
    if not (folder/'resume.json').exists(): return 'failed'
    if (folder/'worker-exit.json').exists():
        exit_record=json.loads((folder/'worker-exit.json').read_text())
        if exit_record['result'].get('state')=='memory_deferred':
            if (folder/'memory-deferral.json').exists():
                required=json.loads((folder/'memory-deferral.json').read_text())['forecast_bytes']
                if required>task.get('memory_gib',64)*2**30*.85: return 'memory_blocked'
            return 'pending'
        if exit_record['result'].get('state')=='paused': return 'pending'
    return 'pending' if attempt<3 else 'failed'


def enforce_deadline(rec,task,folder):
    """Bound stuck invocations, including adopted workers; keep committed restart."""
    hard=task.get('hard_seconds',task.get('seconds',72*3600)+3600)
    if time.time()-rec['started']<hard or process_identity(rec['pid'])!=rec['birth']: return
    receipt=Path(folder)/f"hard-timeout-attempt{rec['attempt']}.json"
    if not receipt.exists():
        json_write(receipt,dict(pid=rec['pid'],birth=rec['birth'],updated=time.time(),hard_seconds=hard))
    try: os.kill(rec['pid'],signal.SIGKILL)
    except ProcessLookupError: pass


def storage_capacity(used_bytes,free_bytes,active_count,pending_count,max_workers,
                     budget_gib=64.,case_work_gib=2.,final_case_mib=6.,reserve_gib=128.,audit_count=0):
    """Reserve active write growth and all pending compact outputs before launch."""
    gib=2**30
    # Reserve two GiB beyond each active case's measured use, including
    # replacement pairs and failure evidence between admission inventories.
    working=active_count*case_work_gib*gib
    pending=pending_count*final_case_mib*2**20
    audit=audit_count*512*2**20
    remaining=min(budget_gib*gib-used_bytes-working-pending-audit,
                  free_bytes-reserve_gib*gib-working-pending-audit)
    slots=max(0,min(max_workers-active_count,int(remaining//(case_work_gib*gib))))
    return slots,dict(campaign_used_bytes=used_bytes,campaign_budget_gib=budget_gib,
                     pending_output_bytes=pending,working_reserve_bytes=working,
                     filesystem_reserve_gib=reserve_gib,disk_slots=slots)


def capacity(root,active,max_workers,peak_gib=24.,disk_gib=2.,storage=None):
    # Ali has no visible cgroup cap in the current inventory. Estimate shared
    # CPU headroom from the live load average and reserve four cores/case plus
    # two host cores. On cgroup-constrained deployments, also honor that quota.
    online=len(os.sched_getaffinity(0)); load=os.getloadavg()[0]
    quota_cores=float(online)
    from .adaptive_memory import snapshot,visible_cgroups
    for cg in visible_cgroups():
        if (cg/'cpu.max').exists():
            quota,period=(cg/'cpu.max').read_text().split()
            if quota!='max': quota_cores=min(quota_cores,float(quota)/float(period))
    background_load=max(0.,load-4*len(active))
    cpu_headroom=max(0.,min(online,quota_cores)-background_load-2.-4*len(active))
    cpu_slots=max(0,min(max_workers-len(active),int(cpu_headroom//4)))
    available=snapshot()['available_bytes']
    rss=[]
    for rec in active.values():
        try: rss.append(int(Path(f"/proc/{rec['pid']}/statm").read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))
        except FileNotFoundError: rss.append(0)
    reserve=16*2**30+sum(max(0,rec.get('memory_reserved_bytes',peak_gib*2**30)-r)
                        for rec,r in zip(active.values(),rss))
    ram_slots=max(0,int((available-reserve)/(peak_gib*2**30)))
    free=shutil.disk_usage(root).free
    disk_slots=max(0,int((free-8*2**30-len(active)*disk_gib*2**30)/(disk_gib*2**30)))
    if storage is not None:
        disk_slots=min(disk_slots,storage['disk_slots'])
    return min(cpu_slots,ram_slots,disk_slots),dict(online_cpus=online,load1=load,cpu_headroom=cpu_headroom,
                 memory_available=available,peak_gib_per_case=peak_gib,disk_free=free,
                 cpu_slots=cpu_slots,ram_slots=ram_slots,disk_slots=disk_slots)


def run(manifest_path,root,max_workers):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    for name in ('cases','launches','logs'): (root/name).mkdir(exist_ok=True)
    lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=json.loads(Path(manifest_path).read_text())
    by_id={t['id']:t for t in tasks}
    if len({t['id'] for t in tasks})!=len(tasks): raise ValueError("Duplicate task IDs")
    stored=root/'tasks.json'
    if stored.exists() and json.loads(stored.read_text())!=tasks: raise ValueError("Manifest changed")
    if not stored.exists(): json_write(stored,tasks)
    active={}; children={}; statuses={}; attempts={}
    registry=root.parent/'queue-reservations'; registry.mkdir(exist_ok=True)
    shared_lock=(registry/'admission.lock').open('a')
    compact=bool(tasks) and all(t.get('adaptive') and t.get('final_state_retention') in ('analysis','full') for t in tasks)
    campaign_bytes=0
    for task in tasks:
        key=task['id']; folder=root/'cases'/key
        launch=root/'launches'/f'{key}.json'
        attempts[key]=json.loads(launch.read_text()).get('attempt',1) if launch.exists() else 0
        if (folder/'result.json').exists(): statuses[key]=completed_status(folder,task); continue
        if launch.exists():
            rec=json.loads(launch.read_text())
            if process_identity(rec['pid'])==rec['birth']:
                active[key]=rec; statuses[key]='running'
            else:
                # Re-admit dead workers through the case's verified restart path.
                statuses[key]=restart_status(folder,attempts[key],task)
        else: statuses[key]='pending'
    while True:
        for key,rec in list(active.items()):
            enforce_deadline(rec,by_id[key],root/'cases'/key)
            child=children.get(key)
            if child is not None: child.poll()
            if process_identity(rec['pid'])!=rec['birth']:
                folder=root/'cases'/key
                if (folder/'result.json').exists():
                    task=next(t for t in tasks if t['id']==key)
                    statuses[key]=completed_status(folder,task)
                else:
                    task=next(t for t in tasks if t['id']==key)
                    statuses[key]=restart_status(folder,attempts.get(key,0),task)
                del active[key]
        fcntl.flock(shared_lock,fcntl.LOCK_EX)
        shared_active={}
        for reservation in registry.glob('*.json'):
            rec=json.loads(reservation.read_text())
            if process_identity(rec['pid'])==rec['birth']: shared_active[str(rec['pid'])]=rec
        for key,rec in active.items():
            if str(rec['pid']) in shared_active: active[key]=shared_active[str(rec['pid'])]
            else: shared_active[str(rec['pid'])]=rec
        storage=None
        if compact:
            # Fresh inventory under the shared admission lock; includes all evidence.
            campaign_bytes=int(subprocess.check_output(['du','-sk',str(root)],text=True).split()[0])*1024
            _,storage=storage_capacity(campaign_bytes,shutil.disk_usage(root).free,len(active),
                sum(s=='pending' for s in statuses.values()),max_workers,
                audit_count=sum(t.get('final_state_retention')=='full' and statuses[t['id']] in ('pending','running') for t in tasks))
        slots,resources=capacity(root,shared_active,max_workers,storage=storage)
        slots=min(slots,max_workers-len(active))
        if storage is not None: resources.update(storage)
        for task in tasks:
            if slots<=0: break
            key=task['id']
            if statuses[key]!='pending': continue
            folder=root/'cases'/key
            reserved=24*2**30
            if (folder/'memory-deferral.json').exists():
                reserved=max(reserved,json.loads((folder/'memory-deferral.json').read_text())['forecast_bytes'])
                # Reserve the requested growth before retrying, rather than racing
                # the same insufficient headroom after every process exit.
                extra=reserved-24*2**30
                if resources['memory_available']-extra<16*2**30: continue
                available=resources['memory_available']
                for rec in shared_active.values():
                    try: rss=int(Path(f"/proc/{rec['pid']}/statm").read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                    except FileNotFoundError: rss=0
                    available-=max(0,rec.get('memory_reserved_bytes',24*2**30)-rss)
                if available-reserved<16*2**30: continue
            if compact:
                from .storage import allocated_bytes
                if folder.exists() and allocated_bytes(folder)>CASE_LIMIT-512*2**20:
                    statuses[key]='storage_blocked'; continue
            if task.get('adaptive',False):
                code=Path(__file__).resolve().parents[1]
                command=[sys.executable,'-B','-u','-c',
                    'from rydberg_kagome.queue import adaptive_worker; import sys; adaptive_worker(*sys.argv[1:])',
                    str(manifest_path),str(root),key]
                env_extra={'PYTHONPATH':str(code)}
            else:
                command=[sys.executable,'-B','-u','-m','rydberg_kagome','case','--L',str(task.get('L',4)),
                    '--output',str(folder),'--sample',str(task['sample']),'--model',task['model'],
                    '--delta',str(task['delta']),'--omega',repr(task['omega']),'--chi',str(task['chi']),
                    '--dimensions',*map(str,task['dimensions']),'--pairs',str(task.get('pairs',20000))]
                command += ['--checkpoint-interval',str(task.get('checkpoint_interval',30))]
                if task.get('no_exact',False): command.append('--no-exact')
                if task.get('fixed_production',False): command.append('--fixed-production')
                env_extra={'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
            logs=root/'logs'; logs.mkdir(exist_ok=True)
            env=os.environ.copy(); env.update(env_extra)
            for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
                env[name]=str(task.get('threads',4 if task.get('adaptive',False) else 1))
            env['PYTHONDONTWRITEBYTECODE']='1'
            if (folder/'worker-exit.json').exists():
                # Keep prior controlled-exit evidence without mistaking it for
                # the terminal state of this new invocation.
                (folder/'worker-exit.json').rename(folder/f'worker-exit-attempt{attempts[key]}.json')
            if compact:
                env['RYDBERG_CASE_ROOT']=str(folder)
                env['RYDBERG_RESERVATIONS']=str(registry)
            with (logs/f'{key}.log').open('ab') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                       env=env,start_new_session=True)
            rec=dict(pid=child.pid,birth=process_identity(child.pid),command=command,started=time.time(),
                     attempt=attempts[key]+1,root=str(root),memory_reserved_bytes=reserved)
            json_write(root/'launches'/f'{key}.json',rec)
            json_write(registry/f'{child.pid}.json',rec)
            active[key]=rec; children[key]=child; statuses[key]='running'; slots-=1
            shared_active[str(child.pid)]=rec
            # Recalculate admission after every spawn, including its reservation.
            slots=min(slots,capacity(root,shared_active,max_workers,storage=storage)[0])
            # Worker growth reservation waits until its controller record commits.
            attempts[key]=attempts.get(key,0)+1
        fcntl.flock(shared_lock,fcntl.LOCK_UN)
        counts={s:list(statuses.values()).count(s) for s in set(statuses.values())}
        json_write(root/'status.json',dict(updated=time.time(),controller=os.getpid(),counts=counts,
                    active=active,resources=resources,deadline=None,max_workers=max_workers))
        if not active and not any(s=='pending' for s in statuses.values()): break
        time.sleep(10)


def adaptive_worker(manifest_path,root,task_id):
    """Run one manifest task with fixed four-thread numerical libraries."""
    root=Path(root).resolve(); tasks=json.loads(Path(manifest_path).read_text())
    matches=[t for t in tasks if t['id']==task_id]
    if len(matches)!=1: raise ValueError('Task ID must occur exactly once in manifest')
    task=matches[0]
    limit=int(task.get('memory_gib',64)*2**30)
    resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    # This also bounds a stuck worker while its controller is absent.
    signal.alarm(int(task.get('hard_seconds',task.get('seconds',72*3600)+3600)))
    from .storage import BoundedLog
    sys.stdout=sys.stderr=BoundedLog(root/'logs'/f'{task_id}.log')
    os.environ['RYDBERG_CASE_ROOT']=str(root/'cases'/task_id)
    # Barrier: do not race controller creation of the admission receipt.
    registry=Path(os.environ['RYDBERG_RESERVATIONS'])
    with (registry/'admission.lock').open('a') as barrier:
        fcntl.flock(barrier,fcntl.LOCK_EX)
    os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]=str(task.get('threads',4))
    from .campaign import run_case
    folder=root/'cases'/task['id']
    result=run_case(str(folder),task.get('L',6),task['sample'],task['model'],task['delta'],task['omega'],
        task['chi'],task['dimensions'],task['pairs'],task.get('max_sweeps',120),False,False,True,
        task.get('seconds',72*3600),None,task.get('memory_gib',64),None,
        task.get('checkpoint_interval',30),task.get('final_state_retention','full'))
    print(json.dumps({k:result.get(k) for k in ('state','completed','solver_pass','sampling_moment_check','seconds')}),flush=True)
    if not result.get('completed'): json_write(folder/'worker-exit.json',dict(updated=time.time(),result=result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    run(a.manifest,a.output,a.workers)
