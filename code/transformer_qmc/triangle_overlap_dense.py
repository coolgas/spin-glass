"""Resumable L4/L6/L8 m8 chi3/16 campaign with per-group pilot gates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .triangle_correlation_dmrg import case, digest, write, ResourceDeferred
from .triangle_correlation_grid import scan_manifest
from .triangle_correlation_highchi import allocation, guard, profile, worker_limits
from .triangle_correlation_parallel import alive, key
from .triangle_overlap_report import load_moments, difference, within

MODULE = 'transformer_qmc.triangle_overlap_dense'


def task(L, chi, gamma=.5, sid=60000):
    return dict(L=L, m=8, chi=chi, gamma=gamma, sample_id=sid,
                validate=L == 4 and gamma == .5 and sid == 60000)


def jobs():
    gates = [dict(mode='case', task=task(4,c)) for c in (16,3)]
    profiles = [dict(mode='profile',task=task(L,c)) for L in (4,6,8) for c in (16,3)]
    pilots = [dict(mode='case',task=task(L,c,g,s)) for g in (.1,.5,1.)
              for s in (60000,60001) for L in (4,6,8) for c in (16,3)]
    manifest = scan_manifest()
    dense = [dict(mode='case',task=task(p['L'],c,p['gamma'],p['sample_id']))
             for p in manifest['points'] for c in (16,3)]
    # Pilot ordering gives both retention settings and sizes early coverage.
    unique = {}
    for job in gates+profiles+pilots+dense:
        unique.setdefault(name(job),job)
    return list(unique.values())


def name(job):
    return job['mode']+'_'+key(job['task'])


def artifact(root, job):
    return root/('profiles' if job['mode']=='profile' else 'cases')/key(job['task'])/(
        'profile.json' if job['mode']=='profile' else 'result.json')


def audit(folder):
    """Audit original solver status, all saved hashes, raw q2, and companions."""
    row = json.loads((folder/'result.json').read_text())
    assert row['h'] == .15 and row['lambda_rg'] == 1.
    for filename, expected in row['hashes'].items():
        assert digest(folder/filename) == expected, folder/filename
    records, bonds = [], {}
    for record in row['records']:
        filename = f"seed{record['seed']}_D{record['D_MPS']}.npz"
        values,_ = load_moments(folder/filename,row['hashes'][filename],bonds,(row['L'],row['sample_id']))
        records.append(dict(record,observables=values))
    ends = [next(r for r in reversed(records) if r['seed']==s) for s in (0,1)]
    ladder = []
    for s in (0,1):
        history = [r for r in records if r['seed']==s]
        ladder.append(difference(history[-1]['observables'],history[-2]['observables']) if len(history)>1 else None)
    starts = difference(ends[0]['observables'],ends[1]['observables'])
    checks = dict(ladder=ladder,starts=starts)
    qualified = row['solver_passed'] and all(d is not None and within(d) for d in ladder)
    qualified = qualified and within(starts,chi_tolerance=.01,q2_tolerance=.01)
    if row['exact'] is not None:
        exact,_ = load_moments(folder/'exact.npz',row['hashes']['exact.npz'],bonds)
        checks['exact'] = [difference(r['observables'],exact) for r in ends]
        qualified = qualified and all(within(d,chi_tolerance=.01,q2_tolerance=.01) for d in checks['exact'])
    selected = min(ends,key=lambda r:r['energy'])
    return dict(qualified=bool(qualified),checks=checks,result_sha256=digest(folder/'result.json'),
                observables=selected['observables'],result_path=str(folder/'result.json'),
                exact_validation=row['exact'] is not None,
                **{k:row[k] for k in ('L','m','chi','gamma','sample_id','solver_passed')})


def external_workers(roots):
    """Account for other correlation owners, including their future growth."""
    workers = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode().split('\0')
            if '--task-json' not in argv or not any(str(r) in argv for r in roots):
                continue
            if not alive(int(proc.name)):
                continue
            t = json.loads(argv[argv.index('--task-json')+1])
            env = dict(v.split('=',1) for v in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in v)
            chi = t['chi']
            memory = float(env.get('HIGHCHI_MEMORY_GIB',{2:3,3:6,4:12}.get(chi,72)))
            rss = next(int(l.split()[1])/2**20 for l in proc.joinpath('status').read_text().splitlines() if l.startswith('VmRSS:'))
            workers.append(dict(pid=int(proc.name),task=t,threads=int(env.get('OMP_NUM_THREADS','1')),
                                memory_gib=memory,rss_gib=rss,disk_gib={2:.1,3:.4,4:.8}.get(chi,1.)))
        except (FileNotFoundError,ProcessLookupError):
            continue
    return workers


def dependencies(job, passed, profile_exists):
    """Gate every size's dense grid on matched pilots at both retentions."""
    t = job['task']; L,c = t['L'],t['chi']
    gate = task(4,c)
    if job['mode']=='case' and t == gate:
        return []
    if not passed(gate,exact=True):
        return ['exact L4 validation']
    if job['mode']=='profile':
        return []
    if not profile_exists(task(L,c)):
        return ['per-size cost profile']
    pilot = t['gamma'] in (.1,.5,1.) and t['sample_id'] in (60000,60001)
    if not pilot and not all(passed(task(L,pc,g,s)) for pc in (3,16)
                            for g in (.1,.5,1.) for s in (60000,60001)):
        return ['twelve qualified matched per-size pilots']
    return []


def controller(args):
    if args.dispatch_lock:
        with args.dispatch_lock.open('a') as lock:
            return _controller(args, lock)
    return _controller(args, None)


def _controller(args, dispatch_lock):
    root = args.output
    for directory in ('cases','logs','profiles'):
        (root/directory).mkdir(exist_ok=True)
    policy = dict(scan=scan_manifest(),sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                  reuse_roots=list(map(str,args.reuse)),cpu_budget=16,memory_budget_gib=72,
                  memory_headroom_gib=16,disk_headroom_gib=3,no_deadline=True,
                   automatic_retries=False,backend=str(args.backend),
                   profile_threads=args.profile_threads,dispatch_lock=str(args.dispatch_lock))
    manifest = root/'execution_manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == policy
    else:
        write(manifest,policy)
    backend = json.loads(args.backend.read_text())
    assert backend['validated'] and backend['selected'] in ('cpu','cuda')
    selected_backend = backend['selected']
    write(root/'backend_provenance.json',dict(source=str(args.backend),sha256=digest(args.backend),benchmark=backend))
    ledger = root/'attempts.json'
    attempts = json.loads(ledger.read_text()) if ledger.exists() else {}
    running, processes, audits = {}, {}, {}
    status_path = root/'status.json'
    if status_path.exists():
        for n,j in json.loads(status_path.read_text()).get('running',{}).items():
            if alive(j['pid']):
                command = Path(f"/proc/{j['pid']}/cmdline").read_bytes().decode()
                assert MODULE in command and str(root) in command
                running[n] = j
    all_jobs = jobs()
    def checked(t):
        p = root/'cases'/key(t)/'result.json'
        if not p.exists():
            return None
        if key(t) not in audits:
            a = audit(p.parent)
            assert all(a[k]==t[k] for k in ('L','m','chi','gamma','sample_id'))
            audits[key(t)] = a
        return audits[key(t)]
    def passed(t, exact=False):
        a = checked(t)
        return a and a['qualified'] and (not exact or a['exact_validation'])
    def profile_exists(t):
        return artifact(root,dict(mode='profile',task=t)).exists()
    last_export = 0.
    while True:
        if dispatch_lock:
            fcntl.flock(dispatch_lock,fcntl.LOCK_EX)
        for n,j in list(running.items()):
            if n in processes:
                processes[n].poll()
            if alive(j['pid']):
                continue
            exit_path = root/'logs'/f'{n}.exit.json'
            attempts[n] = dict(j,finished=time.time(),
                               outcome='complete' if artifact(root,j).exists() else 'unresolved',
                               exit=json.loads(exit_path.read_text()) if exit_path.exists() else {'missing_exit_record':True})
            write(ledger,attempts)
            del running[n]
        external = external_workers(args.reuse)
        owned = {key(w['task']) for w in external}
        # Reuse completed cases without copying their large checkpoints. An
        # unfinished externally owned or deferred case is never duplicated.
        external_unfinished = set()
        for source in args.reuse:
            for folder in (source/'cases').glob('*'):
                destination = root/'cases'/folder.name
                if destination.exists() or destination.is_symlink():
                    continue
                if (folder/'result.json').exists():
                    r = json.loads((folder/'result.json').read_text())
                    if r['L'] in (4,6,8) and r['m']==8 and r['chi'] in (3,16):
                        audit(folder)
                        destination.symlink_to(folder.resolve(),target_is_directory=True)
                else:
                    external_unfinished.add(folder.name)
        blocked, eligible = {}, []
        for j in all_jobs:
            n = name(j)
            if artifact(root,j).exists():
                if j['mode']=='case': checked(j['task'])
                continue
            if n in running or n in attempts:
                continue
            if j['mode']=='case' and key(j['task']) in owned | external_unfinished:
                blocked[n] = ['external owner or preserved unfinished case']
                continue
            reasons = dependencies(j,passed,profile_exists)
            if reasons: blocked[n] = reasons
            else: eligible.append(j)
        for j in eligible:
            t = j['task']; n = name(j)
            threads,memory,disk = 2,4.,.6
            if j['mode']=='profile':
                threads,memory,disk = args.profile_threads,8. if t['chi']==16 else 4.,1.
            elif t != task(4,t['chi']):
                p = json.loads(artifact(root,dict(mode='profile',task=task(t['L'],t['chi']))).read_text())
                limits = worker_limits(p,allocation()['available_gib'])
                if not limits: continue
                threads = p['selected_threads']
                memory,disk = limits['memory_gib'],limits['disk_gib']
            if sum(w['threads'] for w in external)+sum(w['threads'] for w in running.values())+threads > 16:
                continue
            if sum(w['memory_gib'] for w in external)+sum(w['memory_gib'] for w in running.values())+memory > 72:
                continue
            extra = memory+sum(max(0,w['memory_gib']-w['rss_gib']) for w in external)
            for w in running.values():
                try:
                    rss = next(int(l.split()[1])/2**20 for l in Path(f"/proc/{w['pid']}/status").read_text().splitlines() if l.startswith('VmRSS:'))
                except FileNotFoundError:
                    rss = 0.
                extra += max(0,w['memory_gib']-rss)
            if allocation()['available_gib'] < 16+extra:
                continue
            if shutil.disk_usage(root).free/2**30 < 3+disk+sum(w['disk_gib'] for w in external+list(running.values())):
                continue
            env = dict(os.environ,**{v:str(min(threads,8)) for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},HIGHCHI_MEMORY_GIB=str(memory),
                       DENSE_PROFILE_THREADS=str(args.profile_threads))
            cmd = [sys.executable,'-u','-m',MODULE,'--output',str(root),'--mode',j['mode'],
                   '--task-json',json.dumps(t),'--selected-backend',selected_backend]
            with (root/'logs'/f'{n}.log').open('ab') as log:
                proc = subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            running[n] = dict(j,pid=proc.pid,threads=threads,memory_gib=memory,disk_gib=disk)
            processes[n] = proc
            write(status_path,dict(state='running',running=running))
            print('Started',n,proc.pid,flush=True)
        missing = [name(j) for j in all_jobs if not artifact(root,j).exists()]
        state = 'running' if running else 'waiting_for_resources'
        if not running and not eligible:
            state = 'waiting_for_external_cases' if external else 'complete_with_unresolved_cases'
        if not missing:
            state = 'complete' if all(a['qualified'] for a in audits.values()) else 'complete_with_unresolved_cases'
        write(status_path,dict(state=state,utc=time.time(),running=running,external_workers=external,
            worker_cpu_threads=sum(w['threads'] for w in external+list(running.values())),
            completed_cases=len(audits),qualified_cases=sum(a['qualified'] for a in audits.values()),
            target_cases=2736,pending_jobs=len(missing),blocked_dependencies=blocked,
            attempts=attempts,allocation=allocation(),disk_free_gib=shutil.disk_usage(root).free/2**30))
        if time.monotonic()-last_export > 60 or state.startswith('complete'):
            write(root/'audited_cases.json',list(audits.values()))
            last_export = time.monotonic()
        if state.startswith('complete'):
            if dispatch_lock:
                fcntl.flock(dispatch_lock,fcntl.LOCK_UN)
            return
        if dispatch_lock:
            fcntl.flock(dispatch_lock,fcntl.LOCK_UN)
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse',type=Path,nargs='*',default=[])
    parser.add_argument('--backend',type=Path)
    parser.add_argument('--mode',choices=('controller','profile','case'),default='controller')
    parser.add_argument('--task-json')
    parser.add_argument('--selected-backend',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--profile-threads',type=int,choices=(1,2,4,8),default=8)
    parser.add_argument('--dispatch-lock',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    code,failure = 0,None
    try:
        if args.mode=='controller':
            with (args.output/'controller.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                controller(args)
        elif args.mode=='profile':
            profile(args.output,json.loads(args.task_json),args.selected_backend,
                    max_threads=int(os.environ.get('DENSE_PROFILE_THREADS','8')))
        else:
            case(args.output,**json.loads(args.task_json),resume=True,stage_guard=guard,
                 eigensolver=args.selected_backend,qualify_overlap=True)
    except ResourceDeferred:
        code = 75
        raise SystemExit(code)
    except Exception:
        code,failure = 1,traceback.format_exc()
        print(failure,file=sys.stderr,flush=True)
        if args.mode=='controller': (args.output/'controller_failure.txt').write_text(failure)
        raise
    finally:
        if args.mode!='controller':
            (args.output/'logs').mkdir(exist_ok=True)
            write(args.output/'logs'/f'{args.mode}_{key(json.loads(args.task_json))}.exit.json',
                  dict(exit_code=code,failure=failure,utc=time.time()))


if __name__=='__main__':
    main()
