"""Matched L6 chi8 scan, with pilot-gated or explicitly D64-capped execution."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .triangle_l6_dense import GAMMAS, IDS, digest, key, write

MODULE = 'transformer_qmc.triangle_l6_chi8_dense'
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
RESERVATION = 13.5
STORAGE_RESERVATION = .77


def task(gamma, sid):
    return dict(L=6, m=6, chi=8, gamma=gamma, sample_id=sid,
                validate=False, qualify_overlap=True)


def folder(root, t):
    return root/'cases'/f"L6_g{t['gamma']:g}_s{t['sample_id']}_m6_c8"


def alive(job):
    path = Path(f"/proc/{job['pid']}/cmdline")
    try:
        return [s for s in path.read_bytes().decode().split('\0') if s] == job['command']
    except (OSError, UnicodeError):
        return False


def rss(job):
    try:
        return next(int(line.split()[1])/2**20 for line in
                    Path(f"/proc/{job['pid']}/status").read_text().splitlines()
                    if line.startswith('VmRSS:'))
    except (OSError, StopIteration):
        return 0.


def audit(root, t, baseline):
    import numpy as np
    from .triangle_overlap_report import load_moments, difference, within
    f = folder(root, t)
    row = json.loads((f/'result.json').read_text())
    assert (row['L'], row['m'], row['chi'], row['h'], row['lambda_rg']) == (6,6,8,.15,1.)
    assert row['overlap_gates_enabled']
    if row.get('max_bond_dimension') == 64:
        assert row['D_ladder'] == [32,64] and row['computation_complete']
        assert all(a['D_MPS'] <= 64 and a['max_bond'] <= 64 for a in row['records'])
    for name, expected in row['hashes'].items():
        assert digest(f/name) == expected, name
    ref = json.loads((baseline/'audits'/(key(t['gamma'],t['sample_id'])+'.json')).read_text())
    assert ref['qualified'] and digest(Path(ref['result_path'])) == ref['result_sha256']
    bonds = {(6,t['sample_id']):ref['bonds_sha256']}
    records = []
    for rec in row['records']:
        name = f"seed{rec['seed']}_D{rec['D_MPS']}.npz"
        obs, source = load_moments(f/name, row['hashes'][name], bonds, (6,t['sample_id']))
        with np.load(f/name) as a:
            np.testing.assert_array_equal(a['disorder_entropy'],[20261027,6,t['sample_id'],1])
            assert np.all((-a['J'] >= .5) & (-a['J'] <= 1.5))
        assert abs(obs['chi_sg_raw_equal_time']-108*obs['q2_raw']) < 1e-10
        records.append(dict(rec,observables=obs))
    ends = [next(a for a in reversed(records) if a['seed']==s) for s in (0,1)]
    deltas = []
    for s in (0,1):
        history = [a for a in records if a['seed']==s]
        deltas.append(difference(history[-1]['observables'],history[-2]['observables'])
                      if len(history)>1 else None)
    starts = difference(ends[0]['observables'],ends[1]['observables'])
    qualified = bool(row['solver_passed'] and all(d is not None and within(d) for d in deltas)
                     and within(starts,chi_tolerance=.01,q2_tolerance=.01))
    best = min(ends,key=lambda a:a['energy'])
    shift = difference(best['observables'],ref['observables'])
    result = dict(task=t,qualified=qualified,observables=best['observables'],
        computation_complete=True,max_bond_dimension=row.get('max_bond_dimension',512),
        numerical_status='qualified' if qualified else 'unqualified_at_bond_cap',
        physical_accuracy_qualified=False,
        baseline_observables=ref['observables'],baseline_audit_sha256=digest(baseline/'audits'/(key(t['gamma'],t['sample_id'])+'.json')),
        bonds_sha256=ref['bonds_sha256'],start_difference=starts,ladder_difference=deltas,
        retention_difference=shift,within_retention_tolerance=bool(within(shift,.01,.02,.02)),
        selected_seed=best['seed'],D_MPS=[a['D_MPS'] for a in ends],
        seconds=row['seconds'],peak_rss_kib=row['peak_rss_kib'],
        archived_bytes=sum(p.stat().st_size for p in f.iterdir() if p.is_file()),
        result_path=str(f/'result.json'),result_sha256=digest(f/'result.json'))
    write(root/'audits'/(key(t['gamma'],t['sample_id'])+'.json'),result)
    return result


def worker(args):
    from .triangle_correlation_dmrg import case, ResourceDeferred
    from .triangle_correlation_highchi import guard
    t = task(args.gamma,args.sample_id)
    out = args.output/'attempts'/(key(args.gamma,args.sample_id)+'.json')
    try:
        result = case(args.output,**t,resume=True,stage_guard=guard,eigensolver='cpu',
                      max_bond_dimension=args.max_bond_dimension)
        write(out,dict(task=t,state='finished',solver_passed=result['solver_passed'],
                       max_bond_dimension=args.max_bond_dimension,utc=time.time()))
    except ResourceDeferred:
        write(out,dict(task=t,state='resource_deferred',error=traceback.format_exc(),utc=time.time()))
        raise SystemExit(75)
    except Exception:
        write(out,dict(task=t,state='failed',error=traceback.format_exc(),utc=time.time()))
        raise


def controller(args):
    from .triangle_correlation_highchi import allocation
    root = args.output
    capped = args.max_bond_dimension == 64
    reservation = 3. if capped else RESERVATION
    storage_reservation = .05 if capped else STORAGE_RESERVATION
    for name in ('audits','attempts','logs'):
        (root/name).mkdir(parents=True,exist_ok=True)
    lock = (root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch_path = root.parent/'pilot_launch.json'
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {'jobs':[]}
    running = {key(j['task']['gamma'],j['task']['sample_id']):j for j in launch['jobs'] if alive(j)}
    status_path = root/'status.json'
    if status_path.exists():
        for name,job in json.loads(status_path.read_text()).get('running',{}).items():
            if alive(job):running[name]=job
    if capped:
        for job in running.values():
            assert '--max-bond-dimension' in job['command']
            assert job['command'][job['command'].index('--max-bond-dimension')+1] == '64'
    processes = {}
    v = args.validation_result or root/'cases/L4_g0.5_s60000_m6_c8/result.json'
    validation = json.loads(v.read_text())
    assert validation['solver_passed'] and validation['overlap_gates_enabled']
    for name,expected in validation['hashes'].items():
        assert digest(v.parent/name)==expected
    pilot_tasks = [task(.1,60000),task(1.,60003),task(1.5,60000),task(1.,60000)]
    priority = [.1,1.5] if capped else [1.,1.025,.975,1.05,.95,.1,1.5]
    all_tasks = [task(g,s) for g in priority+[g for g in GAMMAS if g not in priority] for s in IDS]
    if capped:
        existing = [t for t in all_tasks if (folder(root,t)/'model.pkl').exists()]
        all_tasks = existing+[t for t in all_tasks if t not in existing]
    manifest = dict(scope=dict(L=6,m=6,chi=8,gammas=list(GAMMAS),disorders=list(IDS),cases=456),
        h=.15,lambda_rg=1.,worker_limit=4,worker_threads=4,worker_memory_gib=reservation,
        disk_reserve_gib=3,memory_reserve_gib=16,pilots=pilot_tasks,
        full_dispatch=('user-authorized capped scan; numerical failures are retained and flagged; storage admission still required'
                       if capped else 'requires all four qualified pilots and measured archive projection within disk budget'),
        max_bond_dimension=args.max_bond_dimension,
        measured_storage_projection_factor=1.25,checkpoint_admission_allowance_gib=storage_reservation,
        allowance_is_not_expected_saved_data=True,raw_location='AutoDL only',
        solver=('D32 warmup then D64 cap; two starts; original and overlap gates recorded without escalation'
                if capped else 'unchanged D32,64,128,256,512; two starts; original and overlap gates'),
        runtime='finite queue, no deadline, one attempt per case',baseline=str(args.baseline),
        validation_path=str(v),validation_sha256=digest(v),
        storage_floor_case_bytes=args.storage_floor_case_bytes,
        sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    write(root/'manifest.json',manifest)
    audits = {p.stem:json.loads(p.read_text()) for p in (root/'audits').glob('*.json')}
    phase = 'D64_capped_scan' if capped else 'pilots'
    def external():
        p = args.chi16_root/'pilot_launch.json'
        return [j for j in json.loads(p.read_text())['jobs'] if alive(j)] if p.exists() else []
    def snapshot(state, reason=None, projection=None):
        write(status_path,dict(utc=time.time(),state=state,phase=phase,target=456,
            max_bond_dimension=args.max_bond_dimension,
            completed=len(audits),qualified=sum(a['qualified'] for a in audits.values()),
            excluded=sum(not a['qualified'] for a in audits.values()),running=running,
            attempts={p.stem:json.loads(p.read_text()) for p in (root/'attempts').glob('*.json')},
            reason=reason,projection=projection,disk_free_gib=shutil.disk_usage(root).free/2**30))
    try:
        while True:
            for name,proc in list(processes.items()):
                if proc.poll() is not None:del processes[name]
            for name,job in list(running.items()):
                if not alive(job):
                    if not (folder(root,job['task'])/'result.json').exists() and not (root/'attempts'/(name+'.json')).exists():
                        write(root/'attempts'/(name+'.json'),dict(task=job['task'],state='exited_without_final_result',utc=time.time()))
                    del running[name]
            for t in all_tasks:
                name = key(t['gamma'],t['sample_id'])
                if name not in audits and (folder(root,t)/'result.json').exists():
                    audits[name] = audit(root,t,args.baseline)
            pilot_names = [key(t['gamma'],t['sample_id']) for t in pilot_tasks]
            pilot_done = capped or all(n in audits and audits[n]['qualified'] for n in pilot_names)
            projection = None
            reason = None
            if pilot_done:
                phase = 'D64_capped_scan' if capped else 'scan'
                selection_path = root.parent/'selection.json'
                if not selection_path.exists() or json.loads(selection_path.read_text()).get('selected_chi') != 8:
                    reason = 'four pilots complete; waiting for measured chi16 versus chi8 selection'
                maximum = max([args.storage_floor_case_bytes]+[a['archived_bytes'] for a in audits.values()])
                assert maximum > 0, 'Measured or structural archive estimate required before dispatch'
                remaining = 456-len(audits)
                projected = 1.25*maximum*remaining/2**30
                projection = dict(case_bytes_for_projection=maximum,
                                  max_measured_case_bytes=max([0]+[a['archived_bytes'] for a in audits.values()]),
                                  structural_floor_case_bytes=args.storage_floor_case_bytes,remaining_cases=remaining,
                                  projected_remaining_gib=projected,safety_factor=1.25)
                if projected+3+storage_reservation*4 > shutil.disk_usage(root).free/2**30:
                    reason = 'measured whole-scan archive projection exceeds available disk headroom'
            else:
                for n in pilot_names:
                    if (n in audits and not audits[n]['qualified']) or (n not in audits and n not in running and (root/'attempts'/(n+'.json')).exists()):
                        reason = 'pilot unresolved; preserve diagnostics before extending the grid'
            queue = [t for t in (all_tasks if pilot_done else pilot_tasks)
                     if key(t['gamma'],t['sample_id']) not in audits
                     and key(t['gamma'],t['sample_id']) not in running
                     and not (root/'attempts'/(key(t['gamma'],t['sample_id'])+'.json')).exists()]
            while queue and len(running)<4 and reason is None:
                others = external()
                cpu = sum(j.get('threads',4) for j in running.values())+sum(j.get('threads',4) for j in others)
                if cpu+4>16:break
                jobs = list(running.values())+others
                future = sum(max(0,j.get('memory_reservation_gib',reservation)-rss(j)) for j in jobs)
                if allocation()['available_gib'] < 16+future+reservation:break
                if shutil.disk_usage(root).free/2**30 < 3+storage_reservation*(len(jobs)+1):
                    reason='disk reservation for active checkpoint growth';break
                t=queue.pop(0);name=key(t['gamma'],t['sample_id'])
                env=dict(os.environ,HIGHCHI_MEMORY_GIB=str(reservation),**{k:'4' for k in THREAD_VARS})
                cmd=[sys.executable,'-u','-m',MODULE,'--output',str(root),'--worker','--gamma',str(t['gamma']),'--sample-id',str(t['sample_id']),
                     '--max-bond-dimension',str(args.max_bond_dimension)]
                with (root/'logs'/(name+'.log')).open('ab') as stream:
                    proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                running[name]=dict(task=t,pid=proc.pid,command=cmd,threads=4,memory_reservation_gib=reservation)
                processes[name]=proc
            state = 'running' if reason is None else 'draining_before_hold'
            if not running:
                if reason:state='held'
                elif not queue:state='complete' if len(audits)==456 and all(a['qualified'] for a in audits.values()) else 'complete_with_exclusions'
                else:state='resource_deferred'
            snapshot(state,reason,projection)
            if state in ('held','resource_deferred','complete','complete_with_exclusions'):
                return
            time.sleep(10)
    except Exception:
        snapshot('controller_error',traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--chi16-root',type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--gamma',type=float)
    parser.add_argument('--sample-id',type=int)
    parser.add_argument('--max-bond-dimension',type=int,choices=(64,512),default=512)
    parser.add_argument('--validation-result',type=Path)
    parser.add_argument('--storage-floor-case-bytes',type=int,default=0)
    args=parser.parse_args()
    if args.worker:worker(args)
    else:
        assert args.baseline and args.chi16_root
        controller(args)


if __name__=='__main__':
    main()
