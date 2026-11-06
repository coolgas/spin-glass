"""Finite chi=8,16,32 L8 study, admitted after the original retention campaign.

One owner per case; short profiles select threads, production retains the original
DMRG gates. A resource deferral is distinct from a numerical solver failure.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .triangle_correlation_dmrg import case, digest, write, ResourceDeferred
from .triangle_correlation_parallel import alive, key, memory_headroom

MS = [6, 8, 4]
CHIS = [8, 16, 32]
FIELDS = [.1, .5, 1.]
IDS = [60000, 60001]
MODULE = 'transformer_qmc.triangle_correlation_highchi'


Deferred = ResourceDeferred


def allocation():
    cg = Path('/sys/fs/cgroup')
    quota, period = (cg/'cpu.max').read_text().split()
    return dict(cpu_cores=int(quota)/int(period),
                memory_gib=int((cg/'memory.max').read_text())/2**30,
                available_gib=memory_headroom(),
                unused_gib=(int((cg/'memory.max').read_text())-
                            int((cg/'memory.current').read_text()))/2**30,
                memory_availability='unused plus reclaimable inactive file pages')


def rss_gib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])/2**20


def estimates(model, mpo, dimension):
    """Conservative storage and contraction workspace, including mixer growth.

    TeNPy's generic estimate assumes dense MPO channels even where its sparse
    block structure avoids them. Admission uses sparse contractions
    plus 24 Lanczos/work vectors and a 1.5 workspace margin. Profiles and stage
    RSS are retained to make subsequent tuning reviewable.
    """
    count = len(model['sites'])
    chi = model['decoders'][0].shape[1]
    bonds = [min(2*dimension, chi**min(i, count-i)) for i in range(count+1)]
    mps = 8*sum(chi*bonds[i]*bonds[i+1] for i in range(count))
    env = 16*sum(max(mpo.H_MPO.chi[i:i+2])*bonds[i]**2 for i in range(count))
    vector = 8*max(bonds[i]*chi*chi*bonds[i+2] for i in range(count-1))
    ram = 2. + 1.5*(3*mps+env+24*vector)/2**30
    # All levels and both starts remain archived; at most 1+1/4+... per start.
    checkpoints = 2.8*mps/2**30 + .4
    return dict(memory_gib=ram, checkpoint_gib=checkpoints,
                effective_D_with_mixer=2*dimension)


def guard(folder, model, mpo, psi, dimension, stage):
    if dimension is None:
        chi = model['decoders'][0].shape[1]
        estimate = dict(memory_gib=2.+32*chi**4/2**30,
                        checkpoint_gib=.5+32*chi**4/2**30)
    else:
        estimate = estimates(model, mpo, dimension)
    available = allocation()['available_gib']
    need = max(0., estimate['memory_gib']-rss_gib())
    disk = shutil.disk_usage(folder).free/2**30
    row = dict(stage=stage, **estimate, rss_gib=rss_gib(),
               available_gib=available, disk_free_gib=disk, utc=time.time())
    with (folder/'resource_stages.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    reason = None
    reservation = float(os.environ.get('HIGHCHI_MEMORY_GIB', '72'))
    if estimate['memory_gib'] > reservation:
        reason = 'next stage exceeds this worker memory reservation'
    elif available < 16+need:
        reason = 'memory headroom before next DMRG stage'
    elif disk < 3+estimate['checkpoint_gib']:
        reason = 'disk headroom for archived checkpoints'
    if reason:
        write(folder/'deferred.json', dict(reason=reason, **row))
        raise Deferred(reason)
    write(folder/'progress.json', row)


def backend_benchmark(root):
    import torch
    from scipy.linalg import eigh
    from ising.hamiltonian import Hamiltonian
    from .physics import make_af_sample
    from .triangle_effective_rg import triangles, Operator
    sample = make_af_sample(4, 60000)
    records = []
    # Actual isolated superblocks, same matrices for CPU and GPU.
    for m in (6, 8):
        first = triangles(Hamiltonian(48, sample.bonds, sample.J, h=.15), .5, m)
        anchored = {}
        for t, sites in enumerate(first['sites']):
            anchor, = [i//3 for i in sites if i % 3 == 0]
            anchored[anchor % 4, anchor//4] = t
        group = [anchored[r, s] for s in range(2) for r in range(2)]
        inverse = {t:a for a,t in enumerate(group)}
        terms = [(c, [(inverse[t], matrix) for t,matrix in factors])
                 for c,factors in first['operator'].terms
                 if all(t in inverse for t,_ in factors)]
        matrix = Operator([m]*4, terms).dense()
        with threadpool_limits(limits=4):
            start = time.monotonic()
            cpu_values, cpu_vectors = eigh(matrix)
            cpu_seconds = time.monotonic()-start
        row = dict(m=m, cpu_threads=4, cpu_seconds=cpu_seconds)
        try:
            torch.cuda.synchronize()
            start = time.monotonic()
            device_matrix = torch.as_tensor(matrix, dtype=torch.float64, device='cuda')
            values, vectors = torch.linalg.eigh(device_matrix)
            gpu_values, gpu_vectors = values.cpu().numpy(), vectors.cpu().numpy()
            torch.cuda.synchronize()
            gpu_seconds = time.monotonic()-start
            subspaces = {str(k):float(np.max(abs(np.linalg.svd(
                cpu_vectors[:, :k].T@gpu_vectors[:, :k], compute_uv=False)-1)))
                for k in CHIS}
            k = 32
            row.update(cuda_seconds=gpu_seconds, subspaces=subspaces,
            eigenvalue_error=float(np.max(abs(cpu_values-gpu_values))),
            retained_subspace_error=max(subspaces.values()),
            residual=float(np.linalg.norm(matrix@gpu_vectors[:, :k]-
                                           gpu_vectors[:, :k]*gpu_values[:k])),
            orthogonality_error=float(np.linalg.norm(gpu_vectors[:, :k].T@gpu_vectors[:, :k]-np.eye(k))))
            assert row['eigenvalue_error'] < 1e-9, row
            assert row['residual'] < 1e-8, row
            assert row['retained_subspace_error'] < 1e-8, row
            assert row['orthogonality_error'] < 1e-9, row
            row['cuda_validated'] = True
            del device_matrix, values, vectors
        except Exception:
            row.update(cuda_validated=False, cuda_failure=traceback.format_exc())
        records.append(row)
        write(root/'backend_benchmark.json', dict(records=records, validated=len(records)==2,
            selected='cuda' if all(r['cuda_validated'] for r in records) and
                               sum(r['cuda_seconds'] for r in records) <
                               sum(r['cpu_seconds'] for r in records) else 'cpu'))
        print(json.dumps(row), flush=True)


def profile(root, task, backend, max_threads=8):
    from ising.hamiltonian import Hamiltonian
    from tenpy.algorithms import dmrg
    from .physics import make_af_sample
    from .triangle_effective_rg import hierarchy
    from .triangle_effective_mps import build, initial_state
    folder = root/'profiles'/key(task)
    folder.mkdir(parents=True, exist_ok=True)
    write(folder/'inputs.json', task)
    sample = make_af_sample(task['L'], task['sample_id'])
    start = time.monotonic()
    model = hierarchy(Hamiltonian(3*task['L']**2, sample.bonds, sample.J, h=.15),
                      task['gamma'], task['m'], task['chi'], eigensolver=backend)
    mpo, order = build(model)
    construction_seconds = time.monotonic()-start
    # Keep the profile state separate from every scientific production start.
    psi = initial_state(mpo, 211)
    checkpoint = pickle.dumps(psi)
    records = []
    for threads in (1, 2, 4, 8):
        if threads > max_threads:
            continue
        with threadpool_limits(limits=threads):
            state = pickle.loads(checkpoint)
            engine = dmrg.TwoSiteDMRGEngine(state, mpo, dict(mixer=False,
                min_sweeps=2, max_sweeps=2, max_E_err=0., max_S_err=0.,
                max_trunc_err=None,
                trunc_params=dict(chi_max=16, svd_min=1e-12)))
            begin = time.monotonic()
            energy, state = engine.run()
            records.append(dict(threads=threads, seconds=time.monotonic()-begin,
                                energy=float(np.real(energy)), threadpools=threadpool_info(),
                                max_truncation_error=float(max(engine.trunc_err_list)),
                                canonical_error=float(np.linalg.norm(state.norm_test()))))
            print(json.dumps({k:records[-1][k] for k in ('threads','seconds','energy')}), flush=True)
    disagreement = max(r['energy'] for r in records)-min(r['energy'] for r in records)
    assert disagreement/(3*task['L']**2) < 1e-8, disagreement
    # Prefer throughput when several full workers fit; prefer latency otherwise.
    affordable = [estimates(model,mpo,d) for d in (32,64,128,256,512)
                  if estimates(model,mpo,d)['memory_gib'] <= 72]
    reservation = affordable[-1]['memory_gib'] if affordable else 72.
    # Account for both CPU slots and the memory available for full workers.
    def throughput(row):
        slots = max(1, min(16//row['threads'], int(72/reservation)))
        return slots/row['seconds']
    selected = max(records, key=throughput)['threads']
    result = dict(task=task, backend=backend, construction_seconds=construction_seconds,
        profile_source=str(Path(__file__).resolve()), profile_source_sha256=digest(Path(__file__)),
        truncation_policy='D16 timing only: record truncation and canonical errors; production gates unchanged',
        records=records, selected_threads=selected, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        production_estimates={str(d):estimates(model,mpo,d) for d in (32,64,128,256,512)},
        mpo_bonds=list(map(int,mpo.H_MPO.chi)), scope='D16 two-sweep performance only; no scientific convergence claim')
    write(folder/'profile.json', result)


def worker_limits(profile_record, available):
    """Reserve the largest affordable stage; guards enforce that reservation."""
    candidates = [(int(d), e) for d,e in profile_record['production_estimates'].items()
                  if e['memory_gib'] <= min(72., available-16)]
    if not candidates:
        return None
    dimension, estimate = max(candidates)
    return dict(memory_gib=estimate['memory_gib'], disk_gib=estimate['checkpoint_gib'],
                admitted_dimension=dimension)


def worker(args):
    if args.mode == 'backend':
        backend_benchmark(args.output)
        return
    task = json.loads(args.task_json)
    if args.mode == 'profile':
        profile(args.output, task, args.backend)
        return
    try:
        case(args.output, **task, resume=True, stage_guard=guard, eigensolver=args.backend)
    except Deferred:
        raise SystemExit(75)


def controller(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    jobs = []
    # Validation first for each local dimension, followed by an L8 cost profile.
    # Round-robin fields/disorders avoids completing only the easiest field.
    for chi in CHIS:
        for m in MS:
            validation = dict(L=4,gamma=.5,sample_id=60000,m=m,chi=chi,validate=True)
            jobs.append(dict(mode='case',task=validation))
            jobs.append(dict(mode='profile',task=dict(validation,L=8,validate=False)))
        for gamma in FIELDS:
            for sid in IDS:
                for m in MS:
                    jobs.append(dict(mode='case',task=dict(L=8,gamma=gamma,sample_id=sid,m=m,chi=chi,validate=False)))
    policy = dict(scope=dict(L=8,m=MS,chi=CHIS,gammas=FIELDS,disorders=IDS),
        target_L8=54, validation='exact reduced-space L4 at Gamma=.5, seed 60000, each m/chi',
        predecessors=args.predecessor, memory_reserve_gib=16,disk_reserve_gib=3,
        worker_cpu_budget=16, max_worker_memory_gib=72, dense_scan='held',
        gpu='validated and timed float64 local eigensolver; TeNPy DMRG remains CPU',
        convergence='unchanged two-start D32..D512 gates; resource deferrals stay unresolved',
        sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    if (root/'manifest.json').exists():
        assert json.loads((root/'manifest.json').read_text()) == policy
    else:
        write(root/'manifest.json',policy)
    ledger_path = root/'attempts.json'
    attempts = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    running = {}
    processes = {}
    status_path = root/'status.json'
    if status_path.exists():
        for name,job in json.loads(status_path.read_text()).get('running',{}).items():
            if alive(job['pid']):
                command=Path(f"/proc/{job['pid']}/cmdline").read_bytes().decode()
                assert MODULE in command and str(root) in command
                running[name] = job
    def complete(job):
        if job['mode']=='backend':
            p=root/'backend_benchmark.json'
            return p.exists() and json.loads(p.read_text()).get('validated',False)
        t=job['task']
        p=root/('profiles' if job['mode']=='profile' else 'cases')/key(t)/('profile.json' if job['mode']=='profile' else 'result.json')
        return p.exists()
    def gate_passed(t):
        gate=dict(t,L=4,gamma=.5,sample_id=60000,validate=True)
        p=root/'cases'/key(gate)/'result.json'
        return p.exists() and json.loads(p.read_text())['solver_passed']
    while True:
        for name,job in list(running.items()):
            if name in processes:
                processes[name].poll()
            if alive(job['pid']):
                continue
            terminal = root/'logs'/f'{name}.exit.json'
            exit_record = json.loads(terminal.read_text()) if terminal.exists() else {}
            attempts[name]=dict(job, outcome='complete' if complete(job) else 'unresolved',
                                exit=exit_record, finished=time.time())
            write(ledger_path,attempts)
            del running[name]
        predecessors=[pid for pid in args.predecessor if alive(pid)]
        backend_path=root/'backend_benchmark.json'
        backend_ready=backend_path.exists() and json.loads(backend_path.read_text()).get('validated',False)
        backend=json.loads(backend_path.read_text())['selected'] if backend_ready else 'cpu'
        eligible = []
        if not predecessors:
            if not backend_ready:
                eligible=[] if 'backend' in attempts else [dict(mode='backend')]
            else:
                for job in jobs:
                    t=job['task']
                    name=job['mode']+'_'+key(t)
                    if complete(job) or name in running or name in attempts:
                        continue
                    if (t['L']==8 or job['mode']=='profile') and not gate_passed(t):
                        continue
                    if t['L']==8 and job['mode']=='case':
                        prof=root/'profiles'/key(dict(t,gamma=.5,sample_id=60000))/'profile.json'
                        if not prof.exists():
                            continue
                    eligible.append(job)
                if eligible:
                    # Finish the first actionable chi group before admitting later ones.
                    first_chi = min(j['task']['chi'] for j in eligible)
                    eligible = [j for j in eligible if j['task']['chi'] == first_chi]
        for job in eligible:
            t=job.get('task')
            name=job['mode']+('_'+key(t) if t else '')
            if name in attempts or name in running:
                continue
            threads, memory, disk = 4, 8., .6
            if t and t['L']==4:
                memory={8:8.,16:12.,32:24.}[t['chi']]
            if job['mode']=='profile':
                # One profile owns the whole 16-thread worker budget while timing.
                threads,memory,disk=16,16.,1.
            elif t and t['L']==8:
                prof=root/'profiles'/key(dict(t,gamma=.5,sample_id=60000))/'profile.json'
                p=json.loads(prof.read_text())
                threads=p['selected_threads']
                limits=worker_limits(p, allocation()['available_gib'])
                if limits is None:
                    continue
                memory=limits['memory_gib']
                disk=limits['disk_gib']
            # Case stages pause before an unaffordable D; reserve what can fit.
            if sum(j['threads'] for j in running.values())+threads>16:
                continue
            if sum(j['memory_gib'] for j in running.values())+memory>72:
                continue
            extra=memory
            for current in running.values():
                try:
                    lines=Path(f"/proc/{current['pid']}/status").read_text().splitlines()
                    rss=next(int(l.split()[1])/2**20 for l in lines if l.startswith('VmRSS:'))
                    extra+=max(0.,current['memory_gib']-rss)
                except FileNotFoundError:
                    pass
            if allocation()['available_gib']<16+extra:
                continue
            if shutil.disk_usage(root).free/2**30<3+disk+sum(j['disk_gib'] for j in running.values()):
                # Permit one resumable case to use the dimensions that fit.
                if running or shutil.disk_usage(root).free/2**30<4:
                    continue
            env=dict(os.environ,OMP_NUM_THREADS=str(min(threads,8)),OPENBLAS_NUM_THREADS=str(min(threads,8)),
                     MKL_NUM_THREADS=str(min(threads,8)),NUMEXPR_NUM_THREADS=str(min(threads,8)),
                     HIGHCHI_MEMORY_GIB=str(memory))
            command=[sys.executable,'-u','-m',MODULE,'--output',str(root),'--mode',job['mode'],'--backend',backend]
            if t:command+=['--task-json',json.dumps(t)]
            with (root/'logs'/f'{name}.log').open('ab') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=log,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            running[name]=dict(job,pid=process.pid,threads=threads,memory_gib=memory,disk_gib=disk)
            processes[name]=process
            write(status_path,dict(state='running',running=running))
            print('Started',name,process.pid,'threads',threads,flush=True)
        rows=[json.loads(p.read_text()) for p in (root/'cases').glob('*/result.json')]
        pending=[job for job in jobs if not complete(job)]
        blocked=[]
        for job in pending:
            t=job['task']
            if t['L']==8 and not gate_passed(t):
                blocked.append(dict(mode=job['mode'],task=t,reason='L4 validation not passing or unfinished'))
            elif t['L']==8 and job['mode']=='case' and not (
                    root/'profiles'/key(dict(t,gamma=.5,sample_id=60000))/'profile.json').exists():
                blocked.append(dict(mode=job['mode'],task=t,reason='thread profile unfinished or failed'))
        active=bool(running)
        actionable=bool(eligible)
        state='waiting_for_predecessors' if predecessors else ('running' if active else 'waiting_for_resources')
        if not predecessors and not active and not actionable:
            state='complete' if not pending and all(r['solver_passed'] for r in rows) else 'complete_with_unresolved_cases'
        write(status_path,dict(state=state,predecessors_active=predecessors,
            running=running,solver_passing_L8=sum(r['L']==8 and r['solver_passed'] for r in rows),
            completed_L8=sum(r['L']==8 for r in rows),target_L8=54,
            validations_passed=sum(r['L']==4 and r['solver_passed'] for r in rows),
            pending_jobs=len(pending),blocked_dependencies=blocked,attempts=attempts,allocation=allocation(),
            disk_free_gib=shutil.disk_usage(root).free/2**30,backend=backend,dense_scan='held',utc=time.time()))
        if state.startswith('complete'):
            return
        time.sleep(15)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['controller','backend','profile','case'],default='controller')
    parser.add_argument('--predecessor',type=int,nargs='*',default=[])
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--task-json')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    exit_code = 0
    failure = None
    try:
        if args.mode=='controller':
            with (args.output/'controller.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                controller(args)
        else:worker(args)
    except SystemExit as error:
        exit_code = error.code
        raise
    except Exception:
        exit_code = 1
        failure = traceback.format_exc()
        print(traceback.format_exc(),file=sys.stderr,flush=True)
        if args.mode=='controller':
            (args.output/'controller_failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if args.mode != 'controller':
            name=args.mode+('_'+key(json.loads(args.task_json)) if args.task_json else '')
            (args.output/'logs').mkdir(exist_ok=True)
            write(args.output/'logs'/f'{name}.exit.json',
                  dict(exit_code=exit_code, failure=failure, utc=time.time()))


if __name__=='__main__':
    main()
