"""Finite D128 continuation of both starts of every completed L6 D64 case.

The D64 archive is read-only. New checkpoints are losslessly gzip compressed.
Qualification compares D64 to D128 and retains prior truncation alarms.
"""
import argparse
import fcntl
import gzip
import json
import os
import pickle
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from .triangle_correlation_dmrg import digest, write
from .l6_d64_diagnostics import compare
from .triangle_l6_chi8_dense import alive, rss, THREAD_VARS

MODULE = 'transformer_qmc.triangle_l6_d128'
PILOTS = [(0.1, 60000), (0.95, 60001), (1.5, 60000)]
TASKS = PILOTS + [(i/1000, s) for i in range(100, 1501, 25)
                 for s in range(60000, 60008) if (i/1000, s) not in PILOTS]


def name(g, s):
    return f'L6_g{g:g}_s{s}_m6_c8'


def checked(path, expected):
    assert digest(path) == expected, str(path)
    return path


def run_case(root, baseline, gamma, sid):
    from .triangle_effective_mps import optimize, physical_moments
    from .triangle_correlation_observables import spectra, ratio, overlap_observables
    from .triangle_correlation_highchi import estimates, allocation
    started = time.monotonic()
    src = baseline/'cases'/name(gamma, sid)
    dst = root/'cases'/src.name
    dst.mkdir(parents=True, exist_ok=True)
    lock = (dst/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = json.loads((src/'result.json').read_text())
    assert (old['L'], old['m'], old['chi'], old['h'], old['lambda_rg'],
            old['gamma'], old['sample_id'], old['max_bond_dimension']) == (6,6,8,.15,1.,gamma,sid,64)
    assert old['computation_complete'] and old['overlap_gates_enabled']
    assert {(r['seed'], r['D_MPS']) for r in old['records']} == {(s,d) for s in (0,1) for d in (32,64)}
    for filename in ('triangle_effective_mps.py', 'triangle_effective_rg.py'):
        assert digest(Path(__file__).with_name(filename)) == old['solver_source_sha256'][filename]
    provenance = dict(baseline_result=str(src/'result.json'), baseline_sha256=digest(src/'result.json'),
                      baseline_hashes=old['hashes'], forced_dimension=128, seeds=[0,1],
                      source_sha256={p.name:digest(p) for p in [Path(__file__), Path(__file__).with_name('l6_d64_diagnostics.py')]})
    if (dst/'ancestry.json').exists():
        assert json.loads((dst/'ancestry.json').read_text()) == provenance
    else:
        write(dst/'ancestry.json', provenance)
    if (dst/'result.json').exists():
        result = json.loads((dst/'result.json').read_text())
        for filename, sha in result['hashes'].items():
            checked(dst/filename, sha)
        return result
    with checked(src/'model.pkl', old['hashes']['model.pkl']).open('rb') as stream:
        saved = pickle.load(stream)
    model, mpo, order = saved['model'], saved['mpo'], saved['order']
    estimate = estimates(model, mpo, 128)
    assert estimate['memory_gib'] <= 4., estimate
    with np.load(checked(src/'inputs.npz', old['hashes']['inputs.npz'])) as a:
        inputs = {k:a[k] for k in a.files}
    np.testing.assert_array_equal(inputs['disorder_entropy'], [20261027,6,sid,1])
    records = list(old['records'])
    comparisons = []
    for seed in (0,1):
        prior = next(r for r in old['records'] if r['seed']==seed and r['D_MPS']==64)
        stem = f'seed{seed}_D128'
        row_path = dst/(stem+'.json')
        if row_path.exists():
            row = json.loads(row_path.read_text())
            for filename, sha in row['checkpoint_hashes'].items():
                checked(dst/filename, sha)
        else:
            assert allocation()['available_gib'] > 16+estimate['memory_gib'], 'RAM headroom'
            assert shutil.disk_usage(root).free > 3.05*2**30, 'Disk headroom'
            p = checked(src/f'seed{seed}_D64.pkl', old['hashes'][f'seed{seed}_D64.pkl'])
            with p.open('rb') as stream:
                psi = pickle.load(stream)['psi']
            # Replay the actual saved input state before continuation.
            m0, zz0 = physical_moments(psi, model, order)
            with np.load(checked(src/f'seed{seed}_D64.npz', old['hashes'][f'seed{seed}_D64.npz'])) as a:
                np.testing.assert_allclose(m0, a['magnetization'], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(zz0, a['zz'], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(float(np.real(mpo.H_MPO.expectation_value(psi))), prior['energy'], atol=1e-9, rtol=0)
            write(dst/'progress.json', dict(seed=seed, dimension=128, utc=time.time(),
                  input_checkpoint_sha256=digest(p), input_moments_replayed=True, estimate=estimate))
            begin = time.monotonic()
            info = optimize(psi, mpo, 128)
            m, zz = physical_moments(psi, model, order)
            spectrum = spectra(m, zz, inputs['points'], 6)
            rr = float(ratio(spectrum))
            row = dict(seed=seed, D_MPS=128, energy=float(np.real(mpo.H_MPO.expectation_value(psi))),
                       observables=overlap_observables(m, zz), spectra=spectrum.tolist(),
                       R=rr if np.isfinite(rr) else None, max_bond=max(psi.chi),
                       entropy_max=float(max(psi.entanglement_entropy())),
                       sweeps=int(info['sweep_statistics']['sweep'][-1]),
                       truncation_failure=info['truncation_failure'], canonical_error=info['canonical_error'],
                       last_energy_change=float(abs(np.diff(info['sweep_statistics']['E'])[-1])),
                       seconds=time.monotonic()-begin, parent_checkpoint_sha256=digest(p),
                       solver_source_sha256=provenance['source_sha256'])
            assert row['max_bond'] <= 128
            checkpoint = dst/(stem+'.pkl.gz')
            temporary = dst/(stem+'.pkl.gz.tmp')
            with gzip.open(temporary, 'wb', compresslevel=6) as stream:
                pickle.dump(dict(psi=psi, info=info), stream, protocol=5)
            with gzip.open(temporary, 'rb') as stream:
                restored = pickle.load(stream)
            assert restored['psi'].chi == psi.chi
            np.testing.assert_allclose(float(np.real(mpo.H_MPO.expectation_value(restored['psi']))), row['energy'], atol=1e-10, rtol=0)
            temporary.replace(checkpoint)
            np.savez_compressed(dst/(stem+'.npz'), magnetization=m, zz=zz, spectra=spectrum, **inputs)
            row['checkpoint_hashes'] = {p.name:digest(p) for p in (checkpoint, dst/(stem+'.npz'))}
            row['ladder_stable'] = compare(row, prior, 'ladder', gamma, sid, seed)['all_pass']
            write(row_path, row)
        comparison = compare(row, prior, 'ladder', gamma, sid, seed)
        assert comparison['all_pass'] == row['ladder_stable']
        comparisons.append(comparison)
        records.append(row)
    ends = [next(r for r in records if r['seed']==s and r['D_MPS']==128) for s in (0,1)]
    comparisons.append(compare(*ends, 'starts', gamma, sid, -1))
    passed = all(c['all_pass'] for c in comparisons)
    result = dict(L=6,m=6,chi=8,h=.15,lambda_rg=1.,gamma=gamma,sample_id=sid,
                  records=records,comparisons=comparisons,solver_passed=passed,
                  start_agreement=comparisons[-1]['all_pass'],max_bond_dimension=128,D_ladder=[32,64,128],
                  computation_complete=True,overlap_gates_enabled=True,retention_qualified=False,
                  numerical_status='qualified' if passed else 'unqualified_at_bond_cap',
                  ancestry=provenance,seconds=time.monotonic()-started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  hashes={p.name:digest(p) for p in dst.iterdir() if p.name.endswith(('.npz','.pkl.gz'))})
    write(dst/'result.json', result)
    return result


def controller(args):
    from .triangle_correlation_highchi import allocation
    root = args.output
    for sub in ('cases','logs','attempts'):
        (root/sub).mkdir(parents=True,exist_ok=True)
    lock = (root/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(baseline=str(args.baseline),target=456,D_MPS=128,seeds=[0,1],
                  worker_limit=args.workers,threads=args.threads,memory_per_worker_gib=4,
                  disk_reserve_gib=3,memory_reserve_gib=16,runtime='finite queue; no deadline; one attempt per case',
                  tasks=TASKS,source_sha256=digest(Path(__file__)))
    write(root/'manifest.json',config)
    running, processes = {}, {}
    status_path = root/'status.json'
    if status_path.exists():
        running = {k:v for k,v in json.loads(status_path.read_text()).get('running',{}).items() if alive(v)}
        for job in running.values():
            assert job['threads'] == args.threads
    start = time.time()
    while True:
        for k, proc in list(processes.items()):
            if proc.poll() is not None:
                del processes[k]
        for k, job in list(running.items()):
            if not alive(job):
                if not (root/'cases'/k/'result.json').exists() and not (root/'attempts'/(k+'.json')).exists():
                    write(root/'attempts'/(k+'.json'),dict(state='exited_without_result',job=job))
                del running[k]
        results = [json.loads(p.read_text()) for p in (root/'cases').glob('*/result.json')]
        done = {name(r['gamma'],r['sample_id']) for r in results}
        attempts = {p.stem:json.loads(p.read_text()) for p in (root/'attempts').glob('*.json')}
        queue = [(g,s) for g,s in TASKS if name(g,s) not in done|running.keys()|attempts.keys()]
        # Use full archived bytes of measured D128 pilots, plus a conservative
        # 12 MiB floor, to admit the remaining finite archive before broad dispatch.
        measured = [sum(p.stat().st_size for p in (root/'cases'/k).iterdir() if p.is_file()) for k in done]
        case_bytes = max([12*2**20]+measured)
        projected = 1.25*case_bytes*(456-len(done))/2**30
        disk = shutil.disk_usage(root).free/2**30
        pilots_done = all(name(g,s) in done for g,s in PILOTS)
        reason = None
        if pilots_done and disk < 3+projected+.05*args.workers:
            reason = 'whole-campaign checkpoint projection exceeds disk headroom'
        eligible = queue if pilots_done else [(g,s) for g,s in queue if (g,s) in PILOTS]
        while eligible and len(running)<args.workers and reason is None:
            alloc = allocation()
            assert args.workers*args.threads <= min(16,alloc['cpu_cores']-4)
            future = sum(max(0,4-rss(j)) for j in running.values())
            if alloc['available_gib'] < 16+future+4:
                reason = 'RAM reservation';break
            if disk < 3+.05*(len(running)+1):
                reason = 'active checkpoint disk reservation';break
            g,s = eligible.pop(0);k=name(g,s)
            command = [sys.executable,'-u','-m',MODULE,'--output',str(root),'--baseline',str(args.baseline),
                       '--worker','--gamma',str(g),'--sample-id',str(s)]
            env = dict(os.environ,**{v:str(args.threads) for v in THREAD_VARS})
            with (root/'logs'/(k+'.log')).open('ab') as stream:
                proc = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            running[k]=dict(pid=proc.pid,command=command,threads=args.threads,gamma=g,sample_id=s)
            processes[k]=proc
        failures = {k:v for k,v in attempts.items() if k not in done}
        state = 'running'
        if not running:
            state = 'held' if reason or queue else ('complete' if len(done)==456 else 'finished_with_failures')
        status = dict(utc=time.time(),state=state,target=456,completed=len(done),
                      qualified=sum(r['solver_passed'] for r in results),
                      excluded=sum(not r['solver_passed'] for r in results),running=running,
                      failures=failures,reason=reason,disk_free_gib=disk,projected_remaining_gib=projected,
                      case_bytes_projection=case_bytes,elapsed_seconds=time.time()-start)
        write(status_path,status)
        if state != 'running':return
        time.sleep(10)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--gamma',type=float)
    parser.add_argument('--sample-id',type=int)
    args=parser.parse_args()
    if args.worker:
        try:
            row=run_case(args.output,args.baseline,args.gamma,args.sample_id)
            write(args.output/'attempts'/(name(args.gamma,args.sample_id)+'.json'),
                  dict(state='finished',solver_passed=row['solver_passed'],utc=time.time()))
        except Exception:
            write(args.output/'attempts'/(name(args.gamma,args.sample_id)+'.json'),
                  dict(state='failed',error=traceback.format_exc(),utc=time.time()))
            raise
    else:
        controller(args)


if __name__=='__main__':
    main()
