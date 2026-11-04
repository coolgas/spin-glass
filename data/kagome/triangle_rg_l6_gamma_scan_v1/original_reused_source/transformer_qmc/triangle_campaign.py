"""Versioned triangle-propagator benchmark, sharing checkpointed GFMC estimators.

Selected L2 contexts are algorithm diagnostics, not a disorder ensemble. Frozen
guides are unchanged. Exact enumeration occurs only in guide lookup and offline
reference calculations, never in the scalable triangle proposal definition.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from threadpoolctl import threadpool_limits

from . import collective_campaign as base
from .correlation_lengths import length, sha
from .independent_replica_projector import spectrum_from_moments
from .physics import basis, make_af_sample
from .production_protocol import exact_spectra, geometry
from .triangle_projector import TriangleProposal


METHODS = [('ct', .1), ('guided', .1), ('triangle', .1), ('triangle', .05), ('triangle', .2)]


def tasks(stage):
    rows = []
    for gamma, sid in base.CONTEXTS:
        for guide in ['original', 'corrected']:
            schedules = ([('pilot', 512, 8., 8., 4, 'guide', 2)] if stage == 'pilot' else
                         [('main', 1024, 32., 32., 16, 'guide', 4)] if stage == 'main' else
                         [('population', 4096, 32., 32., 16, 'guide', 2),
                          ('time', 1024, 64., 64., 16, 'guide', 2),
                          ('start', 1024, 32., 32., 16, 'uniform', 2)])
            methods = METHODS if stage != 'audit' else [('ct', .1), ('triangle', .1), ('triangle', .05)]
            for method, delta in methods:
                for audit, walkers, equil, forward, cycles, start, repeats in schedules:
                    for repeat in range(repeats):
                        rows.append(dict(L=2, h=.15, gamma=gamma, sample_id=sid, guide=guide,
                            method=method, delta=delta, epsilon=0., walkers=walkers,
                            equil=equil, forward=forward, cycles=cycles, start=start,
                            repeat=repeat, audit=audit))
    return rows


def reference_one(context, inputs, stages):
    """Actual triangle split targets, including finite projection and initialization."""
    with threadpool_limits(limits=1):
        gamma, sid = context
        ham = make_af_sample(2, sid).hamiltonian(.15)
        spins = basis(12); lattice = geometry(2)
        gp = Path(inputs)/'guides'/(base.label(gamma, sid)+'.npz')
        with np.load(gp) as data:
            exact = data['exact_amplitude'].copy()
        truth = exact_spectra(exact**2, spins, lattice)['connected_spectrum']
        guide, _ = base.load_guide(inputs, dict(gamma=gamma, sample_id=sid, guide='original'))
        rows = []
        selected = [t for stage in stages for t in tasks(stage)
                    if (t['gamma'],t['sample_id'],t['method'],t['repeat'])==(gamma,sid,'triangle',0)]
        for delta in sorted({t['delta'] for t in selected}):
            kernel = TriangleProposal(ham, gamma, guide, delta)
            # A common scalar factor avoids overflowing long powers.
            scale = np.linalg.norm(kernel.action(exact))
            action = lambda x: kernel.action(x) / scale
            operator = LinearOperator((4096,4096), matvec=action, dtype=float)
            values, vectors = eigsh(operator, k=1, which='LA', v0=exact, tol=1e-12)
            ground = vectors[:,0]
            residual = float(np.linalg.norm(action(ground)-values[0]*ground))
            if residual >= 1e-9:
                raise RuntimeError(f'Triangle eigenvector residual {residual}')
            target = exact_spectra(ground**2, spins, lattice)['connected_spectrum']
            def advance(vector, duration):
                for _ in range(round(duration/delta)):
                    vector = action(vector)
                    vector /= np.linalg.norm(vector)
                return vector
            finite = []
            for task in [t for t in selected if t['delta']==delta]:
                current_guide, _ = base.load_guide(inputs, task)
                amplitude = np.exp(current_guide.logs-current_guide.logs.max())
                amplitude /= np.linalg.norm(amplitude)
                initial = amplitude if task['start']=='guide' else 1/amplitude
                initial /= np.linalg.norm(initial)
                left = advance(initial, task['equil'])
                right = advance(amplitude, task['forward'])
                spectrum = base.finite_moments(left, right, lambda x:advance(x,task['forward']),
                                              task['cycles'], spins, lattice)
                finite.append(dict(guide=task['guide'], audit=task['audit'], spectrum=spectrum.tolist()))
            rows.append(dict(delta=delta, residual=residual, ground_spectrum=target.tolist(),
                             finite_projection=finite))
        return dict(gamma=gamma, sample_id=sid, ground_spectrum=truth.tolist(),
                    ground_xi_over_L=float(length(truth,2)/2), triangle=rows)


def assess(output, baseline):
    root = Path(output)
    references = json.loads((root/'triangle-references.json').read_text())['rows']
    old = json.loads(Path(baseline).read_text())['rows']
    groups = {}
    for path in sorted((root/'runs').glob('*.json')):
        record = json.loads(path.read_text())
        assert record['npz_sha256'] == sha(path.with_suffix('.npz'))
        assert record['provenance']['source_sha256'] == base.source_hashes()
        with np.load(path.with_suffix('.npz')) as arrays:
            spectrum = spectrum_from_moments(arrays['magnetization_cycles'].mean(1),
                                            arrays['pair_cycles'].mean(1),geometry(2))
        np.testing.assert_allclose(spectrum, record['spectrum'], atol=1e-12)
        t = record['task']
        key = tuple(t[k] for k in ['gamma','sample_id','guide','method','delta','audit'])
        groups.setdefault(key, []).append(record)
    rows = []; rng = np.random.default_rng(20261103)
    for (gamma,sid,guide,method,delta,audit), records in sorted(groups.items()):
        reference = next(r for r in references if (r['gamma'],r['sample_id'])==(gamma,sid))
        truth = np.array(reference['ground_spectrum'])
        raw = np.array([r['spectrum'] for r in records]); mean = raw.mean(0)
        if method == 'triangle':
            split = next(r for r in reference['triangle'] if r['delta']==delta)
            split_ground = np.array(split['ground_spectrum'])
            finite = next(r for r in split['finite_projection'] if (r['guide'],r['audit'])==(guide,audit))
        else:
            ref = next(r for r in old if (r['gamma'],r['sample_id'])==(gamma,sid))
            split_ground = truth if method=='ct' else np.array(next(r['ground_spectrum'] for r in ref['split'] if r['delta']==delta))
            finite = next(r for r in ref['finite_projection'] if r['guide']==guide
                          and r['audit']==('main' if audit=='population' else audit)
                          and (r.get('delta') is None if method=='ct' else r.get('delta')==delta))
        finite = np.array(finite['spectrum'])
        boot = raw[rng.integers(len(raw),size=(2000,len(raw)))].mean(1)
        lens = length(boot,2)/2; valid = np.isfinite(lens); xi = length(mean,2)/2
        rows.append(dict(gamma=gamma,sample_id=sid,guide=guide,method=method,delta=delta,
            audit=audit,repeats=len(raw),spectrum=mean.tolist(),
            max_relative_bias=float(np.max(abs(mean-truth)/truth)),
            relative_bias=((mean-truth)/truth).tolist(),
            finite_projection_spectrum=finite.tolist(),
            relative_sampling_error=((mean-finite)/finite).tolist(),
            split_ground_relative_bias=((split_ground-truth)/truth).tolist(),
            finite_projection_relative_bias=((finite-split_ground)/split_ground).tolist(),
            spectrum_se=boot.std(0,ddof=1).tolist() if len(raw)>1 else None,
            xi_over_L=float(xi) if np.isfinite(xi) else None,
            xi_interval=np.quantile(lens[valid],[.025,.975]).tolist() if len(raw)>1 and valid.any() else None,
            invalid_length_fraction=float(1-valid.mean()),
            alarms=sum(r['ancestry_alarm'] for r in records),
            minimum_descendant_ess=min(r['minimum_descendant_ess'] for r in records),
            maximum_lineage=max(r['maximum_lineage'] for r in records),
            total_seconds=sum(r['seconds'] for r in records),ground_state_qualified=False))
    result = dict(completed=sum(len(rs) for rs in groups.values()),rows=rows,
        uncertainty='Whole-run bootstrap, selected L2 diagnostics only; small repeat counts do not establish convergence.')
    base.write(root/'review.json', result)
    lines = ['# Hamiltonian-aware triangle GFMC', '', result['uncertainty'], '',
             '| Gamma | Disorder | Guide | Rule | Step | Audit | Runs | Max spectral error | Alarms | CPU s |',
             '|---:|---:|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['gamma']} | {r['sample_id']} | {r['guide']} | {r['method']} | {r['delta']} | {r['audit']} | {r['repeats']} | {r['max_relative_bias']:.2%} | {r['alarms']} | {r['total_seconds']:.1f} |")
    (root/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--wait-lock',type=Path)
    parser.add_argument('--stages',nargs='+',choices=['pilot','main','audit'],default=['pilot','main','audit'])
    parser.add_argument('--workers',type=int,default=18)
    parser.add_argument('--hours',type=float,default=6.)
    parser.add_argument('--assess',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.assess:
        assess(args.output,args.baseline); return
    # Hold the inherited CPU allocation lock for the whole follow-up.
    allocation = None
    if args.wait_lock:
        base.write(args.output/'status.json',dict(state='waiting_for_primary_cpu_batch'))
        allocation = args.wait_lock.open('a')
        fcntl.flock(allocation,fcntl.LOCK_EX)
    with (args.output/'controller.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = dict(source_sha256=base.source_hashes(),stages=args.stages,
            methods=METHODS,workers=args.workers,hours=args.hours,
            inputs={base.label(g,s):sha(args.inputs/'guides'/(base.label(g,s)+'.npz')) for g,s in base.CONTEXTS})
        path = args.output/'manifest.json'
        if path.exists():
            previous = json.loads(path.read_text())
            for field in ['source_sha256','inputs','stages']:
                assert previous[field]==manifest[field], f'{field} changed; use a new output'
        else:
            base.write(path,manifest)
        deadline = time.monotonic()+args.hours*3600
        for stage in args.stages:
            pending = iter(tasks(stage)); done=0
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                def submit(task):
                    futures[pool.submit(base.run_one,task,args.output,args.inputs,1800,deadline)] = task
                for _ in range(args.workers):
                    task = next(pending,None)
                    if task is not None: submit(task)
                while futures:
                    ready,_ = wait(futures,return_when=FIRST_COMPLETED)
                    for future in ready:
                        task = futures.pop(future)
                        try:
                            record = future.result()
                        except Exception as error:
                            base.write(args.output/'failure.json',dict(task=task,error=repr(error)))
                            raise
                        if record['status']=='completed': done+=1
                        print(json.dumps(dict(stage=stage,completed=done,task=task,status=record['status'],seconds=record['seconds'])),flush=True)
                        if time.monotonic()<deadline:
                            task = next(pending,None) if record['status']=='completed' else task
                            if task is not None: submit(task)
                    base.write(args.output/'status.json',dict(state='running',stage=stage,completed=done,
                        total=len(tasks(stage)),active=len(futures),updated_unix=time.time()))
            if time.monotonic()>=deadline:
                base.write(args.output/'status.json',dict(state='paused_time_cap',stage=stage,completed=done))
                return
        reference_path = args.output/'triangle-references.json'
        if not reference_path.exists():
            base.write(args.output/'status.json',dict(state='computing_exact_references'))
            with ProcessPoolExecutor(max_workers=min(4,args.workers)) as pool:
                rows = list(pool.map(reference_one,base.CONTEXTS,[args.inputs]*4,[args.stages]*4))
            base.write(reference_path,dict(rows=rows,scope='Offline L2 evaluation only',source_sha256=base.source_hashes()))
        result = assess(args.output,args.baseline)
        base.write(args.output/'status.json',dict(state='completed',completed=result['completed']))
    if allocation is not None: allocation.close()


if __name__=='__main__':
    main()
