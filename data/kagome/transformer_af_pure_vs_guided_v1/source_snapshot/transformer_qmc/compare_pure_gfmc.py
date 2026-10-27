"""Matched constant-guide / frozen-network GFMC benchmark against exact L2.

Pure here means an unguided constant trial amplitude. Both methods use the
same continuous-time projector and four-replica forward-walking estimators.
Each population is atomic and resumable; exact amplitudes are references only.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from gfmc.guide import ConstantGuide
from .experiment import write_json
from .frozen_scan import SELECTED_SHA256, digest, key, population_summary, stage_lock
from .model import EnumeratedGuide
from .physics import basis, correlations, make_af_sample
from .projector import project, rates_and_local


class BatchedConstantGuide(ConstantGuide):
    """Uniform positive guide, without an enumerated table or neural inference."""

    def __init__(self, n_sites):
        self.n_sites = n_sites

    def log_amplitudes(self, states):
        return np.zeros(len(states))

    def all_flip_log_ratios(self, states):
        return np.zeros(np.asarray(states).shape)

    def sample(self, count, seed):
        return (1 - 2*np.random.default_rng(seed).integers(
            0, 2, size=(count, self.n_sites))).astype(np.int8)


def design(preview):
    contexts = [(.5, .3, 1.), (.5, .3, 2.)]
    if not preview:
        contexts += [(2., .3, 1.), (.25, .3, 1.)]
    return dict(
        contexts=[dict(gamma=g, h=h, K=K) for g, h, K in contexts],
        sample_ids=[10000] if preview else [10000, 10001],
        walkers=[512] if preview else [512, 2048],
        forward_Ktau=[2.] if preview else [2., 8.], branch_Ktau=.1,
        equilibration_Ktau=8. if preview else 16.,
        measurement_Ktau=2. if preview else 4., cycles=6 if preview else 12,
        seeds=[98200+i for i in range(4)], guides=['constant', 'network'])


def identity(args, settings):
    source = Path(__file__).parent
    sources = [source/'compare_pure_gfmc.py', source/'projector.py',
               source/'physics.py', source/'model.py', source/'experiment.py',
               source/'frozen_scan.py', source.parent/'qmc_numerics.py',
               source.parent/'gfmc/guide.py']
    caches = {}
    for c in settings['contexts']:
        for sid in settings['sample_ids']:
            name = key(2, sid, c['gamma'], c['h'], c['K'])
            for suffix in ('.json', '.npz'):
                path = args.scan/'exact'/(name+suffix)
                caches[path.name] = digest(path)
    scan_manifest = json.loads((args.scan/'manifest.json').read_text())
    if scan_manifest['checkpoint_sha256'] != SELECTED_SHA256:
        raise RuntimeError('unexpected frozen guide identity')
    return dict(
        checkpoint_sha256=SELECTED_SHA256, design=settings,
        Hamiltonian='sum K_ij ZiZj - Gamma sum Xi - h sum Zi; PBC',
        disorder='matched K_ij=K*(1+0.5*u_ij)',
        pure_definition='constant guide psi_T(s)=1; uniform initial samples',
        network_definition='frozen seed501 psi_T=sqrt(p); initial samples from p',
        estimates='mixed energy; four-replica forward-walking q2 and connected chi',
        uncertainty='SE across four independent populations; all cycles retained',
        timing='propagation plus guide sampling; excludes cache loading and creation',
        cost_limit='network uses L2 lookup; direct inference overhead excluded',
        reserved_test_ids_inspected=False,
        source_sha256={str(p.relative_to(source.parent)): digest(p) for p in sources},
        input_sha256=caches)


def run_population(task):
    torch.set_num_threads(1)
    c, sid, W, forward, kind, seed, settings, scan, output = task
    name = key(2, sid, c['gamma'], c['h'], c['K'])
    path = Path(output)/'populations'/f'{name}_{kind}_W{W}_f{forward:g}_seed{seed}.json'
    if path.exists():
        return str(path)
    ham = make_af_sample(2, sid, scale=c['K']).hamiltonian(c['h'])
    if kind == 'constant':
        guide = BatchedConstantGuide(12)
    else:
        with np.load(Path(scan)/'exact'/(name+'.npz')) as cache:
            guide = EnumeratedGuide(cache['logs'].copy(), 12)
    start = time.perf_counter()
    block = settings['branch_Ktau']
    record = project(
        ham, c['gamma'], guide, n_walkers=W, n_replicas=4,
        tau_block=block/c['K'],
        n_equilibrate=round(settings['equilibration_Ktau']/block),
        n_measure=round(settings['measurement_Ktau']/block),
        tau_forward=forward/c['K'], n_cycles=settings['cycles'], seed=seed)
    record.update(seconds=time.perf_counter()-start, **c, sample_id=sid,
                  guide=kind, forward_Ktau=forward, checkpoint_sha256=SELECTED_SHA256)
    write_json(path, record)
    return str(path)


def summarize(args, manifest, tasks, started):
    settings = manifest['design']
    rows = []
    completed = sum(Path(run_path(args.output, t)).exists() for t in tasks)
    states = basis(12)
    for c in settings['contexts']:
        for sid in settings['sample_ids']:
            name = key(2, sid, c['gamma'], c['h'], c['K'])
            with np.load(args.scan/'exact'/(name+'.npz')) as cache:
                target = correlations(cache['exact']**2, states)
                network_logs = cache['logs'].copy()
            exact = json.loads((args.scan/'exact'/(name+'.json')).read_text())
            target['energy'] = exact['exact_energy']
            ham = make_af_sample(2, sid, scale=c['K']).hamiltonian(c['h'])
            for W in settings['walkers']:
                for f in settings['forward_Ktau']:
                    for kind in settings['guides']:
                        paths = [args.output/'populations'/f'{name}_{kind}_W{W}_f{f:g}_seed{seed}.json'
                                 for seed in settings['seeds']]
                        if not all(p.exists() for p in paths):
                            continue
                        records = [json.loads(p.read_text()) for p in paths]
                        guide = BatchedConstantGuide(12) if kind == 'constant' else EnumeratedGuide(network_logs, 12)
                        probability = np.ones(len(states)) if kind == 'constant' else np.exp(2*network_logs)
                        probability /= probability.sum()
                        _, local = rates_and_local(ham, c['gamma'], guide, states[:, None, :])
                        e = float(probability@local)
                        initial_psi = np.sqrt(probability)
                        with np.load(args.scan/'exact'/(name+'.npz')) as cache:
                            initial_fidelity = float((initial_psi@cache['exact'])**2)
                        rows.append(dict(
                            **c, L=2, sample_id=sid, guide=kind, walkers=W,
                            forward_Ktau=f, branch_Ktau=settings['branch_Ktau'],
                            equilibration_Ktau=settings['equilibration_Ktau'],
                            independent_populations=len(records),
                            observables=population_summary(records, target),
                            initial_fidelity=initial_fidelity,
                            initial_local_energy_variance=float(probability@((local-e)**2)),
                            min_descendant_ess=min(min(r['descendant_ess']) for r in records),
                            max_largest_lineage=max(max(r['largest_lineage']) for r in records),
                            min_weight_ess=min(min(r['weight_ess']) for r in records),
                            min_unique_ancestors=min(min(r['unique_ancestors']) for r in records),
                            seconds=sum(r['seconds'] for r in records)))
    result = dict(manifest=manifest, rows=rows, completed_populations=completed,
                  expected_populations=len(tasks), completed=completed == len(tasks),
                  elapsed_seconds=time.time()-started, updated_unix=time.time())
    write_json(args.output/'summary.json', result)
    return result


def run_path(output, task):
    c, sid, W, forward, kind, seed, *_ = task
    name = key(2, sid, c['gamma'], c['h'], c['K'])
    return Path(output)/'populations'/f'{name}_{kind}_W{W}_f{forward:g}_seed{seed}.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--time-cap-hours', type=float, default=2.)
    args = parser.parse_args()
    if args.workers < 1 or args.time_cap_hours <= 0:
        parser.error('workers and time cap must be positive')
    args.scan, args.output = args.scan.resolve(), args.output.resolve()
    torch.set_num_threads(1)
    with stage_lock(args.output, 'comparison'):
        settings = design(args.preview)
        manifest = identity(args, settings)
        path = args.output/'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError('comparison identity changed; use a new output directory')
        write_json(path, manifest)
        (args.output/'populations').mkdir(exist_ok=True)
        tasks = [(c, sid, W, f, kind, seed, settings, str(args.scan), str(args.output))
                 for c in settings['contexts'] for sid in settings['sample_ids']
                 for W in settings['walkers'] for f in settings['forward_Ktau']
                 for seed in settings['seeds'] for kind in settings['guides']]
        started = time.time()
        deadline = started+args.time_cap_hours*3600
        pending = iter(t for t in tasks if not run_path(args.output, t).exists())
        summary = summarize(args, manifest, tasks, started)
        def save(state, **extra):
            write_json(args.output/'status.json', dict(
                state=state, pid=os.getpid(), updated_unix=time.time(),
                completed_populations=summary['completed_populations'],
                expected_populations=len(tasks), **extra))
        save('running')
        # Stop dispatching at the cap; at most workers in-flight populations finish.
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                active = {pool.submit(run_population, t) for t in
                          [next(pending, None) for _ in range(args.workers)] if t is not None}
                while active:
                    future = next(as_completed(active))
                    active.remove(future)
                    print('POPULATION', future.result(), flush=True)
                    summary = summarize(args, manifest, tasks, started)
                    save('running')
                    if time.time() < deadline:
                        task = next(pending, None)
                        if task is not None:
                            active.add(pool.submit(run_population, task))
            save('completed' if summary['completed'] else 'time_cap_reached')
        except BaseException as error:
            save('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
