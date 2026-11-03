"""Finite higher-field L6 comparison using the anchor training/GFMC protocol."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path

from .collective_campaign import write
from .correlation_lengths import sha
from .l6_convergence_campaign import milestone, projection_specs
from .parallel_disorder_scaling import configure_training_worker
from .projected_overlap_histograms import configure, run_one


def contexts():
    return [dict(L=6, gamma=g, sample_id=d, kind='rbm_2', seed=s, mode='anneal')
            for g in [.85, 1.] for d in [60000, 60001] for s in [0, 1]]


def plan(campaign, workers):
    return dict(contexts=contexts(), extra_updates=[0, 1200, 3600],
        target_updates=[400, 1600, 4000], training_workers=1,
        projector_workers=workers, planned_milestones=24, maximum_projection_runs=48,
        campaign=str(campaign.resolve()), deadline=None,
        baseline_schedules={'0.85': [1.5, .85], '1.00': [1.5, 1.]},
        protocol='Same RBM alpha2, Adam .001, chains256, temperatures8, sweeps/update2, frozen random/up/down checks, four GFMC populations, walkers512, equil16, forward16, cycles8, repeats0/1 as lower-field anchors. Continuations preserve Adam, chains and RNG. Annealing never visits a field below target.',
        hypothesis='If low-field difficulty reflects slow exploration in a glassy regime, operational seed/repeat agreement and ancestry may improve at higher Gamma. These fields are not assumed paramagnetic.',
        selection='Fixed disorders60000/60001, both seeds, all budgets; no selection by energy, overlap or sampling difficulty. Equal disorder weights. Sampling alarms block only that checkpoint projection.',
        interpretation='Field-specific optimized guides and warmup schedules differ. Improvement supports field-dependent computational difficulty; it cannot establish a spin-glass phase or locate a phase transition. No automatic production promotion.',
        screening=dict(energy_per_site=.001, q2=.01, histogram_tv=.05,
            spectrum_relative=.1, spectrum_absolute_floor=.001,
            ancestry='ESS<8 or largest lineage>.5; retain full distributions',
            uncertainty='Two repeats have limited precision. Agreement requires uncertainty resolving tolerances. Walkers/descendant copies are not independent units; matched budgets are correlated.'),
        sources={f.name: sha(f) for f in Path(__file__).parent.glob('*.py')},
        numerics_sha256=sha(Path(__file__).parent.parent/'qmc_numerics.py'),
        ground_state_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('Positive projector worker count required')
    manifest = plan(args.campaign, args.workers)
    if args.validate_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != 'sources'}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        write(path, manifest)
    pending = deque((t, 0, None) for t in contexts())
    projections = deque()
    active_train, active_project = {}, {}
    rows, results, failures, exclusions = [], [], [], []
    ctx = mp.get_context('spawn')
    def status(state):
        write(args.output/'status.json', dict(state=state,
            updated_utc=datetime.now(timezone.utc).isoformat(), training_workers=1,
            projector_workers=args.workers, planned_milestones=24, maximum_projection_runs=48,
            completed_milestones=len(rows), completed_projections=len(results),
            pending_training=len(pending), pending_projectors=len(projections),
            active_training=list(active_train.values()), active_projectors=list(active_project.values()),
            failures=failures, excluded_projections=exclusions, ground_state_qualified=False))
    with ProcessPoolExecutor(1, mp_context=ctx, initializer=configure_training_worker) as gpu, \
         ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=configure) as cpu:
        while pending or projections or active_train or active_project:
            if pending and not active_train:
                task, extra, parent = pending.popleft()
                future = gpu.submit(milestone, task, extra, parent,
                    str(args.campaign), str(args.output/'training'))
                active_train[future] = dict(task=task, extra_updates=extra)
            while projections and len(active_project) < args.workers:
                spec, output = projections.popleft()
                active_project[cpu.submit(run_one, spec, output)] = dict(spec=spec, output=output)
            status('running')
            done, _ = wait(list(active_train)+list(active_project), timeout=30,
                           return_when=FIRST_COMPLETED)
            for future in done:
                training = future in active_train
                item = (active_train if training else active_project).pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    failure = dict(phase='training' if training else 'projection',
                                   item=item, error=repr(error))
                    failures.append(failure)
                    print(json.dumps(failure), flush=True)
                    write(args.output/'failures.json', failures)
                    continue
                if training:
                    rows.append(row)
                    if row['record']['sampling_alarms']:
                        exclusions.append(dict(task=row['task'], extra_updates=row['extra_updates'],
                                               alarms=row['record']['sampling_alarms']))
                    projections.extend(projection_specs(row, args.output))
                    if row['extra_updates'] < 3600:
                        pending.append((row['task'], 1200 if row['extra_updates'] == 0 else 3600, row['model']))
                    write(args.output/'training_summary.json', dict(rows=rows, failures=failures))
                else:
                    results.append(dict(**item, result=row))
                    write(args.output/'projection_summary.json', dict(rows=results, failures=failures))
    write(args.output/'summary.json', dict(training=rows, projections=results,
        failures=failures, excluded_projections=exclusions, ground_state_qualified=False))
    status('completed_with_failures' if failures else 'completed_unqualified')


if __name__ == '__main__':
    main()
