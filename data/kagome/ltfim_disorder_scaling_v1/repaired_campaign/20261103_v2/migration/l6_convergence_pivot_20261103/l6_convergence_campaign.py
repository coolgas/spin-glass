"""Finite L6 anchor study with milestone training and bounded GFMC dispatch."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import shutil

from .collective_campaign import write
from .correlation_lengths import sha
from .extended_rbm_training import extend
from .parallel_disorder_scaling import configure_training_worker, fit_name
from .projected_overlap_histograms import configure, run_one
from .sampled_simple_guides import train


def contexts():
    # The already running extension owns .55 / 60000 / both seeds.
    return [dict(L=6, gamma=g, sample_id=s, kind='rbm_2', seed=k, mode='anneal')
            for g in [.55, .60, .65, .70] for s in [60000, 60001] for k in [0, 1]
            if (g, s) != (.55, 60000)]


def milestone(task, extra, parent, campaign, output):
    root = Path(output)/fit_name(task)/f'extra_{extra}'
    root.mkdir(parents=True, exist_ok=True)
    if extra:
        record, model = extend(parent, root, 1200 if extra == 1200 else 2400)
    else:
        source = Path(campaign)/'results/training'/fit_name(task)
        record_path = source.with_suffix('.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record['status'] == 'completed' and record['task'] == task
            assert sha(source.with_suffix('.pt')) == record['checkpoint_sha256']
            assert sha(source.with_suffix('.npz')) == record['npz_sha256']
            for suffix in ['.pt', '.npz', '.json']:
                dest = root/(source.name+suffix)
                if dest.exists():
                    assert sha(dest) == sha(source.with_suffix(suffix))
                else:
                    shutil.copy2(source.with_suffix(suffix), dest)
        args = argparse.Namespace(device='cuda', chains=256, temperatures=8,
            steps=400, sweeps_per_step=2, diagnostic_sweeps=200,
            learning_rate=.001, inputs=Path(output)/'inputs')
        record = train(task, args, root, float('inf'))
        model = root/(fit_name(task)+'.pt')
    return dict(task=task, extra_updates=extra, model=str(model),
                model_sha256=sha(model), record=record)


def projection_specs(row, output):
    if row['record']['sampling_alarms']:
        return []
    t = row['task']
    dest = str(Path(output)/'projector'/fit_name(t)/f"extra_{row['extra_updates']}")
    specs = []
    for repeat in [0, 1]:
        task = dict(L=6, gamma=t['gamma'], sample_id=t['sample_id'], kind='rbm_2',
            training_seed=t['seed'], audit='l6_convergence_training', walkers=512,
            equil=16., forward=16., start='guide', delta=.1, cycles=8, repeat=repeat)
        specs.append((dict(task=task, model=row['model']), dest))
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    tasks = contexts()
    assert len(tasks) == 14
    manifest = dict(contexts=tasks, extra_updates=[0, 1200, 3600],
        target_updates=[400, 1600, 4000], training_workers=2, projector_workers=2,
        planned_milestones=42, maximum_projection_runs=84, deadline=None,
        existing_extension='/root/autodl-tmp/extended-rbm-20261103-v1',
        existing_field_controls='/root/autodl-tmp/ancestry-diagnostic-20261103-v1',
        scope='Stage 1 training-budget/seed/field screening. Population, independent equilibration/forward-time, starts and learning-rate qualification remain pending. No automatic production promotion.',
        sources={f.name: sha(f) for f in Path(__file__).parent.glob('*.py')},
        numerics_sha256=sha(Path(__file__).parent.parent/'qmc_numerics.py'),
        ground_state_qualified=False)
    if a.validate_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != 'sources'}, indent=2))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        write(path, manifest)
    pending = deque((t, 0, None) for t in tasks)
    projections = deque()
    active_train = {}
    active_project = {}
    rows = []
    results = []
    failures = []
    exclusions = []
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(2, mp_context=ctx, initializer=configure_training_worker) as gpu, \
         ProcessPoolExecutor(2, mp_context=ctx, initializer=configure) as cpu:
        while pending or projections or active_train or active_project:
            while pending and len(active_train) < 2:
                t, extra, parent = pending.popleft()
                f = gpu.submit(milestone, t, extra, parent, str(a.campaign), str(a.output/'training'))
                active_train[f] = dict(task=t, extra_updates=extra)
            while projections and len(active_project) < 2:
                spec, dest = projections.popleft()
                active_project[cpu.submit(run_one, spec, dest)] = dict(spec=spec, output=dest)
            write(a.output/'status.json', dict(state='running', updated_utc=datetime.now(timezone.utc).isoformat(),
                active_training=list(active_train.values()), active_projectors=list(active_project.values()),
                pending_training=len(pending), pending_projectors=len(projections),
                completed_milestones=len(rows), completed_projections=len(results),
                failures=failures, excluded_projections=exclusions, ground_state_qualified=False))
            done, _ = wait(list(active_train)+list(active_project), timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                training = f in active_train
                item = (active_train if training else active_project).pop(f)
                try:
                    row = f.result()
                except Exception as error:
                    failures.append(dict(phase='training' if training else 'projection', item=item, error=repr(error)))
                    write(a.output/'failures.json', failures)
                    continue
                if training:
                    rows.append(row)
                    if row['record']['sampling_alarms']:
                        exclusions.append(dict(task=row['task'], extra_updates=row['extra_updates'], alarms=row['record']['sampling_alarms']))
                    projections.extend(projection_specs(row, a.output))
                    if row['extra_updates'] < 3600:
                        pending.append((row['task'], 1200 if row['extra_updates'] == 0 else 3600, row['model']))
                    write(a.output/'training_summary.json', dict(rows=rows, failures=failures))
                else:
                    results.append(dict(**item, result=row))
    write(a.output/'summary.json', dict(training=rows, projections=results, failures=failures,
        excluded_projections=exclusions, ground_state_qualified=False))
    write(a.output/'status.json', dict(state='completed_with_failures' if failures else 'completed_unqualified',
        updated_utc=datetime.now(timezone.utc).isoformat(), completed_milestones=len(rows),
        completed_projections=len(results), failures=failures, excluded_projections=exclusions,
        ground_state_qualified=False))


if __name__ == '__main__':
    main()
