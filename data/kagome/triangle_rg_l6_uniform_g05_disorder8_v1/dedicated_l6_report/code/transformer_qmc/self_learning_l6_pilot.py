"""Finite matched Gamma=.60 L6 feedback diagnostic, never production promotion."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import write
from .correlation_lengths import sha
from .parallel_disorder_scaling import fit_name
from .physics import make_af_sample
from .projected_overlap_histograms import run_one
from .sampled_simple_guides import freeze_diagnostics, sampling_alarms
from .self_learning_rbm import run as learn
from .simple_guides import RBM


def contexts():
    return [dict(L=6, gamma=.6, sample_id=sid, kind='rbm_2', seed=seed, mode='anneal')
            for sid in [60000, 60001] for seed in [0, 1]]


def measurement_task(task, repeat):
    # Same task identity (and RNG seed) as the paired fixed-guide control.
    # Model and output paths differ; provenance prevents accidental reuse.
    return dict(L=6, gamma=.6, sample_id=task['sample_id'], kind='rbm_2',
        training_seed=task['seed'], audit='l6_convergence_training', walkers=512,
        equil=16., forward=16., start='guide', delta=.1, cycles=8, repeat=repeat)


def settings(initial, output, task):
    return argparse.Namespace(output=output, initial=initial, L=6, gamma=.6,
        sample_id=task['sample_id'], seed=task['seed'], stints=6, walkers=1024,
        equil=8., collect=12., delta=.1, sample_walkers=64, updates=1000,
        batch=256, gibbs=10, initial_sweeps=400, lr=.001, device='cuda')


def validate_l2(root):
    training = json.loads((root/'training_report.json').read_text())
    measurements = json.loads((root/'measurements.json').read_text())['rows']
    status = json.loads((root/'status.json').read_text())
    assert training['completed_stints'] == 12
    assert status['state'] == 'completed_unqualified' and status['completed'] == 4
    assert {(r['task']['training_seed'], r['task']['repeat']) for r in measurements} == {(s,r) for s in [0,1] for r in [0,1]}
    for row in measurements:
        error = row['errors']
        assert not row['ancestry_alarm']
        assert abs(error['energy_error_per_site']) <= .001
        assert abs(error['q2_error']) <= .01 and error['histogram_tv'] <= .05
        assert error['spectrum_max_tolerance_ratio'] <= 1
    return dict(files={n:sha(root/n) for n in ['training_report.json', 'measurements.json', 'summary.json', 'guides.json']},
        rationale='Completed exact L2 implementation and independent observable screen; sufficient for a diagnostic L6 pilot only.',
        uncertainty='Two L2 repeats and frozen sampling checks do not qualify population, time or training-chain convergence.',
        ground_state_qualified=False)


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert sha(destination) == sha(source), str(destination)
    else:
        shutil.copy2(source, destination)


def run(args):
    approval = validate_l2(args.validation)
    args.output.mkdir(parents=True, exist_ok=True)
    items = []
    for task in contexts():
        name = fit_name(task)
        baseline = args.anchor/'training'/name/'extra_0'/(name+'.pt')
        record = json.loads(baseline.with_suffix('.json').read_text())
        assert record['task'] == task and record['status'] == 'completed'
        assert sha(baseline) == record['checkpoint_sha256']
        assert sha(baseline.with_suffix('.npz')) == record['npz_sha256']
        control = args.output/'controls'/name
        for suffix in ['.pt', '.json', '.npz']:
            copy_verified(baseline.with_suffix(suffix), control/(name+suffix))
        records = []
        for path in sorted((args.anchor/'projector'/name/'extra_0/runs').glob('*.json')):
            row = json.loads(path.read_text())
            assert row['task'] in [measurement_task(task, r) for r in [0, 1]]
            assert row['provenance']['model_sha256'] == sha(baseline)
            assert sha(path.with_suffix('.npz')) == row['raw_sha256']
            for suffix in ['.json', '.npz']:
                copy_verified(path.with_suffix(suffix), control/'runs'/(path.stem+suffix))
            records.append(dict(record=str(path), sha256=sha(path), raw_sha256=row['raw_sha256']))
        assert len(records) == 2
        items.append(dict(task=task, initial=str(control/(name+'.pt')),
            initial_sha256=sha(baseline), control_records=records))
    manifest = dict(format=1, contexts=items, l2_review=approval,
        training_settings={k:v for k,v in vars(settings(None,None,contexts()[0])).items()
                           if k not in ['output','initial','sample_id','seed']},
        contexts_planned=4, planned_stints=24, planned_final_projections=8,
        workers=1, deadline=None,
        optimizer='Fresh likelihood Adam; VMC optimizer not reused across a different objective.',
        selection='Retain both seeds and disorders, all stints, failures and alarms. Measure both final guides even if sampling checks alarm.',
        measurements='Established four-population fixed-final-guide protocol; no tags or averages cross adaptation.',
        scope='Finite Gamma=.60 pilot. No broad L6, L8/L10, production promotion, or automatic further training.',
        sources={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
        numerics_sha256=sha(Path(__file__).parent.parent/'qmc_numerics.py'), ground_state_qualified=False)
    path = args.output/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        write(path, manifest)
    rows = []; failures = []
    def status(state, current=None):
        write(args.output/'status.json', dict(state=state,current=current,
            updated_utc=datetime.now(timezone.utc).isoformat(), completed_contexts=len(rows),
            planned_contexts=4, planned_stints=24, planned_final_projections=8,
            failures=failures, ground_state_qualified=False))
    for item in items:
        task = item['task']; directory = args.output/'training'/fit_name(task)
        try:
            status('adapting', task)
            learn(settings(Path(item['initial']), directory, task))
            final = directory/'stint_005/model.pt'
            status('frozen_sampling_checks', task)
            diagnostic_path = directory/'frozen.json'
            if diagnostic_path.exists():
                diagnostic = json.loads(diagnostic_path.read_text())
                assert diagnostic['model_sha256'] == sha(final)
                assert diagnostic['raw_sha256'] == sha(diagnostic_path.with_suffix('.npz'))
            else:
                model = RBM(108, 2).to('cuda')
                model.load_state_dict(torch.load(final,map_location='cuda',weights_only=False)['model'])
                ham = make_af_sample(6, task['sample_id']).hamiltonian(.15)
                checks, raw = freeze_diagnostics(model, ham, .6, 256, 8, 200, 20261104+task['seed'])
                np.savez_compressed(diagnostic_path.with_suffix('.npz'), **raw)
                diagnostic = dict(rows=checks, sampling_alarms=sampling_alarms(checks,256),
                    model_sha256=sha(final), raw_sha256=sha(diagnostic_path.with_suffix('.npz')),
                    ground_state_qualified=False)
                write(diagnostic_path, diagnostic)
            measured = []
            for repeat in [0, 1]:
                status('projecting',dict(**task,repeat=repeat))
                result = run_one(dict(model=str(final),task=measurement_task(task, repeat)),
                                 args.output/'projector'/fit_name(task))
                measured.append(result)
            rows.append(dict(task=task,training=json.loads((directory/'summary.json').read_text()),
                             frozen=diagnostic,measurements=measured))
        except Exception as error:
            failures.append(dict(task=task,error=repr(error)))
            write(args.output/'failures.json',failures)
        write(args.output/'summary.json',dict(rows=rows,failures=failures,ground_state_qualified=False))
    status('completed_with_failures' if failures else 'completed_unqualified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        run(args)


if __name__ == '__main__':
    main()
