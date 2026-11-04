"""Finite L6 negative-persistence and field-curriculum comparison on AutoDL."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import write
from .correlation_lengths import sha
from .feedback_frontier import audit_training, verify_measurement
from .parallel_disorder_scaling import fit_name
from .physics import make_af_sample
from .projected_overlap_histograms import run_one
from .sampled_simple_guides import train, freeze_diagnostics, sampling_alarms
from .self_learning_l6_pilot import contexts, measurement_task
from .self_learning_rbm import run as learn
from .simple_guides import RBM


ARMS = ['curriculum_pcd', 'direct_pcd', 'direct_reset']
CURRICULUM = [g for g in [4., 3., 2., 1.5, 1.2, 1., .85, .75, .65] for _ in range(2)]+[.6]*6
MILESTONES = [21, 24]


def settings(arm, task, initial, output):
    if arm not in ARMS:
        raise ValueError('unknown L6 comparison arm')
    return argparse.Namespace(output=output, initial=initial, L=6, gamma=.6,
        sample_id=task['sample_id'], seed=task['seed'], stints=24, walkers=1024,
        islands=4, negative_temperatures=16, step_rms=.025, step_max=.15,
        equil=8., collect=12., delta=.1, sample_walkers=64, updates=1000,
        batch=256, gibbs=10, initial_sweeps=400, lr=.001, device='cuda',
        projection_device='cuda', persistent_negative=arm!='direct_reset',
        gamma_schedule=CURRICULUM if arm=='curriculum_pcd' else [.6]*24)


def audit_persistence(folder):
    """Check actual negative bank/ladder identity across every stint boundary."""
    cfg = json.loads((folder/'manifest.json').read_text())['settings']
    rows = json.loads((folder/'summary.json').read_text())['rows']
    result = []
    for row in rows:
        stint = row['stint']; path = folder/f'stint_{stint:03d}'
        assert sha(path/'negative_initial.npz') == row['negative_initial_sha256']
        with np.load(path/'negative_initial.npz') as initial:
            if stint and cfg['persistent_negative']:
                with np.load(folder/f'stint_{stint-1:03d}'/'fit.npz') as previous:
                    np.testing.assert_array_equal(initial['negative'], previous['negative_samples'])
                    np.testing.assert_array_equal(initial['ladder'], previous['negative_ladder'])
                assert row['negative_carried']
            else:
                assert not row['negative_carried']
        result.append(dict(stint=stint, gamma=row['gamma'], carried=row['negative_carried']))
    write(folder/'persistence_audit.json', dict(rows=result, passed=True))
    return result


def compare_records(a, b):
    spectrum_a = np.asarray(a['spectrum']); spectrum_b = np.asarray(b['spectrum'])
    return dict(energy_per_site_difference=(a['energy']-b['energy'])/108,
        q2_difference=a['q2']-b['q2'], histogram_tv=float(np.abs(np.asarray(a['mass'])-b['mass']).sum()/2),
        spectrum_max_tolerance_ratio=float(np.max(np.abs(spectrum_a-spectrum_b)/
            np.maximum(.001, .1*np.maximum(np.abs(spectrum_a), np.abs(spectrum_b))))),
        ancestry_alarm=bool(a['ancestry_alarm'] or b['ancestry_alarm']),
        interpretation='Descriptive difference; correlated training budgets and two repeats do not bound systematic error.')


def report(root):
    training = []; measurements = []
    for p in sorted(root.glob('training/*/*/summary.json')):
        arm = p.parent.parent.name; name = p.parent.name
        for row in json.loads(p.read_text())['rows']:
            with np.load(p.parent/f"stint_{row['stint']:03d}"/'walkers.npz') as raw:
                # Shape: time, (mixed/growth), island.
                energy = raw['energies'][80:, 0, :]/108
            training.append(dict(arm=arm, context=name, **row,
                mixed_energy_per_site=float(energy.mean()), island_energy_means=energy.mean(0).tolist(),
                island_energy_range=float(np.ptp(energy.mean(0)))))
    for p in sorted(root.glob('measurements/*/*/*/result.json')):
        measurements.append(json.loads(p.read_text()))
    comparisons = []
    indexed = {(r['arm'], r['task']['sample_id'], r['task']['seed'], r['stints']): r for r in measurements}
    for key, group in indexed.items():
        arm, sid, seed, stint = key
        comparisons.append(dict(kind='repeat', arm=arm, sample_id=sid, seed=seed, stints=stint,
            **compare_records(*group['measurements'])))
        other = indexed.get((arm, sid, 1, stint)) if seed==0 else None
        if other:
            for repeat in [0, 1]:
                comparisons.append(dict(kind='seed', arm=arm, sample_id=sid, stints=stint, repeat=repeat,
                    **compare_records(group['measurements'][repeat], other['measurements'][repeat])))
        for other_arm in ARMS[ARMS.index(arm)+1:]:
            other = indexed.get((other_arm, sid, seed, stint))
            if other:
                for repeat in [0, 1]:
                    comparisons.append(dict(kind='arm', arm=arm, other_arm=other_arm, sample_id=sid,
                        seed=seed, stints=stint, repeat=repeat,
                        **compare_records(group['measurements'][repeat], other['measurements'][repeat])))
        if stint==24:
            other = indexed.get((arm, sid, seed, 21))
            if other:
                for repeat in [0, 1]:
                    comparisons.append(dict(kind='budget', arm=arm, sample_id=sid, seed=seed, repeat=repeat,
                        **compare_records(group['measurements'][repeat], other['measurements'][repeat])))
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), training=training,
        measurements=measurements, comparisons=comparisons, ground_state_qualified=False,
        limitations='Matched update count, not identical propagation cost; curriculum spends 18 stints at higher fields. '
        'No ancestry-based selection. A high starting field is chosen heuristically; paramagnetic phase is not proved. '
        'PCD persistence does not ensure barrier crossing. Population/time/start convergence remains required.')
    write(root/'research_report.json', result)
    from .notebook_l6_pcd import build
    build(root)
    return result


def run(args):
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    regression = json.loads(args.regressions.read_text())
    assert regression['passed'] and regression['skipped']==0
    for name, digest in regression['source_hashes'].items():
        assert sha(Path(__file__).parent/name)==digest, name
    manifest = dict(format=1, contexts=contexts(), arms=ARMS,
        settings={arm:{k:v for k,v in vars(settings(arm, contexts()[0], None, None)).items()
                       if k not in ['output','initial']} for arm in ARMS},
        sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        regression_sha256=sha(args.regressions), planned_stints=288,
        planned_new_measurements=48, workers=1, deadline=None,
        initialization='One shared Gamma4 VMC400 checkpoint per disorder/seed, Adam .001; fresh likelihood optimizer for each arm.',
        comparison='Direct reset vs direct PCD isolates persistence across stints. Direct PCD vs curriculum PCD changes fields only.',
        scheduling='One-stint round robin: all contexts reach the first high-field feedback stint promptly; '
                   'measure each context/arm at global stints21 and24 immediately after its milestone.',
        retain='All seeds/disorders, raw samples, failures, ancestry and frozen-start alarms; no retries.',
        ground_state_qualified=False)
    path = root/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text())==manifest
    else:
        write(path, manifest)
    failure_path = root/'failures.json'
    failures = json.loads(failure_path.read_text()) if failure_path.exists() else []
    failed = {r['key'] for r in failures}
    def status(state, current=None):
        write(root/'status.json', dict(state=state, current=current,
            updated_utc=datetime.now(timezone.utc).isoformat(), failures=failures,
            planned_stints=288, planned_new_measurements=48, deadline=None, ground_state_qualified=False))
    def failure(key, error):
        record = dict(key=key, error=repr(error), traceback=traceback.format_exc(),
                      utc=datetime.now(timezone.utc).isoformat())
        failures.append(record); failed.add(key); write(failure_path, failures)
        print(json.dumps(record), flush=True)
    def initialize(task):
        high = dict(task, gamma=4., mode='direct'); name = fit_name(high)
        folder = root/'initial'/fit_name(task); folder.mkdir(parents=True, exist_ok=True)
        status('high_field_vmc', task)
        cfg = argparse.Namespace(chains=256, temperatures=8, steps=400, sweeps_per_step=2,
            diagnostic_sweeps=200, learning_rate=.001, device='cuda')
        result = train(high, cfg, folder, float('inf'))
        assert result['status']=='completed'
        checkpoint = folder/(name+'.pt')
        assert sha(checkpoint)==result['checkpoint_sha256']
        return checkpoint
    initials = {}
    for target in range(1, 25):
        for task in contexts():
            name = fit_name(task)
            if 'initial/'+name in failed:
                continue
            try:
                if name not in initials:
                    initials[name] = initialize(task)
                initial = initials[name]
            except Exception as error:
                failure('initial/'+name, error); continue
            for arm in ARMS:
                key = arm+'/'+name
                if key in failed:
                    continue
                folder = root/'training'/arm/name
                try:
                    state_file = folder/'status.json'
                    done = json.loads(state_file.read_text())['completed_stints'] if state_file.exists() else 0
                    if done < target:
                        status('adapting', dict(arm=arm, target_stints=target, **task))
                        learn(settings(arm, task, initial, folder), stop_after_stints=target-done)
                    if target in MILESTONES:
                        audit_training(folder, 6, task['sample_id']); audit_persistence(folder)
                        measure = root/'measurements'/arm/name/f'stints_{target:02d}'
                        measure.mkdir(parents=True, exist_ok=True)
                        model_path = folder/f'stint_{target-1:03d}'/'model.pt'
                        status('frozen_checks', dict(arm=arm, stints=target, **task))
                        frozen_path = measure/'frozen.json'
                        if frozen_path.exists():
                            diagnostic = json.loads(frozen_path.read_text())
                            assert diagnostic['model_sha256']==sha(model_path)
                            assert diagnostic['raw_sha256']==sha(frozen_path.with_suffix('.npz'))
                        else:
                            model = RBM(108, 2).to('cuda')
                            model.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=False)['model'])
                            checks, raw = freeze_diagnostics(model, make_af_sample(6,task['sample_id']).hamiltonian(.15),
                                                            .6, 256, 8, 200, 20261106+task['seed'])
                            np.savez_compressed(frozen_path.with_suffix('.npz'), **raw)
                            diagnostic = dict(rows=checks, sampling_alarms=sampling_alarms(checks,256),
                                model_sha256=sha(model_path), raw_sha256=sha(frozen_path.with_suffix('.npz')))
                            write(frozen_path, diagnostic)
                        records = []
                        for repeat in [0,1]:
                            status('projecting', dict(arm=arm, stints=target, repeat=repeat, **task))
                            record = run_one(dict(model=str(model_path),task=measurement_task(task,repeat),
                                                  projection_device='cuda'), measure/'projector')
                            record['verification'] = verify_measurement(record, measure/'projector')
                            records.append(record)
                        write(measure/'result.json', dict(arm=arm, task=task, stints=target,
                            target_field_stints=target-18 if arm=='curriculum_pcd' else target,
                            frozen=diagnostic, measurements=records, ground_state_qualified=False))
                except Exception as error:
                    failure(key, error)
            report(root)
    status('completed_with_failures' if failures else 'completed_unqualified')
    report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regressions', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    try:
        with threadpool_limits(limits=1):
            run(args)
    except Exception as error:
        write(args.output/'fatal_failure.json', dict(error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
