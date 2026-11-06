"""Extract saved L4 contrast evidence on AutoDL; never launch simulations."""
import argparse
import ast
import datetime
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from monte_carlo_failure_report import extract_run, frozen_helpers, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/root/autodl-tmp'))
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed1-root', type=Path,
                        help='Completed four-case seed1 extension; omit to reproduce seed0-only evidence')
    args = parser.parse_args()
    root = args.base / 'l4-production-20261103-v1/results'
    replay, helpers = frozen_helpers(args.base)
    ids = ['3921146fbe6c832880b0875f', '4dedf474c02be1ad8fd38418',
           '694b96715c9bbc02afde361d', '25f5bb04de74628f9a9e81f4']
    runs = [extract_run(root / 'projector/runs' / f'{key}.json',
                        'l4_distribution_contrast', replay) for key in ids]
    extension = None
    if args.seed1_root:
        manifest_path = args.seed1_root / 'manifest.json'
        status_path = args.seed1_root / 'status.json'
        manifest = json.loads(manifest_path.read_text())
        status = json.loads(status_path.read_text())
        assert status['state'] == 'completed' and status['source_hashes_preserved']
        assert len(status['completed']) == 4
        for path, expected in manifest['sources'].items():
            assert sha(Path(path)) == expected, path
        added = [extract_run(Path(item['path']), 'l4_distribution_contrast', replay)
                 for item in status['completed']]
        assert {(r['task']['sample_id'], r['task']['training_seed'], r['task']['repeat'])
                for r in added} == {(d, 1, repeat) for d in [60000, 60001] for repeat in [0, 1]}
        for run in added:
            paired = next(r for r in runs if r['task']['sample_id'] == run['task']['sample_id']
                          and r['task']['repeat'] == run['task']['repeat'])
            assert run['task'] == dict(paired['task'], training_seed=1)
            run['data_origin'] = 'New projection of archived seed1 guide; no retraining'
        for run in runs:
            run['data_origin'] = 'Original archived seed0 measurement'
        runs += added
        assert len({r['raw_sha256'] for r in runs}) == 8
        extension = dict(root=str(args.seed1_root), manifest=manifest, status=status,
                         manifest_sha256=sha(manifest_path), status_sha256=sha(status_path))
        histogram_path = root.parent / 'code/transformer_qmc/projected_overlap_histograms.py'
        namespace = dict(np=np)
        function = next(node for node in ast.parse(histogram_path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'histogram')
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(histogram_path), 'exec'), namespace)
        for run in runs:
            assert run['record']['provenance']['source_sha256'] == sha(histogram_path)
            with np.load(run['raw_path'], allow_pickle=False) as arrays:
                grid, mass = namespace['histogram'](arrays['tags'], arrays['counts'])
            np.testing.assert_allclose(grid, run['record']['q_grid'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(mass, run['record']['mass'], rtol=0, atol=1e-12)
            run['checks'].append('full_histogram_reconstruction_from_raw_tags_counts')
        helpers[str(histogram_path)] = sha(histogram_path)
    baseline_bytes = args.baseline.read_bytes()
    if args.baseline.suffix == '.gz':
        baseline_bytes = gzip.decompress(baseline_bytes)
    baseline = json.loads(baseline_bytes)
    l6 = [r for r in baseline['runs'] if r['campaign'] == 'fixed_low'
          and r['task']['gamma'] == .6 and r['target_updates'] == 400]
    assert len(l6) == 8
    training = []
    for disorder in [60000, 60001]:
        for seed in [0, 1]:
            path = root / 'training' / f'L4_s{disorder}_g0p6_rbm_2_anneal_seed{seed}.json'
            record = json.loads(path.read_text())
            model = path.with_suffix('.pt')
            model_hash = sha(model)
            assert model_hash == record['checkpoint_sha256']
            assert record['status'] == 'completed'
            assert record['signature']['learning_rate'] == .01
            assert record['signature']['steps'] == 400
            training.append(dict(path=str(path), json_sha256=sha(path),
                                 model_path=str(model), model_sha256=model_hash,
                                 record={k: v for k, v in record.items() if k != 'history'},
                                 history_rows=len(record.get('history', [])),
                                 stages=sorted({(r['stage'], r['gamma'])
                                                for r in record.get('history', [])})))
    for run in runs:
        task = run['task']
        guide = next(t for t in training if t['record']['task']['sample_id'] == task['sample_id']
                     and t['record']['task']['seed'] == task['training_seed'])
        assert guide['model_sha256'] == run['record']['provenance']['model_sha256']
        reference = next(r for r in l6 if r['task']['sample_id'] == task['sample_id']
                         and r['task']['training_seed'] == task['training_seed']
                         and r['task']['repeat'] == task['repeat'])
        for key in ['gamma', 'kind', 'walkers', 'equil', 'forward', 'start', 'delta', 'cycles']:
            assert task[key] == reference['task'][key], key
        for key in ['numerics', 'numerics_sha256', 'sources']:
            assert run['record']['provenance'][key] == reference['record']['provenance'][key], key
        run['checks'] += ['training_checkpoint_hash_link', 'L6_projection_protocol_match',
                          'L6_numerical_source_hash_match']
        run['training_path'] = guide['path']
        run['target_updates'] = 400
    selections = []
    for path in sorted((root / 'selections').glob('*g0p6_rbm*json')):
        if any(f's{s}_' in path.name for s in [60000, 60001]):
            selections.append(dict(path=str(path), sha256=sha(path),
                                   record=json.loads(path.read_text())))
    assert len(selections) == 2
    for selection in selections:
        record = selection['record']
        selected = record['selected']
        assert selected['seed'] == min(record['candidate_energies'], key=lambda r: r['energy'])['seed'] == 0
        guide = next(t for t in training if t['record']['task'] == selected)
        assert record['checkpoint_sha256'] == guide['model_sha256']
    l6_training = []
    for disorder in [60000, 60001]:
        for seed in [0, 1]:
            name = f'L6_s{disorder}_g0p6_rbm_2_anneal_seed{seed}'
            path = args.base / 'l6-convergence-20261103-v1/results/training' / name / 'extra_0' / f'{name}.json'
            record = json.loads(path.read_text())
            guide_hash = sha(path.with_suffix('.pt'))
            assert guide_hash == record['checkpoint_sha256']
            assert record['signature']['learning_rate'] == .001
            l4 = next(t for t in training if t['record']['task']['sample_id'] == disorder
                      and t['record']['task']['seed'] == seed)
            for key in ['chains', 'temperatures', 'steps', 'optimizer', 'gradient_norm_cap',
                        'sweeps_per_step', 'diagnostic_sweeps', 'sources']:
                assert record['signature'][key] == l4['record']['signature'][key], key
            for run in l6:
                if run['task']['sample_id'] == disorder and run['task']['training_seed'] == seed:
                    assert run['record']['provenance']['model_sha256'] == guide_hash
            stages = sorted({(r['stage'], r['gamma']) for r in record['history']})
            assert stages == l4['stages']
            l6_training.append(dict(path=str(path), json_sha256=sha(path),
                                    model_sha256=guide_hash, signature=record['signature'], stages=stages))
    comparisons = []
    for size, group in [(4, runs), (6, l6)]:
        for disorder in [60000, 60001]:
            for seed in sorted({r['task']['training_seed'] for r in group}):
                pair = sorted([r for r in group if r['task']['sample_id'] == disorder
                               and r['task']['training_seed'] == seed], key=lambda r: r['task']['repeat'])
                assert [r['task']['repeat'] for r in pair] == [0, 1]
                comparisons.append(dict(L=size, sample_id=disorder, seed=seed,
                                        repeat_tv=float(.5 * np.abs(np.array(pair[0]['record']['mass'])
                                                                   - pair[1]['record']['mass']).sum()),
                                        repeat_q2_difference=abs(pair[0]['record']['q2'] - pair[1]['record']['q2']),
                                        ancestry_alarms=sum(r['record']['ancestry_alarm'] for r in pair)))
    seed_comparisons = []
    if extension:
        for size, group in [(4, runs), (6, l6)]:
            for disorder in [60000, 60001]:
                means = []
                for seed in [0, 1]:
                    pair = [r for r in group if r['task']['sample_id'] == disorder
                            and r['task']['training_seed'] == seed]
                    assert len(pair) == 2
                    means.append(dict(mass=np.mean([r['record']['mass'] for r in pair], axis=0),
                                      q2=float(np.mean([r['record']['q2'] for r in pair]))))
                seed_comparisons.append(dict(L=size, sample_id=disorder,
                    seed0_mean_q2=means[0]['q2'], seed1_mean_q2=means[1]['q2'],
                    q2_difference=abs(means[0]['q2']-means[1]['q2']),
                    histogram_tv=float(.5*np.abs(means[0]['mass']-means[1]['mass']).sum())))
    scope = ('L4 both training seeds and projection repeats; original seed0 plus four new seed1 measurements'
             if extension else 'Saved L4 seed0 projection repeats; not a fully matched two-training-seed study')
    limitations = ['L4 Adam .01 versus L6 Adam .001',
                   'Same disorder ID across L does not mean identical bonds']
    if not extension:
        limitations.append('Only energy-selected L4 seed0 has projection histograms in these contexts')
    report = dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope=scope, extension=extension, seed_comparisons=seed_comparisons,
                  runs=runs, training=training, l6_training=l6_training,
                  selections=selections, comparisons=comparisons,
                  baseline_report_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
                  helper_sources=helpers, extractor_sha256=sha(Path(__file__)),
                  limitations=limitations)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode()
    packed = gzip.compress(payload, mtime=0)
    (args.output / 'report.json').write_bytes(payload)
    (args.output / 'report.json.gz').write_bytes(packed)
    hashes = {'report.json': hashlib.sha256(payload).hexdigest(),
              'report.json.gz': hashlib.sha256(packed).hexdigest()}
    (args.output / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), hashes=hashes, comparisons=comparisons,
                         seed_comparisons=seed_comparisons,
                         runs=[dict(task=r['task'], q2=r['record']['q2'],
                                    ancestry_alarm=r['record']['ancestry_alarm'],
                                    minimum_ess=r['descendant_ess_min']) for r in runs]), indent=2))


if __name__ == '__main__':
    main()
