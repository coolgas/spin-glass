"""One-shot analysis of preserved Monte Carlo campaigns; never launches simulations.

Run on AutoDL, with a single BLAS thread. Only the compact JSON evidence and
executed notebook are intended for local transfer; raw arrays stay remote.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')


def frozen_helpers(base):
    source = base / 'l6-convergence-20261103-v1/code/transformer_qmc'
    namespace = dict(np=np, combinations=combinations)
    provenance = {}
    for name, wanted in [
        ('production_protocol.py', ['geometry']),
        ('independent_replica_projector.py', ['cross_connected', 'spectrum_from_moments']),
        ('disorder_scaling.py', ['observables']),
    ]:
        path = source / name
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        provenance[str(path)] = sha(path)
    return namespace['observables'], provenance


def extract_run(path, campaign, replay):
    record = read(path)
    task = record.get('task', record.get('case'))
    raw = path.with_suffix('.npz')
    assert sha(raw) == record['raw_sha256'], raw
    with np.load(raw, allow_pickle=False) as arrays:
        counts = arrays['counts'].astype(float)
        weights = arrays['weight_ess']
        tags = arrays['tags'].astype(float)
        walkers = task['walkers']
        assert counts.shape == tags.shape[:-1]
        assert tags.shape[1] == 4 and tags.shape[-1] == 3 * task['L']**2
        np.testing.assert_allclose(counts.sum(-1), walkers, rtol=0, atol=0)
        ess = walkers**2 / (counts**2).sum(-1)
        lineage = counts.max(-1) / walkers
        np.testing.assert_allclose(ess, arrays['descendant_ess'])
        np.testing.assert_allclose(lineage, arrays['largest_lineage'])
        np.testing.assert_allclose(ess.min(), record['minimum_descendant_ess'])
        np.testing.assert_allclose(lineage.max(), record['largest_lineage'])
        alarm = (ess < 8) | (lineage > .5)
        assert bool(alarm.any()) == record['ancestry_alarm']
        for values in [ess, lineage, weights, arrays['energy']]:
            assert np.isfinite(values).all()
        assert (weights > 0).all() and (weights <= walkers * (1+1e-12)).all()
        checks = ['raw_sha256', 'descendant_counts', 'ancestry_alarm', 'finite_diagnostics']
        if 'mass' in record:
            mass, grid = np.asarray(record['mass']), np.asarray(record['q_grid'])
            assert np.isfinite(mass).all() and (mass >= 0).all()
            np.testing.assert_allclose(mass.sum(), 1, atol=1e-12)
            np.testing.assert_allclose([grid @ mass, grid**2 @ mass],
                                       [record['q'], record['q2']], atol=1e-12)
            probabilities = counts / walkers
            magnetization = np.einsum('crw,crwn->crn', probabilities, tags).mean(0)
            pair = np.zeros((4, tags.shape[-1], tags.shape[-1]))
            for c in range(tags.shape[0]):
                for replica in range(4):
                    spins = tags[c, replica]
                    pair[replica] += spins.T @ (probabilities[c, replica, :, None] * spins) / tags.shape[0]
            result = replay(magnetization, pair, task['L'])
            for key in ['q', 'q2', 'spectrum']:
                np.testing.assert_allclose(result[key], record[key], rtol=2e-8, atol=2e-10)
            checks += ['histogram_normalization_and_moments', 'raw_q_q2_connected_spectrum']
        stats = dict(descendant_ess=ess.tolist(), largest_lineage=lineage.tolist(),
                     descendant_ess_min=float(ess.min()), descendant_ess_median=float(np.median(ess)),
                     largest_lineage_max=float(lineage.max()),
                     alarmed_population_cycles=int(alarm.sum()), population_cycles=int(alarm.size),
                     weight_ess_quantiles=np.quantile(weights, [0, .1, .5, .9, 1]).tolist(),
                     weight_ess_fraction_median=float(np.median(weights) / walkers))
    relative = str(path)
    budget_match = re.search(r'(?:extra_|extra)(\d+)', relative)
    stints_match = re.search(r'stints_(\d+)', relative)
    arm = None
    if '/measurements/' in relative:
        arm = relative.split('/measurements/')[1].split('/')[0]
    return dict(campaign=campaign, path=str(path), json_sha256=sha(path), raw_path=str(raw),
                raw_sha256=record['raw_sha256'], task=task, record=record, checks=checks,
                target_updates=400+int(budget_match[1]) if budget_match else None,
                stints=int(stints_match[1]) if stints_match else None, arm=arm, **stats)


def extract_training(path, campaign):
    manifest_path = path.parent / 'manifest.json'
    manifest = read(manifest_path)
    settings = manifest['settings']
    rows = []
    for record in read(path)['rows']:
        folder = path.parent / f"stint_{record['stint']:03d}"
        raw = folder / 'walkers.npz'
        assert sha(raw) == record['raw_sha256'], raw
        model = folder / 'model.pt'
        assert sha(model) == record['model_sha256'], model
        assert sha(folder / 'input_model.pt') == record['input_model_sha256']
        assert sha(folder / 'fit.npz') == record['fit_sha256']
        with np.load(raw, allow_pickle=False) as arrays:
            start = round(settings['equil'] / settings['delta'])
            lineages = arrays['lineages'][start:]
            ess, lineage = lineages[..., 0], lineages[..., 1]
            np.testing.assert_allclose(ess.min(), record['minimum_training_lineage_ess'])
            np.testing.assert_allclose(lineage.max(), record['largest_training_lineage'])
            assert bool(ess.min() < 8 or lineage.max() > .5) == record['training_ancestry_alarm']
            weight_fraction = float(np.median(arrays['weight_ess'][start:]) / settings['walkers'])
        rows.append(dict(record, collection_weight_ess_fraction_median=weight_fraction,
                         path=str(folder / 'summary.json'), raw_path=str(raw)))
    return dict(campaign=campaign, arm=path.parent.parent.name, context=path.parent.name,
                path=str(path), json_sha256=sha(path), manifest=manifest,
                manifest_sha256=sha(manifest_path), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/root/autodl-tmp'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base, args.output
    output.mkdir(parents=True, exist_ok=True)
    replay, helper_sources = frozen_helpers(base)
    specifications = [
        ('fixed_low', 'l6-convergence-20261103-v1/results', '**/runs/*.json'),
        ('fixed_extension', 'extended-rbm-20261103-v1/results', '**/runs/*.json'),
        ('reference_fields', 'l6-reference-fields-20261103-v1/results', '**/runs/*.json'),
        ('ancestry_controls', 'ancestry-diagnostic-20261103-v1/results', '**/runs/*.json'),
        ('production_audits', 'disorder-scaling-repaired-20261103-v2/results', 'projector/runs/*.json'),
        ('feedback_original', 'self-learning-l6-20261103-v1/results', 'projector/**/runs/*.json'),
        ('feedback_frontier', 'self-learning-frontier-20261103-v1/scaling/L6', 'measurements/**/runs/*.json'),
        ('pcd_curriculum', 'l6-pcd-curriculum-20261103-v1/results', 'measurements/**/runs/*.json'),
        ('small_exact', 'self-learning-rbm-20261103-v1/validation', 'projector/runs/*.json'),
        ('strong_field', 'rbm-repair-20261103-v1/validation', 'projector/*.json'),
    ]
    campaigns, runs, failures = [], [], []
    for name, relative, pattern in specifications:
        root = base / relative
        selected = []
        for path in sorted(root.glob(pattern)):
            if path.name.endswith('_progress.json'):
                continue
            record = read(path)
            if 'raw_sha256' not in record:
                failures.append(dict(campaign=name, path=str(path), record=record))
                continue
            selected.append(extract_run(path, name, replay))
        status_files = [root / 'status.json', root / 'parallel_status.json']
        statuses = {str(p): dict(sha256=sha(p), record=read(p)) for p in status_files if p.exists()}
        manifests = {str(p): dict(sha256=sha(p), record=read(p)) for p in [root/'manifest.json'] if p.exists()}
        for path, entry in statuses.items():
            for failure in entry['record'].get('failures', []):
                failures.append(dict(campaign=name, path=path, record=failure))
        campaigns.append(dict(name=name, root=str(root), selection=pattern, runs=len(selected),
                              alarms=sum(r['record']['ancestry_alarm'] for r in selected),
                              statuses=statuses, manifests=manifests))
        runs.extend(selected)
        print('EXTRACTED', name, len(selected), 'alarms', campaigns[-1]['alarms'], flush=True)

    training = []
    for campaign, relative, pattern in [
        ('feedback_frontier', 'self-learning-frontier-20261103-v1/scaling/L6', 'training/*/*/summary.json'),
        ('pcd_curriculum', 'l6-pcd-curriculum-20261103-v1/results', 'training/*/*/summary.json'),
    ]:
        for path in sorted((base / relative).glob(pattern)):
            training.append(extract_training(path, campaign))
        print('TRAINING_VERIFIED', campaign, sum(len(x['rows']) for x in training if x['campaign']==campaign), flush=True)

    coverage = []
    for L in [6, 8, 10]:
        root = base / 'self-learning-frontier-20261103-v1/scaling' / f'L{L}'
        metadata = {}
        for name in ['status.json', 'user_stop.json', 'manifest.json']:
            path = root / name
            if path.exists():
                metadata[name] = dict(path=str(path), sha256=sha(path), record=read(path))
        coverage.append(dict(L=L, new_measurements=len(list(root.glob('measurements/**/runs/*.json'))),
                             training_stints=sum(len(read(p)['rows']) for p in root.glob('training/*/*/summary.json')),
                             metadata=metadata))
    matched_initials = []
    for campaign in ['feedback_frontier', 'pcd_curriculum']:
        contexts = sorted({t['context'] for t in training if t['campaign']==campaign})
        for context in contexts:
            groups = [t for t in training if t['campaign']==campaign and t['context']==context]
            initial = {t['manifest']['initial_sha256'] for t in groups}
            assert len(initial) == 1, (campaign, context)
            matched_initials.append(dict(campaign=campaign,context=context,
                                         arms=[t['arm'] for t in groups],initial_sha256=next(iter(initial))))
    for run in runs:
        if run['campaign'] not in ['feedback_frontier', 'pcd_curriculum']:
            continue
        groups = [t for t in training if t['campaign']==run['campaign'] and t['arm']==run['arm']
                  and t['manifest']['settings']['sample_id']==run['task']['sample_id']
                  and t['manifest']['settings']['seed']==run['task']['training_seed']]
        assert len(groups)==1
        row = groups[0]['rows'][run['stints']-1]
        assert row['stint']==run['stints']-1
        assert row['model_sha256']==run['record']['provenance']['model_sha256']
        run['checks'].append('frozen_measurement_model_matches_training_milestone')

    # Freeze measured comparisons before notebook plotting; repeated measurements
    # and seeds are shown individually, never treated as independent disorders.
    budgets = [r for r in runs if r['campaign'] in ['fixed_low', 'fixed_extension', 'reference_fields']]
    comparisons = []
    def mean(group, key):
        return np.mean([r['record'][key] for r in group], axis=0)
    def compare(left, right, info):
        assert len(left) == len(right) == 2
        for group in [left, right]:
            assert sorted(r['task']['repeat'] for r in group) == [0, 1]
        a, b = mean(left, 'spectrum'), mean(right, 'spectrum')
        # Existing protocol: 10% of the larger magnitude, absolute floor .001.
        tolerance = np.maximum(.1*np.maximum(np.abs(a), np.abs(b)), .001)
        comparisons.append(dict(info, energy_per_site_difference=float(abs(mean(left,'energy')-mean(right,'energy'))/108),
                                q2_difference=float(abs(mean(left,'q2')-mean(right,'q2'))),
                                histogram_tv=float(.5*np.abs(mean(left,'mass')-mean(right,'mass')).sum()),
                                spectrum_max_tolerance_ratio=float(np.max(np.abs(a-b)/tolerance)),
                                ancestry_alarm=any(r['record']['ancestry_alarm'] for r in left+right),
                                left_paths=[r['path'] for r in left], right_paths=[r['path'] for r in right]))
    for gamma in sorted({r['task']['gamma'] for r in budgets}):
        for disorder in [60000, 60001]:
            for updates in [400, 1600, 4000]:
                groups=[[r for r in budgets if r['task']['gamma']==gamma and r['task']['sample_id']==disorder
                         and r['target_updates']==updates and r['task']['training_seed']==seed] for seed in [0,1]]
                compare(*groups, dict(kind='between_seed', gamma=gamma, sample_id=disorder, target_updates=updates))
            for seed in [0,1]:
                groups=[[r for r in budgets if r['task']['gamma']==gamma and r['task']['sample_id']==disorder
                         and r['target_updates']==updates and r['task']['training_seed']==seed] for updates in [1600,4000]]
                compare(*groups, dict(kind='last_two_budgets', gamma=gamma, sample_id=disorder, seed=seed))

    # Identify reuse rather than treating controls copied between campaigns as
    # additional independent simulations. No reused frontier controls are counted.
    reused = Counter(r['raw_sha256'] for r in runs)
    duplicates = {digest:[r['path'] for r in runs if r['raw_sha256']==digest]
                  for digest,count in reused.items() if count > 1}
    control_links = []
    for path in sorted((base/'self-learning-frontier-20261103-v1/scaling/L6').glob('controls/*/runs/*.json')):
        record = read(path)
        matches=[r['path'] for r in runs if r['raw_sha256']==record['raw_sha256']]
        assert matches, path
        assert sha(path.with_suffix('.npz')) == record['raw_sha256']
        control_links.append(dict(path=str(path), sha256=sha(path), matched_runs=matches))

    # Reconstruct the deterministic bond realization from the archived generator.
    # This is an auditable source/seed check, not a claim that GFMC stored bonds.
    source = base / 'l6-convergence-20261103-v1/code'
    sys.path.insert(0, str(source))
    from transformer_qmc.physics import make_af_sample
    bonds = []
    for L, disorder in sorted({(r['task']['L'],r['task']['sample_id']) for r in runs}):
        sample = make_af_sample(L, disorder)
        arrays = {name:np.asarray(getattr(sample,name)) for name in ['bonds','J']}
        bonds.append(dict(L=L, sample_id=disorder, seed_entropy=[20261027,L,disorder,1],
                          provenance='Reconstructed from archived make_af_sample; GFMC raw does not store a separate bond array',
                          arrays={name:dict(shape=list(value.shape),dtype=str(value.dtype),
                                            sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name,value in arrays.items()}))

    supplements = {}
    for relative in [
        'self-learning-rbm-20261103-v1/validation/summary.json',
        'l6-pcd-curriculum-20261103-v1/results/research_report.json',
        'rbm-repair-20261103-v1/failed_start_reproduction.json',
        'rbm-repair-20261103-v1/validation/projection_summary.json',
    ]:
        path = base / relative
        entry = read(path)
        if relative.endswith('research_report.json'):
            entry = {k:v for k,v in entry.items() if k in ['updated_utc','comparisons','limitations','ground_state_qualified']}
        supplements[relative] = dict(path=str(path),sha256=sha(path),record=entry)
    # Original repaired launch errors remain discoverable, distinct from completed
    # projection records. These JSON artifacts are small and kept verbatim.
    def contains_error(value):
        if isinstance(value, dict):
            return 'error' in value or any(contains_error(v) for v in value.values())
        return isinstance(value, list) and any(contains_error(v) for v in value)
    for path in sorted((base/'rbm-repair-20261103-v1/validation').glob('**/*summary*.json')):
        entry = read(path)
        if contains_error(entry):
            failures.append(dict(campaign='strong_field',path=str(path),sha256=sha(path),record=entry))

    report = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),
                  scope='All saved runs in the explicitly listed campaign roots; no new simulation',
                  thresholds=dict(descendant_ess=8,largest_lineage=.5,energy_per_site=.001,q2=.01,histogram_tv=.05,
                                  spectrum_relative=.1,spectrum_absolute_floor=.001),
                  campaigns=campaigns,runs=runs,training=training,size_coverage=coverage,
                  comparisons=comparisons,failures=failures,supplements=supplements,
                  reused_controls=control_links,duplicate_raw_files=duplicates,bond_reconstruction=bonds,
                  matched_initial_models=matched_initials,
                  helper_sources=helper_sources,extractor_sha256=sha(__file__),
                  ground_state_qualified=False)
    write(output/'report.json',report)
    write(output/'sha256.json',{'report.json':sha(output/'report.json')})
    print(json.dumps(dict(output=str(output),runs=len(runs),training_stints=sum(len(t['rows']) for t in training),
                          comparisons=len(comparisons),bytes=(output/'report.json').stat().st_size,
                          duplicate_raw_groups=len(duplicates)),indent=2),flush=True)


if __name__ == '__main__':
    main()
