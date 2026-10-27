"""Independent verification of scan archives and reported numerical summaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent


def main():
    campaign = json.loads((ROOT/'campaign.json').read_text())
    source = ROOT.parent/'issue9-energy2048-20261026'/'source'
    for base, records in ((ROOT, campaign['study_sha256']), (source, campaign['source_sha256'])):
        for name, digest in records.items():
            assert hashlib.sha256((base/name).read_bytes()).hexdigest() == digest, name
    manifest_path = ROOT/'remote_sha256.json'
    if manifest_path.exists():
        for name, digest in json.loads(manifest_path.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    final = json.loads((ROOT/'final.json').read_text())
    grouped, bonds, truncated_total = {}, {}, 0
    seeds = set()
    for task in campaign['tasks']:
        path = ROOT/task['archive']
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata['task'] == task
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256']
        with np.load(path, allow_pickle=False) as archive:
            p = {key: archive[key] for key in archive.files}
        settings = task['settings']
        for key in ('n_walkers', 'n_replicas', 'n_equilibrate', 'n_measure', 'n_forward_cycles', 'tau_block'):
            assert p[key].item() == settings[key], (task['id'], key)
        assert p['tau_forward_requested'] == settings['tau_forward']
        assert p['floor_unique'] == 2 and p['floor_effective'] == 8
        assert int(p['n_sites']) == 3*task['L']**2
        for key, expected in dict(sample_L=task['L'], sample_id=task['sample_id'],
                                  gamma=task['gamma'], sigma=1., h=0., sample_periodic=True,
                                  task_id=task['id'], variant=task['variant']).items():
            assert p[key].item() == expected, (task['id'], key)
        np.testing.assert_equal(p['sample_seed_entropy'], [1000, task['L'], task['sample_id'], 1])
        np.testing.assert_equal(p['solver_seed_entropy'], task['seed_entropy'])
        assert tuple(task['seed_entropy']) not in seeds
        seeds.add(tuple(task['seed_entropy']))
        expected_j = np.random.default_rng([1000, task['L'], task['sample_id'], 1]).normal(
            size=len(p['sample_bonds']))
        np.testing.assert_equal(p['sample_J'], expected_j)
        identity = task['L'], task['sample_id']
        if identity in bonds:
            for a, b in zip(bonds[identity], (p['sample_bonds'], p['sample_J'])):
                np.testing.assert_equal(a, b)
        else:
            bonds[identity] = p['sample_bonds'], p['sample_J']
        for phase, start, count in (('equilibration', 1, settings['n_equilibrate']),
                                    ('measure', settings['n_equilibrate']+1, settings['n_measure'])):
            trace = p[f'energy_{phase}_trace']
            assert trace.shape == (count,) and np.isfinite(trace).all()
            np.testing.assert_allclose(p[f'energy_{phase}_time'], np.arange(start, start+count)*.2)
        for name in ('q', 'q2', 'q3', 'q4'):
            values = p[name+'_series']
            assert values.shape == (8,) and np.isfinite(values).all()
            assert np.all(np.abs(values) <= 1)
            np.testing.assert_allclose(p[name], values.mean())
        assert np.all(p['q2_series'] >= 0)
        assert np.all(p['q4_series'] >= p['q2_series']**2-1e-14)
        assert np.all(p['q4_series'] <= p['q2_series']+1e-14)
        np.testing.assert_allclose(p['chi_sg'], int(p['n_sites'])*p['q2'])
        achieved = p['forward_times']
        assert np.all((achieved >= 0) & (achieved <= settings['tau_forward']+1e-12))
        truncated = achieved+1e-8 < settings['tau_forward']
        assert int(p['n_forward_truncated']) == int(truncated.sum())
        assert bool(p['projection_complete']) == (not truncated.any())
        assert np.array_equal(p['cycle_stop'] != 0, truncated)
        assert np.all(p['cycle_unique'] >= 2) and np.all(p['cycle_effective'] >= 8)
        for i, checkpoints in enumerate(p['cycle_checkpoints']):
            valid = checkpoints[np.isfinite(checkpoints[:, 0])]
            assert valid[0, 0] == 0 and np.all(np.diff(valid[:, 0]) > 0)
            np.testing.assert_allclose(valid[-1], [achieved[i], p['cycle_unique'][i],
                p['cycle_effective'][i], p['cycle_weight_ess'][i], p['cycle_max_fraction'][i]])
        truncated_total += int(truncated.sum())
        key = f"L{task['L']}_g{task['gamma']:g}_s{task['sample_id']}_{task['variant']}"
        grouped.setdefault(key, []).append(p)
    for key, parts in grouped.items():
        row = final['groups'][key]
        matrix = np.array([[p[name+'_series'].mean() for name in ('q', 'q2', 'q3', 'q4')] for p in parts])
        np.testing.assert_allclose(row['population_moments'], matrix)
        np.testing.assert_allclose(row['moment_mean_covariance'], np.cov(matrix, rowvar=False)/4)
        n = int(parts[0]['n_sites'])
        metrics = dict(q2=matrix[:, 1], q4=matrix[:, 3], chi_sg=n*matrix[:, 1],
            energy_per_spin=[p['energy_measure_trace'].mean()/n for p in parts],
            energy_late_minus_early_per_spin=[np.diff(p['energy_measure_trace'].reshape(2, -1).mean(axis=1))[0]/n for p in parts])
        for name, values in metrics.items():
            mean, se = np.mean(values), np.std(values, ddof=1)/2
            np.testing.assert_allclose(row[name]['mean'], mean)
            np.testing.assert_allclose(row[name]['se'], se)
            np.testing.assert_allclose(row[name]['ci95'], mean+np.array([-1, 1])*t.ppf(.975, 3)*se)
        np.testing.assert_allclose(row['binder'], .5*(3-matrix[:, 3].mean()/matrix[:, 1].mean()**2))
        assert row['truncated_cycles'] == sum(int(p['n_forward_truncated']) for p in parts)
        for metric in ('min_unique_ancestors', 'min_effective_ancestors'):
            np.testing.assert_allclose(row[metric], min(float(p[metric]) for p in parts))
        if row['variant'] != 'base':
            base = final['groups'][key.rsplit('_', 1)[0]+'_base']
            for metric in ('chi_sg', 'q2', 'q4', 'energy_per_spin'):
                a, b = np.array(base[metric]['values']), np.array(row[metric]['values'])
                v = np.array([a.var(ddof=1)/4, b.var(ddof=1)/4])
                center, se = b.mean()-a.mean(), np.sqrt(v.sum())
                df = v.sum()**2/np.sum(v*v/3) if se else None
                half = t.ppf(.975, df)*se if se else 0
                record = final['convergence_contrasts'][key][metric]
                np.testing.assert_allclose(record['mean'], center)
                np.testing.assert_allclose(record['ci95'], [center-half, center+half])
    for row in final['curves']:
        size, gamma = row['L'], row['gamma_over_sigma']
        sample_rows = [final['groups'][f'L{size}_g{gamma:g}_s{s}_base'] for s in (0, 1)]
        q2 = np.array([r['q2']['mean'] for r in sample_rows])
        q4 = np.array([r['q4']['mean'] for r in sample_rows])
        np.testing.assert_allclose(row['q_ea_squared'], q2.mean())
        np.testing.assert_allclose(row['q_ea'], np.sqrt(q2.mean()))
        np.testing.assert_allclose(row['chi_sg'], 3*size*size*q2.mean())
        np.testing.assert_allclose(row['binder_ratio_of_disorder_moments'], .5*(3-q4.mean()/q2.mean()**2))
        np.testing.assert_allclose(row['q2_projector_se'], np.linalg.norm([r['q2']['se'] for r in sample_rows])/2)
        np.testing.assert_allclose(row['q2_sample_scatter_se'], q2.std(ddof=1)/np.sqrt(2))
        np.testing.assert_allclose(row['q2_disorder_se_estimate'],
            np.sqrt(max(0, q2.var(ddof=1)-np.mean([r['q2']['se']**2 for r in sample_rows]))/2))
        ids = np.random.default_rng([2026102604, size, 999]).integers(0, 2, size=(10000, 2))
        bs2, bs4 = q2[ids].mean(axis=1), q4[ids].mean(axis=1)
        for key, values in dict(chi_sg=3*size*size*bs2, q_ea=np.sqrt(bs2), binder=.5*(3-bs4/bs2**2)).items():
            np.testing.assert_allclose(row[key+'_disorder_bootstrap_ci95'], np.quantile(values, [.025, .975]))
    assert json.loads((ROOT/'exit.json').read_text())['exit_code'] == 0
    result = dict(populations_verified=len(seeds), independent_disorder_samples=len(bonds),
                  truncated_cycles=truncated_total, statistics_verified=True,
                  metadata_and_lineage_verified=True, transition_established=False)
    (ROOT/'local_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
