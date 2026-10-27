"""Verify downloaded overlap archives independently of the campaign runner."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent


def check_estimate(values, record):
    mean = np.mean(values)
    se = np.std(values, ddof=1)/np.sqrt(len(values))
    interval = mean + np.array([-1, 1])*t.ppf(.975, len(values)-1)*se
    for name, expected in dict(mean=mean, se=se, ci95=interval, values=values).items():
        np.testing.assert_allclose(record[name], expected, rtol=1e-12, atol=1e-10)


def main():
    manifest = json.loads((ROOT/'download_manifest.json').read_text())
    remote = json.loads((ROOT/'remote_sha256.json').read_text())
    assert {row['remote_path'] for row in manifest.values()} == set(remote)
    for name, row in manifest.items():
        content = (ROOT/name).read_bytes()
        assert len(content) == row['bytes']
        assert hashlib.sha256(content).hexdigest() == row['sha256'] == remote[row['remote_path']]
    campaign = json.loads((ROOT/'campaign.json').read_text())
    source = ROOT.parent/'issue9-energy2048-20261026'/'source'
    for name, digest in campaign['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    for name, digest in campaign['study_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    final = json.loads((ROOT/'final.json').read_text())
    assert final['settings'] == campaign['settings']
    assert json.loads((ROOT/'exit.json').read_text())['exit_code'] == 0
    assert final['disorder_samples'] == 1 and not final['phase_established']
    values, truncated_total = [], 0
    for index, duration in enumerate(campaign['settings']['forward_times']):
        parts = []
        for child in range(8*index, 8*(index+1)):
            stem = f't{index}_p{child}'
            with np.load(ROOT/'runs'/f'{stem}.npz', allow_pickle=False) as archive:
                part = {name: archive[name] for name in archive.files}
            assert len(part) == 67
            for name, expected in dict(n_walkers=10000, n_replicas=2, n_sites=108,
                                       n_equilibrate=32, n_measure=32, n_populations=1,
                                       tau_block=.2, floor_unique=2, floor_effective=8,
                                       n_forward_cycles=8, tau_forward_requested=duration).items():
                assert part[name].item() == expected, (stem, name)
            for phase, start in (('equilibration', 1), ('measure', 33)):
                trace = part[f'energy_{phase}_trace']
                assert trace.shape == (32,) and np.isfinite(trace).all()
                np.testing.assert_allclose(part[f'energy_{phase}_time'], np.arange(start, start+32)*.2)
            for power, name in ((1, 'q'), (2, 'q2'), (3, 'q3'), (4, 'q4')):
                series = part[name+'_series']
                assert series.shape == (8,) and np.isfinite(series).all()
                assert np.all(np.abs(series) <= 1)
                if power % 2 == 0:
                    assert np.all(series >= 0)
                np.testing.assert_allclose(part[name], series.mean())
            np.testing.assert_allclose(part['q2_series'], part['cycle_q2'])
            assert np.all(part['q4_series'] >= part['q2_series']**2-1e-14)
            assert np.all(part['q4_series'] <= part['q2_series']+1e-14)
            np.testing.assert_allclose(part['chi_sg'], 108*part['q2'])
            np.testing.assert_allclose(part['chi_sg_error'], 108*part['q2_error'])
            achieved = part['forward_times']
            assert achieved.shape == (8,) and np.all((achieved >= 0) & (achieved <= duration+1e-12))
            truncated = achieved+1e-8 < duration
            assert int(part['n_forward_truncated']) == int(truncated.sum())
            assert bool(part['projection_complete']) == (not truncated.any())
            assert np.array_equal(part['cycle_stop'] != 0, truncated)
            np.testing.assert_allclose(part['tau_forward'], achieved.min())
            assert np.all(part['cycle_unique'] >= 2)
            assert np.all(part['cycle_effective'] >= 8)
            assert int(part['min_unique_ancestors']) == int(part['cycle_unique'].min())
            np.testing.assert_allclose(part['min_effective_ancestors'], part['cycle_effective'].min())
            for cycle, checkpoints in enumerate(part['cycle_checkpoints']):
                valid = checkpoints[np.isfinite(checkpoints[:, 0])]
                assert valid[0, 0] == 0 and np.all(np.diff(valid[:, 0]) > 0)
                np.testing.assert_allclose(valid[-1], [achieved[cycle], part['cycle_unique'][cycle],
                    part['cycle_effective'][cycle], part['cycle_weight_ess'][cycle],
                    part['cycle_max_fraction'][cycle]])
            meta = json.loads((ROOT/'runs'/f'{stem}.json').read_text())
            assert (meta['index'], meta['child'], meta['duration']) == (index, child, duration)
            parts.append(part)
        record = final['summaries'][str(duration)]
        q2 = np.array([part['q2_series'].mean() for part in parts])
        q4 = np.array([part['q4_series'].mean() for part in parts])
        check_estimate(q2, record['q2'])
        check_estimate(q4, record['q4'])
        check_estimate(108*q2, record['chi_sg'])
        np.testing.assert_allclose(record['q_ea_overlap_rms'], np.sqrt(q2.mean()))
        np.testing.assert_allclose(record['q_ea_overlap_rms_ci95'],
                                   np.sqrt(np.maximum(0, record['q2']['ci95'])))
        np.testing.assert_allclose(record['achieved_forward_times'], [part['forward_times'] for part in parts])
        assert record['truncated_cycles'] == sum(int(p['n_forward_truncated']) for p in parts)
        assert record['all_projection_complete'] == all(bool(p['projection_complete']) for p in parts)
        assert record['min_unique_ancestors'] == min(int(p['min_unique_ancestors']) for p in parts)
        np.testing.assert_allclose(record['min_effective_ancestors'], min(float(p['min_effective_ancestors']) for p in parts))
        np.testing.assert_allclose(record['max_descendant_fraction'], max(float(p['cycle_max_fraction'].max()) for p in parts))
        truncated_total += record['truncated_cycles']
        values.append(108*q2)
    for index in range(1, len(values)):
        a, b = values[index-1:index+1]
        variance = np.array([a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)])
        df = variance.sum()**2/np.sum(variance**2/np.array([len(a)-1, len(b)-1]))
        mean, se = b.mean()-a.mean(), np.sqrt(variance.sum())
        interval = mean+np.array([-1, 1])*t.ppf(.975, df)*se
        durations = campaign['settings']['forward_times']
        key = f'{durations[index]}_minus_{durations[index-1]}'
        for name, expected in dict(mean=mean, se=se, df=df, ci95=interval).items():
            np.testing.assert_allclose(final['chi_forward_contrasts'][key][name], expected, rtol=1e-12, atol=1e-10)
    output = dict(sha256_verified=True, download_files=len(manifest), complete_populations=32,
                  forward_cycles=256, truncated_cycles=truncated_total,
                  independent_statistics_verified=True, achieved_times_and_lineage_verified=True,
                  susceptibility_identity_verified=True, overlap_definition_verified=True)
    (ROOT/'local_verification.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
