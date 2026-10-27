"""Independently verify downloaded high-field archives and their statistics."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent


def check_estimate(values, saved):
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)/np.sqrt(len(values)))
    ci = mean + np.array([-1, 1])*t.ppf(0.975, len(values)-1)*se
    for key, value in dict(mean=mean, se=se, ci95=ci, values=values).items():
        np.testing.assert_allclose(value, saved[key], rtol=1e-12, atol=1e-12)


def main():
    manifest = json.loads((ROOT/'download_manifest.json').read_text())
    remote = json.loads((ROOT/'remote_sha256.json').read_text())
    assert {r['remote_path'] for r in manifest.values()} == set(remote)
    for name, row in manifest.items():
        payload = (ROOT/name).read_bytes()
        assert len(payload) == row['bytes']
        assert hashlib.sha256(payload).hexdigest() == row['sha256'] == remote[row['remote_path']]
    campaign = json.loads((ROOT/'campaign.json').read_text())
    source = ROOT.parent/'issue9-energy2048-20261026'/'source'
    for name, digest in campaign['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    for name, digest in campaign['study_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    saved = json.loads((ROOT/'final.json').read_text())
    assert saved['settings'] == campaign['settings']
    assert json.loads((ROOT/'exit.json').read_text())['exit_code'] == 0
    assert not saved['production_allowed'] and not saved['phase_established']
    sys.path.insert(0, str(source/'code'))
    from kagome_qea.sample import make_sample
    ham = make_sample(6, True, 0).hamiltonian(0.0)
    reference = json.loads((ROOT/'reference_state.json').read_text())
    spins, angles = (np.asarray(reference[key]) for key in ('spins', 'product_angles'))
    assert spins.shape == angles.shape == (108,) and set(spins) <= {-1, 1}
    z, x = spins*np.cos(angles), np.sin(angles)
    np.testing.assert_allclose(z*z+x*x, 1, atol=1e-14)
    matrices = {}
    for i, gamma in enumerate((3.0, 5.0)):
        product = -sum(j*z[a]*z[b] for (a, b), j in zip(ham.bonds, ham.J))-gamma*sum(x)
        refs = saved['references'][str(gamma)]
        for key, value in dict(all_x_upper_bound=-108*gamma,
                               saved_product_upper_bound=product,
                               variational_upper_bound=min(-108*gamma, product),
                               second_order_high_field_energy=-108*gamma-sum(ham.J**2)/(4*gamma)).items():
            np.testing.assert_allclose(value, refs[key], rtol=0, atol=1e-10)
        for j, walkers in enumerate((5000, 10000)):
            rows = []
            for child in range(8*(2*i+j), 8*(2*i+j+1)):
                stem = f'g{gamma:g}_m{walkers}_p{child}'
                with np.load(ROOT/'runs'/f'{stem}.npz', allow_pickle=False) as data:
                    fields = {key: data[key] for key in data.files}
                assert len(fields) == 21
                for key, expected in dict(n_walkers=walkers, n_replicas=2, n_sites=108,
                                          n_equilibrate=0, n_measure=64, n_energy_blocks=64,
                                          n_populations=1, tau_block=0.2,
                                          floor_unique=2, floor_effective=8).items():
                    assert fields[key].item() == expected, (stem, key)
                energy, times = fields['energy_measure_trace'], fields['energy_measure_time']
                assert energy.shape == times.shape == (64,) and np.isfinite(energy).all()
                np.testing.assert_allclose(times, np.arange(1, 65)*0.2)
                assert fields['energy_equilibration_trace'].size == fields['energy_equilibration_time'].size == 0
                meta = json.loads((ROOT/'runs'/f'{stem}.json').read_text())
                assert (meta['gamma'], meta['walkers'], meta['child'], meta['blocks']) == (gamma, walkers, child, 64)
                rows.append(energy)
            matrix = np.stack(rows)
            matrices[gamma, walkers] = matrix
            row = saved['summaries'][f'g{gamma:g}_m{walkers}']
            late = matrix[:, 48:].mean(axis=1)
            check_estimate(late, row['late_energy'])
            check_estimate(late-matrix[:, 32:48].mean(axis=1), row['late_drift'])
            np.testing.assert_allclose([late.min(), late.max()], row['population_range'])
            np.testing.assert_allclose(late.mean()-refs['variational_upper_bound'], row['late_mean_minus_variational_bound'])
            assert row['drift_within_working_tolerance'] == (max(abs(v) for v in row['late_drift']['ci95']) <= 0.108)
            for stop in (16, 32, 64):
                check_estimate(matrix[:, stop//2:stop].mean(axis=1), row['windows'][str(stop)])
        a, b = [matrices[gamma, walkers][:, 48:].mean(axis=1) for walkers in (5000, 10000)]
        variances = np.array([a.var(ddof=1), b.var(ddof=1)])/8
        se = np.sqrt(variances.sum())
        df = variances.sum()**2/sum(variances**2/7)
        mean = b.mean()-a.mean()
        ci = mean+np.array([-1, 1])*t.ppf(0.975, df)*se
        row = saved['walker_contrasts'][str(gamma)]
        for key, value in dict(mean=mean, se=se, df=df, ci95=ci).items():
            # Student-t quantiles differ slightly across local/remote SciPy builds.
            np.testing.assert_allclose(value, row[key], rtol=1e-12, atol=1e-10)
        assert row['within_working_tolerance'] == (max(abs(ci)) <= 0.108)
    output = dict(sha256_verified=True, download_files=len(manifest), complete_populations=32,
                  independent_statistics_verified=True, field_specific_references_verified=True,
                  production_allowed=False, phase_established=False)
    (ROOT/'local_verification.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
