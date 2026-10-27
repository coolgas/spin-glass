"""Recompute completed-campaign checks locally; never run the projector."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "issue9-energy2048-20261026"


def estimate(values):
    values = np.asarray(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(8))
    half = float(t.ppf(0.975, 7) * se)
    return dict(mean=mean, se=se, ci95=[mean-half, mean+half], values=values.tolist())


def check_estimate(values, saved):
    row = estimate(values)
    for key in row:
        np.testing.assert_allclose(row[key], saved[key], rtol=1e-12, atol=1e-12)
    return row


def main():
    manifest = json.loads((ROOT / "download_manifest.json").read_text())
    remote = json.loads((ROOT / "remote_sha256.json").read_text())
    assert {row["remote_path"] for row in manifest.values()} == set(remote)
    for name, row in manifest.items():
        payload = (ROOT / name).read_bytes()
        assert len(payload) == row["bytes"], name
        assert hashlib.sha256(payload).hexdigest() == row["sha256"] == remote[row["remote_path"]], name
    campaign = json.loads((ROOT / "campaign.json").read_text())
    for name, digest in campaign["source_sha256"].items():
        assert hashlib.sha256((PREVIOUS / "source" / name).read_bytes()).hexdigest() == digest, name
    for name, digest in campaign["study_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    saved = json.loads((ROOT / "final.json").read_text())
    assert saved["settings"] == campaign["settings"]
    assert json.loads((ROOT / "exit.json").read_text())["exit_code"] == 0
    assert saved["production_allowed"] is False
    assert saved["recommended_equilibration_blocks"] is None
    matrices = {}
    for walkers in (5000, 10000, 20000):
        rows = []
        blocks = 20480 if walkers == 5000 else 2560
        for child in range(8):
            stem = f"m{walkers}_p{child}"
            with np.load(ROOT / "runs" / f"{stem}.npz", allow_pickle=False) as data:
                # Access every field so ZIP CRC and object-free loading are checked.
                fields = {key: data[key] for key in data.files}
                assert len(fields) == 21
                energy = fields["energy_measure_trace"]
                times = fields["energy_measure_time"]
                assert energy.shape == times.shape == (blocks,)
                assert np.isfinite(energy).all() and np.isfinite(times).all()
                np.testing.assert_allclose(times, np.arange(1, blocks+1)*0.2)
                for key, expected in dict(n_walkers=walkers, n_replicas=2, n_sites=108,
                                          n_equilibrate=0, n_measure=blocks,
                                          n_energy_blocks=blocks, n_populations=1,
                                          tau_block=0.2, floor_unique=2, floor_effective=8).items():
                    assert fields[key].item() == expected, (stem, key)
                assert fields["energy_equilibration_trace"].size == 0
                assert fields["energy_equilibration_time"].size == 0
                metadata = json.loads((ROOT / "runs" / f"{stem}.json").read_text())
                assert metadata["blocks"] == blocks and metadata["walkers"] == walkers
                assert metadata["child"] == child
                if walkers == 5000:
                    with np.load(PREVIOUS / "runs" / f"p{child}.npz", allow_pickle=False) as old:
                        np.testing.assert_array_equal(energy[:10240], old["energy_measure_trace"])
                        np.testing.assert_array_equal(times[:10240], old["energy_measure_time"])
                with np.load(ROOT / "partial" / f"{stem}.npz", allow_pickle=False) as partial:
                    assert len(partial["energy"]) == blocks-320
                    np.testing.assert_array_equal(partial["energy"], energy[:blocks-320])
                    np.testing.assert_array_equal(partial["time"], times[:blocks-320])
                rows.append(energy.copy())
        matrices[walkers] = np.stack(rows)
    long = matrices[5000]
    late = long[:, 15360:].mean(axis=1)
    long_saved = saved["long_block_summary"]
    check_estimate(late, long_saved["late_energy"])
    check_estimate(late-long[:, 10240:15360].mean(axis=1), long_saved["primary_late_drift"])
    check_estimate(late-long[:, 7680:10240].mean(axis=1), long_saved["change_from_previous_late"])
    np.testing.assert_allclose([late.min(), late.max()], long_saved["population_range"])
    for walkers, matrix in matrices.items():
        matrix = matrix[:, :2560]
        row = saved["walker_summaries"][str(walkers)]
        check_estimate(matrix[:, 1920:].mean(axis=1), row["late_energy"])
        check_estimate(matrix[:, 1920:].mean(axis=1)-matrix[:, 1280:1920].mean(axis=1), row["late_drift"])
        for stop in (320, 640, 1280, 2560):
            check_estimate(matrix[:, stop//2:stop].mean(axis=1), row["block_windows"][str(stop)])
    for smaller, larger in ((5000, 10000), (10000, 20000), (5000, 20000)):
        values = matrices[larger][:, 1920:2560].mean(axis=1)-matrices[smaller][:, 1920:2560].mean(axis=1)
        row = saved["walker_contrasts"][f"{larger}_minus_{smaller}"]
        check_estimate(values, row)
        assert row["within_working_tolerance"] == bool(max(abs(v) for v in row["ci95"]) <= 0.108)

    # Evaluate the saved normalized product state directly, without optimization.
    sys.path.insert(0, str(PREVIOUS / "source" / "code"))
    from kagome_qea.sample import make_sample
    ham = make_sample(6, True, 0).hamiltonian(0.0)
    bound = json.loads((ROOT / "bound_sample0.json").read_text())
    assert bound == saved["independent_bound"]
    spins = np.asarray(bound["spins"])
    angles = np.asarray(bound["product_angles"])
    assert spins.shape == angles.shape == (108,) and set(spins) <= {-1, 1}
    z = spins*np.cos(angles)
    x = np.sin(angles)
    np.testing.assert_allclose(x*x+z*z, 1, atol=1e-14)
    classical = -sum(float(j)*int(spins[a])*int(spins[b]) for (a, b), j in zip(ham.bonds, ham.J))
    product = -sum(float(j)*z[a]*z[b] for (a, b), j in zip(ham.bonds, ham.J))-0.05*sum(x)
    np.testing.assert_allclose(classical, bound["classical_energy"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(product, bound["product_energy"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(min(classical, product), bound["quantum_variational_upper_bound"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(bound["quantum_lower_bound"], bound["classical_dual_bound"]-108*0.05)
    assert late.min() > product
    result = dict(download_files=len(manifest), download_bytes=sum(row["bytes"] for row in manifest.values()),
                  sha256_verified=True, complete_populations=24, exact_time2048_prefix_checks=8,
                  exact_partial_trace_checks=24, independent_analysis_verified=True,
                  independently_evaluated_product_energy=float(product),
                  solver_patch_sha256=campaign["solver_patch_sha256"],
                  production_allowed=False, recommended_equilibration_blocks=None)
    (ROOT / "local_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
