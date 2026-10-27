"""Bounded stronger-field energy screen using the unchanged flat projector."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "issue9-energy2048-20261026"
sys.path.insert(0, str(PREVIOUS / "source" / "code"))
from kagome_qea import pqmc
from kagome_qea.sample import make_sample

SETTINGS = dict(L=6, sample_id=0, sigma=1.0, h=0.0, gammas=[3.0, 5.0],
                walker_counts=[5000, 10000], populations=8, n_replicas=2,
                tau_block=0.2, n_blocks=64, n_equilibrate=0, n_forward_cycles=0,
                seed=20261026, spawned_children=32, workers=16,
                working_energy_tolerance_per_spin=0.001)


def write_json(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def project(gamma, walkers, child, blocks, uncoupled=False):
    ham = make_sample(6, True, 0).hamiltonian(0.0)
    if uncoupled:
        ham = ham.with_classical_scale(0.0)
    rng = np.random.default_rng(np.random.SeedSequence(SETTINGS["seed"]).spawn(32)[child])
    return pqmc._project_once(ham, gamma, rng, n_walkers=walkers, n_replicas=2,
                              tau_block=0.2, n_equilibrate=0, n_measure=blocks,
                              n_forward_cycles=0)


def smoke():
    start = time.monotonic()
    a = project(3.0, 32, 0, 4)
    b = project(3.0, 32, 0, 4)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_equal(a[key], b[key], err_msg=key)
    for gamma in SETTINGS["gammas"]:
        result = project(gamma, 64, 0, 8, uncoupled=True)
        np.testing.assert_allclose(result["energy_measure_trace"], -108*gamma, rtol=1e-12)
        np.testing.assert_allclose(result["energy_growth"], -108*gamma, rtol=1e-12)
    benchmark_start = time.monotonic()
    project(5.0, 128, 0, 8)
    benchmark_seconds = time.monotonic()-benchmark_start
    row = dict(exact_repeatability=True, result_fields=len(a), uncoupled_exact_fields=[3, 5],
               benchmark_seconds=benchmark_seconds, benchmark_walkers=128,
               benchmark_blocks=8, benchmark_gamma=5,
               elapsed_seconds=time.monotonic()-start)
    write_json(ROOT / "smoke.json", row)
    print("SMOKE", json.dumps(row), flush=True)


def run_population(gamma, walkers, child):
    stem = f"g{gamma:g}_m{walkers}_p{child}"
    path = ROOT / "runs" / f"{stem}.npz"
    assert not path.exists(), stem
    start = time.monotonic()
    result = project(gamma, walkers, child, SETTINGS["n_blocks"])
    assert result["energy_measure_trace"].shape == (64,)
    assert np.isfinite(result["energy_measure_trace"]).all()
    np.testing.assert_allclose(result["energy_measure_time"], np.arange(1, 65)*0.2)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **result)
    temporary.replace(path)
    row = dict(gamma=gamma, walkers=walkers, child=child, blocks=64,
               seconds=time.monotonic()-start, finished=datetime.now(timezone.utc).isoformat())
    write_json(path.with_suffix(".json"), row)
    return row


def estimate(values):
    values = np.asarray(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    half = float(t.ppf(0.975, len(values)-1)*se)
    return dict(mean=mean, se=se, ci95=[mean-half, mean+half], values=values.tolist())


def analyze():
    ham = make_sample(6, True, 0).hamiltonian(0.0)
    bound = json.loads((ROOT / "reference_state.json").read_text())
    spins = np.asarray(bound["spins"])
    angles = np.asarray(bound["product_angles"])
    z = spins*np.cos(angles)
    product_diagonal = -np.dot(ham.J, z[ham.bonds[:, 0]]*z[ham.bonds[:, 1]])
    product_x = np.sin(angles).sum()
    matrices, summaries, contrasts, references = {}, {}, {}, {}
    for field_index, gamma in enumerate(SETTINGS["gammas"]):
        references[str(gamma)] = dict(
            all_x_upper_bound=-108*gamma,
            saved_product_upper_bound=float(product_diagonal-gamma*product_x),
            variational_upper_bound=min(-108*gamma, float(product_diagonal-gamma*product_x)),
            second_order_high_field_energy=float(-108*gamma-np.dot(ham.J, ham.J)/(4*gamma)))
        for walker_index, walkers in enumerate(SETTINGS["walker_counts"]):
            rows = []
            for child in range(8*(2*field_index+walker_index), 8*(2*field_index+walker_index+1)):
                with np.load(ROOT / "runs" / f"g{gamma:g}_m{walkers}_p{child}.npz", allow_pickle=False) as data:
                    rows.append(data["energy_measure_trace"].copy())
            matrix = np.stack(rows)
            matrices[(gamma, walkers)] = matrix
            late = matrix[:, 48:].mean(axis=1)
            drift = estimate(late-matrix[:, 32:48].mean(axis=1))
            summaries[f"g{gamma:g}_m{walkers}"] = dict(
                late_energy=estimate(late), late_drift=drift,
                drift_within_working_tolerance=bool(max(abs(x) for x in drift["ci95"]) <= 0.108),
                population_range=[float(late.min()), float(late.max())],
                windows={str(stop): estimate(matrix[:, stop//2:stop].mean(axis=1)) for stop in (16, 32, 64)},
                late_mean_minus_variational_bound=float(late.mean()-references[str(gamma)]["variational_upper_bound"]))
        a = matrices[(gamma, 5000)][:, 48:].mean(axis=1)
        b = matrices[(gamma, 10000)][:, 48:].mean(axis=1)
        va, vb = a.var(ddof=1)/8, b.var(ddof=1)/8
        se = float(np.sqrt(va+vb))
        df = float((va+vb)**2/(va*va/7+vb*vb/7))
        difference = float(b.mean()-a.mean())
        half = float(t.ppf(0.975, df)*se)
        contrasts[str(gamma)] = dict(mean=difference, se=se, df=df,
                                     ci95=[difference-half, difference+half],
                                     within_working_tolerance=bool(abs(difference)+half <= 0.108))
    output = dict(settings=SETTINGS, summaries=summaries, walker_contrasts=contrasts,
                  references=references, production_allowed=False, phase_established=False,
                  finished=datetime.now(timezone.utc).isoformat())
    write_json(ROOT / "final.json", output)
    lines = ["# L6 stronger-field energy screen", "",
             "Flat guide, same sample0; total energies per replica. Independent populations supply uncertainties.", "",
             "| Gamma/sigma | Walkers | Late energy | SE | Late drift | Drift 95% interval |",
             "|---|---:|---:|---:|---:|---|"]
    for key, row in summaries.items():
        lines.append(f"| {key} | | {row['late_energy']['mean']:.6f} | {row['late_energy']['se']:.6f} | {row['late_drift']['mean']:.6f} | {row['late_drift']['ci95']} |")
    lines += ["", "Walker contrasts (10,000 minus 5,000), independent-population Welch intervals:",
              json.dumps(contrasts, indent=2), "", "Field-specific references:", json.dumps(references, indent=2),
              "", "Energy stability is not a phase determination. No production settings are qualified by this first screen."]
    (ROOT / "REPORT.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    tm = np.arange(1, 65)*0.2
    for i, gamma in enumerate(SETTINGS["gammas"]):
        for j, walkers in enumerate(SETTINGS["walker_counts"]):
            ax = axes[i, j]
            matrix = matrices[(gamma, walkers)]/108
            for row in matrix:
                ax.plot(tm, row, lw=0.7, alpha=0.6)
            ax.plot(tm, matrix.mean(axis=0), color="black", lw=2, label="Population mean")
            ax.axhline(references[str(gamma)]["variational_upper_bound"]/108, color="red", ls="--", label="Product-state upper bound")
            ax.axhline(references[str(gamma)]["second_order_high_field_energy"]/108, color="gray", ls=":", label="Second-order estimate")
            ax.axvline(6.4, color="gray", alpha=0.4)
            ax.axvline(9.6, color="gray", alpha=0.4)
            ax.set(title=f"Gamma/sigma={gamma:g}; {walkers:,} walkers", xlabel="Imaginary time τ", ylabel="Energy per spin")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    for suffix in ("png", "svg"):
        fig.savefig(ROOT / f"field_trace.{suffix}", dpi=150)
    plt.close(fig)
    print("FINAL", json.dumps(output), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    campaign = json.loads((ROOT / "campaign.json").read_text())
    assert campaign["settings"] == SETTINGS
    for name, digest in campaign["source_sha256"].items():
        assert hashlib.sha256((PREVIOUS / "source" / name).read_bytes()).hexdigest() == digest, name
    for name, digest in campaign["study_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    (ROOT / "runs").mkdir(exist_ok=True)
    if args.analyze_only:
        analyze()
        return
    smoke()
    if args.smoke_only:
        return
    start = time.monotonic()
    jobs = [(gamma, walkers, child)
            for i, gamma in enumerate(SETTINGS["gammas"])
            for j, walkers in enumerate(SETTINGS["walker_counts"])
            for child in range(8*(2*i+j), 8*(2*i+j+1))]
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(run_population, *job) for job in jobs]
        for future in as_completed(futures):
            print("DONE", json.dumps(future.result()), flush=True)
    analyze()
    write_json(ROOT / "exit.json", dict(exit_code=0, elapsed_seconds=time.monotonic()-start,
                                       finished=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        write_json(ROOT / "exit.json", dict(exit_code=1, finished=datetime.now(timezone.utc).isoformat()))
        (ROOT / "failure.txt").write_text(traceback.format_exc())
        raise
