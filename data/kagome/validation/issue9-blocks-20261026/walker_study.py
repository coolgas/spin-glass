"""Bounded flat-guide walker and block study, using the unchanged projector."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import inspect
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "issue9-energy2048-20261026"
SOURCE = PREVIOUS / "source"
sys.path.insert(0, str(SOURCE / "code"))
import kagome_qea.pqmc as pq
from kagome_qea.sample import make_sample

SETTINGS = dict(L=6, sample_id=0, gamma=0.05, sigma=1.0, n_replicas=2,
                tau_block=0.2, n_blocks=2560, populations=8,
                walker_counts=[10000, 20000], long_blocks=20480, long_walkers=5000, seed=20261025,
                spawned_children=32, n_equilibrate=0, n_forward_cycles=0,
                workers=16, working_energy_tolerance_per_spin=0.001)


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def save_result(path, **value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **value)
    temp.replace(path)


def projector(walkers, child, blocks, observed):
    rng = np.random.default_rng(np.random.SeedSequence(SETTINGS["seed"]).spawn(32)[child])
    original = pq._branch_dt
    start = time.monotonic()
    last_saved = 0
    def observer(population, limit):
        nonlocal last_saved
        frame = inspect.currentframe().f_back
        try:
            if frame.f_code.co_name == "one_block" and frame.f_locals["covered"] == 0:
                energy = frame.f_locals["measure_energy"]
                times = frame.f_locals["measure_time"]
                count = len(energy)
                if count and count % 320 == 0 and count != last_saved:
                    stem = f"m{walkers}_p{child}"
                    save_result(ROOT/"partial"/f"{stem}.npz", energy=energy, time=times)
                    write_json(ROOT/"partial"/f"{stem}.json", dict(
                        walkers=walkers, child=child, blocks=count,
                        tau=float(times[-1]), elapsed_seconds=time.monotonic()-start))
                    print(f"PROGRESS {stem} blocks={count} tau={times[-1]:.1f}", flush=True)
                    last_saved = count
        finally:
            del frame
        return original(population, limit)
    if observed:
        pq._branch_dt = observer
    try:
        return pq._project_once(make_sample(6, True, 0).hamiltonian(0.0), 0.05, rng,
                               n_walkers=walkers, n_replicas=2, tau_block=0.2,
                               n_equilibrate=0, n_measure=blocks, n_forward_cycles=0)
    finally:
        pq._branch_dt = original


def smoke():
    # Cross a real partial-write boundary to verify the observation path itself.
    plain = projector(32, 0, 321, False)
    observed = projector(32, 0, 321, True)
    assert plain.keys() == observed.keys()
    for key in plain:
        np.testing.assert_equal(plain[key], observed[key], err_msg=key)
    with np.load(ROOT/"partial"/"m32_p0.npz") as data:
        np.testing.assert_array_equal(data["energy"], plain["energy_measure_trace"][:320])
        np.testing.assert_array_equal(data["time"], plain["energy_measure_time"][:320])
    row = dict(result_fields=len(plain), exact_observer_match=True,
               partial_boundary_verified=True, finished=datetime.now(timezone.utc).isoformat())
    write_json(ROOT/"smoke.json", row)
    print("SMOKE", json.dumps(row), flush=True)


def run_population(walkers, child, blocks):
    start = time.monotonic()
    stem = f"m{walkers}_p{child}"
    assert not (ROOT/"runs"/f"{stem}.npz").exists(), stem
    result = projector(walkers, child, blocks, True)
    energy, times = result["energy_measure_trace"], result["energy_measure_time"]
    assert energy.shape == times.shape == (blocks,)
    assert np.isfinite(energy).all()
    np.testing.assert_allclose(times, np.arange(1, len(times)+1)*0.2)
    if walkers == 5000:
        with np.load(PREVIOUS/"runs"/f"p{child}.npz", allow_pickle=False) as previous:
            np.testing.assert_array_equal(energy[:10240], previous["energy_measure_trace"])
            np.testing.assert_array_equal(times[:10240], previous["energy_measure_time"])
    save_result(ROOT/"runs"/f"{stem}.npz", **result)
    row = dict(walkers=walkers, child=child, blocks=len(times), tau=float(times[-1]),
               exact_10240_block_prefix=(walkers == 5000),
               elapsed_seconds=time.monotonic()-start, finished=datetime.now(timezone.utc).isoformat())
    write_json(ROOT/"runs"/f"{stem}.json", row)
    return row


def estimate(values):
    values = np.asarray(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    half = float(student_t.ppf(0.975, len(values)-1)*se)
    return dict(mean=mean, se=se, ci95=[mean-half, mean+half], values=values.tolist())


def analyze():
    long_rows = []
    prefix_checks = []
    old_late = []
    for child in range(8):
        with np.load(ROOT/"runs"/f"m5000_p{child}.npz", allow_pickle=False) as data:
            energy = data["energy_measure_trace"].copy()
            times = data["energy_measure_time"].copy()
        assert energy.shape == times.shape == (20480,) and np.isfinite(energy).all()
        np.testing.assert_allclose(times, np.arange(1, 20481)*0.2)
        with np.load(PREVIOUS/"runs"/f"p{child}.npz", allow_pickle=False) as previous:
            np.testing.assert_array_equal(energy[:10240], previous["energy_measure_trace"])
            np.testing.assert_array_equal(times[:10240], previous["energy_measure_time"])
            old_late.append(previous["energy_measure_trace"][7680:].mean())
        long_rows.append(energy)
        prefix_checks.append(dict(child=child, exact_energy_time_prefix=True, blocks=10240))
    long_matrix = np.stack(long_rows)
    long_late = long_matrix[:, 15360:].mean(axis=1)
    long_summary = dict(
        blocks=20480, tau=4096, late_energy=estimate(long_late),
        primary_late_drift=estimate(long_late-long_matrix[:, 10240:15360].mean(axis=1)),
        change_from_previous_late=estimate(long_late-np.asarray(old_late)),
        population_range=[float(long_late.min()), float(long_late.max())],
        prefix_checks=prefix_checks)
    matrices = {}
    for walkers in (5000, 10000, 20000):
        rows = []
        for child in range(8):
            path = (PREVIOUS/"runs"/f"p{child}.npz" if walkers == 5000
                    else ROOT/"runs"/f"m{walkers}_p{child}.npz")
            with np.load(path, allow_pickle=False) as data:
                rows.append(data["energy_measure_trace"][:2560].copy())
        matrices[walkers] = np.stack(rows)
    summaries = {}
    for walkers, matrix in matrices.items():
        late = matrix[:, 1920:2560].mean(axis=1)
        drift = late-matrix[:, 1280:1920].mean(axis=1)
        summaries[str(walkers)] = dict(
            late_energy=estimate(late), late_drift=estimate(drift),
            population_range=[float(late.min()), float(late.max())],
            block_windows={str(stop): estimate(matrix[:, stop//2:stop].mean(axis=1))
                           for stop in (320, 640, 1280, 2560)})
    contrasts = {}
    tolerance = 108*SETTINGS["working_energy_tolerance_per_spin"]
    for smaller, larger in ((5000, 10000), (10000, 20000), (5000, 20000)):
        row = estimate(matrices[larger][:, 1920:].mean(axis=1)-matrices[smaller][:, 1920:].mean(axis=1))
        row["within_working_tolerance"] = bool(max(abs(v) for v in row["ci95"]) <= tolerance)
        contrasts[f"{larger}_minus_{smaller}"] = row
    bound_path = ROOT/"bound_sample0.json"
    bound = json.loads(bound_path.read_text()) if bound_path.exists() else None
    if bound:
        upper = bound["quantum_variational_upper_bound"]
        for row in summaries.values():
            row["late_mean_minus_variational_upper_bound"] = row["late_energy"]["mean"]-upper
            row["late_ci_wholly_above_variational_bound"] = bool(row["late_energy"]["ci95"][0] > upper)
    if bound:
        long_summary["late_mean_minus_variational_upper_bound"] = float(long_late.mean()-upper)
        long_summary["late_ci_wholly_above_variational_bound"] = bool(long_summary["late_energy"]["ci95"][0] > upper)
    result = dict(settings=SETTINGS, long_block_summary=long_summary, walker_summaries=summaries,
                  walker_contrasts=contrasts, independent_bound=bound,
                  production_allowed=False, recommended_equilibration_blocks=None,
                  decision="Exploratory screen only; require absolute-reference, start-policy, held-out and observable convergence checks.",
                  finished=datetime.now(timezone.utc).isoformat())
    write_json(ROOT/"final.json", result)
    lines = ["# L6 block and walker study", "", result["decision"], "",
             "Total energy per replica; eight populations on the same sample. All intervals are descriptive 95% Student-t intervals.", "",
             "| Walkers | Energy, blocks 1920–2560 | SE | Late drift | Population range |",
             "| --- | ---: | ---: | ---: | --- |"]
    for walkers, row in summaries.items():
        lines.append(f"| {walkers} | {row['late_energy']['mean']:.6f} | {row['late_energy']['se']:.6f} | {row['late_drift']['mean']:.6f} | {row['population_range']} |")
    lines += ["", "| Walker contrast | Late energy change | 95% interval |", "| --- | ---: | --- |"]
    for name, row in contrasts.items():
        lines.append(f"| {name} | {row['mean']:.6f} | {row['ci95']} |")
    if bound:
        lines += ["", f"Independent variational upper bound: E0 <= {bound['quantum_variational_upper_bound']:.9f}."]
    lines += ["", "## Doubling the blocks at 5,000 walkers", "",
              f"20,480 blocks, tau=4096: late energy {long_summary['late_energy']}; primary late drift {long_summary['primary_late_drift']}; change from the time-2048 late window {long_summary['change_from_previous_late']}.",
              "All eight energy/time prefixes through 10,240 blocks reproduce exactly."]
    lines += ["", "No production block count has been qualified by this exploratory stage. See PROTOCOL.md, final.json and the unsmoothed population archives.", ""]
    (ROOT/"REPORT.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, constrained_layout=True)
    for ax, (walkers, matrix) in zip(axes, matrices.items()):
        y = matrix.reshape(8, -1, 40).mean(axis=2)/108
        x = np.arange(20.5, 2560, 40)*0.2
        for row in y:
            ax.plot(x, row, lw=0.8, alpha=0.6)
        ax.plot(x, y.mean(axis=0), color="black", lw=2)
        if bound:
            ax.axhline(bound["quantum_variational_upper_bound"]/108, ls="--", color="red", label="Variational upper bound")
        ax.set(title=f"{walkers:,} walkers", xlabel="Imaginary time τ")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Energy per spin")
    if bound:
        axes[0].legend(fontsize=7)
    for suffix in ("png", "svg"):
        fig.savefig(ROOT/f"walker_trace.{suffix}", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(40.5, 20480, 80)*0.2
    y = long_matrix.reshape(8, -1, 80).mean(axis=2)/108
    for ax in axes:
        for child, row in enumerate(y):
            ax.plot(x, row, lw=0.8, alpha=0.65, label=f"Population {child}")
        ax.plot(x, y.mean(axis=0), color="black", lw=2, label="Mean")
        if bound:
            ax.axhline(upper/108, ls="--", color="red", label="Variational upper bound")
        ax.set(xlabel="Imaginary time τ", ylabel="Energy per spin")
        ax.grid(alpha=0.2)
    axes[0].set_title("20,480 blocks at 5,000 walkers")
    axes[1].set(xlim=(2048, 4096), title="Late projection")
    late_plot = y[:, x >= 2048]
    lo = min(float(late_plot.min()), upper/108 if bound else float(late_plot.min()))
    axes[1].set_ylim(lo-0.005, float(late_plot.max())+0.005)
    axes[1].legend(fontsize=7, ncol=2)
    for suffix in ("png", "svg"):
        fig.savefig(ROOT/f"block_trace.{suffix}", dpi=150)
    plt.close(fig)
    print("FINAL", json.dumps({k:v for k,v in result.items() if k not in ("independent_bound", "walker_summaries")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    for name in ("runs", "partial"):
        (ROOT/name).mkdir(exist_ok=True)
    campaign = json.loads((ROOT/"campaign.json").read_text())
    assert campaign["settings"] == SETTINGS
    for name, digest in campaign["source_sha256"].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest, name
    for name, digest in campaign["study_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    if args.analyze_only:
        analyze()
        return
    smoke()
    if args.smoke_only:
        return
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=16) as pool:
        jobs = [(5000, child, SETTINGS["long_blocks"]) for child in range(8)]
        jobs += [(walkers, child, SETTINGS["n_blocks"])
                 for walkers in (20000, 10000) for child in range(8)]
        futures = [pool.submit(run_population, *job) for job in jobs]
        for future in as_completed(futures):
            print("DONE", json.dumps(future.result()), flush=True)
    analyze()
    write_json(ROOT/"exit.json", dict(exit_code=0, elapsed_seconds=time.monotonic()-start,
                                     finished=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        write_json(ROOT/"exit.json", dict(exit_code=1, finished=datetime.now(timezone.utc).isoformat()))
        (ROOT/"failure.txt").write_text(traceback.format_exc())
        raise
