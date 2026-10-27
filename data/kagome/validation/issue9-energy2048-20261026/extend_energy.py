"""Predeclared extension of the frozen L6 flat-PQMC energy study."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "issue9-energy-20261026"
sys.path.insert(0, str(PREVIOUS))
import energy_study as study

study.ROOT = ROOT
study.SOURCE = ROOT / "source"
study.SETTINGS.update(n_blocks=10240, window_blocks=1280)


def analyze():
    matrix = []
    checks = []
    for child in range(8):
        with np.load(ROOT / "runs" / f"p{child}.npz") as data:
            energy = data["energy_measure_trace"].copy()
            times = data["energy_measure_time"].copy()
        assert energy.shape == times.shape == (10240,)
        assert np.isfinite(energy).all()
        assert np.allclose(times, np.arange(1, 10241) * 0.2)
        with np.load(PREVIOUS / "runs" / f"p{child}.npz") as old:
            assert np.array_equal(energy[:2560], old["energy_measure_trace"])
            assert np.array_equal(times[:2560], old["energy_measure_time"])
        checks.append(dict(child=child, exact_energy_and_time_prefix=True, blocks=2560))
        matrix.append(energy)
    matrix = np.stack(matrix)
    windows = []
    means = []
    for start in range(0, 10240, 1280):
        values = matrix[:, start:start+1280].mean(axis=1)
        means.append(values)
        windows.append(dict(start_time=start*0.2, end_time=(start+1280)*0.2,
                            population_means=values.tolist(), **study.estimate(values)))
    early = matrix[:, 5120:7680].mean(axis=1)
    late = matrix[:, 7680:10240].mean(axis=1)
    primary = study.difference(early, late, "tau1024_1536_to_tau1536_2048")
    shifts = [study.difference(means[i-1], means[i], f"window_{i}_to_{i+1}")
              for i in range(1, 8)]
    x = times[5120:]
    centered = matrix[:, 5120:] - matrix[:, 5120:].mean(axis=1)[:, None]
    slopes = (centered @ (x-x.mean())) / np.sum((x-x.mean())**2)
    result = dict(settings=study.SETTINGS, windows=windows, paired_window_shifts=shifts,
        primary_late_shift=primary, final_short_shift=shifts[-1],
        population_early_means=early.tolist(), population_late_means=late.tolist(),
        late_slope=study.estimate(slopes), population_late_slopes=slopes.tolist(),
        prefix_checks=checks, ground_state_qualified=False, susceptibility_qualified=False,
        decision="resolved late-time energy drift" if primary["resolved_drift"] else
                 "no primary late-time drift resolved at current precision; plateau not certified",
        finished=datetime.now(timezone.utc).isoformat(), failures=[])
    study.write_json(ROOT / "final.json", result)
    report(result)
    plot(matrix, times)
    return result


def report(s):
    p = s["primary_late_shift"]
    out = ["# L6 energy check through imaginary time 2048", "",
        f"**{s['decision']}. Ground-state convergence is not established.**", "",
        "Eight matched populations; 108 spins, flat sampling, 5,000 walkers, two replicas, Gamma=0.05. Energies refer to one replica. Intervals are descriptive 95% Student-t intervals across populations.", "",
        "| Imaginary-time window | Mean energy | SE | 95% interval |",
        "| --- | ---: | ---: | --- |"]
    for w in s["windows"]:
        out.append(f"| {w['start_time']:.0f}–{w['end_time']:.0f} | {w['mean']:.6f} | {w['se']:.6f} | [{w['ci95'][0]:.6f}, {w['ci95'][1]:.6f}] |")
    out += ["", f"Primary paired change, time 1024–1536 to 1536–2048: **{p['mean']:.6f} ± {p['se']:.6f} SE**, 95% interval [{p['ci95'][0]:.6f}, {p['ci95'][1]:.6f}].",
        f"Per spin: {p['per_spin_mean']:.8f}, 95% interval [{p['per_spin_ci95'][0]:.8f}, {p['per_spin_ci95'][1]:.8f}].", "",
        "| Population | Energy 1024–1536 | Energy 1536–2048 | Change |",
        "| --- | ---: | ---: | ---: |"]
    for child, (a, b) in enumerate(zip(s["population_early_means"], s["population_late_means"])):
        out.append(f"| {child} | {a:.6f} | {b:.6f} | {b-a:.6f} |")
    for label, row in [("Final time-256 window change", s["final_short_shift"]),
                       ("Slope over time 1024–2048", s["late_slope"])]:
        out += ["", f"{label}: {row['mean']:.8f} ± {row['se']:.8f} SE; 95% interval [{row['ci95'][0]:.8f}, {row['ci95'][1]:.8f}]."]
    out += ["", "All eight energy and time prefixes through time 512 reproduce the preceding campaign exactly. See campaign.json for frozen source and inherited-harness hashes.", "",
        "This extension was selected after resolved drift in the previous campaign. Intervals do not control error across sequential studies. No equivalence tolerance is specified; an interval containing zero does not certify a plateau. Population disagreement, trapping and finite-walker bias remain relevant. Susceptibility is not qualified. Production remains on hold.", "",
        "Artifacts: energy_trace.png, energy_trace.svg, final.json, PROTOCOL.md, runs/ and partial/.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(out))


def plot(matrix, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    width = 80
    x = times.reshape(-1, width).mean(axis=1)
    y = matrix.reshape(8, -1, width).mean(axis=2)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, scale in zip(axes, (1, 108)):
        for child, row in enumerate(y):
            ax.plot(x, row/scale, alpha=0.6, linewidth=0.8, label=f"Population {child}")
        ax.plot(x, y.mean(axis=0)/scale, color="black", linewidth=2, label="Mean")
        ax.axvline(1024, color="gray", linestyle="--")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Total energy (one replica)")
    axes[1].set_ylabel("Energy per spin")
    axes[1].set_xlabel("Cumulative imaginary time")
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle("L6/sample0, flat PQMC, Gamma=0.05, 5000 walkers\n80-block smoothing; unsmoothed traces retained")
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(ROOT / f"energy_trace.{suffix}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    campaign = json.loads((ROOT / "campaign.json").read_text())
    for name, digest in campaign["inherited_harness_sha256"].items():
        assert hashlib.sha256((PREVIOUS / name).read_bytes()).hexdigest() == digest, name
    for name, key in [("extend_energy.py", "harness_sha256"), ("PROTOCOL.md", "protocol_sha256")]:
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == campaign[key], name
    study.analyze = analyze
    study.main()
