"""Disorder average of the dense L=6 scan, written to a PNG and a table.

Reads every ``sample*_seed0.npz`` under the L=6 pqmc tree. One point on the
figure is the jackknife mean over the samples present at that (σ, Γ/σ).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kagome_qea.observables import (
    jackknife_disorder_binder,
    jackknife_mean_error,
    spin_glass_susceptibility,
)
from kagome_qea.trial_overlap import N_POPULATIONS, repo_root

N_SITES = 108
COLORS = {
    0.05: "#332288",
    0.1: "#88CCEE",
    0.2: "#44AA99",
    0.5: "#117733",
    1.0: "#DDCC77",
    2.0: "#CC6677",
}


def color_for(sigma):
    for key, color in COLORS.items():
        if abs(key - sigma) < 1e-8:
            return color
    return "C0"


def scan_root():
    return (
        repo_root()
        / "data"
        / "kagome"
        / "gaussian_nn"
        / "pbc"
        / "L6"
        / f"pqmc_k{N_POPULATIONS}"
        / "h0"
    )


def load_rows():
    rows = []
    for path in sorted(scan_root().glob("sigma*/gamma*/sample*_seed0.npz")):
        with np.load(path) as data:
            sigma = float(data["sigma"])
            gamma = float(data["gamma"])
            rows.append(
                {
                    "sigma": sigma,
                    "gamma": gamma,
                    "ratio": gamma / sigma,
                    "sample": int(data["sample_id"]),
                    "q2": float(data["q2"]),
                    "q4": float(data["q4"]),
                    "ancestors": int(data["min_unique_ancestors"]),
                }
            )
    return rows


def averages(rows):
    keys = sorted({(row["sigma"], round(row["ratio"], 6)) for row in rows})
    points = []
    for sigma, ratio in keys:
        group = [row for row in rows if row["sigma"] == sigma and abs(row["ratio"] - ratio) < 1e-6]
        q2 = np.array([row["q2"] for row in group], dtype=np.float64)
        q4 = np.array([row["q4"] for row in group], dtype=np.float64)
        q2_error = jackknife_mean_error(q2) if q2.size >= 2 else float("nan")
        if q2.size >= 2 and np.all(q2 > 0.0):
            binder, binder_error = jackknife_disorder_binder(q2, q4)
        else:
            binder, binder_error = float("nan"), float("nan")
        chi_sg = spin_glass_susceptibility(float(q2.mean()), N_SITES)
        chi_sg_error = (
            float(N_SITES) * q2_error if np.isfinite(q2_error) else float("nan")
        )
        points.append(
            {
                "sigma": sigma,
                "ratio": ratio,
                "gamma": float(np.mean([row["gamma"] for row in group])),
                "q2": float(q2.mean()),
                "q2_error": q2_error,
                "chi_sg": chi_sg,
                "chi_sg_error": chi_sg_error,
                "binder": binder,
                "binder_error": binder_error,
                "n": int(q2.size),
                "ancestors": int(min(row["ancestors"] for row in group)),
            }
        )
    return points


def write_table(points, path):
    lines = [
        f"{'sigma':>6} {'Gamma':>8} {'ratio':>7} {'q2':>8} {'err':>8} "
        f"{'chi':>8} {'cerr':>8} {'B':>8} {'Berr':>8} {'n':>3} {'anc':>4}"
    ]
    for row in points:
        lines.append(
            f"{row['sigma']:6g} {row['gamma']:8.4f} {row['ratio']:7.3f} "
            f"{row['q2']:8.4f} {row['q2_error']:8.4f} "
            f"{row['chi_sg']:8.3f} {row['chi_sg_error']:8.3f} "
            f"{row['binder']:8.4f} {row['binder_error']:8.4f} "
            f"{row['n']:3d} {row['ancestors']:4d}"
        )
    path.write_text("\n".join(lines) + "\n")


def _panel(axis, group, x_key, y_key, yerr_key, color, label):
    axis.errorbar(
        [row[x_key] for row in group],
        [row[y_key] for row in group],
        yerr=[row[yerr_key] for row in group],
        fmt="o-",
        capsize=3,
        color=color,
        label=label,
    )


def draw(points, path):
    figure, axes = plt.subplots(3, 2, figsize=(9.2, 10.4))
    sigmas = sorted({row["sigma"] for row in points})
    for sigma in sigmas:
        group = sorted(
            [row for row in points if row["sigma"] == sigma],
            key=lambda row: row["ratio"],
        )
        color = color_for(sigma)
        label = rf"$\sigma = {sigma:g}$"
        _panel(axes[0, 0], group, "gamma", "q2", "q2_error", color, label)
        _panel(axes[0, 1], group, "ratio", "q2", "q2_error", color, label)
        _panel(axes[1, 0], group, "gamma", "chi_sg", "chi_sg_error", color, label)
        _panel(axes[1, 1], group, "ratio", "chi_sg", "chi_sg_error", color, label)
        _panel(axes[2, 0], group, "gamma", "binder", "binder_error", color, label)
        _panel(axes[2, 1], group, "ratio", "binder", "binder_error", color, label)
    for axis in (axes[0, 0], axes[0, 1]):
        axis.axhline(1.0 / N_SITES, color="0.45", linestyle="--", label="$1/N$")
        axis.set_ylabel(r"$[\langle q^{2}\rangle]$")
        axis.legend()
    for axis in (axes[1, 0], axes[1, 1]):
        axis.axhline(1.0, color="0.45", linestyle="--", label="$1$")
        axis.set_ylabel(r"$[\chi_{\mathrm{SG}}]$")
        axis.legend()
    for axis in (axes[2, 0], axes[2, 1]):
        axis.set_ylabel("$[B]$")
        axis.legend()
    for axis in axes[:, 0]:
        axis.set_xlabel(r"$\Gamma$")
    for axis in axes[:, 1]:
        axis.set_xlabel(r"$\Gamma/\sigma$")
    figure.suptitle(r"$6\times 6$ Kagome torus, four coupling draws, $h = 0$")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit("no L=6 pqmc files")
    points = averages(rows)
    out_dir = repo_root()
    table = out_dir / "l6_scan_table.txt"
    figure = out_dir / "l6_scan.png"
    write_table(points, table)
    draw(points, figure)
    print(f"points {len(points)} files {len(rows)}", flush=True)
    print(f"table {table}", flush=True)
    print(f"figure {figure}", flush=True)


if __name__ == "__main__":
    main()
