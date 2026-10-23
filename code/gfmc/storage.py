"""On-disk layout for TFIM GFMC runs.

One run is one file:

    data/tfim/gfmc/{lattice}/{boundary}/{size}/J{J}/gamma{gamma}/h{h}/seed{seed}.npz

``lattice`` is ``chain`` or ``square``. ``boundary`` is ``obc`` or ``pbc``.
``size`` is ``n8`` or ``nx3_ny2``. Dots in parameter values are written as ``p``
and a leading minus sign as ``m``, so ``γ = 0.8`` is the directory ``gamma0p8``.
"""

from pathlib import Path

import numpy as np


def _token(value):
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def run_path(root, lattice, boundary, size, J, gamma, h, seed):
    """Path of one run. Parent directories are not created."""
    if boundary not in ("obc", "pbc"):
        raise ValueError("boundary must be obc or pbc")
    relative = Path("tfim") / "gfmc" / lattice / boundary / size
    relative = relative / f"J{_token(J)}" / f"gamma{_token(gamma)}" / f"h{_token(h)}"
    return Path(root) / relative / f"seed{int(seed)}.npz"


def save_run(root, record):
    """Write one run. ``record`` holds the scalars listed below."""
    required = (
        "lattice",
        "boundary",
        "size",
        "n_sites",
        "J",
        "gamma",
        "h",
        "seed",
        "n_walkers",
        "tau_block",
        "n_equilibrate",
        "n_measure",
        "energy",
        "energy_error",
        "energy_growth",
        "energy_growth_error",
        "exact_energy",
        "guide",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("run record is missing " + ", ".join(missing))
    path = run_path(
        root,
        record["lattice"],
        record["boundary"],
        record["size"],
        record["J"],
        record["gamma"],
        record["h"],
        record["seed"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lattice": np.array(record["lattice"]),
        "boundary": np.array(record["boundary"]),
        "size": np.array(record["size"]),
        "guide": np.array(record["guide"]),
        "n_sites": np.int64(record["n_sites"]),
        "J": np.float64(record["J"]),
        "gamma": np.float64(record["gamma"]),
        "h": np.float64(record["h"]),
        "seed": np.int64(record["seed"]),
        "n_walkers": np.int64(record["n_walkers"]),
        "tau_block": np.float64(record["tau_block"]),
        "n_equilibrate": np.int64(record["n_equilibrate"]),
        "n_measure": np.int64(record["n_measure"]),
        "energy": np.float64(record["energy"]),
        "energy_error": np.float64(record["energy_error"]),
        "energy_growth": np.float64(record["energy_growth"]),
        "energy_growth_error": np.float64(record["energy_growth_error"]),
        "exact_energy": np.float64(record["exact_energy"]),
    }
    np.savez_compressed(path, **payload)
    return path


def load_run(path):
    """Read one file written by :func:`save_run`."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].item() for key in data.files}


def iter_runs(root):
    """Saved TFIM GFMC runs, in path order."""
    directory = Path(root) / "tfim" / "gfmc"
    if not directory.exists():
        return []
    return [load_run(path) for path in sorted(directory.rglob("seed*.npz"))]
