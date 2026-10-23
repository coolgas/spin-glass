"""On-disk layout for TFIM GFMC runs.

One run is one file:

    data/tfim/gfmc/{lattice}/{boundary}/{size}/J{J}/gamma{gamma}/h{h}/seed{seed}_{id}.npz

``lattice`` is ``chain`` or ``square``. ``boundary`` is ``obc`` or ``pbc``.
``size`` is ``n8`` or ``nx3_ny2``. Dots in directory names are written as ``p``
and a leading minus sign as ``m``, so ``γ = 0.8`` is the directory ``gamma0p8``.
``id`` is 12 hex characters from the exact run settings, so two nearby values
of Γ do not share a file.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _token(value):
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text.replace("-", "m").replace(".", "p")


_SETTINGS = (
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
    "guide",
)


def _json_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("run identity must be finite")
        return number.hex()
    if isinstance(value, str):
        return value
    raise TypeError(f"run identity field has unsupported type {type(value).__name__}")


def _settings_token(record):
    identity = {key: _json_value(record[key]) for key in _SETTINGS}
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_path(root, record):
    """Path of one run. Parent directories are not created."""
    boundary = record["boundary"]
    if boundary not in ("obc", "pbc"):
        raise ValueError("boundary must be obc or pbc")
    relative = Path("tfim") / "gfmc" / record["lattice"] / boundary / record["size"]
    relative = relative / f"J{_token(record['J'])}" / f"gamma{_token(record['gamma'])}" / f"h{_token(record['h'])}"
    name = f"seed{int(record['seed'])}_{_settings_token(record)}.npz"
    return Path(root) / relative / name


def save_run(root, record, *, overwrite=False):
    """Write one run. ``record`` holds the scalars listed below.

    ``overwrite=False`` raises ``FileExistsError`` when that file is already
    present. The filename hash uses the exact settings, not the rounded
    directory names.
    """
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
    path = run_path(root, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"run file already exists: {path}")
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
    for key in ("energy_series", "energy_growth_series"):
        if key in record:
            payload[key] = np.asarray(record[key], dtype=np.float64)
    for key in ("energy_resolved", "energy_growth_resolved"):
        if key in record:
            payload[key] = np.bool_(record[key])
    np.savez_compressed(path, **payload)
    return path


def load_run(path):
    """Read one file written by :func:`save_run`.

    Scalar entries come back as Python values. Series stay as arrays.
    """
    with np.load(path, allow_pickle=False) as data:
        record = {}
        for key in data.files:
            value = data[key]
            if value.shape == ():
                record[key] = value.item()
            else:
                record[key] = np.array(value)
        return record


def iter_runs(root):
    """Saved TFIM GFMC runs, in path order."""
    directory = Path(root) / "tfim" / "gfmc"
    if not directory.exists():
        return []
    return [load_run(path) for path in sorted(directory.rglob("seed*.npz"))]
