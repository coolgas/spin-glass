"""One-sample L=6 trial of the forward-walked overlap.

The Friday figure is L=8. This run is the check that the projector, including
the 1/σ projection time, produces an overlap that falls as Γ/σ grows.

``--gate`` compares L=2 sample 0 with exact diagonalization.
``--run`` projects one (σ, Γ/σ) pair at L=6.
``--farm`` runs the nine trial pairs on a process pool.
``--scan`` runs a dense Γ/σ grid at σ = 1 and 2 for four samples.
``--table`` prints the pairs that have finished.
"""

import argparse
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from kagome_qea.exact import ground_state
from kagome_qea.pqmc import _CHECKPOINT_COUNT, project
from kagome_qea.sample import make_sample

TRIAL_SIZE = 6
TRIAL_SAMPLE = 0
TRIAL_SIGMAS = (0.05, 1.0, 2.0)
TRIAL_RATIOS = (1.0, 2.0, 3.0)
SCAN_SIGMAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
SCAN_RATIOS = (
    0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8,
    1.0, 1.25, 1.5, 2.0, 2.5, 3.0,
)
SCAN_SAMPLES = (0, 1, 2, 3)
PROJECTOR_SEED = 0
N_WALKERS = 5000
N_EQUILIBRATE = 40
N_MEASURE = 40
N_FORWARD_CYCLES = 8
# One population remeasures one survivor. The reported error is the scatter
# of independent populations. Thirty-two is the count used for the L=6 rerun.
N_POPULATIONS = 32
TAU_BLOCK_AT_SIGMA_1 = 0.2
# Time 4 at Gamma/sigma = 0.2 is a finite-time bias of several reported
# standard errors on L=2. Time 8 is the projection that matched the exact
# susceptibility there. A cycle the ancestor floor cuts short is stored
# and is not marked complete.
TAU_FORWARD_AT_SIGMA_1 = 8.0
GATE_SIGMAS = (1.0, 0.05)


def repo_root():
    return Path(__file__).resolve().parents[2]


def token(value):
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def output_path(sigma, gamma, size=TRIAL_SIZE, sample_id=TRIAL_SAMPLE):
    directory = (
        repo_root()
        / "data"
        / "kagome"
        / "gaussian_nn"
        / "pbc"
        / f"L{size}"
        / f"pqmc_k{N_POPULATIONS}"
        / "h0"
        / f"sigma{token(sigma)}"
        / f"gamma{token(gamma)}"
    )
    return directory / f"sample{int(sample_id)}_seed{PROJECTOR_SEED}.npz"


def projector_times(sigma):
    """Block and forward times. Both scale as 1/σ, so the block count does not."""
    scale = 1.0 / float(sigma)
    return TAU_BLOCK_AT_SIGMA_1 * scale, TAU_FORWARD_AT_SIGMA_1 * scale


def project_pair(size, sigma, ratio, sample_id=TRIAL_SAMPLE):
    gamma = float(sigma) * float(ratio)
    tau_block, tau_forward = projector_times(sigma)
    sample = make_sample(size, True, sample_id)
    operator = sample.hamiltonian(0.0).with_classical_scale(sigma)
    started = time.perf_counter()
    result = project(
        operator,
        gamma,
        n_replicas=2,
        n_walkers=N_WALKERS,
        tau_block=tau_block,
        n_equilibrate=N_EQUILIBRATE,
        n_measure=N_MEASURE,
        tau_forward=tau_forward,
        n_forward_cycles=N_FORWARD_CYCLES,
        n_populations=N_POPULATIONS,
        seed=PROJECTOR_SEED,
    )
    seconds = time.perf_counter() - started
    return sample, result, gamma, tau_block, tau_forward, seconds


def git_revision():
    """Current commit, recorded so a saved run names the code that wrote it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_fingerprint():
    """Hash of the solver files that decide a cached run.

    The commit name alone stays valid while those files are edited, so the
    cache also requires this hash.
    """
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for name in ("pqmc.py", "trial_overlap.py"):
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def _stored_text(value):
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array)


def _identity_from_path(path):
    """Job identity encoded by the canonical ``pqmc_k*`` path."""
    path = Path(path)
    stem = path.name[: -len(".npz")] if path.name.endswith(".npz") else ""
    if not stem.startswith("sample") or "_seed" not in stem:
        return None
    sample_text, seed_text = stem.split("_seed", 1)
    try:
        sample_id = int(sample_text.removeprefix("sample"))
        seed = int(seed_text)
        size = int(path.parents[4].name.removeprefix("L"))
    except (ValueError, IndexError):
        return None
    if seed != PROJECTOR_SEED or sample_id < 0 or size < 1:
        return None
    names = [parent.name for parent in path.parents]
    if names[:7] != [
        f"gamma{path.parent.name.removeprefix('gamma')}",
        f"sigma{path.parents[1].name.removeprefix('sigma')}",
        "h0",
        f"pqmc_k{N_POPULATIONS}",
        f"L{size}",
        "pbc",
        "gaussian_nn",
    ]:
        return None
    if not path.parent.name.startswith("gamma") or not path.parents[1].name.startswith("sigma"):
        return None
    return {
        "L": size,
        "sample_id": sample_id,
        "sigma_token": path.parents[1].name.removeprefix("sigma"),
        "gamma_token": path.parent.name.removeprefix("gamma"),
        "sigma": None,
        "gamma": None,
    }


def _close(stored, expected):
    stored = float(stored)
    expected = float(expected)
    return abs(stored - expected) <= 1e-8 * max(1.0, abs(expected))


def _integer_scalar(value):
    """Check a scalar count before any lossy integer coercion."""
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind == "b":
        return False
    number = float(array)
    return math.isfinite(number) and number.is_integer()


def _lineage_diagnostics_valid(unique, effective, weight, fraction, n_walkers):
    """Feasible post-resample ancestry and pre-resample weight diversity."""
    values = [np.asarray(x, dtype=np.float64) for x in (unique, effective, weight, fraction)]
    if not all(np.isfinite(x).all() for x in values):
        return False
    unique, effective, weight, fraction = values
    tolerance = 1e-8
    if np.any(unique != np.floor(unique)) or np.any(unique < 1) or np.any(unique > n_walkers):
        return False
    if np.any(effective < 1 - tolerance) or np.any(effective > unique + tolerance):
        return False
    if np.any(weight < 1 - tolerance) or np.any(weight > n_walkers + tolerance):
        return False
    if np.any(fraction < 1.0 / n_walkers - tolerance) or np.any(fraction > 1 + tolerance):
        return False
    # The largest lineage is an integer number of equal-weight children.
    largest = fraction * n_walkers
    if not np.allclose(largest, np.rint(largest), rtol=0.0, atol=tolerance):
        return False
    if np.any(fraction < 1.0 / unique - tolerance):
        return False
    if np.any(largest > n_walkers - unique + 1 + tolerance):
        return False
    # For lineage proportions p: max(p)^2 <= sum(p^2) <= max(p).
    if np.any(1.0 / effective > fraction + tolerance):
        return False
    if np.any(1.0 / effective < fraction * fraction - tolerance):
        return False
    return True


def result_ready(path, expected=None):
    """True when the archive is this job and its statistics agree with the series.

    ``expected`` is the queue request. Without it, the identity is read from
    a canonical ``pqmc_k*`` path. The stored revision must be the checked-out
    commit. A checkout that cannot name that commit is rejected. The solver-file
    hash must match the files on disk, and the site count must match the
    lattice. Overlap moments stay in ``[0, 1]``. Achieved times are nonnegative
    and do not exceed the requested time. Completion flags are recomputed from
    the achieved times. The production floors are Kish size 8 and two unique
    ancestors. Per-cycle stop codes, rejected steps, and checkpoints have to
    agree with those times. Equilibration and measurement energies are stored
    on their own traces. A shortened walk may be kept, and it is not marked
    time-converged.
    """
    if not path.is_file() or path.stat().st_size < 64:
        return False
    try:
        with np.load(path) as data:
            required = {
                "q2",
                "q4",
                "q2_series",
                "q2_error",
                "chi_sg",
                "chi_sg_error",
                "J",
                "min_unique_ancestors",
                "min_effective_ancestors",
                "n_populations",
                "n_walkers",
                "n_equilibrate",
                "n_measure",
                "n_forward_cycles",
                "n_sites",
                "tau_block",
                "tau_forward",
                "tau_forward_requested",
                "forward_times",
                "projection_complete",
                "n_forward_truncated",
                "sigma",
                "gamma",
                "L",
                "sample_id",
                "seed_entropy",
                "periodic",
                "h",
                "git_revision",
                "code_fingerprint",
                "energy",
                "energy_error",
                "energy_growth",
                "energy_growth_error",
                "floor_unique",
                "floor_effective",
                "cycle_q2",
                "cycle_unique",
                "cycle_effective",
                "cycle_max_fraction",
                "cycle_weight_ess",
                "cycle_stop",
                "cycle_rejected_dt",
                "cycle_rejected_unique",
                "cycle_rejected_effective",
                "cycle_rejected_weight_ess",
                "cycle_rejected_max_fraction",
                "cycle_checkpoints",
                "energy_forward_trace",
                "energy_equilibration_trace",
                "energy_equilibration_time",
                "energy_measure_trace",
                "energy_measure_time",
            }
            if not required <= set(data.files):
                return False
            for name in ("n_populations", "n_walkers", "n_equilibrate", "n_measure",
                         "n_forward_cycles", "n_sites", "L", "sample_id", "periodic"):
                if not _integer_scalar(data[name]):
                    return False
            if int(data["n_populations"]) != N_POPULATIONS:
                return False
            if int(data["n_walkers"]) != N_WALKERS:
                return False
            if int(data["n_equilibrate"]) != N_EQUILIBRATE:
                return False
            if int(data["n_measure"]) != N_MEASURE:
                return False
            if int(data["n_forward_cycles"]) != N_FORWARD_CYCLES:
                return False
            revision = git_revision()
            if revision == "unknown" or _stored_text(data["git_revision"]) != revision:
                return False
            if _stored_text(data["code_fingerprint"]) != code_fingerprint():
                return False
            if int(data["periodic"]) != 1 or not _close(data["h"], 0.0):
                return False
            identity = expected if expected is not None else _identity_from_path(path)
            if identity is None:
                return False
            if int(data["L"]) != int(identity["L"]) or int(data["sample_id"]) != int(identity["sample_id"]):
                return False
            if identity.get("sigma") is None:
                if token(float(data["sigma"])) != identity["sigma_token"]:
                    return False
                if token(float(data["gamma"])) != identity["gamma_token"]:
                    return False
            elif not _close(data["sigma"], identity["sigma"]) or not _close(data["gamma"], identity["gamma"]):
                return False
            sample = make_sample(int(identity["L"]), True, int(identity["sample_id"]))
            if not np.array_equal(
                np.asarray(data["seed_entropy"], dtype=np.int64),
                np.asarray(sample.seed_entropy, dtype=np.int64),
            ):
                return False
            if not np.allclose(
                np.asarray(data["J"], dtype=np.float64),
                np.asarray(sample.J, dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            ):
                return False
            if int(data["n_sites"]) != int(sample.n_sites):
                return False
            series = np.asarray(data["q2_series"], dtype=np.float64)
            times = np.asarray(data["forward_times"], dtype=np.float64)
            statistics = [
                data["q2"],
                data["q4"],
                data["q2_error"],
                data["chi_sg"],
                data["chi_sg_error"],
                data["min_effective_ancestors"],
                data["energy"],
                data["energy_error"],
                data["energy_growth"],
                data["energy_growth_error"],
            ]
            if series.shape != (N_POPULATIONS,) or not np.isfinite(series).all():
                return False
            if times.shape != (N_POPULATIONS, N_FORWARD_CYCLES) or not np.isfinite(times).all():
                return False
            if not all(np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in statistics):
                return False
            if float(data["q2_error"]) < 0.0 or float(data["chi_sg_error"]) < 0.0:
                return False
            q2 = float(data["q2"])
            q4 = float(data["q4"])
            moment_limit = 1.0 + 1e-8
            if q2 < 0.0 or q4 < 0.0 or q2 > moment_limit or q4 > moment_limit:
                return False
            if np.any(series < 0.0) or np.any(series > moment_limit):
                return False
            scatter = float(series.std(ddof=1) / np.sqrt(series.size))
            if not _close(series.mean(), data["q2"]) or not _close(data["q2_error"], scatter):
                return False
            if not _close(float(data["chi_sg"]), int(data["n_sites"]) * float(data["q2"])):
                return False
            if not _close(float(data["chi_sg_error"]), int(data["n_sites"]) * scatter):
                return False
            if not _close(times.min(), data["tau_forward"]):
                return False
            requested = float(data["tau_forward_requested"])
            if not math.isfinite(requested) or requested <= 0.0 or float(data["tau_forward"]) < 0.0 or np.any(times < 0.0):
                return False
            if np.any(times > requested + 1e-8):
                return False
            truncated = int(np.count_nonzero(times + 1e-8 < requested))
            complete = int(times.size > 0 and truncated == 0)
            if not _integer_scalar(data["n_forward_truncated"]) or not _integer_scalar(data["projection_complete"]):
                return False
            if int(data["n_forward_truncated"]) != truncated or int(data["projection_complete"]) != complete:
                return False
            if not _integer_scalar(data["floor_unique"]) or int(data["floor_unique"]) != 2:
                return False
            if not _close(data["floor_effective"], 8.0):
                return False
            cycle_q2 = np.asarray(data["cycle_q2"], dtype=np.float64)
            cycle_unique = np.asarray(data["cycle_unique"], dtype=np.float64)
            cycle_effective = np.asarray(data["cycle_effective"], dtype=np.float64)
            cycle_fraction = np.asarray(data["cycle_max_fraction"], dtype=np.float64)
            cycle_weight = np.asarray(data["cycle_weight_ess"], dtype=np.float64)
            cycle_stop = np.asarray(data["cycle_stop"], dtype=np.float64)
            forward_energy = np.asarray(data["energy_forward_trace"], dtype=np.float64)
            cycle_arrays = (
                cycle_q2,
                cycle_unique,
                cycle_effective,
                cycle_fraction,
                cycle_weight,
                forward_energy,
            )
            if cycle_stop.shape != times.shape or any(array.shape != times.shape for array in cycle_arrays):
                return False
            if not np.isfinite(cycle_q2).all() or np.any(cycle_q2 < 0.0) or np.any(cycle_q2 > moment_limit):
                return False
            if not _close(float(cycle_q2.mean()), q2):
                return False
            if not np.allclose(cycle_q2.mean(axis=1), series, rtol=1e-8, atol=1e-8):
                return False
            if not _integer_scalar(data["min_unique_ancestors"]):
                return False
            if not _lineage_diagnostics_valid(
                cycle_unique, cycle_effective, cycle_weight, cycle_fraction, N_WALKERS
            ):
                return False
            if float(np.min(cycle_unique)) != float(data["min_unique_ancestors"]):
                return False
            if not np.isfinite(cycle_effective).all() or not _close(
                float(np.min(cycle_effective)), data["min_effective_ancestors"]
            ):
                return False
            if np.any(cycle_unique < 2) or np.any(cycle_effective + 1e-8 < 8.0):
                return False
            if not np.isfinite(cycle_stop).all() or np.any(~np.isin(cycle_stop, (0, 1, 2, 3))):
                return False
            reached = times + 1e-8 >= requested
            if not np.array_equal(cycle_stop == 0, reached):
                return False
            if np.any(~np.isin(cycle_stop[~reached], (1, 2, 3))):
                return False
            rejected = [
                np.asarray(data[name], dtype=np.float64)
                for name in (
                    "cycle_rejected_dt",
                    "cycle_rejected_unique",
                    "cycle_rejected_effective",
                    "cycle_rejected_weight_ess",
                    "cycle_rejected_max_fraction",
                )
            ]
            if any(array.shape != times.shape for array in rejected):
                return False
            if reached.any() and not all(np.isnan(array[reached]).all() for array in rejected):
                return False
            if (~reached).any():
                if not all(np.isfinite(array[~reached]).all() for array in rejected):
                    return False
                rejected_dt = rejected[0][~reached]
                rejected_unique = rejected[1][~reached]
                rejected_effective = rejected[2][~reached]
                rejected_weight = rejected[3][~reached]
                rejected_fraction = rejected[4][~reached]
                stop_short = cycle_stop[~reached]
                if np.any(rejected_dt <= 0.0) or np.any(rejected_dt > requested - times[~reached] + 1e-8):
                    return False
                if not _lineage_diagnostics_valid(
                    rejected_unique, rejected_effective, rejected_weight, rejected_fraction, N_WALKERS
                ):
                    return False
                effective_only = (rejected_effective < 8.0) & (rejected_unique >= 2)
                unique_only = (rejected_unique < 2) & (rejected_effective >= 8.0)
                both = (rejected_effective < 8.0) & (rejected_unique < 2)
                if np.any((stop_short == 1) & ~effective_only):
                    return False
                if np.any((stop_short == 2) & ~unique_only):
                    return False
                if np.any((stop_short == 3) & ~both):
                    return False
            if not np.isfinite(forward_energy[times > 0.0]).all():
                return False
            if not np.isnan(forward_energy[times == 0.0]).all():
                return False
            checkpoints = np.asarray(data["cycle_checkpoints"], dtype=np.float64)
            if checkpoints.shape != times.shape + (_CHECKPOINT_COUNT, 5):
                return False
            endpoints = np.stack((times, cycle_unique, cycle_effective, cycle_weight, cycle_fraction), axis=-1)
            for endpoint, rows in zip(endpoints.reshape(-1, 5), checkpoints.reshape(-1, _CHECKPOINT_COUNT, 5)):
                achieved = endpoint[0]
                finite_rows = np.isfinite(rows).all(axis=1)
                blank_rows = np.isnan(rows).all(axis=1)
                if not np.all(finite_rows | blank_rows) or not finite_rows.any():
                    return False
                count = int(finite_rows.sum())
                if not finite_rows[:count].all() or finite_rows[count:].any():
                    return False
                valid = rows[:count]
                elapsed = rows[finite_rows, 0]
                if np.any(elapsed < -1e-12) or np.any(elapsed > float(achieved) + 1e-8):
                    return False
                if not _close(elapsed[0], 0.0) or np.any(np.diff(elapsed) <= 0.0):
                    return False
                if not _lineage_diagnostics_valid(valid[:, 1], valid[:, 2], valid[:, 3], valid[:, 4], N_WALKERS):
                    return False
                if np.any(valid[:, 1] < 2) or np.any(valid[:, 2] + 1e-8 < 8.0):
                    return False
                if not np.allclose(valid[-1], endpoint, rtol=1e-8, atol=1e-8):
                    return False
            equilibration = np.asarray(data["energy_equilibration_trace"], dtype=np.float64)
            measurement = np.asarray(data["energy_measure_trace"], dtype=np.float64)
            equilibration_time = np.asarray(data["energy_equilibration_time"], dtype=np.float64)
            measurement_time = np.asarray(data["energy_measure_time"], dtype=np.float64)
            if equilibration.shape != (N_POPULATIONS, N_EQUILIBRATE) or not np.isfinite(equilibration).all():
                return False
            if measurement.shape != (N_POPULATIONS, N_MEASURE) or not np.isfinite(measurement).all():
                return False
            block_time = float(data["tau_block"])
            expected_equilibration = block_time * np.arange(1, N_EQUILIBRATE + 1)
            expected_measurement = block_time * np.arange(
                N_EQUILIBRATE + 1, N_EQUILIBRATE + N_MEASURE + 1
            )
            if equilibration_time.shape != expected_equilibration.shape:
                return False
            if measurement_time.shape != expected_measurement.shape:
                return False
            if not np.allclose(equilibration_time, expected_equilibration, rtol=1e-6, atol=1e-8):
                return False
            if not np.allclose(measurement_time, expected_measurement, rtol=1e-6, atol=1e-8):
                return False
            block, forward = projector_times(float(data["sigma"]))
            return _close(data["tau_block"], block) and _close(data["tau_forward_requested"], forward)
    except Exception:
        return False


def save_run(path, sample, result, sigma, gamma, tau_block, tau_forward, seconds):
    if not _close(tau_block, result["tau_block"]) or not _close(
        tau_forward, result["tau_forward_requested"]
    ):
        raise ValueError("saved projection settings do not match the run")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.npz")
    np.savez_compressed(
        temporary,
        L=np.int64(sample.L),
        sample_id=np.int64(sample.sample_id),
        sigma=np.float64(sigma),
        gamma=np.float64(gamma),
        tau_block=np.float64(result["tau_block"]),
        tau_forward=np.float64(result["tau_forward"]),
        tau_forward_requested=np.float64(result["tau_forward_requested"]),
        forward_times=np.asarray(result["forward_times"], dtype=np.float64),
        n_forward_truncated=np.int64(result["n_forward_truncated"]),
        projection_complete=np.int64(bool(result["projection_complete"])),
        seconds=np.float64(seconds),
        seed_entropy=np.asarray(sample.seed_entropy, dtype=np.int64),
        periodic=np.int64(int(bool(sample.periodic))),
        h=np.float64(0.0),
        git_revision=np.array(git_revision()),
        code_fingerprint=np.array(code_fingerprint()),
        J=np.asarray(sample.J, dtype=np.float64),
        q2=np.float64(result["q2"]),
        q2_error=np.float64(result["q2_error"]),
        q2_series=np.asarray(result["q2_series"], dtype=np.float64),
        q4=np.float64(result["q4"]),
        binder=np.float64(result["binder"]),
        binder_error=np.float64(result["binder_error"]),
        chi_sg=np.float64(result["chi_sg"]),
        chi_sg_error=np.float64(result["chi_sg_error"]),
        n_populations=np.int64(result["n_populations"]),
        n_walkers=np.int64(result["n_walkers"]),
        n_equilibrate=np.int64(result["n_equilibrate"]),
        n_measure=np.int64(result["n_measure"]),
        n_forward_cycles=np.int64(result["n_forward_cycles"]),
        n_sites=np.int64(result["n_sites"]),
        energy=np.float64(result["energy"]),
        energy_error=np.float64(result["energy_error"]),
        energy_growth=np.float64(result["energy_growth"]),
        energy_growth_error=np.float64(result["energy_growth_error"]),
        min_unique_ancestors=np.int64(result["min_unique_ancestors"]),
        min_effective_ancestors=np.float64(result["min_effective_ancestors"]),
        floor_unique=np.int64(result["floor_unique"]),
        floor_effective=np.float64(result["floor_effective"]),
        cycle_q2=np.asarray(result["cycle_q2"], dtype=np.float64),
        cycle_unique=np.asarray(result["cycle_unique"], dtype=np.int64),
        cycle_effective=np.asarray(result["cycle_effective"], dtype=np.float64),
        cycle_max_fraction=np.asarray(result["cycle_max_fraction"], dtype=np.float64),
        cycle_weight_ess=np.asarray(result["cycle_weight_ess"], dtype=np.float64),
        cycle_stop=np.asarray(result["cycle_stop"], dtype=np.int64),
        cycle_rejected_dt=np.asarray(result["cycle_rejected_dt"], dtype=np.float64),
        cycle_rejected_unique=np.asarray(result["cycle_rejected_unique"], dtype=np.float64),
        cycle_rejected_effective=np.asarray(result["cycle_rejected_effective"], dtype=np.float64),
        cycle_rejected_weight_ess=np.asarray(result["cycle_rejected_weight_ess"], dtype=np.float64),
        cycle_rejected_max_fraction=np.asarray(result["cycle_rejected_max_fraction"], dtype=np.float64),
        cycle_checkpoints=np.asarray(result["cycle_checkpoints"], dtype=np.float64),
        energy_forward_trace=np.asarray(result["energy_forward_trace"], dtype=np.float64),
        energy_equilibration_trace=np.asarray(result["energy_equilibration_trace"], dtype=np.float64),
        energy_equilibration_time=np.asarray(result["energy_equilibration_time"], dtype=np.float64),
        energy_measure_trace=np.asarray(result["energy_measure_trace"], dtype=np.float64),
        energy_measure_time=np.asarray(result["energy_measure_time"], dtype=np.float64),
    )
    os.replace(temporary, path)
    if not result_ready(path):
        path.unlink(missing_ok=True)
        raise RuntimeError(f"saved file is unreadable: {path}")
    backup = repo_root() / "saved" / path.relative_to(repo_root())
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)


def run_one(sigma, ratio, sample_id=TRIAL_SAMPLE):
    path = output_path(sigma, sigma * ratio, sample_id=sample_id)
    if result_ready(path):
        print(f"skip sigma {sigma:g} ratio {ratio:g} sample {sample_id} {path}", flush=True)
        return str(path)
    sample, result, gamma, tau_block, tau_forward, seconds = project_pair(
        TRIAL_SIZE, sigma, ratio, sample_id
    )
    save_run(path, sample, result, sigma, gamma, tau_block, tau_forward, seconds)
    print(
        f"L=6 sigma {sigma:g} ratio {ratio:g} gamma {gamma:.4f} sample {sample_id}  "
        f"q2 {result['q2']:.4f} ± {result['q2_error']:.4f}  "
        f"B {result['binder']:.3f} ± {result['binder_error']:.3f}  "
        f"ancestors {result['min_unique_ancestors']}  "
        f"tau {result['tau_forward']:.3f}/{result['tau_forward_requested']:.3f}  "
        f"complete {int(bool(result['projection_complete']))}  "
        f"energy {result['energy']:.3f} growth {result['energy_growth']:.3f}  "
        f"{seconds:.0f}s",
        flush=True,
    )
    return str(path)


def gate():
    """Exact L=2 overlap at σ=1 and at the small-σ time scaling."""
    failed = False
    for sigma in GATE_SIGMAS:
        gamma = float(sigma)  # ratio 1
        sample, result, gamma, tau_block, tau_forward, seconds = project_pair(2, sigma, 1.0)
        exact = ground_state(sample.hamiltonian(0.0).with_classical_scale(sigma), gamma)
        difference = abs(result["q2"] - exact["q2"])
        limit = max(0.06, 3.0 * result["q2_error"])
        print(
            f"L=2 sigma {sigma:g} gamma {gamma:g}  {seconds:.1f}s  "
            f"tau_block {tau_block:g} tau_forward {tau_forward:g}\n"
            f"exact q2 {exact['q2']:.4f}  projector {result['q2']:.4f} ± {result['q2_error']:.4f}  "
            f"ancestors {result['min_unique_ancestors']}",
            flush=True,
        )
        if result["q2_error"] > 0.06 or difference > limit:
            print(f"FAIL difference {difference:.4f} limit {limit:.4f}", flush=True)
            failed = True
        else:
            print(f"pass difference {difference:.4f} limit {limit:.4f}", flush=True)
    if failed:
        raise SystemExit(1)


def _pool(jobs, workers):
    workers = max(1, min(int(workers), len(jobs)))
    print(f"farm {len(jobs)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, sigma, ratio, sample_id) for sigma, ratio, sample_id in jobs]
        for future in as_completed(futures):
            future.result()


def farm(workers):
    jobs = [
        (sigma, ratio, TRIAL_SAMPLE)
        for sigma in TRIAL_SIGMAS
        for ratio in TRIAL_RATIOS
    ]
    _pool(jobs, workers)


def scan_jobs(sigma=None):
    sigmas = SCAN_SIGMAS if sigma is None else (float(sigma),)
    return [
        (width, ratio, sample_id)
        for width in sigmas
        for ratio in SCAN_RATIOS
        for sample_id in SCAN_SAMPLES
    ]


def scan(workers):
    """Dense Γ/σ grid, six variances, four disorder samples."""
    _pool(scan_jobs(), workers)


def queue(workers, sigma=None):
    """Run every scan job, skip files already saved, then draw the figure.

    The manifest is ``l6_k32_queue.txt`` when 32 populations are used.
    ``l6_k32_queue_status.txt`` is one line rewritten as each job finishes.
    Starting this again continues the same list. An archive is skipped only
    when its settings and finite statistics match this run. Passed ``sigma``
    keeps one width.
    """
    jobs = scan_jobs(sigma)
    pending = []
    manifest = []
    for sigma, ratio, sample_id in jobs:
        path = output_path(sigma, sigma * ratio, sample_id=sample_id)
        state = "done" if result_ready(path) else "queued"
        manifest.append(f"{state}\tsigma={sigma:g}\tratio={ratio:g}\tsample={sample_id}")
        if state == "queued":
            pending.append((sigma, ratio, sample_id))
    manifest_path = repo_root() / f"l6_k{N_POPULATIONS}_queue.txt"
    status_path = repo_root() / f"l6_k{N_POPULATIONS}_queue_status.txt"
    manifest_path.write_text("\n".join(manifest) + "\n")
    total = len(jobs)
    done = total - len(pending)
    print(
        f"queue {total} jobs, {done} already on disk, {len(pending)} pending",
        flush=True,
    )
    status_path.write_text(f"pending={len(pending)} done={done} total={total} failed=0\n")
    failed = 0
    if pending:
        workers = max(1, min(int(workers), len(pending)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_one, sigma, ratio, sample_id): (sigma, ratio, sample_id)
                for sigma, ratio, sample_id in pending
            }
            for future in as_completed(futures):
                sigma, ratio, sample_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed += 1
                    print(
                        f"FAIL sigma {sigma:g} ratio {ratio:g} sample {sample_id} {exc}",
                        flush=True,
                    )
                finished = sum(
                    1
                    for sigma_i, ratio_i, sample_i in jobs
                    if result_ready(output_path(sigma_i, sigma_i * ratio_i, sample_id=sample_i))
                )
                status_path.write_text(
                    f"pending={total - finished} done={finished} total={total} failed={failed}\n"
                )
    from kagome_qea.plot_l6_scan import main as plot_main

    plot_main()
    if failed:
        raise SystemExit(f"{failed} queued jobs failed")


def table():
    print(f"{'sigma':>6} {'ratio':>6} {'gamma':>8} {'q2':>8} {'err':>8} {'anc':>5} {'sec':>7}")
    missing = 0
    for sigma in TRIAL_SIGMAS:
        for ratio in TRIAL_RATIOS:
            path = output_path(sigma, sigma * ratio)
            if not path.exists():
                missing += 1
                print(f"{sigma:6g} {ratio:6g} {'missing':>8}")
                continue
            with np.load(path) as data:
                print(
                    f"{float(data['sigma']):6g} {ratio:6g} {float(data['gamma']):8.4f} "
                    f"{float(data['q2']):8.4f} {float(data['q2_error']):8.4f} "
                    f"{int(data['min_unique_ancestors']):5d} {float(data['seconds']):7.0f}"
                )
    if missing:
        raise SystemExit(f"{missing} trial files still missing")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--farm", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--sample", type=int, default=TRIAL_SAMPLE)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.gate:
        gate()
    elif args.run:
        if args.sigma is None or args.ratio is None:
            parser.error("--run needs --sigma and --ratio")
        run_one(args.sigma, args.ratio, args.sample)
    elif args.farm:
        farm(args.workers)
    elif args.scan:
        scan(args.workers)
    elif args.queue:
        queue(args.workers, args.sigma)
    elif args.table:
        table()
    else:
        parser.error("choose --gate, --run, --farm, or --table")


if __name__ == "__main__":
    main()
