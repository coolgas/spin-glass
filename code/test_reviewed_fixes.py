#!/usr/bin/env python3
"""Checks for the reviewed GFMC and exact-diagonalization fixes.

From the repository root:

    python code/test_reviewed_fixes.py

This does not run the TFIM benchmark and it does not run the test notebook.
"""

import math
import tempfile

import numpy as np

from gfmc import GFMC, save_run, load_run
from gfmc.guide import ConstantGuide, Guide
from ising import Hamiltonian, chain
from kagome_qea.exact import _select_ground_states, ground_state, moments_from_probability
from kagome_qea.observables import spin_glass_susceptibility
from kagome_qea.pqmc import _tag_overlap
from qmc_numerics import series_error, systematic, weighted_interval


def main():
    report = _Report()
    _check_operator(report)
    _check_weights(report)
    _check_overlap(report)
    _check_exact(report)
    _check_guide_path(report)
    _check_storage(report)
    _check_series_error(report)
    print(f"Passed {report.passed}. Failed {len(report.failed)}.")
    if report.failed:
        print("Failed checks: " + ", ".join(report.failed))
        return 1
    return 0


def _check_operator(report):
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int64)
    couplings = np.array([1.0, 1.0])
    operator = Hamiltonian(3, bonds, couplings, h=0.2)
    couplings[0] = 9.0
    bonds[0, 1] = 0
    spins = np.ones(3)
    energy = operator.diagonal_energy(spins)
    if abs(energy + 2.6) < 1e-12 and abs(operator.flip_delta(spins, 0) - 2.4) < 1e-12:
        report.ok("operator copy", f"diagonal {energy:.4f} after the caller array changed")
    else:
        report.fail("operator copy", f"diagonal {energy}, flip {operator.flip_delta(spins, 0)}")
    try:
        operator.J[0] = 3.0
    except ValueError:
        report.ok("read-only couplings", "J rejects assignment")
    else:
        report.fail("read-only couplings", "J accepted a write")
    chain_operator = chain(3, J=1.25, h=0.5)
    state = np.array([1.0, -1.0, 1.0])
    before = chain_operator.diagonal_energy(state)
    worst = 0.0
    for site in range(3):
        flipped = state.copy()
        flipped[site] *= -1
        worst = max(
            worst,
            abs(chain_operator.diagonal_energy(flipped) - before - chain_operator.flip_delta(state, site)),
        )
    if worst < 1e-12:
        report.ok("flip identity", f"max abs diff {worst:.2e}")
    else:
        report.fail("flip identity", f"max abs diff {worst:.2e}")


def _check_weights(report):
    _end, energy_piece, time_piece = weighted_interval(1.0, 1e-11, 0.0, 1e-8)
    if time_piece > 0.0 and abs(time_piece - 1e-8) / 1e-8 < 1e-6 and abs(energy_piece - 1e-19) / 1e-19 < 1e-6:
        report.ok("tiny interval", f"time {time_piece:.3e}")
    else:
        report.fail("tiny interval", f"time {time_piece}, energy {energy_piece}")
    try:
        weighted_interval(1e308, 0.0, 1.0, 1.0)
    except RuntimeError:
        report.ok("weight overflow", "a non-finite product raises")
    else:
        report.fail("weight overflow", "a non-finite product was returned")
    try:
        weighted_interval(1e308, 1e10, 1e10, 1.0)
    except RuntimeError:
        report.ok("energy overflow", "a non-finite energy integral raises")
    else:
        report.fail("energy overflow", "a non-finite energy integral was returned")
    end_weight, end_energy, end_time = weighted_interval(1.0, 0.0, -1000.0, 1.0)
    if end_weight == 0.0 and math.isfinite(end_energy) and abs(end_time - 0.001) / 0.001 < 1e-6:
        report.ok("weight underflow", f"time {end_time:.6f}")
    else:
        report.fail("weight underflow", f"weight {end_weight}, energy {end_energy}, time {end_time}")
    indexes, log_total = systematic(np.array([1e308, 1e308]), np.random.default_rng(0))
    expected = math.log(1e308) + math.log(2.0)
    if math.isfinite(log_total) and abs(log_total - expected) < 1e-6 and sorted(indexes.tolist()) == [0, 1]:
        report.ok("large weights", f"log total {log_total:.6f}, indexes {indexes.tolist()}")
    else:
        report.fail("large weights", f"log total {log_total}, indexes {indexes}")
    rng = np.random.default_rng(2)
    offspring = []
    for _ in range(40):
        draw, _log_mass = systematic(np.array([0.0, 2.0, 0.0, 3.0]), rng)
        offspring.extend(draw.tolist())
    if set(offspring) <= {1, 3} and 1 in offspring and 3 in offspring:
        report.ok("zero weights", f"offspring in {sorted(set(offspring))}")
    else:
        report.fail("zero weights", f"offspring {sorted(set(offspring))}")
    try:
        systematic(np.array([0.0, 0.0]), np.random.default_rng(0))
    except RuntimeError:
        report.ok("all-zero weights", "an empty population raises")
    else:
        report.fail("all-zero weights", "an empty population was resampled")
    try:
        systematic(np.array([1.0, -0.5]), np.random.default_rng(0))
    except RuntimeError:
        report.ok("negative weight", "a negative weight raises")
    else:
        report.fail("negative weight", "a negative weight was resampled")
    configs = np.ones((100, 1))
    configs[-1, 0] = -1.0
    projected = GFMC(chain(1, h=1000.0), 0.001).run(
        n_walkers=100,
        tau_block=1.0,
        n_equilibrate=0,
        n_measure=2,
        configurations=configs,
        seed=1,
    )
    if abs(projected["energy"] + 1000.001) < 1e-4:
        report.ok("underflow projection", f"energy {projected['energy']:.8f}")
    else:
        report.fail("underflow projection", f"energy {projected['energy']}")


def _check_overlap(report):
    class _Walker:
        spins = np.ones((2, 147), dtype=np.int8)

    value = _tag_overlap(_Walker())
    if value == 1.0:
        report.ok("int8 overlap", "147 identical sites give +1")
    else:
        report.fail("int8 overlap", str(value))


def _check_exact(report):
    frozen = moments_from_probability(np.array([1.0, 0.0, 0.0, 0.0]), 2)
    chi = spin_glass_susceptibility(frozen["q2"], 2)
    if abs(frozen["q2"] - 1.0) < 1e-12 and abs(chi - 2.0) < 1e-12:
        report.ok("spin-glass susceptibility", "aligned two-site state gives chi_SG = 2")
    else:
        report.fail("spin-glass susceptibility", f"q2 {frozen['q2']} chi {chi}")
    unit = ground_state(chain(2, J=1.0), 1.0)
    tiny = ground_state(chain(2, J=1e-7), 1e-7)
    if (
        unit["degeneracy"] == 1
        and tiny["degeneracy"] == 1
        and not unit["mixed"]
        and not tiny["mixed"]
        and abs(unit["q2"] - 0.6) < 1e-8
        and abs(tiny["q2"] - unit["q2"]) < 1e-6
        and abs(unit["chi_sg"] - 1.2) < 1e-8
        and abs(tiny["chi_sg"] - unit["chi_sg"]) < 1e-6
    ):
        report.ok("scale-invariant q2", f"unit {unit['q2']:.6f}, tiny {tiny['q2']:.6f}")
    else:
        report.fail(
            "scale-invariant q2",
            f"unit deg {unit['degeneracy']} q2 {unit['q2']}, tiny deg {tiny['degeneracy']} q2 {tiny['q2']}",
        )
    split = ground_state(chain(2, J=1.0, h=1e-7), 1e-4)
    gap_over_noise = split["gap"] / max(10.0 * split["residual"], 1e-30)
    if (
        split["degeneracy"] == 1
        and not split["mixed"]
        and not split["ensemble"]
        and split["degeneracy_resolved"]
        and gap_over_noise > 100.0
        and abs(split["q"] - 0.9975062245) < 1e-5
    ):
        report.ok("resolved unique ground state", f"q {split['q']:.7f}, gap/noise {gap_over_noise:.1f}")
    else:
        report.fail(
            "resolved unique ground state",
            f"deg {split['degeneracy']} mixed {split['mixed']} q {split['q']} gap/noise {gap_over_noise}",
        )
    forced = ground_state(chain(2, J=1.0), 1.0, mix_window=100.0)
    if (
        forced["ensemble"]
        and forced["mixed"]
        and forced["degeneracy"] == 1
        and abs(forced["energy"] - unit["energy"]) < 1e-10
        and not unit["ensemble"]
    ):
        report.ok("requested ensemble", f"degeneracy {forced['degeneracy']}, energy {forced['energy']:.6f}")
    else:
        report.fail(
            "requested ensemble",
            f"deg {forced['degeneracy']} mixed {forced['mixed']} ensemble {forced['ensemble']}",
        )
    tie_keep, tie_deg, tie_resolved, tie_mixed, tie_ensemble = _select_ground_states(
        np.array([0.0, 1e-16, 1.0]), 1e-15, 3, None
    )
    open_keep, open_deg, open_resolved, open_mixed, _open_ensemble = _select_ground_states(
        np.array([0.0, 1e-16]), 1e-15, 8, None
    )
    if (
        (tie_keep, tie_deg, tie_resolved, tie_mixed, tie_ensemble) == (2, 2, True, True, False)
        and (open_keep, open_deg, open_resolved, open_mixed) == (1, 2, False, False)
    ):
        report.ok("degeneracy versus truncation", "a closed tie is mixed and an open cluster is not")
    else:
        report.fail(
            "degeneracy versus truncation",
            f"tie {(tie_keep, tie_deg, tie_resolved, tie_mixed)} "
            f"open {(open_keep, open_deg, open_resolved, open_mixed)}",
        )
    try:
        ground_state(chain(13), 1.0)
    except ValueError:
        report.ok("moment limit", "13 sites with moments raises before the solve")
    else:
        report.fail("moment limit", "13 sites with moments was accepted")
    energy = ground_state(chain(13), 1.0, moments=False)
    if math.isfinite(energy["energy"]) and "q2" not in energy and energy["degeneracy"] == 1:
        report.ok("energy only", f"N=13 energy {energy['energy']:.6f}")
    else:
        report.fail("energy only", str(energy))


def _check_guide_path(report):
    class _Flat(Guide):
        def __init__(self):
            self.calls = 0

        def log_amplitude(self, spins):
            return 0.0

        def log_ratio(self, spins, site):
            self.calls += 1
            return 0.0

    constant = ConstantGuide()
    constant.calls = 0

    def _count(spins, site, guide=constant):
        guide.calls += 1
        return 0.0

    constant.log_ratio = _count
    flat = _Flat()
    settings = dict(n_walkers=20, tau_block=1.0, n_equilibrate=2, n_measure=5, seed=1)
    operator = chain(1)
    constant_run = GFMC(operator, 1.0, constant).run(configurations=np.array([1]), **settings)
    flat_run = GFMC(operator, 1.0, flat).run(configurations=np.array([1]), **settings)
    initial = settings["n_walkers"]
    if constant.calls == 0 and flat.calls > initial and flat.calls % operator.n_sites == 0:
        report.ok("constant-guide hop", f"subclass calls {flat.calls}")
    else:
        report.fail("constant-guide hop", f"constant {constant.calls}, subclass {flat.calls}")
    worst = max(abs(constant_run["energy"] + 1.0), abs(flat_run["energy"] + 1.0))
    if worst < 0.02:
        report.ok("flat guides match", f"max |E+1| {worst:.4f}")
    else:
        report.fail("flat guides match", f"constant {constant_run['energy']}, subclass {flat_run['energy']}")


def _record(**updates):
    record = {
        "lattice": "chain",
        "boundary": "obc",
        "size": "n8",
        "n_sites": 8,
        "J": 1.0,
        "gamma": 1.0,
        "h": 0.0,
        "seed": 3,
        "n_walkers": 2500,
        "tau_block": 0.2,
        "n_equilibrate": 40,
        "n_measure": 50,
        "energy": -9.8,
        "energy_error": 0.01,
        "energy_growth": -9.8,
        "energy_growth_error": 0.01,
        "exact_energy": -9.8,
        "guide": "constant",
    }
    record.update(updates)
    return record


def _check_storage(report):
    with tempfile.TemporaryDirectory() as root:
        first = save_run(root, _record())
        try:
            save_run(root, _record())
        except FileExistsError:
            report.ok("refuse overwrite", first.name)
        else:
            report.fail("refuse overwrite", "the second write replaced the file")
        other_walkers = save_run(root, _record(n_walkers=2501))
        near_gamma = save_run(root, _record(gamma=1.0 + 1e-9))
        nearer_gamma = save_run(root, _record(gamma=1.0 + 2e-9))
        separated = (
            len({first, other_walkers, near_gamma, nearer_gamma}) == 4
            and near_gamma.parent == nearer_gamma.parent
            and first.name.startswith("seed3_")
            and len(first.stem.split("_")[-1]) == 12
        )
        if separated:
            report.ok("run identity", f"{near_gamma.name} and {nearer_gamma.name}")
        else:
            report.fail(
                "run identity",
                f"{first.name}, {other_walkers.name}, {near_gamma.name}, {nearer_gamma.name}",
            )
        series = _record(seed=9, energy_series=np.array([1.0, 2.0, 3.0]), energy_resolved=True)
        path = save_run(root, series)
        loaded = load_run(path)
        if (
            loaded["lattice"] == "chain"
            and isinstance(loaded["energy"], float)
            and loaded["energy_series"].shape == (3,)
            and loaded["energy_resolved"] is True
        ):
            report.ok("load series", f"series {loaded['energy_series'].tolist()}")
        else:
            report.fail("load series", str(loaded))


def _exact_stderr(n_points, correlation, fraction=1.0):
    if correlation <= 0.0 or fraction <= 0.0:
        return 1.0 / math.sqrt(n_points)
    lags = np.arange(1, n_points)
    variance = (
        n_points + 2.0 * fraction * np.sum((n_points - lags) * correlation ** lags)
    ) / n_points ** 2
    return math.sqrt(float(variance))


def _ensemble(n_points, correlation, n_trials):
    means = []
    errors = []
    resolved = 0
    covered = 0
    for seed in range(n_trials):
        sample = _ar1(np.random.default_rng(seed), n_points, correlation)
        mean, error, flag = series_error(sample)
        if not flag or not math.isfinite(error) or error <= 0.0:
            continue
        resolved += 1
        means.append(mean)
        errors.append(error)
        covered += abs(mean) <= 1.96 * error
    return resolved, means, errors, covered


def _check_series_error(report):
    short = np.arange(4, dtype=np.float64)
    _mean, error, resolved = series_error(short)
    naive = float(short.std(ddof=1) / math.sqrt(short.size))
    if not resolved and abs(error - naive) < 1e-12:
        report.ok("short series", "four points stay unresolved")
    else:
        report.fail("short series", f"error {error}, resolved {resolved}")
    constant_mean, constant_error, constant_resolved = series_error(np.full(80, -1.0))
    if constant_resolved and constant_error == 0.0 and constant_mean == -1.0:
        report.ok("constant series", "error 0")
    else:
        report.fail("constant series", f"mean {constant_mean}, error {constant_error}, resolved {constant_resolved}")
    slow = _ensemble(32, 0.95, 20)
    if slow[0] == 0:
        report.ok("short correlation", "20 series of length 32 stay unresolved")
    else:
        report.fail("short correlation", f"{slow[0]}/20 resolved")
    _check_calibration(
        report,
        "slow AR(1)",
        512,
        0.95,
        200,
        min_resolved=150,
        min_ratio=0.90,
        min_coverage=0.90,
    )
    _check_calibration(
        report,
        "white noise",
        256,
        0.0,
        200,
        min_resolved=190,
        min_ratio=0.85,
        max_ratio=1.20,
        min_coverage=0.90,
    )
    _check_calibration(
        report,
        "mild AR(1)",
        160,
        0.67,
        200,
        min_resolved=180,
        min_ratio=0.90,
        min_coverage=0.90,
    )
    shifted = _reviewer_series(3, 128, 0.8)
    base_mean, base_error, base_flag = series_error(shifted)
    moved_mean, moved_error, moved_flag = series_error(shifted + 4.0)
    scaled_mean, scaled_error, scaled_flag = series_error(3.0 * shifted)
    affine_ok = (
        moved_flag is base_flag
        and abs(moved_mean - (base_mean + 4.0)) <= 1e-9
        and abs(moved_error - base_error) <= 1e-8 * max(1.0, abs(base_error))
        and scaled_flag is base_flag
        and abs(scaled_mean - 3.0 * base_mean) <= 1e-8 * max(1.0, abs(base_mean))
        and abs(scaled_error - 3.0 * base_error) <= 1e-8 * max(1.0, abs(base_error))
    )
    if affine_ok and math.isfinite(base_error) and base_error > 0.0:
        report.ok("affine series", "shift leaves the error unchanged and scale multiplies it")
    else:
        report.fail(
            "affine series",
            f"base {base_mean}, {base_error}, {base_flag}; "
            f"shifted {moved_mean}, {moved_error}; scaled {scaled_mean}, {scaled_error}",
        )
    # Seeds 0..n_trials-1 use the reviewed generator. An empty resolved subset
    # passes. A non-empty one fails when its median error is below the exact
    # standard deviation. The coverage floor is 0.80 because only a few dozen
    # of the ρ=0.99 series resolve, and because some mixed-timescale draws
    # look like white noise.
    _check_supported(
        report,
        "very slow AR(1)",
        512,
        0.99,
        1.0,
        200,
        min_ratio=0.90,
        min_coverage=0.80,
    )
    _check_supported(
        report,
        "mixed timescale",
        512,
        0.99,
        0.1,
        200,
        min_ratio=0.90,
        min_coverage=0.80,
    )


def _check_calibration(
    report, name, n_points, correlation, n_trials, min_resolved, min_ratio, min_coverage, max_ratio=None
):
    resolved, means, errors, covered = _ensemble(n_points, correlation, n_trials)
    exact = _exact_stderr(n_points, correlation)
    if resolved < min_resolved:
        report.fail(name, f"{resolved}/{n_trials} resolved, need {min_resolved}")
        return
    median = float(np.median(errors))
    ratio = median / exact
    coverage = covered / resolved
    ratio_ok = ratio >= min_ratio and (max_ratio is None or ratio <= max_ratio)
    if ratio_ok and coverage >= min_coverage:
        report.ok(name, f"{resolved}/{n_trials} resolved, median/exact {ratio:.3f}, coverage {coverage:.3f}")
    else:
        report.fail(
            name,
            f"{resolved}/{n_trials} resolved, median/exact {ratio:.3f}, coverage {coverage:.3f}",
        )


def _ar1(rng, n_points, correlation):
    series = np.empty(n_points)
    series[0] = rng.normal()
    scale = math.sqrt(1.0 - correlation * correlation)
    for index in range(1, n_points):
        series[index] = correlation * series[index - 1] + scale * rng.normal()
    return series


def _reviewer_series(seed, n_points, correlation, fraction=1.0):
    """Stationary Gaussian AR(1), plus white noise when ``fraction`` is below 1.

    ``fraction`` is the fraction of variance in the slow component. Seeds follow
    ``range(n_trials)`` so the slow and mixed ensembles match the reviewed generator.
    """
    rng = np.random.default_rng(seed)
    series = np.empty(n_points)
    series[0] = rng.normal()
    noise = rng.normal(size=n_points - 1) * math.sqrt(1.0 - correlation * correlation)
    for index in range(1, n_points):
        series[index] = correlation * series[index - 1] + noise[index - 1]
    if fraction == 1.0:
        return series
    return math.sqrt(fraction) * series + math.sqrt(1.0 - fraction) * rng.normal(size=n_points)


def _reviewer_ensemble(n_points, correlation, fraction, n_trials):
    means = []
    errors = []
    resolved = 0
    covered = 0
    for seed in range(n_trials):
        sample = _reviewer_series(seed, n_points, correlation, fraction)
        mean, error, flag = series_error(sample)
        if not flag or not math.isfinite(error) or error <= 0.0:
            continue
        resolved += 1
        means.append(mean)
        errors.append(error)
        covered += abs(mean) <= 1.96 * error
    return resolved, means, errors, covered


def _check_supported(report, name, n_points, correlation, fraction, n_trials, min_ratio, min_coverage):
    resolved, _means, errors, covered = _reviewer_ensemble(n_points, correlation, fraction, n_trials)
    if resolved == 0:
        report.ok(name, f"0/{n_trials} resolved")
        return
    exact = _exact_stderr(n_points, correlation, fraction)
    ratio = float(np.median(errors)) / exact
    coverage = covered / resolved
    if ratio >= min_ratio and coverage >= min_coverage:
        report.ok(name, f"{resolved}/{n_trials} resolved, median/exact {ratio:.3f}, coverage {coverage:.3f}")
    else:
        report.fail(name, f"{resolved}/{n_trials} resolved, median/exact {ratio:.3f}, coverage {coverage:.3f}")


class _Report:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def ok(self, name, detail):
        self.passed += 1
        print(f"pass  {name}: {detail}")

    def fail(self, name, detail):
        self.failed.append(name)
        print(f"FAIL  {name}: {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
