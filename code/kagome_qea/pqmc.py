"""Continuous-time projector quantum Monte Carlo with a flat guide.

Walkers carry the ground-state amplitude. With ``ψ_g = 1`` the local energy
is ``E_L = E_diag - λ`` and ``λ`` is ``Γ`` times the number of single-spin
flips. The mixed average of ``E_L`` estimates the ground-state energy of the
simulated replicas. Divide by the replica count for one copy.

The walker histogram is the amplitude, so ``⟨q^n⟩`` is a forward-walked tag:
``q`` is stored at the measurement time, descendants inherit it, and the tag
is read after a further projection time ``tau_forward``. A configuration
dependent guide replaces the uniform flip rate ``Γ`` by
``Γ ψ(s')/ψ(s)`` and the escape rate by the sum of those rates.
"""

import math

import numpy as np

from kagome_qea.observables import (
    binder_from_moments,
    connected_binder,
    jackknife_binder_series,
    overlap,
    spin_glass_susceptibility,
)
from qmc_numerics import series_error, systematic, weighted_interval


class _Walker:
    __slots__ = ("spins", "diagonal", "weight", "tag", "ancestor")

    def __init__(self, spins, diagonal):
        self.spins = spins
        self.diagonal = float(diagonal)
        self.weight = 1.0
        self.tag = 0.0
        self.ancestor = 0


def _initial_walkers(ham, n_walkers, n_replicas, rng):
    walkers = []
    for _ in range(n_walkers):
        bits = rng.integers(0, 2, size=(n_replicas, ham.n_sites))
        spins = np.where(bits == 0, np.int8(1), np.int8(-1))
        diagonal = 0.0
        for replica in range(n_replicas):
            diagonal += ham.diagonal_energy(spins[replica])
        walkers.append(_Walker(spins, diagonal))
    return walkers


def _advance(walker, ham, rate, shift, tau, rng):
    spins = walker.spins
    n_replicas, n_sites = spins.shape
    elapsed = 0.0
    energy_integral = 0.0
    time_integral = 0.0
    while elapsed < tau - 1e-15:
        # Mean waiting time of the constant-guide process is 1/λ.
        waiting = rng.exponential(1.0 / rate)
        hop = True
        if elapsed + waiting > tau:
            waiting = tau - elapsed
            hop = False
        local = walker.diagonal - rate
        walker.weight, energy_piece, time_piece = weighted_interval(
            walker.weight, local, shift, waiting
        )
        energy_integral += energy_piece
        time_integral += time_piece
        elapsed += waiting
        if hop:
            chosen = int(rng.integers(n_replicas * n_sites))
            replica, site = divmod(chosen, n_sites)
            walker.diagonal += ham.flip_delta(spins[replica], site)
            spins[replica, site] = -spins[replica, site]
    return energy_integral, time_integral


def _resample(walkers, rng):
    weights = np.fromiter((walker.weight for walker in walkers), dtype=np.float64)
    indexes, log_total = systematic(weights, rng)
    children = []
    for index in indexes:
        parent = walkers[int(index)]
        child = _Walker(parent.spins.copy(), parent.diagonal)
        child.tag = parent.tag
        child.ancestor = parent.ancestor
        children.append(child)
    return children, log_total


def _tag_overlap(walker):
    spins = walker.spins
    return overlap(spins[0], spins[1])


def _branch_dt(walkers, limit):
    """Longest resampling interval that keeps log-weights of order one.

    One block of length 0.2 multiplies weights by the diagonal-energy spread.
    On 216 bonds that spread is large enough to leave a single ancestor, and
    the forward-walked overlap then jumps from one Γ to the next.
    """
    limit = float(limit)
    if limit <= 0.0:
        raise ValueError("branch interval must be positive")
    diagonals = np.fromiter((walker.diagonal for walker in walkers), dtype=np.float64)
    spread = float(diagonals.std()) if diagonals.size else 0.0
    if not math.isfinite(spread) or spread < 1e-8:
        return limit
    return min(limit, 0.5 / spread)


def _ancestor_count(walkers):
    ancestors = np.fromiter((walker.ancestor for walker in walkers), dtype=np.int64)
    return int(np.unique(ancestors).size)


def _effective_ancestors(walkers):
    """Kish size ``(Σ w)^2 / Σ w^2`` of the descendant weights.

    Unique ancestor ids can stay large while one lineage carries almost
    every walker. Moments then have the variance of that one lineage.
    """
    ancestors = np.fromiter((walker.ancestor for walker in walkers), dtype=np.int64)
    if ancestors.size == 0:
        return 0.0
    _, counts = np.unique(ancestors, return_counts=True)
    weights = counts.astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(total * total / np.dot(weights, weights))


def _distinct_tags(walkers):
    tags = np.fromiter((walker.tag for walker in walkers), dtype=np.float64)
    if tags.size == 0:
        return 0
    return int(np.unique(np.round(tags, decimals=12)).size)


def _effective_tags(walkers):
    """Kish size of the tagged overlaps, grouping copies of the same tag."""
    tags = np.fromiter((walker.tag for walker in walkers), dtype=np.float64)
    if tags.size == 0:
        return 0.0
    _, counts = np.unique(np.round(tags, decimals=12), return_counts=True)
    weights = counts.astype(np.float64)
    total = float(weights.sum())
    return float(total * total / np.dot(weights, weights))


def _clone_population(walkers):
    clones = []
    for walker in walkers:
        child = _Walker(walker.spins.copy(), walker.diagonal)
        child.weight = walker.weight
        child.tag = walker.tag
        child.ancestor = walker.ancestor
        clones.append(child)
    return clones


def _project_once(
    ham,
    gamma,
    rng,
    *,
    n_replicas=2,
    n_walkers=2000,
    tau_block=0.25,
    n_equilibrate=40,
    n_measure=60,
    tau_forward=8.0,
    n_forward_cycles=4,
):
    """One independent population.

    Energy blocks are one mixed estimate each. Overlap moments are one
    forward-walking cycle each. The cycle stops before the tagged population
    falls below ``n_walkers / 40`` lineages, so one walker cannot set the
    cycle's ``q^4``.
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if n_replicas not in (1, 2):
        raise ValueError("phase 1 uses one replica or a pair")
    if n_walkers < 2 or tau_block <= 0.0:
        raise ValueError("need at least two walkers and a positive block")
    rate = float(n_replicas * ham.n_sites * gamma)
    walkers = _initial_walkers(ham, n_walkers, n_replicas, rng)
    shift = float(np.mean([walker.diagonal - rate for walker in walkers]))
    energies = []
    growth = []

    def population_step(dt):
        nonlocal shift, walkers
        energy_sum = 0.0
        time_sum = 0.0
        for walker in walkers:
            energy_piece, time_piece = _advance(walker, ham, rate, shift, dt, rng)
            energy_sum += energy_piece
            time_sum += time_piece
        children, log_total = _resample(walkers, rng)
        log_growth = (log_total - math.log(n_walkers)) / dt
        # Growth estimates the pair. The shift used here is the one that
        # branched this step; the next step uses the damped update.
        growth_value = (shift - log_growth) / n_replicas
        shift = 0.5 * shift + 0.5 * (shift - log_growth)
        walkers = children
        return energy_sum, time_sum, growth_value

    def one_block(record):
        nonlocal walkers
        energy_sum = 0.0
        time_sum = 0.0
        growth_values = []
        covered = 0.0
        while covered < tau_block - 1e-12:
            dt = _branch_dt(walkers, tau_block - covered)
            energy_piece, time_piece, growth_value = population_step(dt)
            energy_sum += energy_piece
            time_sum += time_piece
            growth_values.append(growth_value)
            covered += dt
        if record and time_sum > 0.0:
            energies.append(energy_sum / time_sum / n_replicas)
            growth.append(float(np.mean(growth_values)))

    for _ in range(int(n_equilibrate)):
        one_block(False)
    for _ in range(int(n_measure)):
        one_block(True)

    q_means = []
    q2_means = []
    q3_means = []
    q4_means = []
    unique_ancestors = []
    effective_ancestors = []
    distinct_tags = []
    effective_tag_sizes = []
    forward_times = []
    if n_replicas == 2 and n_forward_cycles:
        # Below this many effective lineages one walker sets q^4, and the
        # Binder ratio moves by tenths from cycle to cycle.
        ancestor_floor = max(2, int(n_walkers) // 40)
        for _ in range(int(n_forward_cycles)):
            for identity, walker in enumerate(walkers):
                walker.tag = _tag_overlap(walker)
                walker.ancestor = identity
            distinct_tags.append(_distinct_tags(walkers))
            effective_tag_sizes.append(_effective_tags(walkers))
            saved = _clone_population(walkers)
            elapsed = 0.0
            while elapsed < float(tau_forward) - 1e-12:
                backup = _clone_population(walkers)
                shift_backup = shift
                dt = _branch_dt(walkers, float(tau_forward) - elapsed)
                energy_piece, time_piece, growth_value = population_step(dt)
                if _ancestor_count(walkers) < ancestor_floor:
                    walkers = backup
                    shift = shift_backup
                    break
                if time_piece > 0.0:
                    energies.append(energy_piece / time_piece / n_replicas)
                    growth.append(growth_value)
                saved = walkers
                elapsed += dt
            tags = np.fromiter((walker.tag for walker in saved), dtype=np.float64)
            q_means.append(float(tags.mean()))
            q2_means.append(float(np.mean(tags ** 2)))
            q3_means.append(float(np.mean(tags ** 3)))
            q4_means.append(float(np.mean(tags ** 4)))
            unique_ancestors.append(_ancestor_count(saved))
            effective_ancestors.append(_effective_ancestors(saved))
            forward_times.append(elapsed)
            walkers = saved

    energy, energy_error, energy_resolved = series_error(energies)
    growth_mean, growth_error, growth_resolved = series_error(growth)
    result = {
        "energy": energy,
        "energy_error": energy_error,
        "energy_resolved": bool(energy_resolved),
        "energy_growth": growth_mean,
        "energy_growth_error": growth_error,
        "energy_growth_resolved": bool(growth_resolved),
        "n_walkers": int(n_walkers),
        "n_replicas": int(n_replicas),
        "n_sites": int(ham.n_sites),
        "n_energy_blocks": len(energies),
        "n_populations": 1,
        "shift": float(shift),
    }
    if q2_means:
        q_mean, q_error, q_resolved = series_error(q_means)
        q2, q2_error, q2_resolved = series_error(q2_means)
        q3, q3_error, q3_resolved = series_error(q3_means)
        q4, q4_error, q4_resolved = series_error(q4_means)
        binder, ratio = binder_from_moments(q2, q4)
        _series_binder, binder_error = jackknife_binder_series(q2_means, q4_means)
        connected, connected_ratio = connected_binder(q_mean, q2, q3, q4)
        result.update(
            q=q_mean,
            q_error=q_error,
            q_resolved=q_resolved,
            q_series=np.asarray(q_means, dtype=np.float64),
            q2=q2,
            q2_error=q2_error,
            q2_resolved=q2_resolved,
            q2_series=np.asarray(q2_means, dtype=np.float64),
            q3=q3,
            q3_error=q3_error,
            q3_resolved=q3_resolved,
            q3_series=np.asarray(q3_means, dtype=np.float64),
            q4=q4,
            q4_error=q4_error,
            q4_resolved=q4_resolved,
            q4_series=np.asarray(q4_means, dtype=np.float64),
            binder=binder,
            binder_error=binder_error,
            ratio=ratio,
            connected_binder=connected,
            connected_ratio=connected_ratio,
            chi_sg=spin_glass_susceptibility(q2, ham.n_sites),
            chi_sg_error=float(ham.n_sites) * float(q2_error),
            tau_forward=float(min(forward_times)),
            n_forward_cycles=len(q2_means),
            min_unique_ancestors=int(min(unique_ancestors)),
            min_effective_ancestors=float(min(effective_ancestors)),
            min_distinct_tags=int(min(distinct_tags)),
            min_effective_tags=float(min(effective_tag_sizes)),
        )
    return result


def _population_stderr(values):
    """Standard error across independent populations."""
    data = np.asarray(values, dtype=np.float64)
    if data.size < 2:
        return float("nan")
    return float(data.std(ddof=1) / np.sqrt(data.size))


def _combine_populations(parts):
    """Average independent populations with equal weight.

    A stuck population remeasures one survivor and reports a tiny internal
    error. Weighting populations by that internal error would keep the
    survivor and drop the runs that found a different overlap. Each
    population is one draw. The reported error is the scatter of those draws.
    """
    first = parts[0]
    energies = np.array([part["energy"] for part in parts], dtype=np.float64)
    growths = np.array([part["energy_growth"] for part in parts], dtype=np.float64)
    result = {
        "energy": float(energies.mean()),
        "energy_error": _population_stderr(energies),
        "energy_resolved": len(parts) >= 8,
        "energy_growth": float(growths.mean()),
        "energy_growth_error": _population_stderr(growths),
        "energy_growth_resolved": len(parts) >= 8,
        "n_walkers": first["n_walkers"],
        "n_replicas": first["n_replicas"],
        "n_sites": int(first["n_sites"]),
        "n_energy_blocks": int(sum(part["n_energy_blocks"] for part in parts)),
        "n_populations": len(parts),
        "shift": float(np.mean([part["shift"] for part in parts])),
    }
    if "q2" not in first:
        return result
    q = np.array([part["q"] for part in parts], dtype=np.float64)
    q2 = np.array([part["q2"] for part in parts], dtype=np.float64)
    q3 = np.array([part["q3"] for part in parts], dtype=np.float64)
    q4 = np.array([part["q4"] for part in parts], dtype=np.float64)
    binder, _ratio = binder_from_moments(float(q2.mean()), float(q4.mean()))
    _series_binder, binder_error = jackknife_binder_series(q2, q4)
    connected, connected_ratio = connected_binder(
        float(q.mean()), float(q2.mean()), float(q3.mean()), float(q4.mean())
    )
    resolved = len(parts) >= 8
    result.update(
        q=float(q.mean()),
        q_error=_population_stderr(q),
        q_resolved=resolved,
        q_series=q,
        q2=float(q2.mean()),
        q2_error=_population_stderr(q2),
        q2_resolved=resolved,
        q2_series=q2,
        q3=float(q3.mean()),
        q3_error=_population_stderr(q3),
        q3_resolved=resolved,
        q3_series=q3,
        q4=float(q4.mean()),
        q4_error=_population_stderr(q4),
        q4_resolved=resolved,
        q4_series=q4,
        binder=binder,
        binder_error=binder_error,
        ratio=float(q4.mean()) / float(q2.mean()) ** 2,
        connected_binder=connected,
        connected_ratio=connected_ratio,
        chi_sg=spin_glass_susceptibility(float(q2.mean()), first["n_sites"]),
        chi_sg_error=float(first["n_sites"]) * _population_stderr(q2),
        tau_forward=float(min(part["tau_forward"] for part in parts)),
        n_forward_cycles=int(first["n_forward_cycles"]),
        min_unique_ancestors=int(min(part["min_unique_ancestors"] for part in parts)),
        min_effective_ancestors=float(min(part["min_effective_ancestors"] for part in parts)),
        min_distinct_tags=int(min(part["min_distinct_tags"] for part in parts)),
        min_effective_tags=float(min(part["min_effective_tags"] for part in parts)),
    )
    return result


def project(
    ham,
    gamma,
    *,
    n_replicas=2,
    n_walkers=2000,
    tau_block=0.25,
    n_equilibrate=40,
    n_measure=60,
    tau_forward=8.0,
    n_forward_cycles=4,
    n_populations=1,
    seed=0,
):
    """Project ``n_replicas`` non-interacting copies.

    ``n_populations`` independent populations are averaged with equal weight.
    One population remeasures its own survivors, so its cycle-to-cycle error
    can be far smaller than the scatter between populations. With two or more
    populations the reported errors are that scatter. Eight or more
    populations set the resolved flags. The flag records the population
    count, not a coverage certificate.
    """
    if isinstance(n_populations, bool) or int(n_populations) < 1:
        raise ValueError("n_populations must be a positive integer")
    n_populations = int(n_populations)
    options = dict(
        n_replicas=n_replicas,
        n_walkers=n_walkers,
        tau_block=tau_block,
        n_equilibrate=n_equilibrate,
        n_measure=n_measure,
        tau_forward=tau_forward,
        n_forward_cycles=n_forward_cycles,
    )
    if n_populations == 1:
        result = _project_once(ham, gamma, np.random.default_rng(seed), **options)
        result["n_populations"] = 1
        return result
    rngs = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(n_populations)
    ]
    parts = [_project_once(ham, gamma, rng, **options) for rng in rngs]
    return _combine_populations(parts)
