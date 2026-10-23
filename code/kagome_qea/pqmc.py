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

from kagome_qea.observables import binder_from_moments, connected_binder, overlap
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
    seed=0,
):
    """Project ``n_replicas`` non-interacting copies.

    Energy blocks are one mixed estimate each. Overlap moments are one
    forward-walking cycle each. Errors use the shared autocorrelation
    estimator. ``q``, ``q2``, ``q3``, and ``q4`` each keep a series and a
    resolved flag. A series shorter than 64 values is unresolved.
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if n_replicas not in (1, 2):
        raise ValueError("phase 1 uses one replica or a pair")
    if n_walkers < 2 or tau_block <= 0.0:
        raise ValueError("need at least two walkers and a positive block")
    rng = np.random.default_rng(seed)
    rate = float(n_replicas * ham.n_sites * gamma)
    walkers = _initial_walkers(ham, n_walkers, n_replicas, rng)
    shift = float(np.mean([walker.diagonal - rate for walker in walkers]))
    energies = []
    growth = []

    def one_block(record):
        nonlocal shift, walkers
        energy_sum = 0.0
        time_sum = 0.0
        for walker in walkers:
            energy_piece, time_piece = _advance(walker, ham, rate, shift, tau_block, rng)
            energy_sum += energy_piece
            time_sum += time_piece
        if record and time_sum > 0.0:
            energies.append(energy_sum / time_sum / n_replicas)
        children, log_total = _resample(walkers, rng)
        log_growth = (log_total - math.log(n_walkers)) / tau_block
        if record:
            # Growth estimates the pair. Store one copy, as `energy` does.
            growth.append((shift - log_growth) / n_replicas)
        # Damped population control. The mixed estimator inside a block does
        # not use this shift; the next block's branching does.
        shift = 0.5 * shift + 0.5 * (shift - log_growth)
        walkers = children

    for _ in range(int(n_equilibrate)):
        one_block(False)
    for _ in range(int(n_measure)):
        one_block(True)

    q_means = []
    q2_means = []
    q3_means = []
    q4_means = []
    unique_ancestors = []
    if n_replicas == 2 and n_forward_cycles:
        n_forward_blocks = max(1, int(round(float(tau_forward) / tau_block)))
        for _ in range(int(n_forward_cycles)):
            for identity, walker in enumerate(walkers):
                walker.tag = _tag_overlap(walker)
                walker.ancestor = identity
            for _block in range(n_forward_blocks):
                one_block(True)
            tags = np.fromiter((walker.tag for walker in walkers), dtype=np.float64)
            ancestors = np.fromiter(
                (walker.ancestor for walker in walkers), dtype=np.int64
            )
            q_means.append(float(tags.mean()))
            q2_means.append(float(np.mean(tags ** 2)))
            q3_means.append(float(np.mean(tags ** 3)))
            q4_means.append(float(np.mean(tags ** 4)))
            unique_ancestors.append(int(np.unique(ancestors).size))

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
        "n_energy_blocks": len(energies),
        "shift": float(shift),
    }
    if q2_means:
        q_mean, q_error, q_resolved = series_error(q_means)
        q2, q2_error, q2_resolved = series_error(q2_means)
        q3, q3_error, q3_resolved = series_error(q3_means)
        q4, q4_error, q4_resolved = series_error(q4_means)
        binder, ratio = binder_from_moments(q2, q4)
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
            ratio=ratio,
            connected_binder=connected,
            connected_ratio=connected_ratio,
            tau_forward=n_forward_blocks * tau_block,
            n_forward_cycles=len(q2_means),
            min_unique_ancestors=int(min(unique_ancestors)),
        )
    return result
