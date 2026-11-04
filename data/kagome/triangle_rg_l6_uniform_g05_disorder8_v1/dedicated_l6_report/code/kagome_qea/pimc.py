"""Continuous-time path-integral Monte Carlo with local kink updates.

A worldline is periodic on ``[0, β)``. Its weight is

.. math::

    W = \\Gamma^{K} \\exp(S), \\qquad
    S = \\int_0^\\beta \\Big(\\sum_b J_b s_i s_j + h \\sum_i s_i\\Big) d\\tau.

An insert draws two times uniformly and flips the open interval when that
interval contains no kink. The reverse deletes one of the ``K - 1``
consecutive pairs that do not wrap through ``τ = 0``. Those two proposals are
the two directions of one kernel, so a kink update picks one of them with
equal probability. A global flip is proposed with probability 1/2, so two
replicas do not flip in lockstep. One sweep makes two kink proposals and one
flip attempt on every site.

The energy estimator is the time average of the diagonal plus ``-K/β``.
Overlap moments use ``q(τ)`` at equal imaginary time, averaged as
``β^{-1}\\int q(τ)^n dτ``. Cluster updates are a later acceleration: they
have to pass the same exact checks, and they have to accept negative ``J``.
"""

import bisect
import math

import numpy as np

from kagome_qea.observables import (
    binder_from_moments,
    blocked_stderr,
    connected_binder,
    spin_glass_susceptibility,
)


class Worldline:
    """One continuous-time spin trajectory."""

    def __init__(self, n_sites, beta):
        if int(n_sites) < 1:
            raise ValueError("n_sites must be positive")
        if float(beta) <= 0.0:
            raise ValueError("beta must be positive")
        self.n_sites = int(n_sites)
        self.beta = float(beta)
        self.s0 = [1] * self.n_sites
        self.kinks = [[] for _ in range(self.n_sites)]

    @property
    def kink_count(self):
        return sum(len(times) for times in self.kinks)

    def spin_at(self, site, time):
        """Spin on the segment that begins at ``time`` (kink at ``time`` already applied)."""
        passed = bisect.bisect_right(self.kinks[site], time)
        spin = self.s0[site]
        return spin if passed % 2 == 0 else -spin

    def integrate_spin(self, site, t1, t2):
        """``∫_{t1}^{t2} s(τ) dτ``."""
        if t2 < t1:
            raise ValueError("time interval is reversed")
        times = self.kinks[site]
        left = bisect.bisect_right(times, t1)
        right = bisect.bisect_left(times, t2)
        spin = self.s0[site] if left % 2 == 0 else -self.s0[site]
        integral = 0.0
        previous = t1
        for time in times[left:right]:
            integral += spin * (time - previous)
            spin = -spin
            previous = time
        integral += spin * (t2 - previous)
        return integral

    def integrate_product(self, left, right, t1, t2):
        """``∫_{t1}^{t2} s_left(τ) s_right(τ) dτ``."""
        if t2 < t1:
            raise ValueError("time interval is reversed")
        times_left = self.kinks[left]
        times_right = self.kinks[right]
        index_left = bisect.bisect_right(times_left, t1)
        index_right = bisect.bisect_right(times_right, t1)
        spin_left = self.s0[left] if index_left % 2 == 0 else -self.s0[left]
        spin_right = self.s0[right] if index_right % 2 == 0 else -self.s0[right]
        integral = 0.0
        previous = t1
        while True:
            next_left = (
                times_left[index_left]
                if index_left < len(times_left) and times_left[index_left] < t2
                else None
            )
            next_right = (
                times_right[index_right]
                if index_right < len(times_right) and times_right[index_right] < t2
                else None
            )
            if next_left is None and next_right is None:
                integral += spin_left * spin_right * (t2 - previous)
                return integral
            if next_right is None or (next_left is not None and next_left <= next_right):
                instant = next_left
                integral += spin_left * spin_right * (instant - previous)
                spin_left = -spin_left
                index_left += 1
                previous = instant
                if next_right is not None and next_right == instant:
                    spin_right = -spin_right
                    index_right += 1
            else:
                instant = next_right
                integral += spin_left * spin_right * (instant - previous)
                spin_right = -spin_right
                index_right += 1
                previous = instant

    def classical_action(self, ham):
        """``S`` on this worldline."""
        action = 0.0
        for bond, (left, right) in enumerate(ham.bonds):
            action += float(ham.J[bond]) * self.integrate_product(
                int(left), int(right), 0.0, self.beta
            )
        if ham.h != 0.0:
            for site in range(self.n_sites):
                action += ham.h * self.integrate_spin(site, 0.0, self.beta)
        return action


def segment_action_change(worldline, ham, site, t1, t2, spin):
    """Change in ``S`` from flipping ``site`` on ``(t1, t2)``, where it was ``spin``."""
    bracket = ham.h * (t2 - t1)
    for neighbor, coupling in ham.neighbors[site]:
        bracket += coupling * worldline.integrate_spin(neighbor, t1, t2)
    return (-2.0 * spin) * bracket


def flip_action_change(worldline, ham, site):
    """Change in ``S`` from flipping every segment of ``site``."""
    bracket = ham.h * worldline.integrate_spin(site, 0.0, worldline.beta)
    for neighbor, coupling in ham.neighbors[site]:
        bracket += coupling * worldline.integrate_product(
            site, neighbor, 0.0, worldline.beta
        )
    return -2.0 * bracket


def _log_uniform(rng):
    draw = float(rng.random())
    if draw == 0.0:
        return -math.inf
    return math.log(draw)


def _attempt_insert(worldline, ham, gamma, site, rng):
    beta = worldline.beta
    t_a = float(rng.random()) * beta
    t_b = float(rng.random()) * beta
    if t_a == t_b:
        return
    t1, t2 = (t_a, t_b) if t_a < t_b else (t_b, t_a)
    times = worldline.kinks[site]
    left = bisect.bisect_right(times, t1)
    if left < len(times) and times[left] < t2:
        return
    spin = worldline.s0[site] if left % 2 == 0 else -worldline.s0[site]
    change = segment_action_change(worldline, ham, site, t1, t2, spin)
    log_accept = (
        2.0 * math.log(gamma)
        + change
        + 2.0 * math.log(beta)
        - math.log(2.0 * (len(times) + 1))
    )
    if _log_uniform(rng) < log_accept:
        bisect.insort(times, t1)
        bisect.insort(times, t2)


def _attempt_delete(worldline, ham, gamma, site, rng):
    times = worldline.kinks[site]
    n_kinks = len(times)
    if n_kinks < 2:
        return
    n_pairs = n_kinks - 1
    pair = int(rng.integers(n_pairs))
    t1 = times[pair]
    t2 = times[pair + 1]
    passed = pair + 1
    spin = worldline.s0[site] if passed % 2 == 0 else -worldline.s0[site]
    change = segment_action_change(worldline, ham, site, t1, t2, spin)
    log_accept = (
        math.log(2.0 * n_pairs)
        + change
        - 2.0 * math.log(gamma)
        - 2.0 * math.log(worldline.beta)
    )
    if _log_uniform(rng) < log_accept:
        del times[pair:pair + 2]


def _attempt_flip(worldline, ham, site, rng):
    # Propose the flipped line with probability 1/2. The reverse proposal has
    # the same probability, so it cancels. Always flipping both replicas would
    # freeze their relative sign and bias odd moments of q.
    if rng.random() < 0.5:
        return
    change = flip_action_change(worldline, ham, site)
    if _log_uniform(rng) < change:
        worldline.s0[site] = -worldline.s0[site]


def sweep(worldline, ham, gamma, rng):
    """Two kink proposals and one flip per site.

    Insert and delete are directions of one kernel. Each kink proposal selects
    one direction with probability 1/2, which cancels in the acceptance ratio.
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if worldline.n_sites != ham.n_sites:
        raise ValueError("worldline and Hamiltonian disagree on the site count")
    for site in rng.permutation(worldline.n_sites):
        chosen = int(site)
        for _ in range(2):
            if rng.random() < 0.5:
                _attempt_insert(worldline, ham, gamma, chosen, rng)
            else:
                _attempt_delete(worldline, ham, gamma, chosen, rng)
        _attempt_flip(worldline, ham, chosen, rng)


def measure_energy(worldline, ham):
    """``⟨H_diag⟩ - ⟨K⟩/β``."""
    return -(worldline.classical_action(ham) + worldline.kink_count) / worldline.beta


def measure_magnetization(worldline):
    """Time average of each spin."""
    beta = worldline.beta
    return np.array(
        [
            worldline.integrate_spin(site, 0.0, beta) / beta
            for site in range(worldline.n_sites)
        ]
    )


def _segment_spins(worldline):
    pieces = [np.array([0.0, worldline.beta], dtype=np.float64)]
    for times in worldline.kinks:
        if times:
            pieces.append(np.asarray(times, dtype=np.float64))
    grid = np.unique(np.concatenate(pieces))
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    spins = np.empty((midpoints.size, worldline.n_sites), dtype=np.float64)
    for site in range(worldline.n_sites):
        times = np.asarray(worldline.kinks[site], dtype=np.float64)
        passed = np.searchsorted(times, midpoints, side="right") if times.size else 0
        spin0 = float(worldline.s0[site])
        spins[:, site] = np.where(np.asarray(passed) % 2 == 0, spin0, -spin0)
    return grid, spins


def measure_overlap(worldline_a, worldline_b):
    """Equal-time moments ``(⟨q⟩, ⟨q^2⟩, ⟨q^3⟩, ⟨q^4⟩)`` along the pair of paths."""
    if worldline_a.n_sites != worldline_b.n_sites or worldline_a.beta != worldline_b.beta:
        raise ValueError("replicas must share the site count and beta")
    grid_a, spins_a = _segment_spins(worldline_a)
    grid_b, spins_b = _segment_spins(worldline_b)
    grid = np.unique(np.concatenate((grid_a, grid_b)))
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    index_a = np.searchsorted(grid_a, midpoints, side="right") - 1
    index_b = np.searchsorted(grid_b, midpoints, side="right") - 1
    values = np.mean(spins_a[index_a] * spins_b[index_b], axis=1)
    widths = np.diff(grid)
    beta = worldline_a.beta
    return tuple(
        float(np.dot(values ** power, widths) / beta) for power in (1, 2, 3, 4)
    )


def rescale_beta(worldline, beta):
    """Stretch kink times so the fractional trajectory is unchanged."""
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    factor = beta / worldline.beta
    worldline.beta = beta
    if factor == 1.0:
        return
    for times in worldline.kinks:
        for index, time in enumerate(times):
            times[index] = time * factor


def _series_result(energies, moments, block_size, n_sites):
    energy = np.asarray(energies, dtype=np.float64)
    result = {
        "energy": float(energy.mean()),
        "energy_error": blocked_stderr(energy, block_size),
        "n_measure": int(energy.size),
        "series_energy": energy,
    }
    if moments:
        stacked = np.asarray(moments, dtype=np.float64)
        q, q2, q3, q4 = (float(stacked[:, power].mean()) for power in range(4))
        binder, ratio = binder_from_moments(q2, q4)
        connected, connected_ratio = connected_binder(q, q2, q3, q4)
        result.update(
            q=q,
            q2=q2,
            q3=q3,
            q4=q4,
            q_error=blocked_stderr(stacked[:, 0], block_size),
            q2_error=blocked_stderr(stacked[:, 1], block_size),
            q4_error=blocked_stderr(stacked[:, 3], block_size),
            binder=binder,
            ratio=ratio,
            connected_binder=connected,
            connected_ratio=connected_ratio,
            chi_sg=spin_glass_susceptibility(q2, n_sites),
            chi_sg_error=float(n_sites) * blocked_stderr(stacked[:, 1], block_size),
            series_q2=stacked[:, 1],
            series_q4=stacked[:, 3],
        )
    return result


def _block_size(n_measure, block_size):
    if block_size is None:
        return max(1, int(n_measure) // 30)
    if int(block_size) < 1:
        raise ValueError("block_size must be positive")
    return int(block_size)


def simulate(
    ham,
    gamma,
    beta,
    *,
    n_replicas=2,
    n_sweeps=1000,
    n_equilibrate=200,
    seed=0,
    measure_every=1,
    block_size=None,
):
    """Equilibrium local-update simulation at fixed ``Γ`` and ``β``."""
    if n_replicas not in (1, 2):
        raise ValueError("phase 1 uses one replica or a pair")
    if measure_every < 1:
        raise ValueError("measure_every must be positive")
    rng = np.random.default_rng(seed)
    replicas = [Worldline(ham.n_sites, beta) for _ in range(n_replicas)]
    for _ in range(int(n_equilibrate)):
        for worldline in replicas:
            sweep(worldline, ham, gamma, rng)
    energies = []
    moments = []
    for step in range(int(n_sweeps)):
        for worldline in replicas:
            sweep(worldline, ham, gamma, rng)
        if step % measure_every == 0:
            reading = [measure_energy(worldline, ham) for worldline in replicas]
            energies.append(float(np.mean(reading)))
            if n_replicas == 2:
                moments.append(measure_overlap(replicas[0], replicas[1]))
    return _series_result(
        energies, moments, _block_size(len(energies), block_size), ham.n_sites
    )


def attempt_temper_swap(worldline_a, gamma_a, worldline_b, gamma_b, rng):
    """Swap two worldlines between neighboring ``Γ`` rungs.

    The classical action does not depend on ``Γ``, so the Metropolis ratio is
    ``(Γ_b / Γ_a)^{K_a - K_b}``.
    """
    if worldline_a.beta != worldline_b.beta:
        raise ValueError("tempering rungs must share beta")
    if gamma_a <= 0.0 or gamma_b <= 0.0:
        raise ValueError("gamma must be positive")
    log_accept = (worldline_a.kink_count - worldline_b.kink_count) * math.log(
        gamma_b / gamma_a
    )
    if _log_uniform(rng) < log_accept:
        worldline_a.s0, worldline_b.s0 = worldline_b.s0, worldline_a.s0
        worldline_a.kinks, worldline_b.kinks = worldline_b.kinks, worldline_a.kinks
        return True
    return False


def _install_spins(worldline, spins):
    values = np.asarray(spins, dtype=np.int8).reshape(-1)
    if values.shape != (worldline.n_sites,) or not np.all(np.abs(values) == 1):
        raise ValueError("initial spins must be ±1 on every site")
    worldline.s0 = [int(spin) for spin in values]
    worldline.kinks = [[] for _ in range(worldline.n_sites)]


def parallel_temper(
    ham,
    gammas,
    beta,
    *,
    n_replicas=1,
    n_sweeps=1000,
    n_equilibrate=200,
    seed=0,
    swap_every=1,
    measure_every=1,
    block_size=None,
    initial_s0=None,
):
    """Ladder in ``Γ`` at one ``β``. Spin replicas at a rung swap independently.

    Tempering replicas are the ladder. They are not the overlap pair. Each
    rung can still hold an overlap pair; those two worldlines swap on their own.
    ``initial_s0`` is a classical product state, shape ``(n_replicas, n_sites)``
    or ``(n_rungs, n_replicas, n_sites)``. It replaces the all-up start. Kinks
    still grow during the sweeps.
    """
    ladder = [float(gamma) for gamma in gammas]
    if any(gamma <= 0.0 for gamma in ladder):
        raise ValueError("gamma must be positive")
    if n_replicas not in (1, 2) or swap_every < 1:
        raise ValueError("unsupported tempering parameters")
    rng = np.random.default_rng(seed)
    rungs = [
        [Worldline(ham.n_sites, beta) for _ in range(n_replicas)] for _ in ladder
    ]
    if initial_s0 is not None:
        seeded = np.asarray(initial_s0, dtype=np.int8)
        if seeded.ndim == 2:
            seeded = np.broadcast_to(seeded, (len(ladder), n_replicas, ham.n_sites))
        if seeded.shape != (len(ladder), n_replicas, ham.n_sites):
            raise ValueError("initial_s0 does not match the ladder and the replica count")
        for rung, worldlines in enumerate(rungs):
            for replica, worldline in enumerate(worldlines):
                _install_spins(worldline, seeded[rung, replica])
    swaps = 0
    attempts = 0

    def update_once():
        for worldlines, gamma in zip(rungs, ladder):
            for worldline in worldlines:
                sweep(worldline, ham, gamma, rng)

    for _ in range(int(n_equilibrate)):
        update_once()
    series = [[] for _ in ladder]
    moments = [[] for _ in ladder]
    parity = 0
    for step in range(int(n_sweeps)):
        update_once()
        if (step + 1) % swap_every == 0:
            for rung in range(parity, len(ladder) - 1, 2):
                for replica in range(n_replicas):
                    attempts += 1
                    swapped = attempt_temper_swap(
                        rungs[rung][replica],
                        ladder[rung],
                        rungs[rung + 1][replica],
                        ladder[rung + 1],
                        rng,
                    )
                    swaps += int(swapped)
            parity ^= 1
        if step % measure_every == 0:
            for rung, worldlines in enumerate(rungs):
                reading = [measure_energy(worldline, ham) for worldline in worldlines]
                series[rung].append(float(np.mean(reading)))
                if n_replicas == 2:
                    moments[rung].append(measure_overlap(worldlines[0], worldlines[1]))
    results = []
    width = _block_size(len(series[0]), block_size)
    for rung, gamma in enumerate(ladder):
        packed = _series_result(series[rung], moments[rung], width, ham.n_sites)
        packed.update(gamma=gamma, beta=float(beta))
        results.append(packed)
    return results, {
        "accept_rate": (swaps / attempts) if attempts else float("nan"),
        "attempts": attempts,
    }
