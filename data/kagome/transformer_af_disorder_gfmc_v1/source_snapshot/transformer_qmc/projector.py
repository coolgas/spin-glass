"""Batched continuous-time guided projection with two/four-replica pure tags.

Holding times are exact exponential draws. Branching occurs at tau_block;
population/cadence and forward-time biases still require convergence studies.
Forward cycles always finish: poor ancestry is reported, never selected away.
"""

import numpy as np

from qmc_numerics import systematic
from .physics import diagonal


def rates_and_local(ham, gamma, guide, states):
    shape = states.shape
    ratios = guide.all_flip_log_ratios(states.reshape(-1, ham.n_sites))
    rates = gamma * np.exp(ratios).reshape(shape)
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise FloatingPointError("guide rates must be finite and strictly positive")
    local = diagonal(ham, states).sum(axis=1) - rates.sum(axis=(1, 2))
    return rates, local


def propagate(ham, gamma, guide, states, shift, duration, rng):
    """Advance all walkers independently, then systematic population resampling."""
    count, replicas, sites = states.shape
    states = states.copy()
    remaining = np.full(count, duration)
    weights = np.ones(count)
    energy_integral = np.zeros(count)
    time_integral = np.zeros(count)
    while True:
        active = np.flatnonzero(remaining > 1e-14)
        if not active.size:
            break
        rates, local = rates_and_local(ham, gamma, guide, states[active])
        flattened = rates.reshape(len(active), -1)
        escape = flattened.sum(axis=1)
        waiting = rng.exponential(1 / escape)
        dt = np.minimum(waiting, remaining[active])
        exponent = -(local - shift) * dt
        if np.any(exponent > 700):
            raise FloatingPointError("weight overflow: reduce branch interval")
        integral_factor = np.ones_like(exponent)
        nonzero = np.abs(exponent) > 1e-8
        integral_factor[nonzero] = np.expm1(exponent[nonzero]) / exponent[nonzero]
        integral_factor[~nonzero] = 1 + exponent[~nonzero] / 2
        piece = weights[active] * dt * integral_factor
        time_integral[active] += piece
        energy_integral[active] += piece * local
        weights[active] *= np.exp(exponent)
        hopping = waiting < remaining[active]
        if np.any(hopping):
            cumulative = np.cumsum(flattened[hopping], axis=1)
            draw = rng.random(hopping.sum()) * escape[hopping]
            choice = (cumulative < draw[:, None]).sum(axis=1)
            which = active[hopping]
            states[which, choice // sites, choice % sites] *= -1
        remaining[active] -= dt
    children, log_total = systematic(weights, rng)
    growth_pair = shift - (log_total - np.log(count)) / duration
    scaled = weights / weights.max()
    ess = scaled.sum()**2 / (scaled @ scaled)
    mixed = energy_integral.sum() / time_integral.sum() / replicas
    return (states[children], children, 0.5 * (shift + growth_pair),
            mixed, growth_pair / replicas, float(ess))


def tags(states, phase=None):
    q = (states[:, 0] * states[:, 1]).mean(axis=1)
    result = dict(q=q, q2=q*q, q4=q**4)
    if states.shape[1] == 4:
        # E[d_i d_j]=C_ij² for four independent physical replicas.
        d = 0.5 * (states[:, 0] - states[:, 1]) * (states[:, 2] - states[:, 3])
        result["chi_connected"] = d.sum(axis=1)**2 / states.shape[-1]
        if phase is not None:
            for k in range(phase.shape[1]):
                result[f"sg_k{k}"] = np.abs(d @ phase[:, k])**2 / states.shape[-1]
    return result


def project(ham, gamma, guide, *, n_walkers=512, n_replicas=2,
            tau_block=0.1, n_equilibrate=40, n_measure=40,
            tau_forward=1.0, n_cycles=8, seed=0, phase=None):
    if (n_replicas not in (2, 4) or not isinstance(n_walkers, int) or n_walkers < 2
            or not np.isfinite(gamma) or gamma <= 0
            or not np.isfinite(tau_block) or tau_block <= 0
            or not np.isfinite(tau_forward) or tau_forward <= 0
            or n_equilibrate < 0 or n_measure < 2 or n_cycles < 2):
        raise ValueError("invalid projector settings")
    rng = np.random.default_rng(seed)
    states = guide.sample(n_walkers * n_replicas, int(rng.integers(2**32)))
    states = states.reshape(n_walkers, n_replicas, ham.n_sites)
    _, initial_local = rates_and_local(ham, gamma, guide, states)
    shift = float(initial_local.mean())
    equilibration, energy, growth, weight_ess = [], [], [], []
    for step in range(n_equilibrate + n_measure):
        states, _, shift, mixed, eg, ess = propagate(
            ham, gamma, guide, states, shift, tau_block, rng)
        if step < n_equilibrate:
            equilibration.append(mixed)
        else:
            energy.append(mixed)
            growth.append(eg)
            weight_ess.append(ess)
    cycles, kish, unique, largest = {}, [], [], []
    for _ in range(n_cycles):
        tagged = tags(states, phase)
        ancestors = np.arange(n_walkers)
        elapsed = 0.0
        while elapsed < tau_forward - 1e-12:
            dt = min(tau_block, tau_forward - elapsed)
            states, children, shift, _, _, _ = propagate(ham, gamma, guide, states, shift, dt, rng)
            ancestors = ancestors[children]
            elapsed += dt
        for name, values in tagged.items():
            cycles.setdefault(name, []).append(float(values[ancestors].mean()))
        counts = np.bincount(ancestors, minlength=n_walkers)
        kish.append(float(n_walkers**2 / (counts @ counts)))
        unique.append(int(np.count_nonzero(counts)))
        largest.append(float(counts.max() / n_walkers))
    result = dict(energy=float(np.mean(energy)), energy_growth=float(np.mean(growth)),
                  energy_trace=np.asarray(energy), growth_trace=np.asarray(growth),
                  equilibration_trace=np.asarray(equilibration), weight_ess=np.asarray(weight_ess),
                  descendant_ess=np.asarray(kish), unique_ancestors=np.asarray(unique),
                  largest_lineage=np.asarray(largest), tau_forward=float(tau_forward),
                  forward_times=np.full(n_cycles, tau_forward),
                  ancestry_healthy=bool(min(kish) >= 8 and min(unique) >= 2),
                  n_walkers=n_walkers, n_replicas=n_replicas, seed=int(seed),
                  n_equilibrate=n_equilibrate, n_measure=n_measure, tau_block=tau_block)
    for name, values in cycles.items():
        result[name] = float(np.mean(values))
        result[name + "_cycles"] = np.asarray(values)
    return result
