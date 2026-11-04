"""Batched continuous-time guided projection with two/four-replica pure tags.

Holding times are exact exponential draws. Branching occurs at tau_block;
population/cadence and forward-time biases still require convergence studies.
Forward cycles always finish: poor ancestry is reported, never selected away.
"""

import numpy as np

from qmc_numerics import systematic_log, log_expm1_ratio
from .physics import diagonal


def rates_and_local(ham, gamma, guide, states):
    shape = states.shape
    ratios = guide.all_flip_log_ratios(states.reshape(-1, ham.n_sites))
    if not np.isfinite(ratios).all():
        raise FloatingPointError('nonfinite guide log flip ratios')
    with np.errstate(over='ignore', under='ignore'):
        rates = gamma * np.exp(ratios).reshape(shape)
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise FloatingPointError("guide rates must be finite and strictly positive; "
                                 f"log-ratio range [{ratios.min()}, {ratios.max()}]")
    local = diagonal(ham, states).sum(axis=1) - rates.sum(axis=(1, 2))
    if not np.isfinite(local).all():
        raise FloatingPointError('nonfinite local energy or total escape rate')
    return rates, local


def propagate(ham, gamma, guide, states, shift, duration, rng, diagnostics=None):
    """Exact event times with log-domain branching and interval integrals.

    The returned estimators retain the original weighted-time convention.
    Numerical stabilization does not cure poor guides or population bias.
    """
    if not np.isfinite(shift) or not np.isfinite(duration) or duration <= 0:
        raise ValueError('finite shift and positive finite duration required')
    count, replicas, sites = states.shape
    states = states.copy()
    remaining = np.full(count, duration)
    log_weights = np.zeros(count)
    log_time_integral = np.full(count, -np.inf)
    interval_energy = np.zeros(count)
    extrema = dict(min_local=np.inf, max_local=-np.inf,
                   min_increment=np.inf, max_increment=-np.inf, events=0)
    while True:
        active = np.flatnonzero(remaining > 1e-14)
        if not active.size:
            break
        rates, local = rates_and_local(ham, gamma, guide, states[active])
        flattened = rates.reshape(len(active), -1)
        escape = flattened.sum(axis=1)
        waiting = rng.exponential(1 / escape)
        dt = np.minimum(waiting, remaining[active])
        if np.any(dt <= 0) or not np.isfinite(dt).all():
            raise FloatingPointError('nonpositive or nonfinite event interval')
        exponent = -(local - shift) * dt
        log_piece = log_weights[active] + np.log(dt) + log_expm1_ratio(exponent)
        combined = np.logaddexp(log_time_integral[active], log_piece)
        interval_energy[active] = (
            np.exp(log_time_integral[active] - combined) * interval_energy[active]
            + np.exp(log_piece - combined) * local)
        log_time_integral[active] = combined
        log_weights[active] += exponent
        if not np.isfinite(log_weights[active]).all() or not np.isfinite(combined).all():
            raise FloatingPointError('nonfinite accumulated log weight or log time integral')
        hopping = waiting < remaining[active]
        if diagnostics is not None:
            extrema['min_local'] = min(extrema['min_local'], float(local.min()))
            extrema['max_local'] = max(extrema['max_local'], float(local.max()))
            extrema['min_increment'] = min(extrema['min_increment'], float(exponent.min()))
            extrema['max_increment'] = max(extrema['max_increment'], float(exponent.max()))
            extrema['events'] += int(hopping.sum())
        if np.any(hopping):
            cumulative = np.cumsum(flattened[hopping], axis=1)
            draw = rng.random(hopping.sum()) * escape[hopping]
            choice = (cumulative < draw[:, None]).sum(axis=1)
            which = active[hopping]
            states[which, choice // sites, choice % sites] *= -1
        remaining[active] -= dt
    children, log_total = systematic_log(log_weights, rng)
    growth_pair = shift - (log_total - np.log(count)) / duration
    scaled = np.exp(log_weights - log_weights.max())
    ess = scaled.sum()**2 / (scaled @ scaled)
    time_scaled = np.exp(log_time_integral - log_time_integral.max())
    mixed = (time_scaled @ interval_energy) / time_scaled.sum() / replicas
    if not np.isfinite(mixed) or not np.isfinite(growth_pair):
        raise FloatingPointError('nonfinite mixed or growth energy after log accumulation')
    if diagnostics is not None:
        diagnostics.update(extrema, min_log_weight=float(log_weights.min()),
                           max_log_weight=float(log_weights.max()), weight_ess=float(ess))
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
            tau_forward=1.0, n_cycles=8, seed=0, phase=None,
            resume=None, checkpoint=None, tagger=None):
    if (n_replicas not in (2, 4) or not isinstance(n_walkers, int) or n_walkers < 2
            or not np.isfinite(gamma) or gamma <= 0
            or not np.isfinite(tau_block) or tau_block <= 0
            or not np.isfinite(tau_forward) or tau_forward <= 0
            or n_equilibrate < 0 or n_measure < 2 or n_cycles < 2):
        raise ValueError("invalid projector settings")
    configuration = dict(n_sites=ham.n_sites, gamma=gamma, n_walkers=n_walkers,
                         numerics='log_weight_v1',
                         n_replicas=n_replicas, tau_block=tau_block,
                         n_equilibrate=n_equilibrate, n_measure=n_measure,
                         tau_forward=tau_forward, n_cycles=n_cycles, seed=int(seed))
    rng = np.random.default_rng(seed)
    if resume is None:
        states = guide.sample(n_walkers * n_replicas, int(rng.integers(2**32)))
        states = states.reshape(n_walkers, n_replicas, ham.n_sites)
        _, initial_local = rates_and_local(ham, gamma, guide, states)
        shift = float(initial_local.mean())
        equilibration, energy, growth, weight_ess = [], [], [], []
        cycles, kish, unique, largest = {}, [], [], []
        next_step, cycle, elapsed = 0, 0, 0.0
        tagged, ancestors = None, None
    else:
        if resume.get('format') != 1 or resume['configuration'] != configuration:
            raise ValueError('projector checkpoint configuration mismatch')
        states = np.asarray(resume['states'], dtype=np.int8)
        if states.shape != (n_walkers, n_replicas, ham.n_sites) or not np.isin(states, [-1, 1]).all():
            raise ValueError('invalid checkpoint spins')
        rng.bit_generator.state = resume['rng']
        shift = float(resume['shift'])
        equilibration, energy, growth, weight_ess = (list(resume[k]) for k in
                                                    ('equilibration', 'energy', 'growth', 'weight_ess'))
        cycles = {k: list(v) for k, v in resume['cycles'].items()}
        kish, unique, largest = (list(resume[k]) for k in ('kish', 'unique', 'largest'))
        next_step, cycle, elapsed = resume['next_step'], resume['cycle'], resume['elapsed']
        tagged = None if resume['tagged'] is None else {k: np.asarray(v) for k, v in resume['tagged'].items()}
        ancestors = None if resume['ancestors'] is None else np.asarray(resume['ancestors'], dtype=np.int64)

    def save():
        if checkpoint is not None:
            checkpoint(dict(format=1, configuration=configuration, states=states,
                            rng=rng.bit_generator.state, shift=shift, next_step=next_step,
                            cycle=cycle, elapsed=elapsed, tagged=tagged, ancestors=ancestors,
                            equilibration=equilibration, energy=energy, growth=growth,
                            weight_ess=weight_ess, cycles=cycles, kish=kish, unique=unique, largest=largest))

    save()
    for step in range(next_step, n_equilibrate + n_measure):
        states, _, shift, mixed, eg, ess = propagate(
            ham, gamma, guide, states, shift, tau_block, rng)
        if step < n_equilibrate:
            equilibration.append(mixed)
        else:
            energy.append(mixed)
            growth.append(eg)
            weight_ess.append(ess)
        next_step = step + 1
        save()
    while cycle < n_cycles:
        if tagged is None:
            tagged = tags(states, phase) if tagger is None else tagger(states)
            ancestors = np.arange(n_walkers)
            elapsed = 0.0
        while elapsed < tau_forward - 1e-12:
            dt = min(tau_block, tau_forward - elapsed)
            states, children, shift, _, _, _ = propagate(ham, gamma, guide, states, shift, dt, rng)
            ancestors = ancestors[children]
            elapsed += dt
            save()
        for name, values in tagged.items():
            mean = values[ancestors].mean(axis=0)
            cycles.setdefault(name, []).append(np.asarray(mean).tolist())
        counts = np.bincount(ancestors, minlength=n_walkers)
        kish.append(float(n_walkers**2 / (counts @ counts)))
        unique.append(int(np.count_nonzero(counts)))
        largest.append(float(counts.max() / n_walkers))
        cycle += 1
        tagged, ancestors, elapsed = None, None, 0.0
        save()
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
        result[name] = np.asarray(values).mean(axis=0).tolist()
        result[name + "_cycles"] = np.asarray(values)
    return result
