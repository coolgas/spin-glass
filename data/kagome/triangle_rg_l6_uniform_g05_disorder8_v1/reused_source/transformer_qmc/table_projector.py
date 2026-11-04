"""Continuous-time GFMC in indexed configurations for small-system validation.

Rates contain only constant or frozen-network amplitudes. Enumerated exact
ground states are never passed here. This is the same event/weight/resampling
algorithm as projector.py, with rates cached once rather than per event.
"""

import numpy as np

from qmc_numerics import systematic
from .physics import basis, diagonal
from .projector import tags


class RateTable:
    def __init__(self, ham, gamma, logs):
        self.n_sites = ham.n_sites
        self.spins = basis(ham.n_sites)
        self.logs = np.asarray(logs, dtype=float)
        index = np.arange(len(self.spins))
        flip = index[:, None] ^ (1 << np.arange(ham.n_sites))
        rates = gamma * np.exp(self.logs[flip]-self.logs[:, None])
        if not np.isfinite(rates).all() or np.any(rates <= 0):
            raise ValueError('invalid cached rates')
        self.cumulative = rates.cumsum(axis=1)
        self.escape = self.cumulative[:, -1]
        self.local = diagonal(ham, self.spins)-self.escape
        p = np.exp(2*(self.logs-self.logs.max()))
        self.probability = p/p.sum()


def propagate(table, states, shift, duration, rng):
    count, replicas = states.shape
    states = states.copy()
    remaining = np.full(count, duration)
    weights = np.ones(count)
    energy_integral = np.zeros(count)
    time_integral = np.zeros(count)
    while True:
        active = np.flatnonzero(remaining > 1e-14)
        if not active.size:
            break
        current = states[active]
        local = table.local[current].sum(axis=1)
        replica_escape = table.escape[current]
        cumulative = replica_escape.cumsum(axis=1)
        escape = cumulative[:, -1]
        waiting = rng.exponential(1/escape)
        dt = np.minimum(waiting, remaining[active])
        exponent = -(local-shift)*dt
        if np.any(exponent > 700):
            raise FloatingPointError('weight overflow: reduce branch interval')
        factor = np.ones_like(exponent)
        nonzero = np.abs(exponent) > 1e-8
        factor[nonzero] = np.expm1(exponent[nonzero])/exponent[nonzero]
        factor[~nonzero] = 1+exponent[~nonzero]/2
        piece = weights[active]*dt*factor
        time_integral[active] += piece
        energy_integral[active] += piece*local
        weights[active] *= np.exp(exponent)
        hopping = waiting < remaining[active]
        if hopping.any():
            draw = rng.random(hopping.sum())*escape[hopping]
            replica = (cumulative[hopping] < draw[:, None]).sum(axis=1)
            offset = np.where(replica > 0,
                              cumulative[hopping][np.arange(len(draw)), np.maximum(replica-1, 0)], 0.)
            which = active[hopping]
            current_index = states[which, replica]
            site = (table.cumulative[current_index] < (draw-offset)[:, None]).sum(axis=1)
            states[which, replica] ^= 1 << site
        remaining[active] -= dt
    children, log_total = systematic(weights, rng)
    growth = shift-(log_total-np.log(count))/duration
    scaled = weights/weights.max()
    ess = scaled.sum()**2/(scaled@scaled)
    mixed = energy_integral.sum()/time_integral.sum()/replicas
    return states[children], children, .5*(shift+growth), mixed, growth/replicas, float(ess)


def project(table, *, walkers, equilibration, measurement, forward, cycles, branch, seed):
    rng = np.random.default_rng(seed)
    states = rng.choice(len(table.spins), size=(walkers, 4), p=table.probability)
    shift = float(table.local[states].sum(axis=1).mean())
    energy, growth, weight_ess, equil = [], [], [], []
    for step in range(round((equilibration+measurement)/branch)):
        states, _, shift, mixed, eg, ess = propagate(table, states, shift, branch, rng)
        if step < round(equilibration/branch):
            equil.append(mixed)
        else:
            energy.append(mixed)
            growth.append(eg)
            weight_ess.append(ess)
    values, kish, unique, largest = {}, [], [], []
    for _ in range(cycles):
        tagged = tags(table.spins[states])
        ancestors = np.arange(walkers)
        elapsed = 0.
        while elapsed < forward-1e-12:
            dt = min(branch, forward-elapsed)
            states, children, shift, _, _, _ = propagate(table, states, shift, dt, rng)
            ancestors = ancestors[children]
            elapsed += dt
        for name, v in tagged.items():
            values.setdefault(name, []).append(float(v[ancestors].mean()))
        counts = np.bincount(ancestors, minlength=walkers)
        kish.append(float(walkers**2/(counts@counts)))
        unique.append(int(np.count_nonzero(counts)))
        largest.append(float(counts.max()/walkers))
    return dict(energy=float(np.mean(energy)), energy_growth=float(np.mean(growth)),
                energy_trace=energy, growth_trace=growth, equilibration_trace=equil,
                weight_ess=weight_ess, descendant_ess=kish, unique_ancestors=unique,
                largest_lineage=largest, **{k:float(np.mean(v)) for k,v in values.items()},
                **{k+'_cycles':v for k,v in values.items()})
