"""Hamiltonian-aware, positive three-spin propagators for periodic Kagome GFMC.

The two triangle layers give H=A+B exactly. Each bond belongs to one triangle;
each site's longitudinal and transverse terms are divided between its two
triangles. The step is exp(-d A/2) exp(-d B) exp(-d A/2). It has O(d**3)
local error, not an additional collective term in the physical Hamiltonian.
"""
from itertools import combinations

import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp

from kagome_qea.lattice import build_kagome
from .collective_projector import SplitProposal, indexes


LOCAL_SPINS = (1 - 2 * ((np.arange(8)[:, None] >> np.arange(3)) & 1)).astype(np.int8)


def triangle_layers(ham):
    """Find the two disjoint layers, and check the complete bond/site cover."""
    size = int(round(np.sqrt(ham.n_sites / 3)))
    if 3 * size * size != ham.n_sites or size < 2:
        raise ValueError('Triangle splitting requires a periodic L>=2 Kagome lattice')
    _, bonds, triangles, _ = build_kagome(size, periodic=True)
    if set(map(tuple, np.sort(ham.bonds, axis=1))) != set(map(tuple, bonds)):
        raise ValueError('Hamiltonian bonds do not match the periodic Kagome geometry')
    memberships = [[] for _ in range(ham.n_sites)]
    for t, sites in enumerate(triangles):
        for site in sites:
            memberships[site].append(t)
    if any(len(ts) != 2 for ts in memberships):
        raise ValueError('Each Kagome site must belong to exactly two triangles')
    colors = {}
    for first in range(len(triangles)):
        if first in colors:
            continue
        colors[first] = 0
        pending = [first]
        while pending:
            t = pending.pop()
            for site in triangles[t]:
                for neighbor in memberships[site]:
                    if neighbor == t:
                        continue
                    if neighbor in colors:
                        if colors[neighbor] == colors[t]:
                            raise ValueError('Triangle adjacency is not bipartite')
                    else:
                        colors[neighbor] = 1 - colors[t]
                        pending.append(neighbor)
    layers = [[tuple(t) for i, t in enumerate(triangles) if colors[i] == color]
              for color in [0, 1]]
    covered = [tuple(sorted(edge)) for t in triangles for edge in combinations(t, 2)]
    if len(covered) != len(set(covered)) or set(covered) != set(map(tuple, bonds)):
        raise ValueError('Triangles must cover every bond exactly once')
    for layer in layers:
        if sorted(site for t in layer for site in t) != list(range(ham.n_sites)):
            raise ValueError('Each triangle layer must cover the sites once')
    return layers


def local_hamiltonian(ham, gamma, sites):
    """8x8 triangle block, including half of each on-site field."""
    coupling = {tuple(sorted(edge)): j for edge, j in zip(ham.bonds, ham.J)}
    potential = -ham.h * LOCAL_SPINS.sum(1) / 2
    for a, b in combinations(range(3), 2):
        potential -= coupling[tuple(sorted((sites[a], sites[b])))] * LOCAL_SPINS[:, a] * LOCAL_SPINS[:, b]
    matrix = np.diag(potential)
    for source in range(8):
        for site in range(3):
            matrix[source ^ (1 << site), source] = -gamma / 2
    return matrix


class TriangleProposal(SplitProposal):
    """Sample a locally guide-adapted path through exact triangle factors.

    For a factor K and current x, q(y|x)=K[y,x] psi(y)/Z(x), where the
    environment outside the triangle is held fixed. Its importance weight is
    Z(x)/psi(x). Multiplying these weights is the exact path correction;
    summing over intermediate paths reconstructs the split operator. No
    endpoint-probability approximation or Metropolis acceptance is used.

    Default execution evaluates only eight local candidates per factor. The
    optional N<=16 cache is strictly a small-system benchmark accelerator.
    """

    def __init__(self, ham, gamma, guide, delta=.1, cache=False):
        if not np.isfinite([gamma, delta]).all() or gamma <= 0 or delta <= 0:
            raise ValueError('gamma and delta must be finite and positive')
        self.ham, self.gamma, self.guide, self.delta = ham, gamma, guide, delta
        self.n = ham.n_sites
        self.layers = triangle_layers(ham)
        self.factors = []
        self.tables = {}
        if cache and (self.n > 16 or not hasattr(guide, 'logs')):
            raise ValueError('Table acceleration requires an enumerated N<=16 guide')
        for layer, duration in [(self.layers[0], delta / 2), (self.layers[1], delta),
                                (list(reversed(self.layers[0])), delta / 2)]:
            for triangle in layer:
                sites = np.asarray(triangle)
                kernel = expm(-duration * local_hamiltonian(ham, gamma, sites))
                if not (np.isfinite(kernel).all() and (kernel > 0).all()):
                    raise FloatingPointError('Triangle exponential lost strict positivity')
                self.factors.append((sites, kernel))
                key = tuple(triangle), duration
                if cache and key not in self.tables:
                    ix = np.arange(1 << self.n)
                    local = ((ix[:, None] >> sites) & 1) @ (1 << np.arange(3))
                    mask = int(np.sum(1 << sites))
                    candidates = (ix[:, None] & ~mask) | ((np.arange(8)[:, None] >> np.arange(3) & 1) @ (1 << sites))
                    logits = np.log(kernel[:, local].T) + guide.logs[candidates] - guide.logs[ix, None]
                    log_weight = logsumexp(logits, axis=1)
                    self.tables[key] = np.exp(logits - log_weight[:, None]), log_weight, candidates
        self.cache = cache

    def draw(self, states, rng):
        current = states.copy()
        log_weight = np.zeros(len(states))
        ix = indexes(current) if self.cache else None
        logs = None if self.cache else self.guide.log_amplitudes(current)
        for factor, (sites, kernel) in enumerate(self.factors):
            duration = self.delta if len(self.layers[0]) <= factor < len(self.layers[0]) + len(self.layers[1]) else self.delta / 2
            if self.cache:
                probabilities, weights, candidates = self.tables[(tuple(sites), duration)]
                p = probabilities[ix]
                increment = weights[ix]
            else:
                local = ((1 - current[:, sites]) // 2) @ (1 << np.arange(3))
                candidates = np.repeat(current[:, None, :], 8, axis=1)
                candidates[:, :, sites] = LOCAL_SPINS
                candidate_logs = self.guide.log_amplitudes(candidates.reshape(-1, self.n)).reshape(-1, 8)
                logits = np.log(kernel[:, local].T) + candidate_logs - logs[:, None]
                increment = logsumexp(logits, axis=1)
                p = np.exp(logits - increment[:, None])
            cumulative = np.cumsum(p, axis=1)
            cumulative[:, -1] = 1.
            choice = np.sum(rng.random(len(states))[:, None] > cumulative, axis=1)
            rows = np.arange(len(states))
            if self.cache:
                ix = candidates[ix, choice]
                current[:, sites] = LOCAL_SPINS[choice]
            else:
                current = candidates[rows, choice]
                logs = candidate_logs[rows, choice]
            log_weight += increment
        if not np.isfinite(log_weight).all():
            raise FloatingPointError('Nonfinite triangle path weights')
        return current, log_weight, np.count_nonzero(current != states, axis=1)

    def action(self, vector):
        """Enumerated operator action, for independent reference evaluation only."""
        if self.n > 16:
            raise ValueError('Exact action is restricted to N<=16')
        result = np.asarray(vector, dtype=float).copy()
        ix = np.arange(1 << self.n)
        for sites, kernel in self.factors:
            mask = int(np.sum(1 << sites))
            outside = ix[(ix & mask) == 0]
            offsets = ((np.arange(8)[:, None] >> np.arange(3)) & 1) @ (1 << sites)
            groups = outside[:, None] | offsets
            result[groups] = result[groups] @ kernel.T
        return result
