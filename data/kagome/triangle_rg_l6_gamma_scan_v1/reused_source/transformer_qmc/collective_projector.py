"""Positive finite-step GFMC with corrected collective endpoint proposals.

The production step uses only a log-amplitude callable, no state enumeration.
An optional N<=16 guide table accelerates the L2 development benchmark.
"""
import numpy as np
from scipy.special import gammaln, logsumexp

from .physics import diagonal


def indexes(states):
    return ((1 - np.asarray(states, dtype=np.int64)) // 2) @ (1 << np.arange(states.shape[-1]))


class SplitProposal:
    """Mixture of guided site-factor sampling and a broad subset proposal.

    Each site is visited once in fixed order. The product of guided two-state
    factors equals the importance-sampled transverse exponential. Broad moves
    are weighted using the *mixture endpoint probability*, not branch probability.
    """

    def __init__(self, ham, gamma, guide, delta=.05, epsilon=0., cache=False, natural=False):
        if gamma <= 0 or delta <= 0 or not 0 <= epsilon < 1:
            raise ValueError('Invalid split proposal settings')
        self.ham, self.gamma, self.guide = ham, gamma, guide
        self.delta, self.epsilon = delta, epsilon
        self.n = ham.n_sites
        self.natural = natural
        self.log_tanh = np.log(np.tanh(gamma * delta))
        self.log_cosh = np.log(np.cosh(gamma * delta))
        x = np.arange(self.n + 1)
        self.log_choose = gammaln(self.n + 1) - gammaln(x + 1) - gammaln(self.n - x + 1)
        self.table = None
        if cache:
            if self.n > 16 or not hasattr(guide, 'logs'):
                raise ValueError('Table acceleration requires an enumerated small-system guide')
            ix = np.arange(1 << self.n)
            logits = (np.full((len(ix), self.n), self.log_tanh) if natural else
                      self.log_tanh + guide.logs[ix[:, None] ^ (1 << np.arange(self.n))] - guide.logs[:, None])
            self.table = -np.logaddexp(0., -logits), -np.logaddexp(0., logits)

    def guided(self, states, rng, destination=None):
        """Sample the sequential proposal, or evaluate its endpoint probability."""
        if self.natural:
            probability = np.exp(self.log_tanh)/(1+np.exp(self.log_tanh))
            take = (rng.random(states.shape)<probability if destination is None
                    else states!=destination)
            current = states*np.where(take,-1,1)
            count = take.sum(1)
            return current, count*np.log(probability)+(self.n-count)*np.log1p(-probability)
        current = states.copy()
        log_q = np.zeros(len(states))
        if self.table is not None:
            ix = indexes(current)
        else:
            logs = self.guide.log_amplitudes(current)
        for site in range(self.n):
            if self.table is not None:
                log_flip, log_stay = (part[ix, site] for part in self.table)
            else:
                flipped = current.copy(); flipped[:, site] *= -1
                proposed_logs = self.guide.log_amplitudes(flipped)
                logits = (np.full(len(states), self.log_tanh) if self.natural else
                          self.log_tanh + proposed_logs - logs)
                log_flip, log_stay = -np.logaddexp(0., -logits), -np.logaddexp(0., logits)
            take = (rng.random(len(states)) < np.exp(log_flip) if destination is None
                    else current[:, site] != destination[:, site])
            log_q += np.where(take, log_flip, log_stay)
            current[take, site] *= -1
            if self.table is not None:
                ix[take] ^= 1 << site
            else:
                logs = np.where(take, proposed_logs, logs)
        return current, log_q

    def draw(self, states, rng):
        proposed, log_guided = self.guided(states, rng)
        broad = rng.random(len(states)) < self.epsilon
        if broad.any():
            rows = np.flatnonzero(broad)
            # Uniform count 1..N, uniform subset conditional on count.
            count = rng.integers(1, self.n + 1, size=len(rows))
            order = np.argsort(rng.random((len(rows), self.n)), axis=1)
            mask = np.zeros((len(rows), self.n), dtype=bool)
            for j in range(self.n):
                active = count > j
                mask[np.flatnonzero(active), order[active, j]] = True
            proposed[rows] = states[rows] * np.where(mask, -1, 1)
            _, log_guided[rows] = self.guided(states[rows], rng, destination=proposed[rows])
        count = np.count_nonzero(proposed != states, axis=1)
        log_q = log_guided
        if self.epsilon:
            log_broad = np.full(len(states), -np.inf)
            nonzero = count > 0
            log_broad[nonzero] = -np.log(self.n) - self.log_choose[count[nonzero]]
            log_q = np.logaddexp(np.log1p(-self.epsilon) + log_guided,
                                 np.log(self.epsilon) + log_broad)
        log_kinetic = self.n * self.log_cosh + count * self.log_tanh
        log_weight = (-self.delta * (diagonal(self.ham, states) + diagonal(self.ham, proposed)) / 2
                      + log_kinetic + self.guide.log_amplitudes(proposed)
                      - self.guide.log_amplitudes(states) - log_q)
        if not np.isfinite(log_weight).all():
            raise FloatingPointError('Nonfinite split-propagator weights')
        return proposed, log_weight, count

    def step(self, states, rng):
        """Independent resampling in each physical replica; return ancestry map."""
        replicas, walkers, n = states.shape
        proposed, log_w, count = self.draw(states.reshape(-1, n), rng)
        proposed = proposed.reshape(states.shape)
        log_w = log_w.reshape(replicas, walkers)
        count = count.reshape(replicas, walkers)
        weights = np.exp(log_w - logsumexp(log_w, axis=1)[:, None])
        parents = np.empty((replicas, walkers), dtype=np.int64)
        for r in range(replicas):
            cumulative = np.cumsum(weights[r]); cumulative[-1] = 1.
            parents[r] = np.searchsorted(cumulative, (rng.random() + np.arange(walkers)) / walkers)
        children = proposed[np.arange(replicas)[:, None], parents]
        stats = dict(ess=1 / (weights * weights).sum(1),
                     proposed=np.bincount(count.ravel(), minlength=n + 1),
                     surviving=np.bincount(count[np.arange(replicas)[:, None], parents].ravel(), minlength=n + 1),
                     log_mean_weight=logsumexp(log_w, axis=1) - np.log(walkers))
        return children, parents, stats
