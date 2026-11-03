"""Independent-population forward walking with cross-replica overlap moments.

Four populations branch independently. Products use disjoint populations, never
the square of a noisy sample covariance. Population and forward biases remain
subject to explicit convergence checks.
"""
import numpy as np

from .projector import propagate, rates_and_local


def cross_connected(magnetization, pair):
    """Symmetrize three disjoint pairings, each with expectation C_ij squared."""
    m, q = np.asarray(magnetization), np.asarray(pair)
    if m.ndim != 2 or m.shape[0] != 4 or q.shape != (4, m.shape[1], m.shape[1]):
        raise ValueError('Expected four independent moment estimates')
    def covariance(a, b):
        return .5 * (q[a] + q[b] - np.outer(m[a], m[b]) - np.outer(m[b], m[a]))
    return sum(covariance(a, b) * covariance(c, d)
               for a, b, c, d in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]) / 3


def spectrum_from_moments(magnetization, pair, lattice):
    L = lattice['L']
    cells = np.array([[2*x+y, np.sqrt(3)*y] for y in range(L) for x in range(L)])
    offsets = np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3)/2]])
    points = (cells[:, None] + offsets[None]).reshape(3*L*L, 2)
    phases = np.exp(-1j * (points @ lattice['momenta'].T))
    squared = cross_connected(magnetization, pair)
    return np.einsum('ik,ij,jk->k', phases, squared, phases.conj()).real / len(points)


def project_independent(ham, gamma, guide, lattice, *, n_walkers=1024,
                        tau_block=.1, n_equilibrate=80, n_measure=10,
                        tau_forward=8., n_cycles=16, seed=0, start='guide'):
    if n_walkers < 2 or n_cycles < 2 or tau_block <= 0 or tau_forward <= 0:
        raise ValueError('Invalid independent-projector settings')
    if start not in ('guide', 'uniform'):
        raise ValueError('Unknown initialization')
    streams = np.random.SeedSequence(seed).spawn(4)
    moments, pairs, ess_all, largest_all, unique_all, energies, weights = [], [], [], [], [], [], []
    for stream in streams:
        rng = np.random.default_rng(stream)
        if start == 'guide':
            states = guide.sample(n_walkers, int(rng.integers(2**32)))
        else:
            states = (1 - 2*rng.integers(2, size=(n_walkers, ham.n_sites))).astype(np.int8)
        states = states[:, None, :]
        _, local = rates_and_local(ham, gamma, guide, states)
        shift = float(local.mean())
        energy, weight = [], []
        for step in range(n_equilibrate + n_measure):
            states, _, shift, mixed, _, ess = propagate(
                ham, gamma, guide, states, shift, tau_block, rng)
            if step >= n_equilibrate:
                energy.append(mixed); weight.append(ess)
        m, q, kish, largest, unique = [], [], [], [], []
        for cycle in range(n_cycles):
            tagged = states[:, 0, :].copy().astype(float)
            ancestors = np.arange(n_walkers)
            elapsed = 0.
            while elapsed < tau_forward - 1e-12:
                dt = min(tau_block, tau_forward - elapsed)
                states, children, shift, _, _, _ = propagate(
                    ham, gamma, guide, states, shift, dt, rng)
                ancestors = ancestors[children]
                elapsed += dt
            counts = np.bincount(ancestors, minlength=n_walkers)
            probability = counts / n_walkers
            m.append(probability @ tagged)
            q.append(tagged.T @ (probability[:, None] * tagged))
            kish.append(float(n_walkers**2 / (counts @ counts)))
            largest.append(float(counts.max() / n_walkers))
            unique.append(int(np.count_nonzero(counts)))
        moments.append(m); pairs.append(q); ess_all.append(kish)
        largest_all.append(largest); unique_all.append(unique)
        energies.append(energy); weights.append(weight)
    moments, pairs = np.asarray(moments), np.asarray(pairs)
    spectrum = spectrum_from_moments(moments.mean(axis=1), pairs.mean(axis=1), lattice)
    return dict(connected_spectrum=spectrum.tolist(), chi_connected=float(spectrum[0]),
                magnetization_cycles=moments.tolist(), pair_cycles=pairs.tolist(),
                descendant_ess=ess_all, largest_lineage=largest_all, unique_ancestors=unique_all,
                energy=float(np.mean(energies)), energy_trace=energies, weight_ess=weights,
                ancestry_alarm=bool(np.min(ess_all) < 8 or np.max(largest_all) > .5),
                estimator='independent_populations_cross_moments', start=start,
                seed=int(seed), n_walkers=n_walkers, n_replicas=4)
