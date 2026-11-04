"""Independent algebra and numerical controls for the correlation remedy."""
from itertools import product
import unittest

import numpy as np

from kagome_qea.exact import operator_matrix
from transformer_qmc.model import EnumeratedGuide
from transformer_qmc.physics import basis, make_af_sample
from transformer_qmc.production_protocol import exact_spectra, geometry
from transformer_qmc.refine_correlation_guide import energy_gradient, features, refine
from transformer_qmc.independent_replica_projector import cross_connected, spectrum_from_moments
from transformer_qmc.projector import propagate, rates_and_local


class CorrelationRemedyTests(unittest.TestCase):
    def test_disjoint_moment_identity_for_unequal_distributions(self):
        spins = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
        rng = np.random.default_rng(902)
        p = rng.dirichlet(np.ones(4), size=4)
        m = p @ spins
        q = np.einsum('rx,xi,xj->rij', p, spins, spins)
        expected = np.zeros((2, 2))
        for ix in product(range(4), repeat=4):
            probability = np.prod([p[r, ix[r]] for r in range(4)])
            s = spins[list(ix)]
            for a, b, c, d in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
                tag = .5 * (s[a]-s[b]) * (s[c]-s[d])
                expected += probability * np.outer(tag, tag) / 3
        np.testing.assert_allclose(cross_connected(m, q), expected, atol=1e-14)

    def test_fourier_spectrum_matches_independent_exact_contraction(self):
        spins = basis(12).astype(float)
        p = np.random.default_rng(41).dirichlet(np.ones(len(spins)))
        m = p @ spins
        q = spins.T @ (p[:, None]*spins)
        actual = spectrum_from_moments(np.tile(m, (4, 1)), np.tile(q, (4, 1, 1)), geometry(2))
        expected = exact_spectra(p, spins, geometry(2))['connected_spectrum']
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_energy_gradient_and_variational_decrease(self):
        spins = basis(12).astype(float)
        h = operator_matrix(make_af_sample(2, 50004).hamiltonian(.15), .25)
        x, _ = features(spins, 2)
        logs = np.zeros(len(spins))
        theta = np.random.default_rng(32).normal(0, .02, x.shape[1])
        initial, gradient = energy_gradient(theta, logs, x, h)
        for index in [0, 11, 12, 77]:
            shift = np.zeros_like(theta); shift[index] = 1e-5
            numeric = (energy_gradient(theta+shift, logs, x, h)[0] -
                       energy_gradient(theta-shift, logs, x, h)[0]) / 2e-5
            self.assertAlmostEqual(numeric, gradient[index], places=6)
        result = refine(logs, spins, h, degree=2, maxiter=500)
        self.assertLess(result['energy'], initial)
        self.assertFalse(result['exact_labels_used'])
        self.assertAlmostEqual(np.exp(2*result['logs']).sum(), 1., places=12)

    def test_constant_local_energy_preserves_all_ancestors(self):
        # H=-gamma sum X: uniform guide is its exact ground state.
        ham = make_af_sample(2, 50000).hamiltonian(0.)
        ham = type(ham)(ham.n_sites, ham.bonds, np.zeros_like(ham.J), 0.)
        guide = EnumeratedGuide(np.zeros(4096), 12)
        states = guide.sample(128, 551)[:, None, :]
        _, local = rates_and_local(ham, .75, guide, states)
        rng = np.random.default_rng(93)
        _, children, _, _, _, ess = propagate(ham, .75, guide, states, local.mean(), .1, rng)
        np.testing.assert_array_equal(children, np.arange(128))
        self.assertAlmostEqual(ess, 128.)


if __name__ == '__main__':
    unittest.main()
