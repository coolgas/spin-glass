"""Validate histogram ancestry weighting and deterministic finite-time references."""
import unittest
import numpy as np
from scipy.linalg import expm

from kagome_qea.exact import operator_matrix
from .guided_overlap_demo import histogram_tags, finite_time_distributions
from kagome_qea.hamiltonian import Hamiltonian
from .assess_overlap_onset import overlap_distribution


class GuidedHistogramTests(unittest.TestCase):
    def test_weighted_histogram_preserves_scalar_moments(self):
        rng = np.random.default_rng(31)
        states = rng.choice([-1, 1], size=(41, 4, 12))
        tagged = histogram_tags(states)
        ancestors = rng.integers(0, 41, size=83)
        hist = tagged['pq'][ancestors].mean(0)
        q = np.linspace(-1, 1, 13)
        self.assertAlmostEqual(hist.sum(), 1)
        for power, key in [(1, 'q'), (2, 'q2'), (4, 'q4')]:
            self.assertAlmostEqual(hist @ q**power, tagged[key][ancestors].mean())
        endpoint = np.ones((2, 4, 12), dtype=int)
        endpoint[0, 1] = -1
        np.testing.assert_array_equal(histogram_tags(endpoint)['pq'].argmax(1), [0, 12])

    def test_finite_time_against_dense_exponential_and_pair_sum(self):
        ham = Hamiltonian(3, np.array([[0, 1], [1, 2], [0, 2]]), np.array([-.7, -1.1, -.9]), .05)
        matrix = operator_matrix(ham, .5).toarray()
        energy = np.linalg.eigvalsh(matrix)[0]
        psi = np.arange(1., 9.); psi /= np.linalg.norm(psi)
        parameters = dict(tau_block=.1, n_equilibrate=2, n_measure=2, tau_forward=.3, n_cycles=2)
        got = finite_time_distributions(ham, .5, energy, psi, parameters)
        for cycle in range(2):
            left = expm(-(.4 + .3 * cycle) * matrix) @ psi
            right = expm(-.3 * matrix) @ psi
            p = left * right; p /= p.sum()
            mass = np.zeros(4)
            for a in range(8):
                for b in range(8):
                    mass[3 - bin(a ^ b).count('1')] += p[a] * p[b]
            np.testing.assert_allclose(got[cycle], mass, atol=1e-12)
        _, ground = np.linalg.eigh(matrix)
        fixed = finite_time_distributions(ham, .5, energy, abs(ground[:, 0]), parameters)
        np.testing.assert_allclose(fixed, np.tile(overlap_distribution(ground[:, 0]**2)[1], (2, 1)), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
