"""Check unguided propagation against the exact imaginary-time kernel."""

import unittest

import numpy as np
from scipy.linalg import expm

from ising import Hamiltonian
from kagome_qea.exact import operator_matrix
from .compare_pure_gfmc import BatchedConstantGuide
from .physics import basis
from .projector import propagate, rates_and_local


class PureGFMCPhysicsTests(unittest.TestCase):
    def test_exact_forward_target_matches_importance_transformed_kernel(self):
        ham = Hamiltonian(3, [[0, 1], [1, 2], [0, 2]], [-.8, -1.1, -.9], .25)
        matrix = operator_matrix(ham, .7).toarray()
        for trial in (np.ones(8), np.exp(np.linspace(-.7, .3, 8))):
            trial /= np.linalg.norm(trial)
            left = expm(-.7*matrix)@trial
            right = expm(-.3*matrix)@trial
            expected = left*right
            expected /= expected.sum()
            kernel = trial[:, None]*expm(-.7*matrix)/trial[None, :]
            population = kernel@(trial**2)
            future = trial[:, None]*expm(-.3*matrix)/trial[None, :]
            ancestor = population*future.sum(axis=0)
            ancestor /= ancestor.sum()
            np.testing.assert_allclose(ancestor, expected, atol=1e-14)

    def test_unguided_kernel_against_exact_propagator(self):
        ham = Hamiltonian(3, [[0, 1], [1, 2], [0, 2]], [-.8, -1.1, -.9], .25)
        gamma, tau, count = .7, .2, 100000
        guide = BatchedConstantGuide(3)
        states = basis(3)
        rates, local = rates_and_local(ham, gamma, guide, states[:, None, :])
        np.testing.assert_allclose(rates, gamma, atol=0, rtol=0)
        matrix = operator_matrix(ham, gamma).toarray()
        np.testing.assert_allclose(local, matrix.sum(axis=1), atol=1e-14)
        initial = np.repeat(states[[0], None, :], count, axis=0)
        evolved, *_ = propagate(ham, gamma, guide, initial, 0., tau,
                                np.random.default_rng(917))
        indexes = ((1-evolved[:, 0])//2) @ (1 << np.arange(3))
        frequency = np.bincount(indexes, minlength=8)/count
        expected = expm(-tau*matrix)[:, 0]
        expected /= expected.sum()
        # Includes both random trajectories and systematic branching fluctuations.
        np.testing.assert_allclose(frequency, expected, atol=.004)

    def test_constant_guide_is_exact_for_independent_transverse_spins(self):
        ham = Hamiltonian(3, [], [], 0.)
        states = basis(3)
        guide = BatchedConstantGuide(3)
        _, local = rates_and_local(ham, .8, guide, states[:, None, :])
        np.testing.assert_allclose(local, -2.4, atol=1e-14)
        initial = np.tile(states[:, None, :], (32, 4, 1))
        evolved, children, shift, mixed, growth, ess = propagate(
            ham, .8, guide, initial, -9.6, .4, np.random.default_rng(81))
        self.assertEqual(evolved.shape, initial.shape)
        self.assertEqual(len(np.unique(children)), len(initial))
        self.assertAlmostEqual(mixed, -2.4)
        self.assertAlmostEqual(growth, -2.4)
        self.assertAlmostEqual(ess, len(initial))


if __name__ == '__main__':
    unittest.main(verbosity=2)
