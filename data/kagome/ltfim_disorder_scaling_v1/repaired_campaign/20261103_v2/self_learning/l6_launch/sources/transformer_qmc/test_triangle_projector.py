"""Independent physical-generator and importance-sampling checks."""
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from kagome_qea.exact import operator_matrix
from .model import EnumeratedGuide
from .physics import basis, make_af_sample
from .triangle_projector import TriangleProposal, local_hamiltonian, triangle_layers


class TriangleTests(unittest.TestCase):
    def setUp(self):
        self.ham = make_af_sample(2, 50025).hamiltonian(.15)
        self.spins = basis(12)
        rng = np.random.default_rng(82)
        self.guide = EnumeratedGuide(rng.normal(0, .15, len(self.spins)), 12)

    def test_decomposition_equals_independent_hamiltonian(self):
        rows, cols, values = [], [], []
        ix = np.arange(4096)
        for layer in triangle_layers(self.ham):
            for sites in layer:
                sites = np.array(sites)
                local = local_hamiltonian(self.ham, .5, sites)
                source = ((ix[:, None] >> sites) & 1) @ (1 << np.arange(3))
                mask = int(sum(1 << sites))
                for target in range(8):
                    offset = ((target >> np.arange(3)) & 1) @ (1 << sites)
                    rows.extend((ix & ~mask) | offset)
                    cols.extend(ix)
                    values.extend(local[target, source])
        reconstructed = csr_matrix((values, (rows, cols)), shape=(4096, 4096))
        error = reconstructed - operator_matrix(self.ham, .5)
        self.assertLess(np.max(abs(error.data), initial=0), 1e-13)

    def test_cached_and_direct_paths(self):
        states = self.spins[np.random.default_rng(4).integers(4096, size=1500)]
        a = TriangleProposal(self.ham, .5, self.guide, .2, cache=False)
        b = TriangleProposal(self.ham, .5, self.guide, .2, cache=True)
        direct = a.draw(states, np.random.default_rng(18))
        cached = b.draw(states, np.random.default_rng(18))
        for x, y in zip(direct, cached):
            np.testing.assert_allclose(x, y, atol=1e-13)

    def test_weighted_paths_reconstruct_full_operator(self):
        # A complete multi-factor path sum is checked through Monte Carlo moments,
        # against deterministic application with all intermediate states summed.
        kernel = TriangleProposal(self.ham, .75, self.guide, .4, cache=True)
        source = 931
        count = 140000
        states, weights, _ = kernel.draw(np.repeat(self.spins[source:source+1], count, axis=0), np.random.default_rng(55))
        observable = np.column_stack([np.ones(count), states[:, 0], states[:, 1] * states[:, 8]])
        weighted = np.exp(weights)[:, None] * observable
        unit = np.zeros(4096); unit[source] = 1.
        target = kernel.action(unit) * np.exp(self.guide.logs - self.guide.logs[source])
        full_observable = np.column_stack([np.ones(4096), self.spins[:, 0], self.spins[:, 1] * self.spins[:, 8]])
        expected = target @ full_observable
        error = abs(weighted.mean(0) - expected)
        self.assertTrue(np.all(error < 6 * weighted.std(0) / np.sqrt(count)), (error, expected))

    def test_second_order_and_symmetric_positive_operator(self):
        h = operator_matrix(self.ham, .5)
        vector = np.random.default_rng(77).uniform(.1, 1., 4096)
        other = np.random.default_rng(78).normal(size=4096)
        errors = []
        for delta in [.04, .02, .01]:
            kernel = TriangleProposal(self.ham, .5, self.guide, delta)
            action = kernel.action(vector)
            exact = expm_multiply(-delta * h, vector)
            errors.append(np.linalg.norm(action-exact)/np.linalg.norm(exact))
            self.assertAlmostEqual(float(other @ action), float(vector @ kernel.action(other)), places=9)
            self.assertTrue(np.all(action > 0))
        self.assertTrue(6 < errors[0]/errors[1] < 10, errors)
        self.assertTrue(6 < errors[1]/errors[2] < 10, errors)

    def test_larger_geometry_cover(self):
        for size in [4, 6, 8, 10]:
            layers = triangle_layers(make_af_sample(size, 1).hamiltonian(.15))
            self.assertEqual([len(layer) for layer in layers], [size*size]*2)


if __name__ == '__main__':
    unittest.main()
