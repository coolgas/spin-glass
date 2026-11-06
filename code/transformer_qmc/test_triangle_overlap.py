"""Independent small-state checks for physical overlap normalizations."""
import unittest

import numpy as np

from .triangle_correlation_observables import overlap_observables, spectra


class PhysicalOverlapTests(unittest.TestCase):
    def check_state(self, state):
        state = np.asarray(state, dtype=float)
        n = int(np.log2(len(state)))
        bits = (np.arange(len(state))[:, None] >> np.arange(n)) & 1
        spins = 1-2*bits
        weights = state**2
        m = weights @ spins
        zz = np.einsum('s,si,sj->ij', weights, spins, spins)
        result = overlap_observables(m, zz)
        # Direct two-replica overlap, independent of the moment contraction.
        overlap = spins @ spins.T/n
        probabilities = np.outer(weights, weights)
        self.assertAlmostEqual(result['q_ea_finite'], np.sum(probabilities*overlap))
        self.assertAlmostEqual(result['q2_raw'], np.sum(probabilities*overlap**2))
        points = np.column_stack((np.arange(n), np.zeros(n)))
        self.assertAlmostEqual(result['chi_sg_connected_equal_time'], spectra(m, zz, points, 2)[0])
        self.assertAlmostEqual(result['chi_sg_raw_equal_time'], n*result['q2_raw'])
        self.assertAlmostEqual(result['chi_sg_connected_equal_time'],
            result['chi_sg_connected_diagonal']+result['chi_sg_connected_off_diagonal'])
        return result

    def test_polarized_product(self):
        result = self.check_state([1, 0, 0, 0])
        self.assertEqual(result['q_ea_finite'], 1)
        self.assertEqual(result['chi_sg_connected_equal_time'], 0)

    def test_transverse_product_and_cat(self):
        transverse = self.check_state(np.ones(4)/2)
        cat = self.check_state(np.array([1, 0, 0, 1])/np.sqrt(2))
        self.assertAlmostEqual(transverse['q_ea_finite'], 0)
        self.assertAlmostEqual(transverse['chi_sg_connected_equal_time'], 1)
        self.assertAlmostEqual(cat['q_ea_finite'], 0)
        self.assertAlmostEqual(cat['chi_sg_connected_equal_time'], 2)

    def test_entangled_biased_state(self):
        self.check_state(np.sqrt([.1, .2, .3, .4]))

    def test_invalid_physical_moments(self):
        with self.assertRaises(ValueError):
            overlap_observables([0, 0], np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            overlap_observables([0, np.nan], np.eye(2))


if __name__ == '__main__':
    unittest.main()
