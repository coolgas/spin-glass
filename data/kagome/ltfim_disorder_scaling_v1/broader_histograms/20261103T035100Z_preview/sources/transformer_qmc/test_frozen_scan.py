"""Physical scale identity and direct-cache context correctness."""

import unittest
import numpy as np
import torch

from kagome_qea.exact import operator_matrix
from .direct_gfmc_pilot import CachedDirectGuide
from .model import TransformerGuide
from .physics import basis, make_af_sample
from .symmetric import SymmetricTransformer


class FrozenScanTests(unittest.TestCase):
    def test_matched_disorder_hamiltonian_scale(self):
        for K in (.5, 2.):
            sample = make_af_sample(2, 10000, scale=K)
            unit = make_af_sample(2, 10000)
            np.testing.assert_array_equal(sample.J, K*unit.J)
            matrix = operator_matrix(sample.hamiltonian(.3), .7)
            scaled = K*operator_matrix(unit.hamiltonian(.3/K), .7/K)
            np.testing.assert_allclose((matrix-scaled).data, 0., atol=1e-13)

    def test_cached_direct_ratios_across_eviction_and_context(self):
        torch.set_num_threads(1)
        torch.manual_seed(51)
        ham = make_af_sample(2, 10000).hamiltonian(.3)
        model = SymmetricTransformer(ham, .5, width=8, layers=1, heads=2).double()
        torch.nn.init.normal_(model.base.output.weight, std=.2)
        direct = TransformerGuide(model)
        cache = CachedDirectGuide(model, max_entries=4)
        states = basis(12)
        for indices in ([0, 0, 1, 2], [2, 1, 3, 4], [10, 0, 10, 1], [0, 1]):
            np.testing.assert_allclose(cache.all_flip_log_ratios(states[indices]),
                                       direct.all_flip_log_ratios(states[indices]), atol=1e-12)
        self.assertLess(cache.evaluated, cache.requested)
        model.set_hamiltonian(make_af_sample(2, 10001).hamiltonian(0.), .5)
        cache = CachedDirectGuide(model)
        np.testing.assert_allclose(cache.all_flip_log_ratios(states[:4]),
                                   direct.all_flip_log_ratios(states[:4]), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
