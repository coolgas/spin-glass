"""Cached rates must preserve event propagation and physical estimators."""

import unittest
import numpy as np

from .model import EnumeratedGuide
from .physics import basis, make_af_sample
from .projector import propagate as direct_propagate
from .table_projector import RateTable, propagate


class TableProjectorTests(unittest.TestCase):
    def test_cached_and_direct_propagation_agree_for_both_guides(self):
        ham = make_af_sample(2, 80001).hamiltonian(.3)
        states = basis(12)
        for logs in (np.zeros(4096), .1*states.sum(axis=1)):
            table = RateTable(ham, .75, logs)
            guide = EnumeratedGuide(logs, 12)
            indices = np.random.default_rng(1).integers(4096, size=(512, 4))
            direct = direct_propagate(ham, .75, guide, states[indices], -40., .1,
                                      np.random.default_rng(22))
            cached = propagate(table, indices, -40., .1, np.random.default_rng(22))
            np.testing.assert_array_equal(direct[0], states[cached[0]])
            np.testing.assert_array_equal(direct[1], cached[1])
            np.testing.assert_allclose(direct[2:], cached[2:], rtol=1e-13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
