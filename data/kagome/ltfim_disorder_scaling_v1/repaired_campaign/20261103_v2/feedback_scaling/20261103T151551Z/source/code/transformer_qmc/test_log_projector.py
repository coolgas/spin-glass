"""Independent analytic controls for log-domain continuous-time projection."""
from types import SimpleNamespace
import unittest

import numpy as np

from qmc_numerics import log_expm1_ratio, systematic, systematic_log
from .projector import propagate, rates_and_local


class ProductGuide:
    def __init__(self, h, gamma):
        self.a = .5*np.arcsinh(h/gamma)

    def all_flip_log_ratios(self, states):
        return -2*self.a*states


class LogProjectorTests(unittest.TestCase):
    def test_signed_mixed_integrals_for_extreme_population_spread(self):
        from unittest.mock import patch
        from scipy.special import logsumexp
        local=np.array([-10000.,-2.,0.,3.,10000.])
        states=np.ones((5,1,1),dtype=np.int8)
        class NoHops:
            def exponential(self,scale):return np.full(len(scale),np.inf)
            def random(self):return .41
        ham=SimpleNamespace(n_sites=1)
        for shift in [-20000.,0.,20000.]:
            x=-(local-shift)*.1
            # Independent long-double integral, whose range covers exp(3000).
            xx=x.astype(np.longdouble)
            integrals=np.ones(5,dtype=np.longdouble)*.1
            nz=xx!=0
            integrals[nz]*=np.expm1(xx[nz])/xx[nz]
            mixed=float((integrals*local).sum()/integrals.sum())
            growth=shift-(logsumexp(x)-np.log(5))/.1
            with patch('transformer_qmc.projector.rates_and_local',return_value=(np.ones((5,1,1)),local)):
                result=propagate(ham,1.,None,states,shift,.1,NoHops())
            np.testing.assert_allclose(result[3:5],[mixed,growth],rtol=1e-12,atol=1e-9)

    def test_resampling_offset_invariance_and_invalid_mass(self):
        logs = np.array([-4., -1., 0., 2., -np.inf])
        expected, total = systematic(np.exp(logs), np.random.default_rng(72))
        for offset in [-10000., 0., 10000.]:
            children, actual = systematic_log(logs+offset, np.random.default_rng(72))
            np.testing.assert_array_equal(children, expected)
            self.assertAlmostEqual(actual-offset, total, places=11)
        for bad in [[np.nan, 0], [np.inf, 0], [-np.inf, -np.inf]]:
            with self.assertRaises(FloatingPointError):
                systematic_log(bad, np.random.default_rng(1))

    def test_interval_log_integral(self):
        x = np.array([-10000., -1000., -30., -1., -1e-9, 0., 1e-9, 1., 30., 1000., 10000.])
        actual = log_expm1_ratio(x)
        self.assertTrue(np.isfinite(actual).all())
        self.assertAlmostEqual(actual[0], -np.log(10000), places=13)
        self.assertAlmostEqual(actual[-1], 10000-np.log(10000), places=11)
        middle = x[2:-2]
        expected = np.ones(len(middle))
        nonzero = middle != 0
        expected[nonzero] = np.expm1(middle[nonzero])/middle[nonzero]
        np.testing.assert_allclose(np.exp(actual[2:-2]), expected, rtol=1e-14)

    def test_l6_exact_product_guide_with_extreme_shift(self):
        # K=0 independent spins: E0=-N sqrt(Gamma^2+h^2), exact for any N.
        # This is a numerical control, not the interacting kagome ensemble.
        n, h, gamma = 108, .15, 4.
        ham = SimpleNamespace(n_sites=n, h=h, bonds=np.empty((0, 2), dtype=int), J=np.empty(0))
        guide = ProductGuide(h, gamma)
        states = (1-2*np.random.default_rng(4).integers(2, size=(32, 1, n))).astype(np.int8)
        truth = -n*np.hypot(h, gamma)
        np.testing.assert_allclose(rates_and_local(ham, gamma, guide, states)[1], truth, atol=2e-13)
        results = []
        for shift in [truth-20000, truth, truth+20000]:
            diagnostics = {}
            result = propagate(ham, gamma, guide, states, shift, .1, np.random.default_rng(19), diagnostics)
            np.testing.assert_allclose(result[3:5], truth, atol=2e-9, rtol=0)
            self.assertAlmostEqual(result[5], len(states), places=10)
            self.assertAlmostEqual(diagnostics['min_log_weight'], (shift-truth)*.1, places=9)
            results.append(result)
        for result in results[1:]:
            np.testing.assert_array_equal(result[0], results[0][0])
            np.testing.assert_array_equal(result[1], results[0][1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
