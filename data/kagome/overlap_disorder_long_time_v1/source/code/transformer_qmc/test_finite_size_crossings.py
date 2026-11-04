import unittest
import numpy as np
from .finite_size_crossings import crossing_brackets,drift_sensitivity


class CrossingTests(unittest.TestCase):
    def test_brackets_retain_multiple_crossings_and_absence(self):
        x=[0,1,2];zeros=[0,0,0]
        result=crossing_brackets(x,[-1,1,-1],zeros)
        self.assertEqual([r['field'] for r in result],[.5,1.5])
        self.assertEqual(crossing_brackets(x,[1,2,3],zeros),[])
        self.assertEqual(len(crossing_brackets(x,[-1,0,1],zeros)),1)
        with self.assertRaises(ValueError):crossing_brackets(x,zeros,zeros)

    def test_recovers_synthetic_drift_with_correlated_crossings(self):
        rng=np.random.default_rng(351)
        pairs=np.array([[4,6],[5,7],[6,8],[7,9],[8,10]])
        length=np.sqrt(pairs.prod(1));noise=rng.normal(0,.002,(1000,1))
        samples=1.3+.7/length**2+noise+rng.normal(0,.0001,(1000,5))
        result=drift_sensitivity(pairs,samples,powers=(2.,))[0]
        self.assertAlmostEqual(result['critical_field_median'],1.3,places=3)
        low,high=result['bootstrap_95_interval'];self.assertLess(low,1.3);self.assertGreater(high,1.3)
        with self.assertRaises(ValueError):drift_sensitivity(pairs[:2],samples[:,:2])


if __name__=='__main__':unittest.main()
