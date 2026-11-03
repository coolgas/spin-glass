"""Independent enumeration and analytic limits for exact P(q)."""
import math
import unittest
import numpy as np
from .assess_overlap_onset import overlap_distribution


class OverlapTests(unittest.TestCase):
    def test_against_explicit_two_copy_enumeration(self):
        rng=np.random.default_rng(53);p=rng.random(16);p/=p.sum()
        q,mass=overlap_distribution(p);explicit=np.zeros(5)
        for a in range(16):
            for b in range(16):explicit[4-bin(a^b).count('1')]+=p[a]*p[b]
        np.testing.assert_allclose(mass,explicit,atol=1e-14)
        np.testing.assert_allclose(q,np.linspace(-1,1,5))

    def test_uniform_and_spin_flip_cat_limits(self):
        q,mass=overlap_distribution(np.ones(4096))
        np.testing.assert_allclose(mass,[math.comb(12,k)/4096 for k in range(13)],atol=1e-14)
        self.assertAlmostEqual(float(mass@q**2),1/12)
        p=np.zeros(4096);p[0]=p[-1]=.5
        q,mass=overlap_distribution(p)
        expected=np.zeros(13);expected[0]=expected[-1]=.5
        np.testing.assert_allclose(mass,expected,atol=1e-14)


if __name__=='__main__':unittest.main()
