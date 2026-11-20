import unittest
import itertools
import numpy as np
from rydberg_kagome.geometry import lattice,distances,sample,masks,blocks


class GeometryTests(unittest.TestCase):
    def test_shortest_images(self):
        _,cell=lattice(4)
        rng=np.random.default_rng(42)
        points=rng.random((30,2)) @ cell.T
        delta=points[:,None]-points[None,:]
        brute=np.min([np.linalg.norm(delta+cell @ shift,axis=-1)
                      for shift in itertools.product(range(-3,4),repeat=2)],axis=0)
        np.fill_diagonal(brute,np.inf)
        np.testing.assert_allclose(distances(points,cell),brute,atol=1e-12)
        shifted=points+(rng.integers(-3,4,size=points.shape) @ cell.T)
        np.testing.assert_allclose(distances(shifted,cell),brute,atol=1e-12)
        finite=np.isfinite(brute)
        self.assertGreater(np.max(distances(points,cell,False)[finite]-brute[finite]),.1)

    def test_ensemble_and_masks(self):
        for L in (4,6,8):
            a=sample(L,0); b=sample(L,0)
            np.testing.assert_array_equal(a['points'],b['points'])
            self.assertGreaterEqual(a['distances'].min(),.5)
            m=masks(L)
            np.testing.assert_array_equal(m['nn'].sum(1),4)
            np.testing.assert_array_equal(m['nn_nnn'].sum(1),8)
            np.testing.assert_array_equal(np.sort(blocks(L).ravel()),np.arange(3*L*L))
