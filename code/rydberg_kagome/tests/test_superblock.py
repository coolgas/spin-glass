import unittest
import numpy as np
from rydberg_kagome.exact import Operator


class ProjectionTests(unittest.TestCase):
    def test_cross_block_projection(self):
        rng=np.random.default_rng(12)
        W=[np.linalg.qr(rng.normal(size=(4,2)))[0] for _ in range(2)]
        z=np.diag([1.,-1,1,-1])
        coupling=.72
        projected=Operator([2,2],[(coupling,[(0,W[0].T @ z @ W[0]),(1,W[1].T @ z @ W[1])])])
        decoder=np.kron(*W); full=coupling*np.kron(z,z)
        expected=decoder.T @ full @ decoder
        actual=np.column_stack([projected.action(e) for e in np.eye(4)])
        np.testing.assert_allclose(actual,expected,atol=1e-13)
