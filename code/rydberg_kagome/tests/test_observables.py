import unittest
import numpy as np
from rydberg_kagome.observables import summarize,physical_vector_moments,vector_moments
from rydberg_kagome.hamiltonian import spin_table


class ObservableTests(unittest.TestCase):
    def test_replica_moment_identities(self):
        rng=np.random.default_rng(8); v=rng.normal(size=16); v/=np.linalg.norm(v)
        z=spin_table(4); m,zz=physical_vector_moments(v,z)
        obs=summarize(m,zz); q=z @ z.T/4
        weights=np.outer(v*v,v*v)
        self.assertAlmostEqual(obs['qea'],np.sum(weights*q))
        self.assertAlmostEqual(obs['q2'],np.sum(weights*q*q))

    def test_same_block_product(self):
        rng=np.random.default_rng(55)
        W=np.linalg.qr(rng.normal(size=(4096,3)))[0]
        v=rng.normal(size=3); v/=np.linalg.norm(v)
        spins=spin_table(12)
        ops=[W.T @ (spins[:,i,None]*W) for i in range(12)]
        model=dict(chi=3,sites=np.array([np.arange(12)]),spins=spins,decoders=[W],zops=[ops])
        m,zz=vector_moments(model,v)
        ref=physical_vector_moments(W @ v,spins)
        np.testing.assert_allclose(m,ref[0],atol=1e-12)
        np.testing.assert_allclose(zz,ref[1],atol=1e-12)
        self.assertGreater(abs(v @ ops[0] @ ops[1] @ v-zz[0,1]),1e-4)
