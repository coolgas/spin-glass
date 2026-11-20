import unittest
import numpy as np
from rydberg_kagome.sampling import sample_tensors,overlap_histogram
from rydberg_kagome.hamiltonian import spin_table


class SamplingTests(unittest.TestCase):
    def test_correlated_blocks(self):
        # Two physical spins in an entangled Bell state, including a minus sign.
        A=np.zeros((1,2,2)); A[0,0,0]=1/np.sqrt(2); A[0,1,1]=-1/np.sqrt(2)
        B=np.zeros((2,2,1)); B[0,0,0]=1; B[1,1,0]=1
        samples=sample_tensors([A,B],[np.eye(2)]*2,[[0],[1]],spin_table(1),20000,8)
        np.testing.assert_array_equal(samples[:,0],samples[:,1])
        self.assertLess(abs(samples.mean()),.03)
        hist=overlap_histogram(samples)
        self.assertAlmostEqual(hist['p'][1],0)
        self.assertLess(abs(hist['qmean']),.04)

    def test_decoder_rotations(self):
        rng=np.random.default_rng(3)
        W=np.linalg.qr(rng.normal(size=(4,2)))[0]
        v=np.array([.6,.8]); T=v.reshape(1,2,1)
        samples=sample_tensors([T],[W],[[0,1]],spin_table(2),60000,9)
        index=((1-samples[:,0])//2)+2*((1-samples[:,1])//2)
        p=np.bincount(index,minlength=4)/len(index)
        np.testing.assert_allclose(p,(W @ v)**2,atol=.008)
