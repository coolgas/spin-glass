import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('tenpy'), 'TeNPy available on compute host')
class DMRGTests(unittest.TestCase):
    def test_mpo_and_decoded_sampling(self):
        from rydberg_kagome.geometry import sample,masks
        from rydberg_kagome.superblock import build
        from rydberg_kagome import dmrg
        from rydberg_kagome.observables import vector_moments,summarize
        from rydberg_kagome.sampling import mps_samples,overlap_histogram
        from rydberg_kagome.convergence import sampling_check
        g=sample(4,2); v=g['interactions']*masks(4)['full']
        rg=build(v,2.,1.3,4,3)
        mpo,order=dmrg.build_mpo(rg)
        psi=dmrg.initial_state(mpo,70)
        vector=dmrg.full_vector(psi,order,3)
        self.assertAlmostEqual(float(mpo.H_MPO.expectation_value(psi)),
                               vector @ rg['operator'].action(vector),places=9)
        m,zz,x=dmrg.moments(psi,rg,order)
        direct_m,direct_zz=vector_moments(rg,vector)
        np.testing.assert_allclose(m,direct_m,atol=1e-10)
        np.testing.assert_allclose(zz,direct_zz,atol=1e-10)
        # Reconstruct the physical occupation energy independently of RG terms.
        nn=(1+m[:,None]+m[None,:]+zz)/4
        physical_energy=1.3*x.sum()/2-2.*np.sum((1+m)/2)+np.sum(v*nn)/2
        projected_energy=float(mpo.H_MPO.expectation_value(psi))+rg['constant']
        self.assertAlmostEqual(physical_energy,projected_energy,places=9)
        hist=overlap_histogram(mps_samples(psi,rg,order,20000,13))
        self.assertTrue(sampling_check(hist,summarize(m,zz)))
        import tempfile
        from pathlib import Path
        from rydberg_kagome.results import checkpoint,load_checkpoint
        saved_vector=dmrg.full_vector(psi,order,3)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pkl.gz'
            checkpoint(path,psi,mpo.lat.mps_sites())
            restored=load_checkpoint(path,mpo.lat.mps_sites())
            np.testing.assert_array_equal(dmrg.full_vector(restored,order,3),saved_vector)
            self.assertAlmostEqual(float(mpo.H_MPO.expectation_value(restored)),
                                   float(mpo.H_MPO.expectation_value(psi)),places=12)
