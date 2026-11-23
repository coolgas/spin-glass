import unittest
import numpy as np
from rydberg_kagome.geometry import sample, masks
from rydberg_kagome.superblock import build
from rydberg_kagome import dmrg
from adaptive_memory import growth_bytes, permitted, GIB


def dense(mpo):
    """Full finite MPO with explicit public boundary identities."""
    from tenpy.linalg import np_conserved as npc
    H=mpo.H_MPO; count=H.L; chi=H.sites[0].dim
    tensor=H.get_W(0).take_slice(H.get_IdL(0),'wL')
    tensor.ireplace_labels(['p','p*'],['p0','p0*'])
    for i in range(1,count):
        w=H.get_W(i).replace_labels(['p','p*'],[f'p{i}',f'p{i}*'])
        tensor=npc.tensordot(tensor,w,axes=['wR','wL'])
    tensor=tensor.take_slice(H.get_IdR(count-1),'wR')
    return tensor.transpose([f'p{i}' for i in range(count)]+[f'p{i}*' for i in range(count)]).to_ndarray().reshape(chi**count,chi**count)


class CompactTests(unittest.TestCase):
    def test_exact_hamiltonian_action_and_moments(self):
        g=sample(4,2); model=build(g['interactions'],2.,1.3,4,3)
        old,order=dmrg.build_mpo(model)
        new,neworder=dmrg.build_mpo(model,interactions=g['interactions'])
        self.assertEqual(order,neworder)
        # Entire dense operator, not just one ground state or expectation.
        A=dense(old); B=dense(new)
        np.testing.assert_allclose(A,B,atol=1e-11,rtol=1e-12)
        C=np.column_stack([model['operator'].action(x) for x in np.eye(81)])
        # Undo snake ordering for independent block-order operator comparison.
        perm=np.arange(81).reshape((3,)*4).transpose(order).ravel()
        np.testing.assert_allclose(B,C[np.ix_(perm,perm)],atol=1e-11,rtol=1e-12)
        for seed in (0,1):
            p=dmrg.initial_state(old,seed); q=dmrg.initial_state(new,seed)
            self.assertAlmostEqual(float(old.H_MPO.expectation_value(p)),float(new.H_MPO.expectation_value(q)),places=10)
            for a,b in zip(dmrg.moments(p,model,order),dmrg.moments(q,model,order)):
                np.testing.assert_allclose(a,b,atol=1e-11)
        self.assertLess(max(new.H_MPO.chi),max(old.H_MPO.chi))

    def test_nine_block_L6_complete_operator(self):
        # Actual L6 geometry and projection, reduced chi=2 makes all 512 basis
        # vectors accessible. This checks the nine-site snake orientation.
        g=sample(6,3); v=g['interactions']; model=build(v,3.15,.9037538790536807,6,2)
        compact,order=dmrg.build_mpo(model,interactions=v)
        A=dense(compact)
        C=np.column_stack([model['operator'].action(x) for x in np.eye(512)])
        perm=np.arange(512).reshape((2,)*9).transpose(order).ravel()
        np.testing.assert_allclose(A,C[np.ix_(perm,perm)],atol=2e-11,rtol=1e-12)
        np.testing.assert_allclose(A,A.T,atol=1e-12)

    def test_masks_and_wrong_matrix_rejection(self):
        g=sample(4,7)
        for mask in ('nn','nn_nnn'):
            with self.subTest(mask=mask):
                v=g['interactions']*masks(4)[mask]; model=build(v,2.,.4,4,2)
                old,_=dmrg.build_mpo(model); new,_=dmrg.build_mpo(model,interactions=v)
                np.testing.assert_allclose(dense(old),dense(new),atol=1e-11,rtol=1e-12)
                bad=v.copy()
                b,c=model['sites'][0][0],model['sites'][1][0]
                bad[b,c]+=1.; bad[c,b]+=1.
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    dmrg.build_mpo(model,interactions=bad)

    def test_checkpoint_uses_compact_sites(self):
        import tempfile
        from pathlib import Path
        from rydberg_kagome.results import checkpoint,load_checkpoint
        g=sample(4,1); model=build(g['interactions'],2.,.6,4,2)
        mpo,order=dmrg.build_mpo(model,interactions=g['interactions'])
        psi=dmrg.initial_state(mpo,5)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pkl.gz'
            checkpoint(path,psi,mpo.lat.mps_sites())
            # Restore against a freshly reconstructed compact MPO.
            rebuilt,_=dmrg.build_mpo(model,interactions=g['interactions'])
            restored=load_checkpoint(path,rebuilt.lat.mps_sites())
            np.testing.assert_array_equal(dmrg.full_vector(psi,order,2),dmrg.full_vector(restored,order,2))
            self.assertAlmostEqual(float(mpo.H_MPO.expectation_value(psi)),float(rebuilt.H_MPO.expectation_value(restored)),places=11)

    def test_memory_admission_reserves_other_workloads(self):
        forecast=growth_bytes(GIB,128,256)
        self.assertEqual(forecast,6*GIB)
        self.assertTrue(permitted(GIB,forecast,24*GIB,8*GIB))
        self.assertFalse(permitted(GIB,forecast,20*GIB,8*GIB))
        self.assertFalse(permitted(GIB,forecast,100*GIB,4*GIB))


if __name__=='__main__': unittest.main()
