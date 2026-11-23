"""Actual TeNPy integration checks on a small, exactly soluble spin chain."""
import unittest
import numpy as np
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from adaptive_d_runtime import disable_mixer, increase_dimension, one_sweep
from adaptive_d_policy import final_window_pass


class RuntimeTests(unittest.TestCase):
    def test_latest_growth_leaves_confirmation_within_120(self):
        model=TFIChain(dict(L=6,J=1.,g=1.,bc_MPS='finite',conserve=None))
        psi=MPS.from_product_state(model.lat.mps_sites(),['up']*6,bc='finite',unit_cell_width=1)
        engine=TwoSiteDMRGEngine(psi,model,dict(mixer=True,
            mixer_params=dict(amplitude=1e-3,decay=2.,disable_after=15),
            N_sweeps_check=1,trunc_params=dict(chi_max=2,svd_min=1e-12)))
        engine.pre_run_initialize()
        history=[]; moments={}; cap=2; stage=0; accepted=[]
        for sweep in range(1,121):
            if sweep-stage>(15 if stage==0 else 5) and engine.mixer is not None:
                disable_mixer(engine)
            mixed=engine.mixer is not None
            energy,_=one_sweep(engine)
            history.append(dict(D=cap,mixer=mixed,energy=float(energy)))
            if sweep%5==0:
                moments[sweep]=dict(D=cap,mixer=mixed,m=psi.expectation_value('Sigmaz').tolist())
                if final_window_pass(history,moments,sweep,6,float(np.linalg.norm(psi.norm_test()))):
                    accepted.append(sweep)
            if sweep in (20,40,60):
                cap+=2; stage=sweep; increase_dimension(engine,cap)
        self.assertEqual(engine.sweeps,120)
        self.assertTrue(accepted)
        self.assertGreaterEqual(min(accepted),115)
        self.assertIn(120,accepted)

    def test_growth_preserves_state_counters_and_improves_energy(self):
        model=TFIChain(dict(L=6,J=1.,g=1.,bc_MPS='finite',conserve=None))
        psi=MPS.from_product_state(model.lat.mps_sites(),['up']*6,bc='finite',unit_cell_width=1)
        engine=TwoSiteDMRGEngine(psi,model,dict(mixer=True,
            mixer_params=dict(amplitude=1e-3,decay=2.,disable_after=2),
            N_sweeps_check=1,trunc_params=dict(chi_max=2,svd_min=1e-12)))
        engine.pre_run_initialize()
        for _ in range(5): one_sweep(engine)
        disable_mixer(engine); disable_mixer(engine)
        e_before=float(model.H_MPO.expectation_value(psi))
        m_before=psi.expectation_value('Sigmaz').copy()
        count=engine.sweeps; state_id=id(psi)
        increase_dimension(engine,8)
        self.assertEqual(engine.sweeps,count)
        self.assertEqual(id(engine.psi),state_id)
        np.testing.assert_allclose(psi.expectation_value('Sigmaz'),m_before,atol=1e-12)
        self.assertAlmostEqual(float(model.H_MPO.expectation_value(psi)),e_before,places=11)
        for _ in range(8): one_sweep(engine)
        disable_mixer(engine)
        self.assertEqual(engine.sweeps,count+8)
        self.assertGreater(max(psi.chi),2)
        self.assertLessEqual(max(psi.chi),8)
        e_after=float(model.H_MPO.expectation_value(psi))
        self.assertLessEqual(e_after,e_before+1e-10)
        # Independent dense open-chain reference, H=-sum XX-sum Z.
        I=np.eye(2); X=np.array([[0.,1.],[1.,0.]]); Z=np.diag([1.,-1.])
        def term(operators):
            out=np.array([[1.]])
            for i in range(6): out=np.kron(out,operators.get(i,I))
            return out
        H=-sum(term({i:X,i+1:X}) for i in range(5))-sum(term({i:Z}) for i in range(6))
        self.assertAlmostEqual(e_after,float(np.linalg.eigvalsh(H)[0]),places=9)
        self.assertLess(float(np.linalg.norm(psi.norm_test())),1e-8)


if __name__=='__main__': unittest.main()
