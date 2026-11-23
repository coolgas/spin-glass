import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from rydberg_kagome.adaptive import engine_state,restore_engine,save_restart,load_restart
from rydberg_kagome.adaptive_runtime import one_sweep,increase_dimension


class ResumeTests(unittest.TestCase):
    def engine(self):
        model=TFIChain(dict(L=6,J=1.,g=1.,bc_MPS='finite',conserve=None))
        psi=MPS.from_product_state(model.lat.mps_sites(),['up']*6,bc='finite',unit_cell_width=1)
        engine=TwoSiteDMRGEngine(psi,model,dict(mixer=True,N_sweeps_check=1,
            mixer_params=dict(amplitude=1e-3,decay=2.,disable_after=15),
            trunc_params=dict(chi_max=2,svd_min=1e-12)))
        engine.pre_run_initialize()
        return model,engine

    def test_active_mixer_and_growth_resume_matches_continuation(self):
        model,engine=self.engine()
        for _ in range(3): one_sweep(engine)
        increase_dimension(engine,4)
        one_sweep(engine)
        from rydberg_kagome.results import checkpoint,load_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pkl.gz'
            checkpoint(path,engine_state(engine),model.lat.mps_sites())
            restored=restore_engine(load_checkpoint(path,model.lat.mps_sites()),model)
            self.assertEqual(engine.sweeps,restored.sweeps)
            self.assertEqual(engine.mixer.amplitude,restored.mixer.amplitude)
            for _ in range(4):
                e,_=one_sweep(engine); r,_=one_sweep(restored)
                self.assertAlmostEqual(float(e),float(r),places=11)
            np.testing.assert_allclose(engine.psi.expectation_value('Sigmaz'),restored.psi.expectation_value('Sigmaz'),atol=1e-11)

    def test_manifest_integrity_and_partial_generation(self):
        model,a=self.engine(); _,b=self.engine(); one_sweep(a)
        state=dict(D=2,history=[[{}],[]],pending_growth=4,stage_start=0)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); rec=save_restart(root,[a,b],state,{'source':'A'})
            (root/'restart/orphan.pkl.gz').write_bytes(b'incomplete unrelated generation')
            engines,resumed=load_restart(root,model,{'source':'A'})
            self.assertEqual([e.sweeps for e in engines],[1,0])
            self.assertEqual(resumed['pending_growth'],4)
            with self.assertRaisesRegex(ValueError,'identity'): load_restart(root,model,{'source':'B'})
            (root/rec['files'][1]['file']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash'): load_restart(root,model,{'source':'A'})

    def test_second_payload_failure_preserves_committed_pair(self):
        model,a=self.engine(); _,b=self.engine(); one_sweep(a)
        state=dict(D=2,history=[[{}],[]],pending_growth=4,stage_start=0)
        from rydberg_kagome.results import checkpoint
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); record=save_restart(root,[a,b],state,{'source':'A'})
            pointer=(root/'resume.json').read_bytes()
            one_sweep(b); advanced=copy.deepcopy(state); advanced['history'][1].append({})
            def interrupted(path,*args):
                if path.name.endswith('-start1.pkl.gz'): raise OSError('simulated write failure')
                return checkpoint(path,*args)
            with patch('rydberg_kagome.adaptive.checkpoint',side_effect=interrupted):
                with self.assertRaisesRegex(OSError,'simulated write failure'):
                    save_restart(root,[a,b],advanced,{'source':'A'})
            self.assertEqual((root/'resume.json').read_bytes(),pointer)
            self.assertEqual(len(list((root/'restart').glob('*.json'))),1)
            engines,resumed=load_restart(root,model,{'source':'A'})
            self.assertEqual([e.sweeps for e in engines],[1,0])
            self.assertEqual(resumed,state)
            self.assertEqual(json.loads(pointer)['generation'],record['generation'])


if __name__=='__main__': unittest.main()
