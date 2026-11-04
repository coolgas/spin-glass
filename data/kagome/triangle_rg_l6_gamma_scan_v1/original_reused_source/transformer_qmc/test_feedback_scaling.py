"""Check the prioritized finite L6/L8/L10 design and GPU measurement path."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .feedback_scaling import contexts,settings,task_for_measurement
from .feedback_frontier import verify_measurement,l2_task
from .self_learning_l6_pilot import measurement_task
from .simple_guides import RBM,SimpleGuide
from .projected_overlap_histograms import run_one
from .physics import make_af_sample
from .projector import rates_and_local


class ScalingTests(unittest.TestCase):
    def test_scope_prioritizes_three_sizes_and_paired_arms(self):
        for L in [6,8,10]:
            items=contexts(L)
            self.assertEqual({(t['sample_id'],t['seed']) for t in items},
                             {(sid,seed) for sid in [60000,60001] for seed in [0,1]})
            for task in items:
                a=settings('baseline',task,Path('same.pt'),Path('a'))
                b=settings('robust',task,Path('same.pt'),Path('b'))
                self.assertEqual(a.initial,b.initial);self.assertEqual(a.stints,b.stints)
                self.assertEqual(a.stints,12);self.assertEqual(b.negative_temperatures,16)
                self.assertEqual(a.islands,1);self.assertEqual(b.islands,4)
                self.assertEqual(a.projection_device,b.projection_device)
                for repeat in [0,1]:
                    m=task_for_measurement(task,repeat)
                    self.assertEqual(m['L'],L);self.assertEqual(m['walkers'],512)
                    self.assertEqual(m['forward'],16.);self.assertEqual(m['cycles'],8)
                    if L==6:self.assertEqual(m,measurement_task(task,repeat))
        with self.assertRaises(ValueError):contexts(2)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_gpu_rates_match_cpu_and_measurement_reconstructs(self):
        for L in [2,6,8,10]:
            ham=make_af_sample(L,60000).hamiltonian(.15)
            cpu=RBM(3*L*L,2);gpu=RBM(3*L*L,2).to('cuda');gpu.load_state_dict(cpu.state_dict())
            states=np.random.default_rng(131).choice([-1,1],size=(8,1,3*L*L)).astype(np.int8)
            cr,ce=rates_and_local(ham,.6,SimpleGuide(cpu),states)
            gr,ge=rates_and_local(ham,.6,SimpleGuide(gpu),states)
            np.testing.assert_allclose(cr,gr,atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(ce,ge,atol=1e-10,rtol=1e-10)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'model.pt'
            torch.save(dict(model=RBM(12,2).state_dict()),path)
            task=dict(l2_task(0,0),walkers=8,equil=.1,forward=.2,cycles=2)
            result=run_one(dict(model=str(path),task=task,projection_device='cuda'),root/'projector')
            self.assertTrue(verify_measurement(result,root/'projector')['reconstructed'])
            self.assertEqual(result['provenance']['projection_device'],'cuda')


if __name__=='__main__':unittest.main()
