"""Regression for negative persistence through field changes and checkpoint restarts."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import self_learning_rbm as feedback
from .l6_pcd_curriculum import audit_persistence, ARMS, CURRICULUM, settings


class CurriculumTests(unittest.TestCase):
    def test_finite_matched_design(self):
        task = dict(sample_id=60000, seed=0)
        configs = [settings(arm, task, None, Path(arm)) for arm in ARMS]
        self.assertEqual(len(CURRICULUM),24)
        self.assertEqual(CURRICULUM[-6:],[.6]*6)
        for config in configs:
            self.assertEqual(config.stints,24)
            self.assertEqual(config.updates,1000)
        a,b,c = [vars(cfg).copy() for cfg in configs]
        for value in [a,b,c]:
            value.pop('output')
        a.pop('gamma_schedule'); b.pop('gamma_schedule')
        self.assertEqual(a,b)
        b = vars(configs[1]).copy(); c = vars(configs[2]).copy()
        for value in [b,c]:
            value.pop('output'); value.pop('persistent_negative')
        self.assertEqual(b,c)

    def check_continuation(self, device, temperatures, interrupt_update=25, persistent=True):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(L=2, gamma=.6, gamma_schedule=[2.,.6],
                persistent_negative=persistent, negative_temperatures=temperatures,
                islands=2, step_rms=.025, step_max=.15, sample_id=60000, seed=1,
                stints=2, walkers=8, equil=.1, collect=.2, delta=.1, sample_walkers=8,
                updates=30, batch=8, gibbs=1, initial_sweeps=1, lr=.001,
                device=device, projection_device=device, initial=None, output=root/'full')
            feedback.run(args)
            full = torch.load(args.output/'checkpoint.pt', weights_only=False)
            audit_persistence(args.output) if temperatures>1 else None
            # Independent raw check for single-temperature PCD too.
            with np.load(args.output/'stint_000/fit.npz') as previous, np.load(args.output/'stint_001/negative_initial.npz') as initial:
                if persistent:
                    np.testing.assert_array_equal(previous['negative_samples'],initial['negative'])
                else:
                    self.assertFalse(np.array_equal(previous['negative_samples'],initial['negative']))
            self.assertEqual([r['gamma'] for r in full['rows']],[2.,.6])
            self.assertEqual([r['negative_carried'] for r in full['rows']],[False,persistent])
            args.output = root/'resumed'
            feedback.run(args, stop_after_stints=1)
            original = feedback.save_checkpoint
            def interrupt(path,state):
                original(path,state)
                if state.get('stint')==1 and state.get('phase')=='fit' and state.get('update')==interrupt_update:
                    raise InterruptedError('intentional field-change checkpoint interruption')
            with patch.object(feedback,'save_checkpoint',interrupt):
                with self.assertRaises(InterruptedError):
                    feedback.run(args)
            feedback.run(args)
            resumed = torch.load(args.output/'checkpoint.pt',weights_only=False)
            for key,value in full['model'].items():
                torch.testing.assert_close(value,resumed['model'][key],atol=0,rtol=0)
            self.assertEqual(full['rows'],resumed['rows'])
            torch.testing.assert_close(full['torch_rng'],resumed['torch_rng'],atol=0,rtol=0)
            torch.testing.assert_close(full['negative'],resumed['negative'],atol=0,rtol=0)
            self.assertEqual(full['rng'],resumed['rng'])
            self.assertEqual(full['island_rngs'],resumed['island_rngs'])
            if temperatures>1:
                self.assertEqual(resumed['sampler']['sweeps'],60 if persistent else 30)
                for key,value in full['sampler'].items():
                    if isinstance(value,torch.Tensor):
                        torch.testing.assert_close(value,resumed['sampler'][key],atol=0,rtol=0)
                    else:
                        self.assertEqual(value,resumed['sampler'][key])
            for index,values in full['optimizer']['state'].items():
                for key,value in values.items():
                    torch.testing.assert_close(value,resumed['optimizer']['state'][index][key],atol=0,rtol=0)

    def test_single_temperature_pcd_carries_through_field_change(self):
        self.check_continuation('cpu',1)

    def test_tempered_pcd_carries_through_field_change(self):
        self.check_continuation('cpu',4)

    def test_tempered_initialization_checkpoint_resumes(self):
        self.check_continuation('cpu',4,interrupt_update=0)

    def test_tempered_reset_restarts_ladder_between_stints(self):
        self.check_continuation('cpu',4,persistent=False)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_pcd_curriculum_checkpoint_continuation(self):
        self.check_continuation('cuda',4)


if __name__=='__main__':
    unittest.main()
