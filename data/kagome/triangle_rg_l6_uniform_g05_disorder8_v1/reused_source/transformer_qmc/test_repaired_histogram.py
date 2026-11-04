"""Interrupted histogram replay includes energies and numerical provenance."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import projected_overlap_histograms as projection
from .simple_guides import Jastrow
from .production_protocol import identity


class RepairedHistogramTests(unittest.TestCase):
    def test_interrupted_replay_and_numerics_mismatch(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'model.pt'
            torch.save(dict(model=Jastrow(12).state_dict()),model)
            task=dict(L=2,gamma=.7,sample_id=60000,kind='jastrow',training_seed=0,
                audit='test',walkers=32,equil=.1,forward=.1,start='uniform',delta=.1,cycles=2,repeat=0)
            spec=dict(task=task,model=str(model));key=identity(task)[:24]
            reference=projection.run_one(spec,root/'full');calls=0;propagate=projection.propagate
            def interrupted(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==9:raise RuntimeError('intentional interruption after checkpoint')
                return propagate(*args,**kwargs)
            with patch.object(projection,'propagate',interrupted):
                with self.assertRaisesRegex(RuntimeError,'intentional interruption'):
                    projection.run_one(spec,root/'resume')
            digest=projection.digest
            def changed(path):
                return 'different_numerics' if Path(path).name=='qmc_numerics.py' else digest(path)
            with patch.object(projection,'digest',changed):
                with self.assertRaises(AssertionError):projection.run_one(spec,root/'resume')
            resumed=projection.run_one(spec,root/'resume')
            for name in ['q','q2','spectrum','mass','energy','growth_energy','minimum_descendant_ess','largest_lineage']:
                np.testing.assert_equal(reference[name],resumed[name])
            with np.load(root/'full/runs'/(key+'.npz')) as a,np.load(root/'resume/runs'/(key+'.npz')) as b:
                for name in a.files:np.testing.assert_array_equal(a[name],b[name])
                self.assertTrue(np.isfinite(a['energy']).all())
                self.assertEqual(a['energy'].shape,(3,4))
            self.assertEqual(resumed['provenance']['numerics'],'log_weight_v1')
            self.assertTrue((root/'resume/progress'/(key+'.json')).exists())


if __name__=='__main__':unittest.main(verbosity=2)
