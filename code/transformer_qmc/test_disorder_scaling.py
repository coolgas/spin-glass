"""Independent observable identities and interrupted-projector reproducibility."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .disorder_scaling import observables,project_task
from .physics import basis
from .production_protocol import exact_spectra,geometry
from .simple_guides import Jastrow
from .projector import propagate


class ScalingTests(unittest.TestCase):
    def test_exact_moment_identities(self):
        spins=basis(12).astype(float);rng=np.random.default_rng(43)
        probability=rng.random(len(spins));probability/=probability.sum()
        m=probability@spins;p=spins.T@(probability[:,None]*spins)
        measured=observables(np.tile(m,(4,1)),np.tile(p,(4,1,1)),2)
        truth=exact_spectra(probability,spins,geometry(2))['connected_spectrum']
        np.testing.assert_allclose(measured['spectrum'],truth,atol=1e-12)
        self.assertAlmostEqual(measured['q'],np.mean(m*m))
        self.assertAlmostEqual(measured['q2'],np.mean(p*p))

    def test_population_permutation(self):
        rng=np.random.default_rng(51);m=rng.random((4,12));p=rng.random((4,12,12))
        a=observables(m,p,2);order=[2,0,3,1];b=observables(m[order],p[order],2)
        for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-12)

    def test_checkpoint_replay(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'model.pt';torch.save(dict(model=Jastrow(12).state_dict()),model)
            task=dict(L=2,gamma=.7,sample_id=60000,kind='jastrow',training_seed=0,
                audit='test',walkers=32,equil=.1,forward=.1,start='uniform',delta=.1,cycles=2,repeat=0)
            reference=project_task(task,model,root/'full');calls=0
            def interrupted(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==9:raise RuntimeError('intentional interruption after checkpoint')
                return propagate(*args,**kwargs)
            with patch('transformer_qmc.disorder_scaling.propagate',interrupted):
                with self.assertRaisesRegex(RuntimeError,'intentional interruption'):
                    project_task(task,model,root/'resume')
            resumed=project_task(task,model,root/'resume')
            for key in ['q','q2','spectrum','minimum_descendant_ess','ancestry_alarm']:
                np.testing.assert_equal(reference[key],resumed[key])
            for path in (root/'resume/runs').glob('*.npz'):
                with np.load(path) as data:
                    raw=observables(data['magnetization_cycles'].mean(0),data['pair_cycles'].mean(0),2)
                np.testing.assert_allclose(raw['spectrum'],resumed['spectrum'])


if __name__=='__main__':unittest.main()
