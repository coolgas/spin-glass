"""A segmented continuation must reproduce uninterrupted Adam and chain states."""
import argparse
from pathlib import Path
import tempfile
import unittest

import torch

from .sampled_simple_guides import train
from .extended_rbm_training import extend
from .parallel_disorder_scaling import fit_name


class ContinuationTest(unittest.TestCase):
    def test_adam_sampler_rng_continuity(self):
        torch.set_num_threads(1)
        task=dict(L=4,gamma=.55,sample_id=60000,kind='rbm_2',seed=0,mode='direct')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'short';b=root/'full';a.mkdir();b.mkdir()
            settings=dict(device='cpu',chains=4,temperatures=2,sweeps_per_step=1,diagnostic_sweeps=4,
                learning_rate=.001,inputs=root/'inputs')
            train(task,argparse.Namespace(**settings,steps=2),a,float('inf'))
            train(task,argparse.Namespace(**settings,steps=4),b,float('inf'))
            _,continued=extend(a/(fit_name(task)+'.pt'),root/'continued',2,'cpu')
            x=torch.load(continued,weights_only=False);y=torch.load(b/(fit_name(task)+'.pt'),weights_only=False)
            def equal(x,y):
                if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
                elif isinstance(x,dict):
                    self.assertEqual(x.keys(),y.keys())
                    for k in x:equal(x[k],y[k])
                elif isinstance(x,list):
                    self.assertEqual(len(x),len(y))
                    for p,q in zip(x,y):equal(p,q)
                else:self.assertEqual(x,y)
            for key in ['model','optimizer','sampler','rng']:equal(x[key],y[key])


if __name__=='__main__':unittest.main()
