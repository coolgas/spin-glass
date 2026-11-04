"""Checkpoint recovery must preserve forward-walking ancestry and RNG state."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .collective_campaign import label, run_one, tasks
from .physics import basis, diagonal, make_af_sample
from .production_protocol import identity


class CampaignTests(unittest.TestCase):
    def test_interrupted_forward_walk_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=root/'inputs';(inputs/'guides').mkdir(parents=True)
            logs=-.05*diagonal(make_af_sample(2,50025).hamiltonian(.15),basis(12))
            np.savez_compressed(inputs/'guides'/(label(.25,50025)+'.npz'),logs=logs,original_logs=logs)
            for method in ['ct','natural','guided','broad','triangle']:
                with self.subTest(method=method):
                    task=next(t for t in tasks('pilot') if t['method']==('guided' if method=='triangle' else method)).copy()
                    task['method']=method
                    task.update(walkers=32,equil=.2,forward=1.,cycles=3,audit='verification')
                    full=root/('full_'+method);resumed=root/('resumed_'+method)
                    run_one(task,full,inputs)
                    clock=itertools.count()
                    with patch('transformer_qmc.collective_campaign.time.monotonic',side_effect=lambda:next(clock)*.001):
                        paused=run_one(task,resumed,inputs,.015)
                    self.assertEqual(paused['status'],'paused_time_cap')
                    key=identity(task)[:24]
                    progress=json.loads((resumed/'progress'/(key+'.json')).read_text())
                    self.assertGreater(progress['step'],round(task['equil']/task['delta']))
                    self.assertLess(progress['step'],round((task['equil']+task['forward'])/task['delta']))
                    run_one(task,resumed,inputs)
                    with np.load(full/'runs'/(key+'.npz')) as a,np.load(resumed/'runs'/(key+'.npz')) as b:
                        for name in a.files:np.testing.assert_array_equal(a[name],b[name])


if __name__=='__main__':unittest.main()
