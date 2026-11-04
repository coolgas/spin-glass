"""Check task isolation, serial equivalence, and checkpoint reuse on CUDA."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from .parallel_disorder_scaling import configure_training_worker,training_worker,fit_name


class ParallelTrainingTests(unittest.TestCase):
    def test_serial_parallel_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);serial=root/'serial';parallel=root/'parallel'
            serial.mkdir();parallel.mkdir()
            tasks=[dict(L=4,gamma=.7,sample_id=60000,kind=kind,seed=0,mode='anneal')
                   for kind in ['jastrow','rbm_2']]
            settings=dict(device='cuda',chains=16,temperatures=3,steps=25,sweeps_per_step=2,
                          diagnostic_sweeps=4,inputs=root/'inputs')
            ctx=mp.get_context('spawn');deadline=time.monotonic()+120
            with ProcessPoolExecutor(max_workers=1,mp_context=ctx,initializer=configure_training_worker) as pool:
                references=[pool.submit(training_worker,t,settings,str(serial),deadline).result() for t in tasks]
            with ProcessPoolExecutor(max_workers=2,mp_context=ctx,initializer=configure_training_worker) as pool:
                futures=[pool.submit(training_worker,t,settings,str(parallel),deadline) for t in tasks]
                results=[f.result() for f in futures]
                reused=pool.submit(training_worker,tasks[0],settings,str(parallel),deadline).result()
            self.assertEqual(reused,results[0])
            for task,a,b in zip(tasks,references,results):
                self.assertEqual(a['signature'],b['signature'])
                ca=torch.load(serial/(fit_name(task)+'.pt'),weights_only=False,map_location='cpu')
                cb=torch.load(parallel/(fit_name(task)+'.pt'),weights_only=False,map_location='cpu')
                for key in ca['model']:
                    torch.testing.assert_close(ca['model'][key],cb['model'][key],rtol=1e-12,atol=1e-12)
                for da,db in zip(a['diagnostics'],b['diagnostics']):
                    np.testing.assert_allclose([da['energy'],da['variance']],
                                               [db['energy'],db['variance']],rtol=1e-12,atol=1e-12)
            self.assertFalse((parallel/'status.json').exists())
            self.assertTrue(list((parallel/'worker_status').glob('*.json')))


if __name__=='__main__':unittest.main()
