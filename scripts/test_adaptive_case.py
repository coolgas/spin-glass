import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rydberg_kagome.campaign import run_case
from rydberg_kagome.results import digest


class CaseTests(unittest.TestCase):
    def test_pause_resume_final_moments_sampling_and_immutable_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'case'
            args=dict(output=str(root),L=4,sid=0,model_name='full',delta=2.,omega=.03,
                chi=2,dimensions=[4,8],pairs=2000,max_sweeps=120,validate=False,adaptive=True)
            result=run_case(**args,invocation_sweeps=11)
            self.assertFalse(result['completed']); self.assertEqual(result['sweeps_per_start'],[6,5])
            paused=json.loads((root/'resume.json').read_text())['state']['moments']
            for moments in paused:
                self.assertTrue(np.isfinite(moments['5']['canonical_error']))
            self.assertFalse((root/'result.json').exists())
            result=run_case(**args)
            self.assertTrue(result['completed']); self.assertTrue(result['solver_pass'])
            self.assertTrue(result['sampling_moment_check'])
            self.assertFalse(result['production_qualified']); self.assertFalse(result['rg_converged'])
            self.assertTrue(all(e['sweeps']<=120 for e in result['endpoints']))
            diagnostics=result['canonical_diagnostics']
            self.assertEqual(diagnostics['interval_sweeps'],5)
            saved_moments=json.loads((root/'resume.json').read_text())['state']['moments']
            for seed,records in enumerate(diagnostics['starts']):
                self.assertEqual(records[0]['canonical_error'],paused[seed]['5']['canonical_error'])
                self.assertEqual([r['sweep'] for r in records],list(range(5,result['endpoints'][seed]['sweeps']+1,5)))
                for record in records:
                    self.assertEqual(record['canonical_error'],saved_moments[seed][str(record['sweep'])]['canonical_error'])
                self.assertLess(records[-1]['canonical_error'],1e-8)
            with np.load(root/'pq.npz') as h: self.assertEqual(int(h['counts'].sum()),2000)
            saved=digest(root/'result.json')
            again=run_case(**args)
            self.assertEqual(again,result); self.assertEqual(digest(root/'result.json'),saved)
            bad=dict(args,pairs=3000)
            with self.assertRaisesRegex(ValueError,'mismatch'): run_case(**bad)
            direct=run_case(**dict(args,output=str(Path(folder)/'uninterrupted')))
            for a,b in zip(result['endpoints'],direct['endpoints']):
                self.assertEqual(a['sweeps'],b['sweeps'])
                self.assertAlmostEqual(a['energy'],b['energy'],places=10)
                self.assertAlmostEqual(a['observables']['qea'],b['observables']['qea'],places=10)
            np.testing.assert_allclose(result['sampling']['qmean'],direct['sampling']['qmean'],atol=1e-12)
            for resumed,direct_records in zip(diagnostics['starts'],direct['canonical_diagnostics']['starts']):
                np.testing.assert_allclose([r['canonical_error'] for r in resumed],
                                           [r['canonical_error'] for r in direct_records],atol=1e-11)

    def test_memory_deferral_is_resumable_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'case'
            args=dict(output=str(root),L=4,sid=1,model_name='full',delta=2.,omega=.03,
                chi=2,dimensions=[4,8],pairs=20,validate=False,adaptive=True)
            with patch('rydberg_kagome.adaptive.snapshot',return_value=dict(rss_bytes=10**8,peak_rss_bytes=10**8,available_bytes=0)):
                result=run_case(**args)
            self.assertEqual(result['state'],'memory_deferred')
            self.assertFalse((root/'result.json').exists())
            self.assertTrue((root/'resume.json').exists())
            result=run_case(**args,invocation_sweeps=1)
            self.assertEqual(result['sweeps_per_start'],[1,0])

    def test_resume_after_interrupted_final_sampling(self):
        # Optimization has completed and final state receipts exist, but a
        # sampling failure must still permit restart without more DMRG sweeps.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'case'
            args=dict(output=str(root),L=4,sid=0,model_name='full',delta=2.,omega=.03,
                chi=2,dimensions=[4,8],pairs=100,validate=False,adaptive=True)
            with patch('rydberg_kagome.adaptive_case.mps_samples',side_effect=RuntimeError('sampling interrupted')):
                with self.assertRaisesRegex(RuntimeError,'sampling interrupted'): run_case(**args)
            self.assertFalse((root/'result.json').exists())
            resume=json.loads((root/'resume.json').read_text())
            self.assertTrue(resume['state']['complete'])
            saved={seed:digest(root/f'start{seed}-final.pkl.gz') for seed in (0,1)}
            with patch('rydberg_kagome.adaptive.one_sweep',side_effect=AssertionError('must not repeat optimization')):
                result=run_case(**args)
            self.assertTrue(result['completed'])
            self.assertEqual(result['resume_generation'],resume['generation'])
            for seed in (0,1):
                self.assertEqual(digest(root/f'start{seed}-final.pkl.gz'),saved[seed])
                self.assertEqual(result['endpoints'][seed]['sweeps'],len(resume['state']['history'][seed]))
            # Immutable-result reuse must also reject a damaged final artifact.
            with (root/'pq.npz').open('ab') as f: f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'Completed output hash mismatch'): run_case(**args)


if __name__=='__main__': unittest.main()
