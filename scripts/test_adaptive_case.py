import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rydberg_kagome.campaign import run_case
from rydberg_kagome.results import digest


class CaseTests(unittest.TestCase):
    def compact_args(self,root):
        return dict(output=str(root),L=4,sid=0,model_name='full',delta=2.,omega=.03,
            chi=2,dimensions=[4,8],pairs=2000,validate=False,adaptive=True,
            final_state_retention='analysis')

    def test_compact_retirement_after_committed_result_and_orphan_preservation(self):
        from rydberg_kagome.adaptive_case import retire_committed_restart
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'case'; args=self.compact_args(root)
            with patch('rydberg_kagome.adaptive_case.retire_committed_restart',side_effect=RuntimeError('after commit')):
                with self.assertRaisesRegex(RuntimeError,'after commit'): run_case(**args)
            result=json.loads((root/'result.json').read_text())
            self.assertTrue(result['sampling_moment_check'])
            orphan=root/'restart'/'orphan.pkl.gz'; orphan.write_bytes(b'evidence')
            self.assertTrue((root/'resume.json').exists())
            with patch('rydberg_kagome.adaptive.optimize',side_effect=AssertionError('must reuse result')):
                self.assertEqual(run_case(**args),result)
            self.assertEqual(orphan.read_bytes(),b'evidence')
            self.assertFalse((root/'resume.json').exists())
            self.assertEqual(list(root.glob('start*-final.pkl.gz')),[])
            self.assertTrue((root/'optimization-history.json').exists())
            full=run_case(**dict(args,output=str(Path(folder)/'full'),final_state_retention='full'))
            for a,b in zip(result['endpoints'],full['endpoints']):
                self.assertAlmostEqual(a['energy'],b['energy'],places=10)
                self.assertEqual(a['observables'],b['observables'])
            with np.load(root/'pq.npz') as a, np.load(Path(folder)/'full'/'pq.npz') as b:
                np.testing.assert_array_equal(a['counts'],b['counts'])
            with (root/'pq.npz').open('ab') as f: f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Completed output hash mismatch'): run_case(**args)

    def test_compact_sampling_failure_retains_restart_and_interruption_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'case'; args=self.compact_args(root)
            with patch('rydberg_kagome.adaptive_case.mps_samples',side_effect=RuntimeError('sampling interrupted')):
                with self.assertRaisesRegex(RuntimeError,'sampling interrupted'): run_case(**args)
            self.assertFalse((root/'result.json').exists())
            self.assertTrue((root/'resume.json').exists())
            with patch('rydberg_kagome.adaptive.one_sweep',side_effect=AssertionError('must not optimize')), \
                 patch('rydberg_kagome.adaptive_case.sampling_check',return_value=False):
                result=run_case(**args)
            self.assertTrue(result['storage']['restart_retained'])
            self.assertEqual(result['retired_artifacts'],{})
            self.assertEqual(len(list((root/'restart').glob('*.pkl.gz'))),2)
            self.assertEqual(run_case(**args),result)

    def test_thirty_sweep_cadence_writes_one_pair_and_forces_pause_save(self):
        from rydberg_kagome import adaptive
        committed=[]
        original=adaptive.save_restart
        def record(root,engines,state,identity):
            committed.append(tuple(len(h) for h in state['history']))
            return original(root,engines,state,identity)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'case'
            with patch('rydberg_kagome.adaptive.save_restart',side_effect=record), \
                 patch('rydberg_kagome.adaptive.final_window_pass',return_value=False), \
                 patch('rydberg_kagome.adaptive.growth_reason',return_value=[]):
                result=run_case(output=str(root),L=4,sid=0,model_name='full',delta=2.,omega=.03,
                    chi=2,dimensions=[4,8],pairs=20,validate=False,adaptive=True,invocation_sweeps=125)
            self.assertEqual(result['sweeps_per_start'],[63,62])
            self.assertEqual(committed,[(0,0),(30,30),(60,60),(63,62)])
            self.assertEqual(json.loads((root/'inputs.json').read_text())['checkpoint_interval'],30)
            self.assertEqual(len(list((root/'restart').glob('*.pkl.gz'))),2)
            self.assertEqual(len(list((root/'restart').glob('*.json'))),1)

    def test_pause_resume_final_moments_sampling_and_immutable_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'case'
            args=dict(output=str(root),L=4,sid=0,model_name='full',delta=2.,omega=.03,
                chi=2,dimensions=[4,8],pairs=2000,max_sweeps=120,validate=False,adaptive=True)
            result=run_case(**args,invocation_sweeps=21)
            self.assertFalse(result['completed']); self.assertEqual(result['sweeps_per_start'],[11,10])
            paused=json.loads((root/'resume.json').read_text())['state']['moments']
            self.assertEqual(len(list((root/'restart').glob('*.pkl.gz'))),2)
            self.assertEqual(len(list((root/'restart').glob('*.json'))),1)
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
            self.assertFalse((root/'restart').exists())
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
