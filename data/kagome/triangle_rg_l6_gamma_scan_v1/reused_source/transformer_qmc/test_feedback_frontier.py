"""Validate the finite campaign design and fail-closed physical screen."""
import copy
import argparse
from pathlib import Path
import tempfile
import unittest
from .feedback_frontier import variants, settings, contexts, measurement_task, l2_task, screen, audit_training, verify_measurement
from .self_learning_rbm import run
from .projected_overlap_histograms import run_one


class FrontierTests(unittest.TestCase):
    def test_end_to_end_raw_training_and_measurement_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            cfg=argparse.Namespace(L=2,gamma=.6,sample_id=60000,seed=0,stints=2,walkers=8,
                equil=.1,collect=.2,delta=.1,sample_walkers=8,updates=3,batch=8,gibbs=1,
                initial_sweeps=1,lr=.001,device='cpu',initial=None,output=root/'training',
                islands=2,negative_temperatures=4,step_rms=.025,step_max=.15)
            run(cfg);audited=audit_training(root/'training',2,60000)
            self.assertEqual(len(audited),2)
            self.assertIn('negative_gradient_exact_rms',audited[-1])
            task=dict(l2_task(0,0),walkers=8,equil=.1,forward=.2,cycles=2)
            measurement=run_one(dict(model=str(root/'training/stint_001/model.pt'),task=task),root/'projector')
            self.assertTrue(verify_measurement(measurement,root/'projector')['reconstructed'])
            bad=copy.deepcopy(measurement);bad['q2']+=.01
            with self.assertRaises(AssertionError):verify_measurement(bad,root/'projector')

    def test_scope_and_population_control_are_matched(self):
        arms=variants();self.assertEqual(len(arms),5)
        self.assertEqual({(t['sample_id'],t['seed']) for t in contexts()},
                         {(sid,seed) for sid in [60000,60001] for seed in [0,1]})
        for L,stints in [(2,6),(6,12)]:
            for variant in arms:
                cfg=settings(variant,L,60000,0,Path('initial.pt'),Path('output'))
                self.assertEqual(cfg.stints,stints);self.assertEqual(cfg.gamma,.6)
                self.assertEqual(cfg.updates,1000);self.assertEqual(cfg.sample_walkers,64)
        large=arms['tempered_bounded_large'];islands=arms['tempered_bounded_islands']
        self.assertEqual(large['walkers']*large['islands'],4096)
        self.assertEqual(islands['walkers']*islands['islands'],4096)
        self.assertEqual(large['negative_temperatures'],islands['negative_temperatures'])
        self.assertEqual(large['step_rms'],islands['step_rms'])

    def test_final_measurement_identities_match_original_controls(self):
        for task in contexts():
            for repeat in [0,1]:
                t=measurement_task(task,repeat)
                self.assertEqual(t,dict(L=6,gamma=.6,sample_id=task['sample_id'],kind='rbm_2',
                    training_seed=task['seed'],audit='l6_convergence_training',walkers=512,
                    equil=16.,forward=16.,start='guide',delta=.1,cycles=8,repeat=repeat))
        for seed in [0,1]:
            for repeat in [0,1]:
                self.assertEqual(l2_task(seed,repeat),dict(L=2,gamma=.6,sample_id=60000,kind='rbm_2',
                    training_seed=seed,audit='self_learning_final_l2',walkers=512,
                    equil=16.,forward=16.,start='guide',delta=.1,cycles=8,repeat=repeat))

    def test_screen_rejects_missing_failed_and_nonfinite_results(self):
        row=dict(seed=0,audit_complete=True,measurements=[dict(task=dict(repeat=r),
            ancestry_alarm=False,errors=dict(energy_error_per_site=0.,q2_error=0.,histogram_tv=0.,
            spectrum_max_tolerance_ratio=.5)) for r in [0,1]],
            tempered=dict(rows=[dict(start=start,round_trips=100,exchange_fraction=[.5]*7,
            gradient_rms_error=.005) for start in ['random','up','down']]))
        rows=[row,copy.deepcopy(row)];rows[1]['seed']=1
        self.assertTrue(screen(rows)['pass_diagnostic_screen'])
        self.assertFalse(screen(rows[:1])['pass_diagnostic_screen'])
        self.assertFalse(screen(rows+[copy.deepcopy(row)])['pass_diagnostic_screen'])
        for kind in ['alarm','nan','missing_repeat','missing_start','spectrum','gradient']:
            bad=copy.deepcopy(rows)
            if kind=='alarm':bad[0]['measurements'][0]['ancestry_alarm']=True
            if kind=='nan':bad[0]['measurements'][0]['errors']['q2_error']=float('nan')
            if kind=='missing_repeat':bad[0]['measurements'].pop()
            if kind=='missing_start':bad[0]['tempered']['rows'].pop()
            if kind=='spectrum':bad[0]['measurements'][0]['errors']['spectrum_max_tolerance_ratio']=1.01
            if kind=='gradient':bad[0]['tempered']['rows'][0]['gradient_rms_error']=.03
            self.assertFalse(screen(bad)['pass_diagnostic_screen'],kind)


if __name__=='__main__':unittest.main()
