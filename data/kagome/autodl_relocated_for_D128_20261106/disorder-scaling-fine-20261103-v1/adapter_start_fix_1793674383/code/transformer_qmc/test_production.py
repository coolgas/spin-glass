"""Checkpoint parity, Fourier conventions, four-replica identities and task identity."""

import copy
import itertools
import unittest

import numpy as np

from .compare_pure_gfmc import BatchedConstantGuide
from .physics import make_af_sample
from .production_protocol import design, exact_spectra, geometry, production_tags, seed_for
from .projector import project, tags


class ProductionTests(unittest.TestCase):
    def test_resume_is_identical_at_every_checkpoint(self):
        ham = make_af_sample(2, 60000).hamiltonian(.15)
        guide = BatchedConstantGuide(12)
        kwargs = dict(n_walkers=16, n_replicas=4, tau_block=.02, n_equilibrate=2,
                      n_measure=2, tau_forward=.04, n_cycles=2, seed=981)
        checkpoints = []
        tagger = lambda s: production_tags(s, geometry(2))
        complete = project(ham, .5, guide, **kwargs, tagger=tagger,
                           checkpoint=lambda s: checkpoints.append(copy.deepcopy(s)))
        for state in checkpoints:
            resumed = project(ham, .5, guide, **kwargs, tagger=tagger, resume=state)
            self.assertEqual(set(resumed), set(complete))
            for key in complete:
                np.testing.assert_array_equal(resumed[key], complete[key], err_msg=key)
        wrong = copy.deepcopy(checkpoints[0]);wrong['configuration']['seed'] += 1
        with self.assertRaises(ValueError):
            project(ham, .5, guide, **kwargs, resume=wrong)

    def test_vector_tags_keep_scalar_results(self):
        rng = np.random.default_rng(77)
        for L in (2, 5, 10):
            s = rng.choice([-1,1], size=(7,4,3*L*L)).astype(np.int8)
            g = geometry(L); got = production_tags(s, g); old = tags(s)
            for key in old:np.testing.assert_array_equal(got[key], old[key])
            np.testing.assert_allclose(got['connected_spectrum'][:,0], got['chi_connected'], atol=1e-12)
            cells = np.array([[2*x+y,np.sqrt(3)*y] for y in range(L) for x in range(L)])
            points = (cells[:,None]+np.array([[0,0],[1,0],[.5,np.sqrt(3)/2]])[None]).reshape(-1,2)
            phase = np.exp(-1j*(points@g['momenta'].T))
            d = .5*(s[:,0]-s[:,1])*(s[:,2]-s[:,3])
            np.testing.assert_allclose(got['connected_spectrum'], abs(d@phase)**2/(3*L*L), atol=1e-11)
            np.testing.assert_allclose(got['spin_spectrum'], (abs(s@phase)**2).mean(1)/(3*L*L), atol=1e-11)

    def test_exact_four_replica_spectra(self):
        # Exhaust all four independent draws from a nontrivial three-state distribution.
        rng=np.random.default_rng(19); spins=rng.choice([-1.,1.],size=(3,12)); p=np.array([.2,.3,.5])
        indexes=np.array(list(itertools.product(range(3),repeat=4)))
        weights=p[indexes].prod(1); lattice=geometry(2)
        values=production_tags(spins[indexes],lattice)
        exact=exact_spectra(p,spins,lattice)
        for name,target in exact.items():
            np.testing.assert_allclose(np.tensordot(weights,values[name],axes=(0,0)), target, atol=2e-14)
        zz=spins@spins.T/12
        self.assertAlmostEqual(weights@values['q4'],float(np.sum(p[:,None]*p[None]*zz**4)))

    def test_size_parity_and_seeds(self):
        d=design();self.assertEqual(d['anchor_runs'],15680);self.assertEqual(d['all_anchor_runs'],62720)
        self.assertIn(10,d['anchor_sizes']);self.assertFalse(d['final_dispatch_enabled'])
        tasks=[dict(L=L, sample_id=60000, gamma=.5,h=0.,precision=precision,repeat=r)
               for L in d['sizes'] for precision in ['float32','float64'] for r in range(4)]
        seeds=[seed_for(t) for t in tasks]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertEqual(seed_for(tasks[0]),seed_for(dict(reversed(list(tasks[0].items())))))

    def test_queue_has_equal_settings_for_all_sizes(self):
        from .production_runner import build_tasks
        from .production_protocol import canonical
        tasks=build_tasks()
        baseline=None
        for L in range(4,11):
            policy={canonical({k:v for k,v in t.items() if k not in ('L','id','seed')})
                    for t in tasks if t['L']==L}
            if baseline is None:baseline=policy
            self.assertEqual(policy,baseline)
        self.assertTrue(all(t['sample_id']<61000 for t in tasks))

    def test_operational_queues_are_disjoint_and_complete(self):
        from .production_dispatch import split_queues
        from .production_runner import build_tasks
        tasks=build_tasks();completed={t['id'] for t in tasks[:3]}
        queues={lane:list(queue) for lane,queue in split_queues(tasks,completed).items()}
        gpu={t['id'] for t in queues['gpu']};cpu={t['id'] for t in queues['cpu']}
        self.assertFalse(gpu & cpu)
        self.assertEqual(gpu | cpu,{t['id'] for t in tasks}-completed)
        self.assertTrue(all(t['method']=='network' for t in queues['gpu']))
        self.assertTrue(all(t['method']!='network' for t in queues['cpu']))

    def test_notebook_stage_b_matches_agreed_pilot(self):
        from .notebook_stage_b import build_tasks
        tasks=build_tasks()
        self.assertEqual(len(tasks),288)
        expected=dict(n_walkers=64,n_replicas=4,tau_block=.1,n_equilibrate=20,
                      n_measure=10,tau_forward=1.,n_cycles=4)
        self.assertTrue(all(t['L']==4 and t['precision']=='float64' and
                            t['parameters']==expected and t['stage']=='notebook_stage_b' for t in tasks))
        actual={(t['sample_id'],t['gamma'],t['h'],t['method'],t['repeat']) for t in tasks}
        wanted=set(itertools.product(range(60000,60004),(.5,1.,2.),(0.,.15,.3),
                                     ('network','constant'),range(4)))
        self.assertEqual(actual,wanted)
        self.assertEqual(len({t['id'] for t in tasks}),288)
        self.assertEqual(len({t['seed'] for t in tasks}),288)

    def test_followup_compares_one_parameter_with_independent_runs(self):
        from .notebook_stage_b import build_tasks as pilot
        from .research_followup import build_tasks
        base=pilot();followup=build_tasks();tasks=base+followup
        self.assertEqual(len(followup),144)
        self.assertEqual(len({t['seed'] for t in tasks}),len(tasks))
        self.assertEqual(len({t['id'] for t in tasks}),len(tasks))
        for gamma,h,method,repeat in itertools.product((.5,2.),(0.,.05),('network','constant'),range(4)):
            selected={t.get('ladder','base'):t for t in tasks if t['sample_id']==60000 and
                      (t['gamma'],t['h'],t['method'],t['repeat'])==(gamma,h,method,repeat)}
            baseline='base' if h==0 else 'baseline_h005'
            for a,b,changed in [(baseline,'population256','n_walkers'),('population256','population1024','n_walkers'),
                                ('population1024','equilibration4','n_equilibrate'),('population1024','forward2','tau_forward')]:
                x,y=selected[a]['parameters'],selected[b]['parameters']
                self.assertEqual({k for k in x if x[k]!=y[k]},{changed})

    def test_followup_waits_for_parent_supervisor_completion(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from .notebook_stage_b import build_tasks
        from .research_followup import verify_parent
        from .production_protocol import identity
        with TemporaryDirectory() as directory:
            root=Path(directory);p=root/'notebook_stage_b';p.mkdir();(p/'failures').mkdir()
            tasks=build_tasks();manifest=dict(tasks_sha256=identity(tasks));manifest['signature']=identity(manifest)
            (p/'tasks.json').write_text(json.dumps(tasks));(p/'manifest.json').write_text(json.dumps(manifest))
            with patch('transformer_qmc.production_runner.verified_result',return_value=True):
                (p/'status.json').write_text(json.dumps(dict(state='running_notebook_stage_b')))
                self.assertFalse(verify_parent(root))
                (p/'status.json').write_text(json.dumps(dict(state='awaiting_stage_b_cost_review')))
                self.assertTrue(verify_parent(root))
                (p/'PAUSE.json').write_text('{}')
                with self.assertRaises(RuntimeError):verify_parent(root)

    def test_monitor_survives_handoff_and_stops_on_failure(self):
        import json,time
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from .watch_production import terminal_state
        with TemporaryDirectory() as directory:
            root=Path(directory);p=root/'notebook_stage_b_escalation';p.mkdir()
            active=dict(stage='notebook_stage_b');status=dict(state='awaiting_stage_b_cost_review')
            self.assertTrue(terminal_state(root,active,status))
            (p/'status.json').write_text(json.dumps(dict(state='waiting_for_base_pilot',updated_unix=time.time())))
            self.assertFalse(terminal_state(root,active,status))
            self.assertTrue(terminal_state(root,active,dict(state='failed')))
            (p/'status.json').write_text(json.dumps(dict(state='waiting_for_base_pilot',updated_unix=time.time()-200)))
            self.assertTrue(terminal_state(root,active,status))

    def test_stage_b_uncertainty_keeps_disorder_and_run_units(self):
        from .assess_stage_b import uncertainty
        values=np.array([[0.,2.,4.,6.],[10.,12.,14.,16.]])
        got=uncertainty(values)
        self.assertAlmostEqual(got['mean'],8.)
        self.assertAlmostEqual(got['disorder_se'],5.)
        self.assertAlmostEqual(got['mc_se'],np.sqrt(5/6))
        self.assertIsNone(uncertainty(values[:1])['disorder_se'])

    def test_physical_equivalence_requires_precision(self):
        from .assess_physical_convergence import difference_stats
        x=np.array([-3.,-1.,1.,3.])
        got=difference_stats(x,x,.002)
        self.assertEqual(got['status'],'unresolved_precision_or_shift')
        self.assertEqual(got['difference'],0.)
        self.assertAlmostEqual(got['independent_run_se'],np.sqrt(10/3))
        self.assertAlmostEqual(got['welch_df'],6.)
        # The notebook normal bound can pass while four-repeat t sensitivity does not.
        got=difference_stats(x/2000,x/2000,.002)
        self.assertEqual(got['status'],'within_pointwise_tolerance')
        self.assertFalse(got['welch_within_tolerance'])
        self.assertEqual(difference_stats(x,x,None)['status'],'no_prespecified_tolerance')
        for bad in ([],[0.,1.,2.],[0.,1.,2.,np.nan]):
            with self.assertRaises(ValueError):difference_stats(bad,x,.002)

    def test_physical_resolved_shift_and_zero_variance(self):
        from .assess_physical_convergence import difference_stats
        for sign in (-1,1):
            got=difference_stats([0.]*4,[sign*.01]*4,.002)
            self.assertEqual(got['status'],'resolved_shift_beyond_tolerance')
            self.assertEqual(got['welch95_low'],sign*.01)
            self.assertEqual(got['welch95_high'],sign*.01)
        got=difference_stats([0.]*4,[.001]*4,.002)
        self.assertEqual(got['status'],'within_pointwise_tolerance')
        self.assertEqual(got['independent_run_se'],0.)

    def test_monitor_waits_for_matching_continuation_outcome(self):
        import json,time
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from .watch_production import terminal_state
        with TemporaryDirectory() as directory:
            root=Path(directory);window=root/'continuations'/'window';window.mkdir(parents=True)
            (root/'active_continuation.json').write_text(json.dumps(dict(launch_path='/remote/continuations/window/launch.json')))
            active=dict(stage='notebook_stage_b_escalation')
            status=dict(state='awaiting_notebook_stage_b_escalation_review',updated_unix=time.time(),deadline_unix=123.)
            self.assertFalse(terminal_state(root,active,status))
            outcome=dict(finished_unix=time.time(),status={**status,'deadline_unix':122.},assessment_returncode=0)
            (window/'outcome.json').write_text(json.dumps(outcome))
            self.assertFalse(terminal_state(root,active,status))
            outcome['status']=status
            (window/'outcome.json').write_text(json.dumps(outcome))
            self.assertTrue(terminal_state(root,active,status))
            # Assessment failure is terminal too; its return code remains visible for notification.
            outcome['assessment_returncode']=1
            (window/'outcome.json').write_text(json.dumps(outcome))
            self.assertTrue(terminal_state(root,active,status))
            (window/'outcome.json').unlink()
            with self.assertRaisesRegex(RuntimeError,'finalization'):
                terminal_state(root,active,{**status,'updated_unix':time.time()-901})
            self.assertFalse(terminal_state(root,active,{**status,'state':'running_notebook_stage_b_escalation'}))

    def test_physical_analysis_omits_partial_sets_and_preserves_signed_overlap(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from .assess_physical_convergence import analyze
        from .production_protocol import identity
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for stage,level,count in [('notebook_stage_b','base',3),
                                      ('notebook_stage_b_escalation','population256',4)]:
                folder=root/stage;(folder/'results').mkdir(parents=True)
                tasks=[dict(id=f'{stage}{r}',sample_id=60000,gamma=.5,h=0.,method='network',
                    L=4,ladder=level,repeat=r,parameters=dict(n_walkers=64 if level=='base' else 256)) for r in range(4)]
                manifest=dict(tasks_sha256=identity(tasks));manifest['signature']=identity(manifest)
                (folder/'tasks.json').write_text(json.dumps(tasks));(folder/'manifest.json').write_text(json.dumps(manifest))
                for task in tasks[:count]:
                    result=dict(energy_per_site=-1.,q2=.1,q4=.02,q_ea_finite=-.01,
                        chi_connected_per_site=.1,connected_spectrum=[4.8,2.4],largest_lineage=[.25])
                    record=dict(task=task,signature=manifest['signature'],state='completed',result=result,
                        ancestry_alarm=False,minimum_descendant_ess=16.)
                    (folder/'results'/(task['id']+'.json')).write_text(json.dumps(record))
            with patch('transformer_qmc.assess_physical_convergence.assess',return_value={'input_signatures':{}}), \
                 patch('transformer_qmc.assess_physical_convergence.plot_estimates') as plot:
                self.assertEqual(analyze(root),[])
                estimates=plot.call_args.args[1]
            self.assertTrue(all(row['level']=='population256' for row in estimates))
            means={row['observable']:row['mean'] for row in estimates}
            self.assertEqual(means['q_ea_finite'],-.01)
            self.assertAlmostEqual(means['connected_spectrum_per_site_0'],.1)
            self.assertAlmostEqual(means['connected_spectrum_per_site_1'],.05)
            summary=json.loads((root/'physical_convergence/summary.json').read_text())
            baseline=next(row for row in summary['coverage'] if
                (row['gamma'],row['h'],row['method'],row['level'])==(.5,0.,'network','baseline'))
            self.assertEqual(baseline['completed'],3)
            self.assertEqual(baseline['minimum_ess_fraction'],.25)


if __name__ == '__main__':
    unittest.main(verbosity=2)
