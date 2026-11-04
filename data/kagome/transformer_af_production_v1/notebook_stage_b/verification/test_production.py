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


if __name__ == '__main__':
    unittest.main(verbosity=2)
