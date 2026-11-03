"""Weak-field queue ordering retains the entire configured campaign."""
import unittest

from .parallel_disorder_scaling import ordered_tasks, context


class PriorityTests(unittest.TestCase):
    def test_lower_fields_across_sizes_precede_higher_l4_fields(self):
        manifest=dict(sizes=[4,6],fields=[.6,1.],
                      fields_by_size={'4':[.55,.56,.67], '6':[.55,.56]},
                      disorders=[60000,60001],kinds=['jastrow','rbm_2'],seeds=[0,1])
        tasks=ordered_tasks(manifest)
        self.assertEqual(len(tasks),40)
        self.assertEqual(len({context(t)+(t['seed'],) for t in tasks}),40)
        self.assertEqual([t['gamma'] for t in tasks],sorted(t['gamma'] for t in tasks))
        self.assertLess(max(i for i,t in enumerate(tasks) if t['L']==6),
                        min(i for i,t in enumerate(tasks) if t['gamma']==.67))
        self.assertEqual({t['gamma'] for t in tasks if t['L']==4},{.55,.56,.67})


if __name__=='__main__':unittest.main(verbosity=2)
