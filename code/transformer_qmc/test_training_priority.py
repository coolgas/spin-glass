"""Weak-field queue ordering retains the entire configured campaign."""
import unittest

from .parallel_disorder_scaling import ordered_tasks, context, projector_priority


class PriorityTests(unittest.TestCase):
    def test_projectors_prioritize_weak_field_main_ensemble(self):
        tasks=[dict(gamma=g,audit=a,sample_id=s,L=L,kind='rbm_2',repeat=0)
               for g in [.6,.55] for a in ['time','main'] for s in [60001,60000] for L in [6,4]]
        ordered=sorted(tasks,key=projector_priority)
        self.assertEqual([t['gamma'] for t in ordered],sorted(t['gamma'] for t in tasks))
        low=[t for t in ordered if t['gamma']==.55]
        self.assertEqual([t['audit'] for t in low],['main']*4+['time']*4)

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
