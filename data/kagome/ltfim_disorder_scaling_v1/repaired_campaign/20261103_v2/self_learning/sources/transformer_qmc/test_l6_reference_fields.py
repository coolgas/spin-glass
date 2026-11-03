"""Check fixed reference coverage and exact lower-field projection matching."""
from pathlib import Path
import unittest

from .l6_reference_fields import contexts, plan
from .l6_convergence_campaign import projection_specs


class ReferenceFieldTests(unittest.TestCase):
    def test_fixed_coverage_and_finite_budget(self):
        tasks = contexts()
        self.assertEqual({(t['gamma'], t['sample_id'], t['seed']) for t in tasks},
                         {(g, d, s) for g in [.85, 1.] for d in [60000, 60001] for s in [0, 1]})
        self.assertEqual(len(tasks), 8)
        manifest = plan(Path('campaign'), 2)
        self.assertEqual(manifest['planned_milestones'], len(tasks)*3)
        self.assertEqual(manifest['maximum_projection_runs'], len(tasks)*3*2)
        self.assertIsNone(manifest['deadline'])
        self.assertFalse(manifest['ground_state_qualified'])

    def test_projection_protocol_and_budget_rng_matching(self):
        row = dict(task=contexts()[0], extra_updates=0, model='baseline.pt',
                   record=dict(sampling_alarms=[]))
        before = projection_specs(row, 'results')
        after = projection_specs(dict(row, extra_updates=1200, model='extended.pt'), 'results')
        self.assertEqual([s['task'] for s, _ in before], [s['task'] for s, _ in after])
        self.assertNotEqual(before[0][1], after[0][1])
        task = before[0][0]['task']
        self.assertEqual((task['walkers'], task['equil'], task['forward'], task['cycles']),
                         (512, 16., 16., 8))
        self.assertEqual(projection_specs(dict(row, record=dict(sampling_alarms=['mixing'])), 'results'), [])


if __name__ == '__main__':
    unittest.main()
