"""Verify anchor coverage and matched-budget projector identities."""
import unittest
from .l6_convergence_campaign import contexts, projection_specs


class ConvergencePlanTest(unittest.TestCase):
    def test_coverage_including_existing_extension(self):
        found = {(t['gamma'], t['sample_id'], t['seed']) for t in contexts()}
        existing = {(.55, 60000, s) for s in [0, 1]}
        expected = {(g, d, s) for g in [.55, .60, .65, .70]
                    for d in [60000, 60001] for s in [0, 1]}
        self.assertFalse(found & existing)
        self.assertEqual(found | existing, expected)

    def test_budget_matching_and_sampling_exclusion(self):
        row = dict(task=contexts()[0], extra_updates=0, model='original.pt',
                   record=dict(sampling_alarms=[]))
        before = projection_specs(row, 'results')
        after = projection_specs(dict(row, extra_updates=1200, model='extended.pt'), 'results')
        self.assertEqual([s['task'] for s, _ in before], [s['task'] for s, _ in after])
        self.assertNotEqual(before[0][1], after[0][1])
        self.assertEqual(projection_specs(dict(row, record=dict(sampling_alarms=['start disagreement'])), 'results'), [])


if __name__ == '__main__':
    unittest.main()
