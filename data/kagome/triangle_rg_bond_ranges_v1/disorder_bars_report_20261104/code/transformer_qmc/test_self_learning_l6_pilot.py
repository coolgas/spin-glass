"""Finite scope and matched measurement identities for the L6 pilot."""
from pathlib import Path
import unittest

from .l6_convergence_campaign import projection_specs
from .self_learning_l6_pilot import contexts, measurement_task, settings


class FeedbackPilotTests(unittest.TestCase):
    def test_four_contexts_and_finite_budget(self):
        tasks = contexts()
        self.assertEqual(len(tasks), 4)
        self.assertEqual({(t['gamma'],t['sample_id'],t['seed']) for t in tasks},
                         {(.6,d,s) for d in [60000,60001] for s in [0,1]})
        configs = [settings(Path('initial.pt'),Path('output'),t) for t in tasks]
        self.assertEqual(sum(a.stints for a in configs),24)
        self.assertEqual(sum(a.stints*a.updates for a in configs),24000)

    def test_final_measurements_match_existing_control_tasks(self):
        for task in contexts():
            baseline = dict(task=task,extra_updates=0,model='baseline.pt',
                            record=dict(sampling_alarms=[]))
            specs = projection_specs(baseline,Path('control'))
            self.assertEqual([s['task'] for s,_ in specs],
                             [measurement_task(task,r) for r in [0,1]])


if __name__ == '__main__':
    unittest.main()
