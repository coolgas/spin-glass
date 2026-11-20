import unittest
from rydberg_kagome.production import tasks


class ProductionTests(unittest.TestCase):
    def test_fixed_complete_manifest(self):
        rows=tasks()
        self.assertEqual(len(rows),23040)
        self.assertEqual(len({r['id'] for r in rows}),23040)
        self.assertEqual({r['sample'] for r in rows},set(range(128)))
        self.assertTrue(all(r['chi']==12 and r['dimensions']==[128] and r['fixed_production'] for r in rows))
        self.assertEqual(sum(not r['no_exact'] for r in rows),18)
