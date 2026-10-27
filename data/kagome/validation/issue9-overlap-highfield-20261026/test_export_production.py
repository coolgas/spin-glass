"""Check the disorder/replica hierarchy with analytically known moments."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from export_production import summarize


class ProductionSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = dict(n_sites=108, samples=[])
        for sample, values in enumerate(((.02, .06), (.14, .18))):
            names = []
            for population, value in enumerate(values):
                name = f's{sample}_p{population}.npz'
                names.append(name)
                np.savez(self.root/name, n_sites=108, n_replicas=2, n_walkers=10000,
                    n_equilibrate=32, n_measure=32, tau_block=.2, n_forward_cycles=8,
                    tau_forward_requested=.4, floor_unique=2, floor_effective=8,
                    q_series=np.zeros(8), q2_series=np.full(8, value),
                    q3_series=np.zeros(8), q4_series=np.full(8, value**2),
                    energy_measure_trace=np.full(32, -552.),
                    projection_complete=(sample == 0), forward_times=np.full(8, .3 if sample else .4),
                    min_effective_ancestors=9.)
            self.spec['samples'].append(dict(sample_id=sample, populations=names))

    def test_disorder_average_precedes_sqrt_and_uncertainties_stay_separate(self):
        result = summarize(self.spec, self.root)
        # Sample moments .04 and .16: sqrt of the disorder mean is sqrt(.10).
        self.assertAlmostEqual(result['q_ea'], np.sqrt(.10))
        self.assertNotAlmostEqual(result['q_ea'], (.2+.4)/2)
        self.assertAlmostEqual(result['chi_sg'], 10.8)
        # Each sample has SE .02; disorder mean has projector SE .02/sqrt(2).
        self.assertAlmostEqual(result['q2_projector_se'], .02/np.sqrt(2))
        self.assertAlmostEqual(result['q2_sample_scatter_se'], .06)
        self.assertAlmostEqual(result['q2_disorder_se_estimate'], np.sqrt((.0072-.0004)/2))
        self.assertFalse(result['all_projection_complete'])
        self.assertEqual(len(result['samples']), 2)

    def test_one_sample_has_no_disorder_error(self):
        self.spec['samples'] = self.spec['samples'][:1]
        result = summarize(self.spec, self.root)
        self.assertIsNone(result['q2_disorder_se_estimate'])
        self.assertIsNone(result['q2_sample_scatter_se'])

    def test_duplicate_population_is_rejected(self):
        names = self.spec['samples'][0]['populations']
        names[1] = names[0]
        with self.assertRaisesRegex(ValueError, 'more than once'):
            summarize(self.spec, self.root)


if __name__ == '__main__':
    unittest.main()
