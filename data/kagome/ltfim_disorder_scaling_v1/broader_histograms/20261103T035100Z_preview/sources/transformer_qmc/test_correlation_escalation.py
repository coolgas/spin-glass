"""Checks for correlation inference and rejecting corrupted campaign artifacts."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .assess_correlation_escalation import bootstrap, comparison, validate
from .correlation_lengths import length


class EscalationStatisticsTests(unittest.TestCase):
    def test_bootstrap_keeps_whole_momentum_vectors_and_equal_disorder_weights(self):
        # Common-scale fluctuations have nonzero spectral errors but no ratio error.
        scale = np.array([[1., 2., 3., 4.], [4., 5., 6., 7.]])
        shape = np.array([2., 1., 1., 1.])
        raw = scale[:, :, None]*shape
        boot, ix = bootstrap(raw, np.random.default_rng(71), draws=2000)
        expected_scale = scale[np.arange(2)[None, :, None], ix].mean(axis=(1, 2))
        np.testing.assert_allclose(boot, expected_scale[:, None]*shape)
        np.testing.assert_allclose(length(boot, 2), length(shape, 2), atol=1e-14)
        self.assertGreater(boot[:, 0].std(), 0)

    def test_independent_setting_contrast_retains_both_run_variances(self):
        rng = np.random.default_rng(19)
        target = np.array([2., 1., 1., 1.])
        a = np.tile(target, (12000, 1)); b = a.copy()
        a[:, 0] += rng.normal(0, .04, len(a))
        b[:, 0] += rng.normal(0, .07, len(b))
        result = comparison(target, b, target, a)
        expected = (length(b, 2)-length(a, 2)).std(ddof=1)
        self.assertAlmostEqual(result['xi_difference_mc_se'], expected, places=12)
        self.assertGreater(result['xi_difference_mc_se'], .95*np.hypot(length(a, 2).std(), length(b, 2).std()))

    def test_invalid_ratios_are_flagged_not_replaced_by_zero(self):
        boot = np.array([[2., 1., 1., 1.], [.5, 1., 1., 1.], [3., 1., 1., 1.]])
        result = comparison(boot.mean(0), boot, np.array([2., 1., 1., 1.]))
        self.assertAlmostEqual(result['invalid_bootstrap_fraction'], 1/3)
        result = comparison(np.array([.5, 1., 1., 1.]), boot, np.array([2., 1., 1., 1.]))
        self.assertIsNone(result['xi_difference'])
        self.assertIsNone(result['xi_over_side_95_bound'])

    def test_paired_cycle_halves_preserve_shared_run_noise(self):
        raw = np.zeros((2, 4, 4))+np.array([2., 1., 1., 1.])
        raw[:, :, 0] += np.arange(8).reshape(2, 4)/100
        boot, _ = bootstrap(raw, np.random.default_rng(8))
        result = comparison(raw.mean((0, 1)), boot, raw.mean((0, 1)), boot)
        self.assertEqual(result['xi_difference_mc_se'], 0)
        self.assertEqual(result['spectral_difference_mc_se'], [0., 0., 0., 0.])


class EscalationArtifactTests(unittest.TestCase):
    def test_real_pilot_detects_duplicate_and_tampered_record(self):
        root = Path(__file__).resolve().parents[2]/'data/kagome'
        source, parent = root/'ltfim_correlation_lengths_v4', root/'ltfim_correlation_lengths_v3'
        paths = sorted((source/'runs').glob('*.json'))
        if not paths:
            self.skipTest('Requires a downloaded v4 pilot')
        temporary = Path('/private/var/folders/pl/l9f5xt416qq5qz_7wfhm_94c0000gn/T/opencode')
        with tempfile.TemporaryDirectory(prefix='correlation-audit-', dir=temporary) as directory:
            fixture = Path(directory)
            (fixture/'runs').mkdir()
            shutil.copy2(source/'manifest.json', fixture/'manifest.json')
            path = fixture/'runs'/paths[0].name
            shutil.copy2(paths[0], path)
            _, records, missing = validate(fixture, parent)
            self.assertEqual((len(records), len(missing)), (1, 383))
            original = json.loads(path.read_text())
            altered = copy.deepcopy(original)
            altered['seed'] += 1
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'Seed mismatch'):
                validate(fixture, parent)
            altered = copy.deepcopy(original)
            altered['guide_sha256'] = '0'*64
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'Reference hash mismatch'):
                validate(fixture, parent)
            path.write_text(json.dumps(original))
            shutil.copy2(path, fixture/'runs'/'duplicate.json')
            with self.assertRaisesRegex(ValueError, 'Duplicate task|Unexpected task'):
                validate(fixture, parent)


if __name__ == '__main__':
    unittest.main(verbosity=2)
