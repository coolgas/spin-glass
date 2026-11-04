"""Analytic limits and finite-sampling bias checks for the overlap decomposition."""
import numpy as np
import unittest

from .overlap_polarization import polarization_components


def test_against_explicit_replica_sum():
    from itertools import combinations, product

    rng = np.random.default_rng(831)
    tags = rng.choice([-1, 1], size=(2, 4, 3, 5))
    counts = np.array([[rng.multinomial(3, [.2, .3, .5]) for _ in range(4)] for _ in range(2)])
    expected = np.zeros(3)
    for a, b in combinations(range(4), 2):
        for c, d, w, v in product(range(2), range(2), range(3), range(3)):
            left, right = tags[c, a, w], tags[d, b, v]
            weight = counts[c, a, w] * counts[d, b, v] / (6 * 2**2 * 3**2)
            expected += weight * np.array([
                np.mean(left * right), left.mean() * right.mean(),
                np.mean((left-left.mean()) * (right-right.mean()))])
    result = polarization_components(tags, counts)
    np.testing.assert_allclose([result[k] for k in ['mean_q', 'uniform_squared', 'nonuniform']],
                               expected, atol=1e-14, rtol=0)


def test_uniform_polarization_and_zero_mean_frozen_pattern():
    counts = np.ones((2, 4, 1))
    uniform = polarization_components(np.ones((2, 4, 1, 6)), counts)
    assert uniform['mean_q'] == uniform['uniform_squared'] == 1
    assert uniform['nonuniform'] == 0
    pattern = np.tile([1, -1, 1, -1, 1, -1], (2, 4, 1, 1))
    frozen = polarization_components(pattern, counts)
    assert frozen['mean_q'] == frozen['nonuniform'] == 1
    assert frozen['uniform_squared'] == 0


def test_symmetric_ferromagnetic_mixture_has_zero_mean_overlap():
    # Snapshot M^2=1, but <M>^2=0. Do not confuse these observables.
    tags = np.ones((1, 4, 2, 6)); tags[:, :, 1] = -1
    result = polarization_components(tags, np.ones((1, 4, 2)))
    assert result['mean_q'] == result['uniform_squared'] == result['nonuniform'] == 0
    assert np.mean(tags.mean(axis=-1)**2) == 1


def test_cross_products_exclude_self_noise_and_pool_cycles_first():
    tags = np.ones((2, 4, 1, 2))
    tags[:, 3] = -1
    result = polarization_components(tags, np.ones((2, 4, 1)))
    assert result['uniform_squared'] == 0
    assert result['uniform_m']**2 == .25  # Pooled square would be biased upward.
    tags[1] = -tags[0]
    pooled = polarization_components(tags, np.ones((2, 4, 1)))
    assert pooled['mean_q'] == pooled['uniform_squared'] == 0


def test_descendant_weights_and_negative_estimates():
    pattern = np.array([1, -1])
    tags = np.tile(pattern, (1, 4, 2, 1))
    tags[:, 2:] *= -1
    result = polarization_components(tags, np.full((1, 4, 2), 1.))
    np.testing.assert_allclose(result['nonuniform'], -1/3)
    counts = np.ones((1, 4, 2)); counts[:, :, 0] = 2; counts[:, :, 1] = 0
    tags[:, :, 1] *= -1
    weighted = polarization_components(tags, counts)
    np.testing.assert_allclose(weighted['nonuniform'], -1/3)
    with np.testing.assert_raises(AssertionError):
        polarization_components(tags, counts*2)


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in list(globals().items())
                               if name.startswith('test_') and callable(test))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
