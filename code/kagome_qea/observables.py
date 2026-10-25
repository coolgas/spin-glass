"""Overlap moments, the Binder ratio, and disorder averages.

Two replicas share ``J`` and do not interact. The equal-time overlap is

.. math::

    q = N^{-1} \\sum_i s_i^{(a)} s_i^{(b)}.

Quantum brackets are one sample. Disorder brackets sit outside them.
At ``h = 0`` the Binder ratio uses ``R = ⟨q^4⟩ / ⟨q^2⟩^2`` and
``B = (3 - R) / 2``, with the ratio formed before the disorder average.
At ``h ≠ 0`` the same construction uses the central moments of ``q``.

The spin-glass susceptibility of one sample is

.. math::

    \\chi_{SG} = N \\langle q^2\\rangle
    = N^{-1}\\sum_{ij}\\langle\\sigma_i^z\\sigma_j^z\\rangle^2.

Its error is ``N`` times the error on ``⟨q^2⟩``. Independent spins give
``χ_SG = 1``.
"""

import numpy as np


def overlap(spins_a, spins_b):
    """Equal-time overlap of two ``±1`` configurations."""
    left = np.asarray(spins_a, dtype=np.float64)
    right = np.asarray(spins_b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("the two replicas must be vectors of the same length")
    return float(np.dot(left, right) / left.size)


def spin_glass_susceptibility(q2, n_sites):
    """``χ_SG = N ⟨q²⟩`` for one sample.

    ``n_sites`` is exact, so an error on ``⟨q²⟩`` is multiplied by the same
    factor. The disorder average is applied afterwards.
    """
    sites = int(n_sites)
    if isinstance(n_sites, bool) or sites < 1:
        raise ValueError("n_sites must be a positive integer")
    second = float(q2)
    if second < 0.0 or not np.isfinite(second):
        raise ValueError("q2 must be finite and nonnegative")
    return sites * second


def binder_from_moments(q2, q4):
    """``B = (3 - R) / 2`` with ``R = q4 / q2^2``."""
    q2 = float(q2)
    q4 = float(q4)
    if q2 <= 0.0:
        raise ValueError("q2 must be positive")
    ratio = q4 / (q2 * q2)
    return 0.5 * (3.0 - ratio), ratio


def connected_cumulants(q, q2, q3, q4):
    """Central moments ``μ2 = ⟨(q-⟨q⟩)^2⟩`` and ``μ4 = ⟨(q-⟨q⟩)^4⟩``."""
    q = float(q)
    q2 = float(q2)
    q3 = float(q3)
    q4 = float(q4)
    mu2 = q2 - q * q
    mu4 = q4 - 4.0 * q * q3 + 6.0 * q * q * q2 - 3.0 * q ** 4
    return mu2, mu4


def connected_binder(q, q2, q3, q4):
    """Binder ratio of the connected moments. At ``⟨q⟩ = 0`` this is ``B``."""
    mu2, mu4 = connected_cumulants(q, q2, q3, q4)
    if mu2 <= 0.0:
        raise ValueError("connected q2 must be positive")
    ratio = mu4 / (mu2 * mu2)
    return 0.5 * (3.0 - ratio), ratio


def disorder_binder(q2, q4):
    """Disorder-averaged Binder ratio.

    ``q2`` and ``q4`` are per-sample quantum expectations. The ratio is
    averaged over samples. Averaging the moments first is a different number.
    """
    second = np.asarray(q2, dtype=np.float64)
    fourth = np.asarray(q4, dtype=np.float64)
    if second.shape != fourth.shape or second.ndim != 1:
        raise ValueError("q2 and q4 must be per-sample vectors")
    if np.any(second <= 0.0):
        raise ValueError("every sample needs a positive q2")
    ratio = fourth / (second * second)
    mean_ratio = float(ratio.mean())
    return 0.5 * (3.0 - mean_ratio), mean_ratio


def jackknife_disorder_binder(q2, q4):
    """Binder ratio and its delete-one jackknife standard error over samples."""
    second = np.asarray(q2, dtype=np.float64)
    fourth = np.asarray(q4, dtype=np.float64)
    n_samples = second.size
    if n_samples < 2:
        raise ValueError("the jackknife needs at least two samples")
    ratio = fourth / (second * second)
    leave = (ratio.sum() - ratio) / (n_samples - 1)
    binder = 0.5 * (3.0 - leave)
    center = float(binder.mean())
    variance = (n_samples - 1) / n_samples * np.sum((binder - center) ** 2)
    mean_binder = 0.5 * (3.0 - float(ratio.mean()))
    return mean_binder, float(np.sqrt(variance))


def jackknife_binder_series(q2_series, q4_series):
    """Delete-one error of the Binder ratio formed from replicate moments.

    Each entry is one forward-walking cycle, or one independent population.
    The ratio uses the mean of the fourth moment and the square of the mean
    of the second. Deleting one replicate at a time is the standard error
    of that ratio.
    """
    second = np.asarray(q2_series, dtype=np.float64)
    fourth = np.asarray(q4_series, dtype=np.float64)
    if second.shape != fourth.shape or second.ndim != 1:
        raise ValueError("q2 and q4 series must be vectors of the same length")
    n_cycles = int(second.size)
    if n_cycles < 2:
        raise ValueError("the jackknife needs at least two cycles")
    total_second = float(second.sum())
    total_fourth = float(fourth.sum())
    leave_second = (total_second - second) / (n_cycles - 1)
    leave_fourth = (total_fourth - fourth) / (n_cycles - 1)
    if np.any(leave_second <= 0.0) or total_second <= 0.0:
        raise ValueError("q2 must stay positive when one cycle is removed")
    leave_ratio = leave_fourth / (leave_second * leave_second)
    leave_binder = 0.5 * (3.0 - leave_ratio)
    center = float(leave_binder.mean())
    variance = (n_cycles - 1) / n_cycles * np.sum((leave_binder - center) ** 2)
    ratio = (total_fourth * n_cycles) / (total_second * total_second)
    binder = 0.5 * (3.0 - ratio)
    return binder, float(np.sqrt(variance))


def jackknife_mean_error(values):
    """Jackknife standard error of a sample mean."""
    data = np.asarray(values, dtype=np.float64)
    n_samples = data.size
    if n_samples < 2:
        raise ValueError("the jackknife needs at least two values")
    leave = (data.sum() - data) / (n_samples - 1)
    center = float(leave.mean())
    variance = (n_samples - 1) / n_samples * np.sum((leave - center) ** 2)
    return float(np.sqrt(variance))


def block_means(series, block_size):
    """Non-overlapping block averages. A trailing partial block is dropped."""
    data = np.asarray(series, dtype=np.float64)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    usable = (data.size // block_size) * block_size
    if usable == 0:
        raise ValueError("the series is shorter than one block")
    return data[:usable].reshape(-1, block_size).mean(axis=1)


def blocked_stderr(series, block_size):
    """Standard error from the scatter of block means."""
    means = block_means(series, block_size)
    if means.size < 2:
        return float("nan")
    return float(means.std(ddof=1) / np.sqrt(means.size))


def bootstrap_indices(n_samples, n_bootstrap, rng):
    """Sample indices of shape ``(n_bootstrap, n_samples)``.

    Reuse one draw across every ``Γ``, field, and method at a fixed
    ``(L, periodic)``. Draw independent indices for each size and boundary.
    """
    if n_samples < 1 or n_bootstrap < 1:
        raise ValueError("bootstrap counts must be positive")
    return rng.integers(0, n_samples, size=(n_bootstrap, n_samples))


def bootstrap_means(values, indices):
    """Mean of ``values`` on each bootstrap replicate in ``indices``."""
    data = np.asarray(values, dtype=np.float64)
    return data[np.asarray(indices)].mean(axis=1)


def crossings(x, y_left, y_right):
    """Linear interpolations where ``y_left - y_right`` changes sign.

    An empty list means the curves do not cross. A grid point where the
    difference is exactly zero is reported. Several zeros are several roots.
    """
    grid = np.asarray(x, dtype=np.float64)
    delta = np.asarray(y_left, dtype=np.float64) - np.asarray(y_right, dtype=np.float64)
    if grid.ndim != 1 or delta.shape != grid.shape:
        raise ValueError("x and the two curves must be 1-d and aligned")
    found = []
    for index in range(delta.size - 1):
        left = delta[index]
        right = delta[index + 1]
        if left == 0.0:
            found.append(float(grid[index]))
        elif left * right < 0.0:
            weight = abs(left) / (abs(left) + abs(right))
            found.append(float(grid[index] + (grid[index + 1] - grid[index]) * weight))
    if delta.size and delta[-1] == 0.0:
        found.append(float(grid[-1]))
    return found
