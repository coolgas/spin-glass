"""Shared numerics for the continuous-time projectors.

Green's function Monte Carlo and the Kagome projector both integrate a
Feynman–Kac weight, resample a population, and estimate an error from a
correlated series. The two packages do not import each other.
"""

import math

import numpy as np
from scipy.stats import t as student_t


def _expm1_ratio(exponent):
    """(exp(x) - 1) / x. The value at zero is 1."""
    if abs(exponent) < 1e-8:
        return 1.0 + exponent * (0.5 + exponent / 6.0)
    return math.expm1(exponent) / exponent


def weighted_interval(weight, local_energy, shift, dt):
    """Integrate one hop-free interval of the Feynman–Kac weight.

    The weight grows as exp(alpha * t) with alpha = -(E_L - E_T). The time
    integral is weight * dt * (exp(alpha * dt) - 1) / (alpha * dt), evaluated
    with expm1 so a tiny alpha * dt does not cancel to zero. A large negative
    exponent may round the final weight to zero. A non-finite weight or a
    non-finite energy contribution is rejected.
    """
    alpha = -(local_energy - shift)
    exponent = alpha * dt
    if exponent > 700.0:
        raise RuntimeError("walker weight overflowed; reduce tau_block or the shift step")
    factor = math.exp(exponent)
    weight_end = weight * factor
    time_integral = weight * dt * _expm1_ratio(exponent)
    energy_integral = local_energy * time_integral
    if (
        not math.isfinite(weight_end)
        or not math.isfinite(time_integral)
        or not math.isfinite(energy_integral)
    ):
        raise RuntimeError("walker weight overflowed; reduce tau_block or the shift step")
    return weight_end, energy_integral, time_integral


def systematic(weights, rng):
    """Systematic resampling. Returns child indexes and log of the total weight.

    The total is a log-sum-exp over the positive mass, so two large finite
    weights do not add to infinity. A zero weight has no offspring. Negative,
    non-finite, and all-zero populations are rejected.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size < 1:
        raise RuntimeError("walker weights left the positive reals")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise RuntimeError("walker weights left the positive reals")
    positive = np.flatnonzero(weights > 0.0)
    if positive.size == 0:
        raise RuntimeError("walker weights left the positive reals")
    logs = np.log(weights[positive])
    peak = float(logs.max())
    scaled = np.exp(logs - peak)
    alive = scaled > 0.0
    if not np.any(alive):
        raise RuntimeError("walker weights left the positive reals")
    positive = positive[alive]
    scaled = scaled[alive]
    sum_scaled = float(scaled.sum())
    if not math.isfinite(sum_scaled) or sum_scaled <= 0.0:
        raise RuntimeError("walker weights left the positive reals")
    log_total = peak + math.log(sum_scaled)
    cdf = np.cumsum(scaled)
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    n_walkers = int(weights.size)
    positions = (rng.random() + np.arange(n_walkers)) / n_walkers
    picked = np.searchsorted(cdf, positions, side="left")
    picked = np.clip(picked, 0, positive.size - 1)
    return positive[picked], log_total


def _autocovariance(centered):
    n_points = centered.size
    return np.correlate(centered, centered, mode="full")[n_points - 1 :] / n_points


def _positive_sequence_variance(autocovariance):
    """Geyer's initial positive sequence of paired autocovariances.

    Returns the summed variance and whether a non-positive pair ended the sum.
    A sequence that stays positive through the available lags is truncated.
    """
    gamma0 = float(autocovariance[0])
    n_pairs = (autocovariance.size - 1) // 2
    if n_pairs < 1:
        return gamma0, False
    pairs = autocovariance[0 : 2 * n_pairs : 2] + autocovariance[1 : 2 * n_pairs : 2]
    last = -1
    for index, pair in enumerate(pairs):
        if pair <= 0.0:
            if last < 0:
                return gamma0, True
            return float(-gamma0 + 2.0 * pairs[: last + 1].sum()), True
        last = index
    return float(-gamma0 + 2.0 * pairs.sum()), False


# An AR(1) of correlation 0.95 has increment variance 0.10 times its variance.
_SMOOTH_REFERENCE = 0.10
_SMOOTH_POWER = 4.0
_SMOOTH_CAP = 8.0
# A modest excess of low-frequency power widens the error; the cap stops one bin from dominating.
_SPECTRUM_POWER = 0.90
_SPECTRUM_CAP = 30.0


def _shape_factor(data, centered, gamma0, rho):
    """Extra width when lag 1 describes the series as shorter than it is.

    Both comparisons are ratios, so they are unchanged by a shift or by a
    rescaling. Green's function Monte Carlo energies are not mean-zero and do
    not have a known variance.

    Smoothness is the mean squared increment divided by the sample variance.
    An AR(1) of correlation 0.95 has this ratio equal to ``0.10``. A smaller
    ratio multiplies the error by ``(0.10 / ratio) ** 4``, and the multiplier
    stops at 8. This is the near-unit-root case: after subtracting the sample
    mean, the lag-1 correlation falls toward about 0.96 while the increments
    stay small, and the lag-1 integrated time is then too short.

    The spectral ratio compares the first Fourier ordinate,
    ``|rfft(x)[1]|**2 / n``, with the lag-1 AR(1) spectrum at frequency
    ``2π/n``. A larger ordinate multiplies the error by ``min(ratio, 30)``
    to the power ``0.90``. A lag-1 model misses a slow component that carries
    only a fraction of the variance. That component shows up at the lowest
    frequency, not at lag 1.

    The factor multiplies the error from the positive sequence or the lag-1
    time. It does not change the resolved flag. A mixed-timescale draw that
    is indistinguishable from white noise has neither small increments nor a
    large first ordinate, so its error can remain short of the exact standard
    deviation. Raising every white-looking series enough to close that tail
    would also inflate ordinary white noise.
    """
    increments = np.diff(data)
    factor = 1.0
    if increments.size > 0 and gamma0 > 0.0:
        smoothness = float(np.dot(increments, increments) / increments.size / gamma0)
        if 0.0 < smoothness < _SMOOTH_REFERENCE:
            factor *= min((_SMOOTH_REFERENCE / smoothness) ** _SMOOTH_POWER, _SMOOTH_CAP)
    n_points = int(centered.size)
    ordinate = float(abs(np.fft.rfft(centered)[1]) ** 2 / n_points)
    if rho <= 0.0:
        predicted = gamma0
    else:
        correlation = min(rho, 0.999)
        omega = 2.0 * math.pi / n_points
        denominator = 1.0 - 2.0 * correlation * math.cos(omega) + correlation * correlation
        if denominator <= 0.0:
            return factor
        predicted = gamma0 * (1.0 - correlation * correlation) / denominator
    if predicted > 0.0 and math.isfinite(ordinate) and math.isfinite(predicted):
        ratio = ordinate / predicted
        if ratio > 1.0:
            factor *= min(ratio, _SPECTRUM_CAP) ** _SPECTRUM_POWER
    return factor


def series_error(values):
    """Mean, autocorrelation-aware error, and a length-and-shape flag.

    The variance is the larger of Geyer's initial positive sequence and the
    lag-1 AR(1) integrated time ``(1+ρ)/(1-ρ)``. ``_shape_factor`` then widens
    that error for extra smoothness or extra low-frequency power. A series of
    at least 64 values whose positive sequence terminated and whose estimated
    number of independent samples is at least 8 is also widened by
    ``t_{0.975, ν} / 1.96``. The flag is false when any of those three checks
    fails. It records the length of the series, not a coverage certificate.
    No finite-series diagnostic establishes convergence for every process. A
    constant series has error 0 and the flag is true. Fewer than 16 values
    keep the ordinary standard error and the flag is false; the shape factor
    is not applied there.
    """
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0 or not np.isfinite(data).all():
        return float("nan"), float("nan"), False
    n_points = int(data.size)
    mean = float(data.mean())
    if n_points < 2:
        return mean, float("nan"), False
    centered = data - mean
    gamma0 = float(np.dot(centered, centered) / n_points)
    if gamma0 <= 0.0:
        return mean, 0.0, True
    naive = math.sqrt(gamma0 / (n_points - 1))
    if n_points < 16:
        return mean, naive, False
    autocovariance = _autocovariance(centered)
    rho = float(autocovariance[1] / autocovariance[0])
    if rho <= 0.0:
        tau_ar = 1.0
    else:
        rho = min(rho, 0.999)
        tau_ar = (1.0 + rho) / (1.0 - rho)
    sigma_ar = gamma0 * tau_ar
    sigma_ips, terminated = _positive_sequence_variance(autocovariance)
    if not math.isfinite(sigma_ips) or sigma_ips <= 0.0:
        sigma_ips = sigma_ar
    sigma = max(sigma_ar, sigma_ips)
    if not math.isfinite(sigma) or sigma <= 0.0:
        return mean, naive, False
    raw = math.sqrt(sigma / n_points)
    degrees = n_points * gamma0 / sigma
    resolved = bool(terminated and n_points >= 64 and degrees >= 8.0)
    error = raw * _shape_factor(data, centered, gamma0, rho)
    if resolved:
        error *= float(student_t.ppf(0.975, degrees) / 1.96)
    return mean, error, resolved
