"""Continuous-time GFMC projector.

One walker is one σ^z configuration and a positive weight. From s it waits an
exponential time of rate

    λ(s) = Σ_i Γ exp(log ψ(s⊕i) − log ψ(s))

and flips a site with probability proportional to that site's term. Over the
wait Δτ the weight is multiplied by exp[−Δτ (E_L − E_T)], with the local energy

    E_L(s) = E_diag(s) − λ(s).

The mixed energy is the time average of E_L. The growth energy is read from
the total weight after systematic resampling. Both estimate the ground-state
energy when the guide is positive. E_T is moved halfway toward the latest
growth estimate so the population stays finite.
"""

import math

import numpy as np

from gfmc.guide import ConstantGuide


class _Walker:
    __slots__ = ("spins", "diagonal", "weight", "ratios", "rate")

    def __init__(self, spins, diagonal, ratios, rate):
        self.spins = spins
        self.diagonal = float(diagonal)
        self.weight = 1.0
        self.ratios = ratios
        self.rate = float(rate)


def _weighted_interval(weight, local_energy, shift, dt):
    """Integrate the Feynman–Kac weight over a hop-free interval."""
    alpha = -(local_energy - shift)
    exponent = alpha * dt
    if exponent > 700.0:
        raise RuntimeError("walker weight overflowed; reduce tau_block or the shift step")
    factor = math.exp(exponent)
    weight_end = weight * factor
    if abs(alpha) < 1e-12:
        time_integral = weight * dt
    else:
        time_integral = weight * (factor - 1.0) / alpha
    return weight_end, local_energy * time_integral, time_integral


def _systematic(weights, rng):
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise RuntimeError("walker weights left the positive reals")
    n_walkers = weights.size
    cdf = np.cumsum(weights) / total
    positions = (rng.random() + np.arange(n_walkers)) / n_walkers
    indexes = np.searchsorted(cdf, positions, side="left")
    return np.clip(indexes, 0, n_walkers - 1), total


class GFMC:
    """Projector for one copy of a longitudinal-transverse Ising operator.

    ``hamiltonian`` supplies ``n_sites`` and the diagonal bonds. ``gamma`` is
    the transverse field Γ. ``h`` is the longitudinal field. ``guide`` defaults
    to ψ = 1.
    """

    def __init__(self, hamiltonian, gamma, guide=None, *, h=None):
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        for name in ("n_sites", "diagonal_energy", "flip_delta"):
            if not hasattr(hamiltonian, name):
                raise TypeError(f"operator is missing {name}")
        if h is not None:
            if not hasattr(hamiltonian, "with_field"):
                raise TypeError("operator cannot set the longitudinal field")
            hamiltonian = hamiltonian.with_field(h)
        self.hamiltonian = hamiltonian
        self.gamma = float(gamma)
        self.h = float(getattr(hamiltonian, "h", 0.0))
        self.guide = ConstantGuide() if guide is None else guide

    def run(
        self,
        n_walkers=2000,
        tau_block=0.25,
        n_equilibrate=40,
        n_measure=60,
        seed=0,
        configurations=None,
    ):
        """Project from ``configurations`` and return the block energies.

        ``configurations`` is ``None`` (random product states), one ``±1``
        vector of length ``n_sites``, or an array of shape
        ``(n_walkers, n_sites)``.
        """
        if n_walkers < 2 or tau_block <= 0.0:
            raise ValueError("need at least two walkers and a positive block")
        if n_equilibrate < 0 or n_measure < 2:
            raise ValueError("need a nonnegative warmup and at least two blocks")
        rng = np.random.default_rng(seed)
        ham = self.hamiltonian
        walkers = self._initial_walkers(n_walkers, configurations, rng)
        shift = float(np.mean([walker.diagonal - walker.rate for walker in walkers]))
        mixed = []
        growth = []

        def one_block(record):
            nonlocal shift, walkers
            energy_sum = 0.0
            time_sum = 0.0
            for walker in walkers:
                energy_piece, time_piece = self._advance(walker, shift, tau_block, rng)
                energy_sum += energy_piece
                time_sum += time_piece
            children, total = self._resample(walkers, rng)
            if record and time_sum > 0.0:
                mixed.append(energy_sum / time_sum)
                growth.append(shift - math.log(total / n_walkers) / tau_block)
            shift = 0.5 * shift + 0.5 * (shift - math.log(total / n_walkers) / tau_block)
            walkers = children

        for _ in range(int(n_equilibrate)):
            one_block(False)
        for _ in range(int(n_measure)):
            one_block(True)

        energy, energy_error = _mean_error(mixed)
        growth_mean, growth_error = _mean_error(growth)
        return {
            "energy": energy,
            "energy_error": energy_error,
            "energy_growth": growth_mean,
            "energy_growth_error": growth_error,
            "shift": float(shift),
            "n_walkers": int(n_walkers),
            "h": self.h,
        }

    def _initial_walkers(self, n_walkers, configurations, rng):
        ham = self.hamiltonian
        spins = _configurations(ham.n_sites, n_walkers, configurations, rng)
        walkers = []
        for replica in range(n_walkers):
            state = np.ascontiguousarray(spins[replica])
            ratios = self._ratios(state)
            walkers.append(
                _Walker(state, ham.diagonal_energy(state), ratios, self.gamma * float(ratios.sum()))
            )
        return walkers

    def _ratios(self, spins):
        ratios = np.empty(spins.size, dtype=np.float64)
        for site in range(spins.size):
            log_ratio = float(self.guide.log_ratio(spins, site))
            if not math.isfinite(log_ratio):
                raise ValueError("guiding-function ratio is not positive")
            ratios[site] = math.exp(log_ratio)
        return ratios

    def _advance(self, walker, shift, tau, rng):
        spins = walker.spins
        elapsed = 0.0
        energy_integral = 0.0
        time_integral = 0.0
        while elapsed < tau - 1e-15:
            rate = walker.rate
            if rate <= 0.0 or not math.isfinite(rate):
                raise ValueError("guiding-function ratio is not positive")
            waiting = rng.exponential(1.0 / rate)
            hop = True
            if elapsed + waiting > tau:
                waiting = tau - elapsed
                hop = False
            local = walker.diagonal - rate
            walker.weight, energy_piece, time_piece = _weighted_interval(
                walker.weight, local, shift, waiting
            )
            energy_integral += energy_piece
            time_integral += time_piece
            elapsed += waiting
            if hop:
                chosen = _choose_site(walker.ratios, rng)
                walker.diagonal += self.hamiltonian.flip_delta(spins, chosen)
                spins[chosen] = -spins[chosen]
                walker.ratios = self._ratios(spins)
                walker.rate = self.gamma * float(walker.ratios.sum())
        return energy_integral, time_integral

    def _resample(self, walkers, rng):
        weights = np.fromiter((walker.weight for walker in walkers), dtype=np.float64)
        indexes, total = _systematic(weights, rng)
        children = []
        for index in indexes:
            parent = walkers[int(index)]
            child = _Walker(
                parent.spins.copy(),
                parent.diagonal,
                parent.ratios.copy(),
                parent.rate,
            )
            children.append(child)
        return children, total


def _choose_site(ratios, rng):
    total = float(ratios.sum())
    pick = rng.random() * total
    cumulative = 0.0
    last = ratios.size - 1
    for site, ratio in enumerate(ratios):
        cumulative += float(ratio)
        if cumulative >= pick:
            return site
    return last


def _configurations(n_sites, n_walkers, configurations, rng):
    if configurations is None:
        bits = rng.integers(0, 2, size=(n_walkers, n_sites))
        return np.where(bits == 0, np.int8(1), np.int8(-1))
    config = np.asarray(configurations)
    if config.shape == (n_sites,):
        config = np.broadcast_to(config, (n_walkers, n_sites))
    elif config.shape != (n_walkers, n_sites):
        raise ValueError("configurations must be one state or one state per walker")
    if not np.all((config == 1) | (config == -1)):
        raise ValueError("configurations must be ±1")
    return np.array(config, dtype=np.int8, copy=True)


def _mean_error(values):
    data = np.asarray(values, dtype=np.float64)
    if data.size < 2:
        return float(data.mean()) if data.size else float("nan"), float("nan")
    error = float(data.std(ddof=1) / math.sqrt(data.size))
    return float(data.mean()), error
