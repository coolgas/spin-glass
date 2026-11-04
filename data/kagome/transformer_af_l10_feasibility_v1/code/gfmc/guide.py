"""Guiding function for continuous-time GFMC.

``log_ratio(spins, site)`` is log ψ(s with site flipped) − log ψ(s). The
propagator multiplies the flip rate Γ by exp of that value. A ratio that is
not a positive real is a sign problem and is rejected.
"""

import math


class Guide:
    """Positive real guiding function on configurations ``s_i = ±1``."""

    def log_amplitude(self, spins):
        """Logarithm of ψ(s)."""
        raise NotImplementedError

    def log_ratio(self, spins, site):
        """Log ψ(s⊕site) − log ψ(s). The default evaluates both amplitudes."""
        flipped = spins.copy()
        flipped[site] = -flipped[site]
        before = float(self.log_amplitude(spins))
        after = float(self.log_amplitude(flipped))
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError("guiding-function ratio is not positive")
        return after - before


class ConstantGuide(Guide):
    """ψ(s) = 1. Every flip rate is Γ."""

    def log_amplitude(self, spins):
        return 0.0

    def log_ratio(self, spins, site):
        return 0.0
