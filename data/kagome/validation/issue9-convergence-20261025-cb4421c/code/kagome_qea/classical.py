"""Classical parallel tempering for two non-interacting replicas.

The transverse-field projector collapses to one walker and never reaches
these states. Temperature swaps move a replica out of a basin without a
guiding wavefunction. The cold rung is the Γ = 0 anchor for the overlap.
"""

import numpy as np

from kagome_qea.observables import overlap


def _incident(ham):
    sites = [[] for _ in range(ham.n_sites)]
    for bond, (left, right) in enumerate(ham.bonds):
        coupling = float(ham.J[bond])
        sites[int(left)].append((int(right), coupling))
        sites[int(right)].append((int(left), coupling))
    return sites


def _energies(ham, spins):
    energy = -float(ham.h) * spins.sum(axis=-1)
    for bond, (left, right) in enumerate(ham.bonds):
        energy = energy - ham.J[bond] * spins[..., int(left)] * spins[..., int(right)]
    return energy


def temper_pair(
    ham,
    *,
    n_temps=16,
    n_anneal=80,
    n_sweeps=2000,
    t_hot=2.0,
    t_cold=0.02,
    seed=0,
):
    """Anneal, then temper two replicas. Returns cold spins, energies, and q.

    ``spins`` has shape ``(2, n_sites)`` with entries ±1. Temperatures are
    multiples of ``rms(J)``.
    """
    if n_temps < 2 or n_sweeps < 1 or n_anneal < 1:
        raise ValueError("tempering needs at least two temperatures and one sweep")
    if t_hot <= t_cold or t_cold <= 0.0:
        raise ValueError("temperature ladder must decrease to a positive cold rung")
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(np.square(ham.J)))) if ham.J.size else 1.0
    if rms == 0.0:
        rms = 1.0
    temps = rms * np.geomspace(float(t_hot), float(t_cold), int(n_temps))
    spins = rng.integers(0, 2, size=(2, int(n_temps), ham.n_sites))
    spins = np.where(spins == 0, np.int8(1), np.int8(-1))
    incident = _incident(ham)
    h = float(ham.h)

    def sweep(temperature):
        scale = temps.reshape((1, -1)) if temperature is None else float(temperature)
        for site in rng.permutation(ham.n_sites):
            field = np.full(spins.shape[:-1], h, dtype=np.float64)
            for neighbor, coupling in incident[site]:
                field += coupling * spins[..., neighbor]
            delta = 2.0 * spins[..., site] * field
            accept = delta <= 0.0
            accept = accept | (rng.random(delta.shape) < np.exp(-np.maximum(delta, 0.0) / scale))
            spins[..., site] = np.where(accept, -spins[..., site], spins[..., site])

    for temperature in rms * np.geomspace(float(t_hot), float(t_cold), int(n_anneal)):
        sweep(float(temperature))

    accepts = 0
    attempts = 0
    for _ in range(int(n_sweeps)):
        sweep(None)
        energy = _energies(ham, spins)
        for parity in (0, 1):
            for rung in range(parity, n_temps - 1, 2):
                # Index rung + 1 is colder. Metropolis factor (β_c - β_h)(E_c - E_h).
                log_accept = (1.0 / temps[rung + 1] - 1.0 / temps[rung]) * (
                    energy[:, rung + 1] - energy[:, rung]
                )
                take = np.log(rng.random(log_accept.shape)) < log_accept
                attempts += int(take.size)
                accepts += int(np.count_nonzero(take))
                hot = spins[:, rung, :].copy()
                cold = spins[:, rung + 1, :].copy()
                spins[:, rung, :] = np.where(take[:, None], cold, hot)
                spins[:, rung + 1, :] = np.where(take[:, None], hot, cold)
                hot_energy = energy[:, rung].copy()
                cold_energy = energy[:, rung + 1].copy()
                energy[:, rung] = np.where(take, cold_energy, hot_energy)
                energy[:, rung + 1] = np.where(take, hot_energy, cold_energy)

    cold_spins = np.array(spins[:, -1, :], dtype=np.int8, copy=True)
    cold_energy = _energies(ham, cold_spins)
    return {
        "spins": cold_spins,
        "energy": cold_energy,
        "q": overlap(cold_spins[0], cold_spins[1]),
        "accept_rate": (accepts / attempts) if attempts else float("nan"),
        "temperature": float(temps[-1]),
    }
