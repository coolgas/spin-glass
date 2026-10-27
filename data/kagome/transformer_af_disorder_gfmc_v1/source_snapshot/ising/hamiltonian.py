"""Sparse transverse-field Ising operator: one coupling per stored bond.

The configuration-basis matrix has dimension 2^N and is not built here.
"""

import numpy as np


class Hamiltonian:
    """Bonds, couplings, and an optional longitudinal field."""

    def __init__(self, n_sites, bonds, J, h=0.0):
        if isinstance(n_sites, bool) or int(n_sites) < 1:
            raise ValueError("n_sites must be a positive integer")
        self.n_sites = int(n_sites)
        # Own the couplings. A later write to the caller's arrays must not
        # change the diagonal while the neighbor list keeps the old values.
        self.bonds = np.array(
            np.asarray(bonds, dtype=np.int64).reshape(-1, 2),
            dtype=np.int64,
            copy=True,
            order="C",
        )
        self.J = np.array(
            np.asarray(J, dtype=np.float64).reshape(-1),
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if self.bonds.shape[0] != self.J.shape[0]:
            raise ValueError("J must have one value per bond")
        if self.bonds.size and (
            int(self.bonds.min()) < 0 or int(self.bonds.max()) >= self.n_sites
        ):
            raise ValueError("bond index out of range")
        self.bonds.setflags(write=False)
        self.J.setflags(write=False)
        self.h = float(h)
        self.neighbors = [[] for _ in range(self.n_sites)]
        for (left, right), coupling in zip(self.bonds, self.J):
            pair = (int(right), float(coupling))
            self.neighbors[int(left)].append(pair)
            self.neighbors[int(right)].append((int(left), float(coupling)))

    def __repr__(self):
        return (
            f"Hamiltonian(n_sites={self.n_sites}, "
            f"n_bonds={self.bonds.shape[0]}, h={self.h})"
        )

    def with_field(self, h):
        """Same bonds and couplings, with a new longitudinal field."""
        return Hamiltonian(self.n_sites, self.bonds, self.J, h)

    def with_classical_scale(self, scale):
        """Multiply every classical coefficient, ``J`` and ``h``, by ``scale``."""
        return Hamiltonian(
            self.n_sites, self.bonds, self.J * float(scale), self.h * float(scale)
        )

    def diagonal_energy(self, spins):
        """Diagonal matrix element ``⟨s|H_diag|s⟩``."""
        state = np.asarray(spins, dtype=np.float64)
        if state.shape != (self.n_sites,):
            raise ValueError("spins must have one entry per site")
        energy = -self.h * float(state.sum())
        if self.bonds.shape[0]:
            bond = state[self.bonds[:, 0]] * state[self.bonds[:, 1]]
            energy -= float(np.sum(self.J * bond))
        return energy

    def flip_delta(self, spins, site):
        """Energy after flipping ``site``, minus the energy before.

        ``ΔE = 2 s_k (∑_{j adjacent to k} J_{kj} s_j + h)``.
        """
        spin = float(spins[site])
        local = self.h
        for neighbor, coupling in self.neighbors[site]:
            local += coupling * float(spins[neighbor])
        return 2.0 * spin * local
