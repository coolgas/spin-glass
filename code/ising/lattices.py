"""Chain and square lattices as sparse Ising bond lists."""

import numpy as np

from ising.hamiltonian import Hamiltonian


def chain(n, J=1.0, periodic=False, h=0.0):
    """One-dimensional chain of ``n`` sites. A periodic chain needs ``n ≥ 3``."""
    if isinstance(n, bool) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    bonds = [(site, site + 1) for site in range(n - 1)]
    if periodic:
        if n < 3:
            raise ValueError("a periodic chain needs at least 3 sites")
        bonds.append((0, n - 1))
    couplings = np.full(len(bonds), float(J), dtype=np.float64)
    return Hamiltonian(n, bonds, couplings, h)


def square(n_x, n_y, J=1.0, periodic=False, h=0.0):
    """Square grid, site ``x + n_x y``. Periodic boundaries need both sides ≥ 2.

    Each unordered nearest-neighbor pair is stored once. On a side of length 2
    the periodic image is that same pair, so the bond is not doubled.
    """
    if isinstance(n_x, bool) or isinstance(n_y, bool) or min(int(n_x), int(n_y)) < 1:
        raise ValueError("grid sides must be positive")
    n_x, n_y = int(n_x), int(n_y)
    if periodic and min(n_x, n_y) < 2:
        raise ValueError("a periodic square needs both sides at least 2")
    edges = set()

    def add(left, right):
        if left == right:
            return
        edges.add((left, right) if left < right else (right, left))

    for y in range(n_y):
        for x in range(n_x):
            site = x + n_x * y
            if x + 1 < n_x:
                add(site, site + 1)
            elif periodic:
                add(site, site - n_x + 1)
            if y + 1 < n_y:
                add(site, site + n_x)
            elif periodic:
                add(site, site - n_x * (n_y - 1))
    bonds = sorted(edges)
    couplings = np.full(len(bonds), float(J), dtype=np.float64)
    return Hamiltonian(n_x * n_y, bonds, couplings, h)
