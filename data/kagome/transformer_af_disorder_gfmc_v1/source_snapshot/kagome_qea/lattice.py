"""Kagome graphs used by the notebook and by every solver.

Sites are ordered by cell row, cell column, then the three basis sites.
Nearest neighbors are pairs at distance 1. Open boundaries use the Euclidean
distance. Periodic boundaries use the minimum image on the torus spanned by
``L a1`` and ``L a2``. Bond order is the upper triangle of that adjacency.
"""

from itertools import combinations

import numpy as np


def build_kagome(size, periodic=False):
    """Kagome lattice of ``size`` by ``size`` three-site cells.

    Sites are ordered by cell row ``j``, cell column ``i``, then the three
    basis sites of the upward triangle. Nearest neighbors are pairs at
    distance 1. ``periodic=False`` gives the open patch drawn by
    ``generate_kagome_4x4.main``. With ``periodic=True`` both cell directions
    wrap to form a torus. Both boundary conditions are supported for the
    first-round solvers.

    Returns ``points`` of shape ``(N, 2)``, ``bonds`` of shape ``(N_b, 2)``,
    a list of corner-index triples, and the boolean adjacency.
    """
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer")
    if periodic and size < 2:
        raise ValueError("The Kagome torus needs size >= 2")
    a1 = np.array([2.0, 0.0])
    a2 = np.array([1.0, np.sqrt(3.0)])
    basis = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2]])
    # Retain all three sites of each unit cell: its upward triangle
    # guarantees the requested condition even at the outer corners.
    points = np.array([
        i * a1 + j * a2 + site
        for j in range(size)
        for i in range(size)
        for site in basis
    ])
    if periodic:
        cell = np.column_stack((size * a1, size * a2))
        delta = points[:, None, :] - points[None, :, :]
        fractional = np.einsum("ij,nmj->nmi", np.linalg.inv(cell), delta)
        fractional -= np.round(fractional)
        displacement = np.einsum("ij,nmj->nmi", cell, fractional)
        distances_sq = np.sum(displacement * displacement, axis=-1)
    else:
        distances_sq = np.sum((points[:, None] - points[None, :]) ** 2, axis=-1)
    adjacency = np.isclose(distances_sq, 1.0, rtol=0.0, atol=1e-8)
    np.fill_diagonal(adjacency, False)
    bonds = np.argwhere(np.triu(adjacency, k=1))
    triangles = [
        (i, j, k)
        for i, j, k in combinations(range(len(points)), 3)
        if adjacency[i, j] and adjacency[j, k] and adjacency[k, i]
    ]
    return points, bonds, triangles, adjacency
