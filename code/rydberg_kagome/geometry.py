"""Conditioned Gaussian disorder on a triangular torus; coordinates in R units."""
from itertools import product
import numpy as np
from .config import Physics

DISTANCE_VERSION = "euclidean-shortest-oblique-v1"


def lattice(L):
    if not isinstance(L, int) or L < 2:
        raise ValueError("L must be an integer >=2")
    a = np.array([[2., 1.], [0., np.sqrt(3)]])
    basis = np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3)/2]])
    points = np.array([a @ [x, y] + b for y in range(L) for x in range(L) for b in basis])
    return points, L*a


def distances(points, cell, shortest=True):
    fractional = (points[:, None] - points[None, :]) @ np.linalg.inv(cell).T
    fractional -= np.rint(fractional)
    best = np.full(fractional.shape[:-1], np.inf)
    # For this 60-degree equal-length cell, rounding gives norm <=sqrt(3)/2
    # of a cell vector. Any image beyond the adjacent 3x3 cells is longer.
    shifts = product((-1, 0, 1), repeat=2) if shortest else [(0, 0)]
    for shift in shifts:
        displacement = (fractional + shift) @ cell.T
        best = np.minimum(best, np.einsum('...i,...i->...', displacement, displacement))
    d = np.sqrt(np.maximum(best, 0))
    np.fill_diagonal(d, np.inf)
    return d


def masks(L):
    base, cell = lattice(L)
    d = distances(base, cell)
    nn = np.isclose(d, 1, rtol=0, atol=1e-8)
    nnn = np.isclose(d, np.sqrt(3), rtol=0, atol=1e-8)
    return dict(full=~np.eye(len(base), dtype=bool), nn=nn, nn_nnn=nn | nnn)


def sample(L, sample_id, physics=Physics()):
    physics.validate()
    if sample_id < 0:
        raise ValueError("Negative sample ID")
    base, cell = lattice(L)
    for attempt in range(physics.max_attempts):
        rng = np.random.default_rng(np.random.SeedSequence([physics.seed_prefix, L, int(sample_id), attempt]))
        displacements = physics.w_um/physics.R_um*rng.normal(size=base.shape)
        fractional = (base + displacements) @ np.linalg.inv(cell).T
        points = (fractional % 1) @ cell.T
        d = distances(points, cell)
        if d.min()*physics.R_um >= physics.minimum_distance_um:
            return dict(points=points, displacements=displacements, cell=cell,
                        distances=d, interactions=d**-6, attempts=attempt+1,
                        sample_id=int(sample_id), L=L, distance_version=DISTANCE_VERSION)
    raise RuntimeError(f"Geometry rejection exhausted: L={L}, sample={sample_id}")


def blocks(L, origin=(0, 0)):
    """Four up triangles per 12-spin block, no triangle truncation."""
    if L < 4 or L % 2:
        raise ValueError("12-spin tiling requires even L>=4")
    result = []
    for y in range(L//2):
        for x in range(L//2):
            result.append([3*(((2*y+dy+origin[1]) % L)*L + (2*x+dx+origin[0]) % L)+a
                           for dy in range(2) for dx in range(2) for a in range(3)])
    return np.array(result)
