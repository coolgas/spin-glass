"""Exact diagonalization for at most 16 sites, and moments for at most 12.

Basis index ``b`` stores site ``i`` in bit ``i``. Bit 0 is spin ``+1`` and
bit 1 is spin ``-1``. For a product of two identical copies,

.. math::

    \\langle q^2\\rangle = N^{-2}\\sum_{ij}\\langle\\sigma_i^z\\sigma_j^z\\rangle^2,

and ``⟨q^4⟩`` is the same contraction of the four-point function. Both
identities hold for a thermal product state ``ρ⊗ρ``, not only for a pure
ground state.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from kagome_qea.observables import binder_from_moments, connected_binder

MAX_SITES = 16
MAX_MOMENT_SITES = 12


def operator_matrix(ham, gamma):
    """Sparse matrix of ``H`` in the ``σ^z`` basis."""
    if ham.n_sites > MAX_SITES:
        raise ValueError(f"exact diagonalization stops at {MAX_SITES} sites")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    dimension = 1 << ham.n_sites
    index = np.arange(dimension, dtype=np.int64)
    bits = (index[:, None] >> np.arange(ham.n_sites, dtype=np.int64)) & 1
    spins = 1.0 - 2.0 * bits
    diagonal = -ham.h * spins.sum(axis=1)
    if ham.bonds.shape[0]:
        bond = spins[:, ham.bonds[:, 0]] * spins[:, ham.bonds[:, 1]]
        diagonal = diagonal - bond @ ham.J
    rows = np.empty(dimension * ham.n_sites, dtype=np.int64)
    cols = np.empty(dimension * ham.n_sites, dtype=np.int64)
    data = np.full(dimension * ham.n_sites, -float(gamma), dtype=np.float64)
    for site in range(ham.n_sites):
        window = slice(site * dimension, (site + 1) * dimension)
        rows[window] = index
        cols[window] = np.bitwise_xor(index, 1 << site)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(dimension, dimension))
    matrix.setdiag(diagonal)
    return matrix


def _lowest(matrix, n_states):
    dimension = matrix.shape[0]
    if n_states < 1 or n_states >= dimension:
        raise ValueError("n_states must be a proper subset of the spectrum")
    if dimension <= 64:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.toarray())
        return eigenvalues[:n_states], eigenvectors[:, :n_states]
    eigenvalues, eigenvectors = eigsh(
        matrix.astype(np.float64),
        k=n_states,
        which="SA",
        tol=1e-12,
        maxiter=max(1000, 40 * dimension),
    )
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def moments_from_probability(probability, n_sites):
    """Raw moments of ``q`` for two independent draws from one distribution."""
    if n_sites > MAX_MOMENT_SITES:
        raise ValueError(f"overlap moments stop at {MAX_MOMENT_SITES} sites")
    probability = np.asarray(probability, dtype=np.float64)
    dimension = 1 << n_sites
    if probability.shape != (dimension,):
        raise ValueError("probability must cover the computational basis")
    weight = float(probability.sum())
    if weight <= 0.0:
        raise ValueError("probability mass must be positive")
    probability = probability / weight
    index = np.arange(dimension)
    spins = np.empty((dimension, n_sites), dtype=np.float64)
    for site in range(n_sites):
        bit = (index >> site) & 1
        spins[:, site] = np.where(bit == 0, 1.0, -1.0)
    weighted = spins * probability[:, None]
    correlator = spins.T @ weighted
    q2 = float(np.sum(correlator * correlator) / n_sites ** 2)
    magnetization = spins.T @ probability
    q_mean = float(np.dot(magnetization, magnetization) / n_sites)
    q3_sum = 0.0
    q4_matrix_source = np.einsum("si,sj->sij", spins, spins).reshape(dimension, -1)
    fourth = q4_matrix_source.T @ (q4_matrix_source * probability[:, None])
    q4 = float(np.sum(fourth * fourth) / n_sites ** 4)
    for site in range(n_sites):
        site_weight = probability * spins[:, site]
        cube = spins.T @ (spins * site_weight[:, None])
        q3_sum += float(np.sum(cube * cube))
    q3 = q3_sum / n_sites ** 3
    return {
        "q": q_mean,
        "q2": q2,
        "q3": q3,
        "q4": q4,
    }


def _pack_moments(moments):
    binder, ratio = binder_from_moments(moments["q2"], moments["q4"])
    connected, connected_ratio = connected_binder(
        moments["q"], moments["q2"], moments["q3"], moments["q4"]
    )
    packed = dict(moments)
    packed.update(
        binder=binder,
        ratio=ratio,
        connected_binder=connected,
        connected_ratio=connected_ratio,
    )
    return packed


def ground_state(ham, gamma, subspace_tol=1e-6):
    """Lowest eigenpair and the two-replica overlap moments.

    A cluster of eigenvalues within ``subspace_tol`` is mixed with equal
    weight. A unique ground state is the usual one-vector projector.
    """
    matrix = operator_matrix(ham, gamma)
    dimension = matrix.shape[0]
    if dimension <= 64:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.toarray())
    else:
        eigenvalues, eigenvectors = _lowest(matrix, min(6, dimension - 2))
    gap = float(eigenvalues[1] - eigenvalues[0])
    keep = eigenvalues <= eigenvalues[0] + subspace_tol
    subspace = eigenvectors[:, keep]
    probability = np.sum(subspace * subspace, axis=1) / float(keep.sum())
    residual = matrix @ subspace[:, 0] - eigenvalues[0] * subspace[:, 0]
    moments = _pack_moments(moments_from_probability(probability, ham.n_sites))
    moments.update(
        energy=float(eigenvalues[0]),
        energy_per_spin=float(eigenvalues[0] / ham.n_sites),
        gap=gap,
        degeneracy=int(keep.sum()),
        residual=float(np.linalg.norm(residual)),
    )
    return moments


def _tail_bound(eigenvalues, dimension, beta):
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    weights = np.exp(-beta * (eigenvalues - eigenvalues[0]))
    kept = float(weights.sum())
    discarded = dimension - int(eigenvalues.size)
    if discarded <= 0:
        return 0.0
    tail = discarded * float(np.exp(-beta * (eigenvalues[-1] - eigenvalues[0])))
    return tail / (kept + tail)


def thermal_observables(ham, gamma, beta, tail_tolerance=1e-4):
    """Thermal moments of one copy, traced in the computational basis.

    ``tail_bound`` is a pessimistic mass of the discarded eigenvalues, valid
    when the computed eigenvalues are the lowest ones. A small system is
    diagonalized in full.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    matrix = operator_matrix(ham, gamma)
    dimension = matrix.shape[0]
    tail = 0.0
    if dimension <= 1024:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.toarray())
    else:
        eigenvalues = eigenvectors = None
        n_states = 32
        while n_states <= min(256, dimension - 2):
            eigenvalues, eigenvectors = _lowest(matrix, n_states)
            tail = _tail_bound(eigenvalues, dimension, beta)
            if tail <= tail_tolerance:
                break
            n_states *= 2
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix.toarray())
            tail = 0.0
    weights = np.exp(-float(beta) * (eigenvalues - eigenvalues[0]))
    probability = np.sum(eigenvectors ** 2 * weights, axis=1)
    moments = _pack_moments(moments_from_probability(probability, ham.n_sites))
    partition = float(weights.sum())
    energy = float(np.dot(weights, eigenvalues) / partition)
    moments.update(
        energy=energy,
        energy_per_spin=energy / ham.n_sites,
        beta=float(beta),
        n_states=int(eigenvalues.size),
        tail_bound=float(tail),
    )
    return moments
