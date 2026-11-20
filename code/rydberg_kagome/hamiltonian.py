"""Ordered-pair occupation and Pauli Hamiltonians; n=(1+Z)/2."""
import numpy as np
from scipy import sparse


def validate_interactions(v):
    v = np.asarray(v, float)
    if v.ndim != 2 or v.shape[0] != v.shape[1] or not np.isfinite(v).all():
        raise ValueError("Expected finite square interaction matrix")
    if not np.allclose(v, v.T, atol=1e-13) or np.any(np.diag(v)) or np.any(v < 0):
        raise ValueError("Expected symmetric nonnegative physical V with zero diagonal")
    return v


def spin_table(n):
    return (1-2*((np.arange(1 << n)[:, None] >> np.arange(n)) & 1)).astype(np.int8)


def fields(v, delta):
    v = validate_interactions(v)
    return delta/2-v.sum(axis=1)/4, -len(v)*delta/2+v.sum()/8


def pauli_matrix(v, effective_fields, omega, constant=0.):
    """Full physical matrix. Both orientations occur in the V/8 contraction."""
    v = validate_interactions(v)
    n = len(v)
    if n > 16:
        raise ValueError("Physical sparse matrix limited to 16 spins")
    z = spin_table(n).astype(float)
    diagonal = -z @ np.asarray(effective_fields) + np.einsum('bi,ij,bj->b', z, v/8, z) + constant
    states = np.arange(1 << n)
    rows = np.tile(states, n)
    cols = np.concatenate([states ^ (1 << i) for i in range(n)])
    return sparse.diags(diagonal, format='csr') + sparse.csr_matrix(
        (np.full(len(rows), omega/2), (rows, cols)), shape=(len(states), len(states)))


def physical_matrix(v, delta, omega):
    h, constant = fields(v, delta)
    return pauli_matrix(v, h, omega, constant)


def occupation_matrix(v, delta, omega):
    """Independent occupation construction for factor/sign validation."""
    v = validate_interactions(v)
    if len(v) > 16:
        raise ValueError("Physical fixture too large")
    n = (1+spin_table(len(v))).astype(float)/2
    diagonal = -delta*n.sum(axis=1) + np.sum((n @ v)*n, axis=1)/2
    d = len(n)
    matrix = sparse.diags(diagonal, format='csr')
    for i in range(len(v)):
        matrix += sparse.csr_matrix((np.full(d, omega/2), (np.arange(d), np.arange(d) ^ (1 << i))), shape=(d,d))
    return matrix
