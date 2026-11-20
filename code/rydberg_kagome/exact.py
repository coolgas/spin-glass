"""Bounded exact reference eigensolvers and tensor-product operator actions."""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh


def apply_axis(tensor, matrix, axis):
    return np.moveaxis(np.tensordot(matrix, tensor, axes=(1, axis)), 0, axis)


class Operator:
    def __init__(self, dims, terms):
        self.dims = tuple(dims)
        self.terms = terms
        self.dimension = int(np.prod(dims))

    def action(self, vector):
        tensor = np.asarray(vector).reshape(self.dims)
        out = np.zeros_like(tensor)
        for coefficient, factors in self.terms:
            value = tensor
            for axis, matrix in factors:
                value = apply_axis(value, matrix, axis)
            out += coefficient*value
        return out.ravel()

    def solve(self, seed=0, tolerance=1e-11):
        if self.dimension > 2**22:
            raise ValueError("Projected exact reference exceeds bounded dimension")
        matrix = LinearOperator((self.dimension,)*2, matvec=self.action, dtype=float)
        return ground_state(matrix, seed, tolerance)


def ground_state(matrix, seed=0, tolerance=1e-11):
    if matrix.shape[0] <= 4:
        a = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
        values, vectors = eigh(a)
    else:
        values, vectors = eigsh(matrix, k=2, which='SA', tol=tolerance, maxiter=30000,
                               v0=np.random.default_rng(seed).normal(size=matrix.shape[0]))
    order = np.argsort(values)
    e, v = float(values[order[0]]), vectors[:, order[0]]
    residual = float(np.linalg.norm(matrix @ v-e*v))
    if residual/max(1, abs(e)) > 1e-9:
        raise RuntimeError(f"Exact solver unresolved: {residual}")
    return dict(energy=e, vector=v, gap=float(values[order[1]]-e), residual=residual)
