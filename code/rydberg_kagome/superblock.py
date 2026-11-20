"""Single projection of complete 12-spin blocks; no triangle-level approximation."""
import numpy as np
from scipy.sparse.linalg import eigsh
from .geometry import blocks
from .hamiltonian import fields, pauli_matrix, spin_table
from .exact import Operator


def build(v, delta, omega, L, chi, origin=(0,0)):
    if not 1 <= chi <= 64:
        raise ValueError("Supported retention: 1..64")
    groups = blocks(L, origin)
    h, constant = fields(v, delta)
    spins = spin_table(12)
    decoders, spectra, zops, terms, residuals = [], [], [], [], []
    for b, sites in enumerate(groups):
        local = pauli_matrix(v[np.ix_(sites, sites)], h[sites], omega)
        # Same deterministic starting vector across chi for nested-space diagnostics.
        values, vectors = eigsh(local, k=chi+1, which='SA', tol=1e-12,
                               maxiter=30000, v0=np.random.default_rng(100+b).normal(size=4096))
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        W = vectors[:, :chi].copy()
        residual = np.linalg.norm(local @ W-W*values[:chi], axis=0).max()
        if residual/max(1, abs(values[0])) > 1e-9:
            raise RuntimeError(f"Block {b} eigenvectors unresolved")
        decoders.append(W); spectra.append(values); residuals.append(float(residual))
        zops.append([W.T @ (spins[:, a, None]*W) for a in range(12)])
        terms.append((1., [(b, np.diag(values[:chi]))]))
    # Exact SVD factorization of each 12x12 coupling matrix avoids hundreds
    # of redundant MPO terms. All singular components are kept (no tolerance).
    for b in range(len(groups)):
        for c in range(b):
            coupling = v[np.ix_(groups[c], groups[b])]/4  # combine ordered V/8 twice
            if not np.any(coupling):
                continue
            u, s, vt = np.linalg.svd(coupling, full_matrices=False)
            for k in range(12):
                left = sum(u[a,k]*zops[c][a] for a in range(12))
                right = sum(vt[k,a]*zops[b][a] for a in range(12))
                terms.append((float(s[k]), [(c,left), (b,right)]))
    return dict(sites=groups, decoders=decoders, spins=spins, zops=zops,
                spectra=spectra, block_residuals=residuals,
                cut_gaps=[float(s[chi]-s[chi-1]) for s in spectra],
                constant=float(constant), L=L, chi=chi, omega=omega,
                operator=Operator([chi]*len(groups), terms))
