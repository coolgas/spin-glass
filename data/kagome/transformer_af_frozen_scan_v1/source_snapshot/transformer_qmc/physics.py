"""Quenched positive AF bonds and exact small-system diagnostics."""

import numpy as np
from scipy.sparse.linalg import eigsh

from kagome_qea.lattice import build_kagome
from kagome_qea.sample import Sample
from kagome_qea.exact import operator_matrix, moments_from_probability


def make_af_sample(L=2, sample_id=0, width=0.5, scale=1.0, periodic=True):
    if not isinstance(L, int) or isinstance(L, bool) or L < 1:
        raise ValueError("L must be a positive integer")
    if not isinstance(sample_id, int) or isinstance(sample_id, bool) or sample_id < 0:
        raise ValueError("sample_id must be a nonnegative integer")
    if not np.isfinite(width) or not 0 <= width < 1:
        raise ValueError("width must be in [0,1)")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    points, bonds, _, _ = build_kagome(L, periodic)
    entropy = np.array([20261027, L, sample_id, int(periodic)], dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    K = scale * (1 + width * rng.uniform(-1, 1, len(bonds)))
    return Sample(L, periodic, sample_id, entropy, points, bonds, -K,
                  f"af_uniform_nn_w{width:g}_K{scale:g}")


def basis(n_sites):
    if not 1 <= n_sites <= 16:
        raise ValueError("enumeration is restricted to N<=16")
    return (1 - 2 * ((np.arange(1 << n_sites)[:, None]
                      >> np.arange(n_sites)) & 1)).astype(np.int8)


def diagonal(ham, spins):
    return (-ham.h * spins.sum(axis=-1)
            - (spins[..., ham.bonds[:, 0]] * spins[..., ham.bonds[:, 1]]) @ ham.J)


def correlations(probability, spins, points=None, L=None):
    """Equal-time connected SG structure factors; no central-q substitution."""
    p = np.asarray(probability, dtype=float)
    p = p / p.sum()
    m = p @ spins
    zz = spins.T @ (p[:, None] * spins)
    connected = zz - np.outer(m, m)
    n = spins.shape[1]
    result = dict(magnetization=float(m.mean()), site_magnetization=m,
                  zz=zz, connected=connected,
                  q_mean=float(m @ m / n), q2=float(np.sum(zz**2) / n**2),
                  chi_connected=float(np.sum(connected**2) / n))
    if points is not None and L is not None:
        # Three equal-length reciprocal directions of the triangular Bravais cell.
        cell = np.array([[2 * L, L], [0, np.sqrt(3) * L]])
        reciprocal = 2 * np.pi * np.linalg.inv(cell).T
        momenta = np.stack([reciprocal[:, 0], reciprocal[:, 1],
                            reciprocal[:, 0] + reciprocal[:, 1]])
        phase = np.exp(1j * (points @ momenta.T))
        sk = np.einsum("ik,ij,jk->k", phase.conj(), connected**2, phase).real / n
        result.update(sg_kmin=sk, kmin=momenta,
                      sg_ratio=float(1 - sk.mean() / result["chi_connected"]))
        # Physical-length continuum second moment, explicitly labeled as such.
        ratio = result["chi_connected"] / sk - 1
        result["xi_second_moment"] = np.sqrt(np.maximum(ratio, 0)) / np.linalg.norm(momenta, axis=1)
    return result


def exact_reference(ham, gamma, points=None, L=None):
    matrix = operator_matrix(ham, gamma)
    values, vectors = eigsh(matrix, k=2, which="SA", tol=1e-12,
                            v0=np.ones(matrix.shape[0]))
    order = np.argsort(values)
    psi = vectors[:, order[0]]
    if psi.sum() < 0:
        psi = -psi
    if np.any(psi <= 0):
        raise RuntimeError("positive ground state unresolved; check eigensolver precision")
    p = psi**2
    states = basis(ham.n_sites)
    result = correlations(p, states, points, L)
    result.update(moments_from_probability(p, ham.n_sites))
    result.update(energy=float(values[order[0]]), gap=float(np.diff(values[order])[0]),
                  residual=float(np.linalg.norm(matrix @ psi - values[order[0]] * psi)),
                  probability=p, amplitude=psi)
    return result
