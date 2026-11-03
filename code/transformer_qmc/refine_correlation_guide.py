"""Hamiltonian-only low-order log-amplitude correction to a frozen neural guide.

Enumeration is an L2 optimization accelerator, not an exact-state training label.
The degree-four correction is not qualified for larger lattices by this module.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits


def features(spins, degree):
    spins = np.asarray(spins, dtype=float)
    subsets = [subset for order in range(1, degree + 1)
               for subset in combinations(range(spins.shape[1]), order)]
    return np.column_stack([np.prod(spins[:, subset], axis=1) for subset in subsets]), subsets


def energy_gradient(theta, original_logs, design, hamiltonian):
    logs = original_logs + design @ theta
    amplitude = np.exp(logs - logs.max())
    amplitude /= np.linalg.norm(amplitude)
    applied = hamiltonian @ amplitude
    energy = float(amplitude @ applied)
    gradient = 2 * design.T @ (amplitude * (applied - energy * amplitude))
    return energy, gradient


def refine(original_logs, spins, hamiltonian, degree=4, maxiter=2000):
    """Minimize <psi|H|psi>/<psi|psi>; no exact reference enters this function."""
    if not 1 <= degree <= 4:
        raise ValueError('Correction degree must be between one and four')
    original_logs = np.asarray(original_logs, dtype=float)
    history, theta = [], np.array([])
    with threadpool_limits(limits=1):
        for order in range(min(2, degree), degree + 1):
            design, subsets = features(spins, order)
            theta = np.pad(theta, (0, design.shape[1] - len(theta)))
            before, _ = energy_gradient(theta, original_logs, design, hamiltonian)
            result = minimize(energy_gradient, theta, args=(original_logs, design, hamiltonian),
                              jac=True, method='L-BFGS-B', options=dict(
                                  maxiter=maxiter, ftol=1e-14, gtol=1e-8, maxls=40, maxcor=30))
            if not np.isfinite(result.fun) or result.fun > before + 1e-10:
                raise RuntimeError('Variational correction did not decrease energy')
            theta = result.x
            history.append(dict(degree=order, energy=float(result.fun),
                                optimizer_converged=bool(result.success),
                                message=str(result.message), iterations=int(result.nit),
                                gradient_max=float(np.max(np.abs(result.jac)))))
    logs = original_logs + design @ theta
    amplitude = np.exp(logs - logs.max())
    amplitude /= np.linalg.norm(amplitude)
    if np.any(amplitude <= 0) or not np.isfinite(amplitude).all():
        raise FloatingPointError('Nonpositive or nonfinite corrected guide')
    energy = float(amplitude @ (hamiltonian @ amplitude))
    variance = float(np.linalg.norm(hamiltonian @ amplitude - energy * amplitude)**2)
    return dict(logs=np.log(amplitude), coefficients=theta, subsets=subsets,
                energy=energy, local_energy_variance=variance, history=history,
                exact_labels_used=False, degree=degree)
