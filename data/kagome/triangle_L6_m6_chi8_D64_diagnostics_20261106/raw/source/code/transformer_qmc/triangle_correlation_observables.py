"""Physical equal-time connected-overlap spectra and disorder bootstrap."""
import numpy as np


def overlap_observables(magnetization, zz):
    """Physical Pauli-Z moments; susceptibility includes i=j and is equal-time."""
    m = np.asarray(magnetization, dtype=float)
    zz = np.asarray(zz, dtype=float)
    if m.ndim != 1 or not len(m) or zz.shape != (len(m), len(m)):
        raise ValueError('Expected N magnetizations and an N by N pair matrix')
    if not (np.isfinite(m).all() and np.isfinite(zz).all()):
        raise ValueError('Nonfinite physical moments')
    if not np.allclose(zz, zz.T, atol=1e-8, rtol=0):
        raise ValueError('Physical pair matrix is not symmetric')
    if not np.allclose(np.diag(zz), 1, atol=1e-8, rtol=0):
        raise ValueError('Expected physical Pauli-Z diagonal <Z_i^2>=1')
    connected = zz - np.outer(m, m)
    n = len(m)
    diagonal = float(np.mean(np.diag(connected)**2))
    susceptibility = float(np.sum(connected**2)/n)
    return dict(q_ea_finite=float(np.mean(m*m)), magnetization=float(m.mean()),
                chi_sg_connected_equal_time=susceptibility,
                chi_sg_connected_diagonal=diagonal,
                chi_sg_connected_off_diagonal=susceptibility-diagonal,
                q2_raw=float(np.mean(zz**2)),
                chi_sg_raw_equal_time=float(np.sum(zz**2)/n))


def spectra(magnetization, zz, points, L):
    m = np.asarray(magnetization)
    connected = np.asarray(zz) - np.outer(m, m)
    cell = np.array([[2., 1.], [0., np.sqrt(3)]])
    reciprocal = 2*np.pi*np.linalg.inv(cell).T
    momenta = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) @ reciprocal.T / L
    phase = np.exp(1j * np.asarray(points) @ momenta.T)
    result = np.einsum('ik,ij,jk->k', phase, connected**2, phase.conj()).real / len(m)
    return result


def ratio(values):
    values = np.asarray(values)
    zero, shell = values[..., 0], values[..., 1:].mean(axis=-1)
    fraction = np.divide(zero, shell, out=np.full_like(zero, np.nan), where=shell > 0)-1
    return np.sqrt(np.where((zero > 0) & (fraction >= 0), fraction, np.nan))*np.sqrt(3)/(2*np.pi)


def disorder_summary(values, indices):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    draws = ratio(values[indices].mean(axis=1))
    invalid = float(np.mean(~np.isfinite(draws)))
    interval = np.quantile(draws, [.025, .975]).tolist() if invalid == 0 else None
    point = float(ratio(mean))
    return dict(spectra=mean.tolist(), R=point if np.isfinite(point) else None,
                R_interval95=interval, invalid_bootstrap_fraction=invalid,
                spectrum_disorder_se=(values.std(axis=0, ddof=1)/np.sqrt(len(values))).tolist())
