"""Shared, versioned GFMC production design and equal-time scaling estimators.

Raw replica moments are retained. Disorder-level ratios and phase-boundary
inference belong to the later qualified analysis, never individual-run tags.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .projector import tags


SCHEMA = 1
GAMMAS = [.25, .5, .75, 1., 1.5, 2., 3.]
FIELDS = [0., .05, .10, .15, .20, .25, .30]
ANCHORS = [4, 6, 8, 10]
SIZES = [4, 5, 6, 7, 8, 9, 10]
DEVELOPMENT_IDS = list(range(60000, 60032))
FINAL_IDS = list(range(61000, 61256))
TOLERANCES = dict(q2=.002, chi_connected_per_site=.002, energy_per_site=.001)
TARGET_SE = dict(q2=.005, chi_connected_per_site=.005, energy_per_site=.002)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed_for(task):
    # Full scientific identity, including size, method, precision and repeat.
    return int(identity(dict(namespace='pbc-gfmc-production-20261027-v1', task=task))[:16], 16)


def source_hashes(root):
    root = Path(root)
    paths = [root/'qmc_numerics.py']
    for name in ('transformer_qmc', 'kagome_qea', 'gfmc', 'ising'):
        paths.extend((root/name).rglob('*.py'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def geometry(L):
    """Triangular cell reciprocal vectors and all allowed cell momenta.

Lattice indexing is (cell y, cell x, sublattice). Fourier arrays use the
same (my, mx) order. Connected kmin averages directions b1,b2,b1+b2.
"""
    cell = np.array([[2., 1.], [0., np.sqrt(3)]])
    reciprocal = 2*np.pi*np.linalg.inv(cell).T
    indices = np.array([(mx, my) for my in range(L) for mx in range(L)])
    momenta = indices @ reciprocal.T / L
    positions = np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3)/2]])
    phases = np.exp(-1j*(momenta @ positions.T))
    return dict(L=L, reciprocal=reciprocal, momenta=momenta,
                indices=indices, phases=phases, kmin_indices=[1, L, L+1])


def production_tags(states, lattice):
    """Four-replica connected overlap and conventional sublattice spectra.

S_conn(k)=sum_ij Cij^2 exp[ik.(ri-rj)]/N, equal time.
S_ab(k)=<F_a(k) F_b(k)*>/L^2 retains kagome basis channels,
including conventional order invisible in the total k=0 magnetization.
"""
    L = lattice['L']
    s = np.asarray(states, dtype=np.float64)
    if s.ndim != 3 or s.shape[1:] != (4, 3*L*L):
        raise ValueError('production requires four replicas and 3 L^2 sites')
    W, _, N = s.shape
    result = tags(s)
    d = .5*(s[:, 0]-s[:, 1])*(s[:, 2]-s[:, 3])
    fourier_d = np.fft.fftn(d.reshape(W, L, L, 3), axes=(1, 2)).reshape(W, L*L, 3)
    amplitude = (fourier_d*lattice['phases'][None]).sum(axis=-1)
    result['connected_spectrum'] = np.abs(amplitude)**2 / N
    for k, index in enumerate(lattice['kmin_indices']):
        result[f'sg_k{k}'] = result['connected_spectrum'][:, index]
    spin = np.fft.fftn(s.reshape(W, 4, L, L, 3), axes=(2, 3)).reshape(W, 4, L*L, 3)
    matrix = np.einsum('wrka,wrkb->wkab', spin, spin.conj(), optimize=True)/(4*L*L)
    result['spin_sublattice_real'] = matrix.real
    result['spin_sublattice_imag'] = matrix.imag
    result['spin_spectrum'] = (np.abs((spin*lattice['phases'][None, None]).sum(-1))**2).mean(1)/N
    result['magnetization'] = s.mean(axis=(1, 2))
    return result


def exact_spectra(probability, spins, lattice):
    """Independent dense contraction for validating Fourier tag conventions."""
    p = np.asarray(probability, dtype=float)
    p = p/p.sum()
    s = np.asarray(spins, dtype=float)
    N = s.shape[1]
    L = lattice['L']
    m = p@s
    zz = s.T@(p[:, None]*s)
    connected = zz-np.outer(m, m)
    cells = np.array([[2*x+y, np.sqrt(3)*y] for y in range(L) for x in range(L)])
    offsets = np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3)/2]])
    points = (cells[:, None]+offsets[None]).reshape(N, 2)
    phase = np.exp(-1j*(points@lattice['momenta'].T))
    conn = np.einsum('ik,ij,jk->k', phase, connected**2, phase.conj()).real/N
    spin = np.einsum('ik,ij,jk->k', phase, zz, phase.conj()).real/N
    cell_phase = np.exp(-1j*(cells@lattice['momenta'].T))
    sub = np.empty((L*L, 3, 3), complex)
    for a in range(3):
        for b in range(3):
            sub[:, a, b] = np.einsum('ik,ij,jk->k', cell_phase, zz[a::3,b::3], cell_phase.conj())/(L*L)
    return dict(connected_spectrum=conn, spin_spectrum=spin,
                spin_sublattice_real=sub.real, spin_sublattice_imag=sub.imag)


def design():
    per_anchor = (64+16)*4*len(GAMMAS)*len(FIELDS)
    return dict(schema=SCHEMA, periodic=True, sizes=SIZES, anchor_sizes=ANCHORS,
                gamma=GAMMAS, h=FIELDS, development_ids=DEVELOPMENT_IDS,
                final_ids=FINAL_IDS, initial_final_disorders=64, repeats=4,
                unguided_subset=FINAL_IDS[:16], final_audit_ids=FINAL_IDS[:8],
                anchor_runs=per_anchor, all_anchor_runs=per_anchor*len(ANCHORS),
                numerical_tolerances=TOLERANCES, target_standard_errors=TARGET_SE,
                pending_crossing_tolerances=True, final_dispatch_enabled=False,
                observables=['energy', 'energy_growth', 'q', 'q2', 'q4',
                             'chi_connected', 'connected_spectrum', 'spin_spectrum',
                             'spin_sublattice_real', 'spin_sublattice_imag', 'magnetization'],
                definitions=dict(q_ea='q; finite-size mean_i <Zi>^2',
                                 chi_sg_equal_time='N*q2',
                                 chi_connected_equal_time='sum_ij Cij^2/N',
                                 binder='0.5*(3-[q4]_D/[q2]_D^2), at h=0',
                                 connected_ratio='1-mean([sg_k0,sg_k1,sg_k2]_D)/[chi_connected]_D',
                                 spectra='momenta indexed by (my,mx), no imaginary-time integration'),
                policy='Same anchor coverage and numerical standards; per-context settings require qualification')
