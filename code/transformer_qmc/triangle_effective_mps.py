"""TeNPy DMRG for the unmodified projected kagome Hamiltonian."""
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.tools.misc import TenpyInconsistencyError


def build(model):
    count = len(model['sites'])
    width = int(round(np.sqrt(count)))
    order = [y*width+x for y in range(width)
             for x in (range(width) if y % 2 == 0 else reversed(range(width)))]
    inverse = {b: i for i, b in enumerate(order)}
    chi = model['decoders'][0].shape[1]
    sites = [Site(npc.LegCharge.from_trivial(chi), sort_charge=False) for _ in order]
    for i, b in enumerate(order):
        D, spins = model['decoders'][b], model['spins']
        for a in range(12):
            sites[i].add_op(f'Z{a}', D.T @ (spins[:, a, None]*D))
            for c in range(a):
                sites[i].add_op(f'P{a}_{c}', D.T @ ((spins[:, a]*spins[:, c])[:, None]*D))
    terms = []
    for index, (coefficient, factors) in enumerate(model['operator'].terms):
        converted = []
        for b, matrix in factors:
            i = inverse[b]
            name = f'T{index}_{i}'
            sites[i].add_op(name, matrix)
            converted.append((i, name))
        terms.append((coefficient, sorted(converted)))
    lattice = Lattice([1], sites, bc='open', bc_MPS='finite')
    coupling = CouplingModel(lattice)
    for coefficient, factors in terms:
        if len(factors) == 1:
            i, op = factors[0]; coupling.add_onsite_term(coefficient, i, op)
        elif len(factors) == 2:
            (i, a), (j, b) = factors
            coupling.add_coupling_term(coefficient, i, j, a, b)
        else:
            raise ValueError('Unexpected term support')
    return MPOModel(lattice, coupling.calc_H_MPO()), order


def initial_state(mpo_model, seed):
    rng = np.random.default_rng(seed)
    sites = mpo_model.lat.mps_sites()
    count = len(sites)
    chi = sites[0].dim
    bonds = [min(16, chi**min(i, count-i)) for i in range(count+1)]
    tensors = [rng.normal(size=(site.dim, bonds[i], bonds[i+1]))
               for i, site in enumerate(sites)]
    psi = MPS.from_Bflat(sites, tensors, form=None, dtype=np.float64, unit_cell_width=1)
    psi.canonical_form()
    return psi


def optimize(psi, mpo_model, bond_dimension):
    engine = dmrg.TwoSiteDMRGEngine(psi, mpo_model, dict(mixer=True, max_sweeps=40,
        mixer_params=dict(amplitude=1e-3, decay=2., disable_after=15),
        min_sweeps=18, max_E_err=1e-10, max_S_err=1e-7,
        trunc_params=dict(chi_max=bond_dimension, svd_min=1e-12)))
    failure = None
    try:
        engine.run()
    except TenpyInconsistencyError as error:
        if 'Maximum truncation error' not in str(error):
            raise
        failure = str(error)
    info = dict(sweep_statistics=engine.sweep_stats, bond_statistics=engine.update_stats,
                truncation_failure=failure)
    psi.canonical_form()
    info['canonical_error'] = float(np.linalg.norm(psi.norm_test()))
    return info


def physical_moments(psi, model, order):
    count, n = len(order), sum(map(len, model['sites']))
    physical = np.asarray([model['sites'][b] for b in order])
    magnetization, zz = np.zeros(n), np.eye(n)
    for a in range(12):
        magnetization[physical[:, a]] = np.real(psi.expectation_value(f'Z{a}'))
        for c in range(a):
            values = np.real(psi.expectation_value(f'P{a}_{c}'))
            zz[physical[:, a], physical[:, c]] = values
            zz[physical[:, c], physical[:, a]] = values
        for c in range(12):
            values = np.real(psi.correlation_function(f'Z{a}', f'Z{c}', autoJW=False))
            for i in range(count):
                for j in range(i):
                    x, y = physical[i, a], physical[j, c]
                    zz[x, y] = zz[y, x] = values[i, j]
    return magnetization, zz


def full_state(psi, order, chi):
    tensor = psi.get_theta(0, n=len(order)).to_ndarray()
    tensor = tensor.reshape((chi,)*len(order))
    return tensor.transpose(np.argsort(order)).reshape(-1)
