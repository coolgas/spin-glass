"""Independent exact-state checks of the physical correlation convention."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh
from kagome_qea.exact import operator_matrix
from .physics import make_af_sample, basis
from .production_protocol import exact_spectra, geometry
from .triangle_correlation_observables import spectra, ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    spins = basis(12)
    lattice = geometry(2)
    for sid in (60000, 60001):
        sample = make_af_sample(2, sid)
        for gamma in (.1, .5, 1., 2.):
            H = operator_matrix(sample.hamiltonian(.15), gamma)
            energy, state = eigsh(H, k=1, which='SA', tol=1e-12,
                                 v0=np.random.default_rng(0).normal(size=4096))
            p = state[:, 0]**2
            m = p@spins; zz = spins.T@(p[:, None]*spins)
            result = spectra(m, zz, sample.points, 2)
            reference = exact_spectra(p, spins, lattice)['connected_spectrum'][[0,1,2,3]]
            np.testing.assert_allclose(result, reference, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(np.diag(zz)-m*m, 1-m*m, atol=1e-12)
            residual = float(np.linalg.norm(H@state[:,0]-energy[0]*state[:,0]))
            assert residual < 1e-9
            rows.append(dict(sample_id=sid,gamma=gamma,max_error=float(max(abs(result-reference))),residual=residual))
        m = np.linspace(-.8,.8,12); zz = np.outer(m,m); np.fill_diagonal(zz,1.)
        product = spectra(m,zz,sample.points,2)
        np.testing.assert_allclose(product, np.mean((1-m*m)**2), atol=1e-12)
        assert abs(float(ratio(product))) < 1e-7
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(exact_checks=rows,product_state_passed=True),indent=2)+'\n')
    print('Eight exact L2 spectral checks and product-state limit passed.')


if __name__ == '__main__':
    main()
