"""Exact L2 low-energy spectrum and frozen-guide spectral weight diagnostics."""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import eigsh

from kagome_qea.exact import operator_matrix
from .experiment import load_checkpoint, write_json
from .physics import basis, make_af_sample
from .symmetric import SymmetricTransformer


def spectrum(matrix, count=16):
    values, vectors = eigsh(matrix, k=count, which='SA', tol=1e-11,
                            v0=np.random.default_rng(731).normal(size=matrix.shape[0]))
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    residuals = np.linalg.norm(matrix @ vectors-vectors*values[None], axis=0)
    if residuals.max() > 1e-8:
        raise RuntimeError('low-energy eigensolver residual too large')
    return values, vectors, float(residuals.max())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = load_checkpoint(args.checkpoint, 'cpu')
    model = SymmetricTransformer(make_af_sample(2).hamiltonian(0.), .5,
                                 **saved['metadata']['architecture']).double().eval()
    model.load_state_dict(saved['model'])
    spins = torch.tensor(basis(12), dtype=torch.long)
    rows = []
    for sample_id in (10000, 10001):
        for gamma, h in ((.5, 0.), (.5, .3), (2., .3)):
            ham = make_af_sample(2, sample_id).hamiltonian(h)
            matrix = operator_matrix(ham, gamma)
            model.set_hamiltonian(ham, gamma)
            psi = model.log_amplitudes(spins).exp().numpy()
            psi /= np.linalg.norm(psi)
            energy = float(psi @ (matrix @ psi))
            parity_gap = None
            if h == 0:
                half = matrix.shape[0]//2
                direct = matrix[:half, :half]
                cross = matrix[:half, np.arange(matrix.shape[0]-1, half-1, -1)]
                values, vectors, residual = spectrum(direct+cross)
                odd, _, odd_residual = spectrum(direct-cross, count=1)
                parity_gap = float(odd[0]-values[0])
                residual = max(residual, odd_residual)
                state = (psi[:half]+psi[half:][::-1])/np.sqrt(2)
            else:
                values, vectors, residual = spectrum(matrix)
                state = psi
            weights = np.abs(vectors.T @ state)**2
            gap = float(values[1]-values[0])
            excess = energy-float(values[0])
            row = dict(sample_id=sample_id, gamma=gamma, h=h,
                       sector='even' if h == 0 else 'full',
                       ground_energy=float(values[0]), guide_energy=energy,
                       energy_excess=excess, energy_error_per_spin=excess/12,
                       accessible_gap=gap, parity_gap=parity_gap,
                       full_gap=min(gap, parity_gap) if h == 0 else gap,
                       excitation_energies=values-values[0], spectral_weights=weights,
                       ground_fidelity=float(weights[0]),
                       first_16_weight=float(weights.sum()),
                       remaining_weight=float(1-weights.sum()),
                       energy_excess_over_gap=excess/gap,
                       fidelity_lower_bound=max(0., 1-excess/gap),
                       max_eigen_residual=residual)
            rows.append(row)
            print(sample_id, gamma, h, 'gap', round(gap, 6),
                  'parity gap', parity_gap, 'dE/gap', round(excess/gap, 3),
                  'F', round(weights[0], 4), 'first16', round(weights.sum(), 4), flush=True)
    write_json(args.output, dict(checkpoint=args.checkpoint.name,
               sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               step=saved['step'], n_sites=12, rows=rows,
               interpretation='Exact finite-size spectra; no thermodynamic phase claim. '
               'At h=0 the guide is even, so its accessible gap is the even-sector gap. '
               'The bound 1-F <= (E-E0)/gap uses total energy, not energy per spin.'))


if __name__ == '__main__':
    main()
