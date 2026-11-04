"""Exact finite-size imaginary-time convergence of frozen shared guides.

This evaluates exp[-tau(H-E0)] psi on all L2 configurations. It measures the
projection time required by each guide without walker population or ancestry
error, and supplies a reference for subsequent stochastic projector studies.
"""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply

from kagome_qea.exact import operator_matrix
from .compare_low_field import connected_matrix
from .diagnose_shared import measure, reference
from .experiment import load_checkpoint, write_json
from .physics import basis, make_af_sample
from .symmetric import SymmetricTransformer


@torch.no_grad()
def diagnose(checkpoint, times=(0., .5, 1., 2., 4., 8., 16.), ids=(10000, 10001)):
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('projection times must start at zero and strictly increase')
    torch.set_num_threads(4)
    saved = load_checkpoint(checkpoint, 'cpu')
    model = SymmetricTransformer(make_af_sample(2).hamiltonian(0.), .5,
                                 **saved['metadata']['architecture']).double().eval()
    model.load_state_dict(saved['model'])
    states = basis(12)
    spins = torch.tensor(states, dtype=torch.long)
    offdiag = ~np.eye(12, dtype=bool)
    rows = []
    for sample_id in ids:
        for gamma, h in ((.5, 0.), (.5, .3), (2., .3)):
            ham = make_af_sample(2, sample_id).hamiltonian(h)
            matrix = operator_matrix(ham, gamma)
            energy, exact, _ = reference(matrix, h)
            exact_c = connected_matrix(exact**2, states)
            model.set_hamiltonian(ham, gamma)
            psi = model.log_amplitudes(spins).exp().numpy()
            if abs(psi@psi-1) > 1e-10:
                raise RuntimeError('initial normalization failed')
            shifted = matrix-energy*eye(len(psi), format='csr')
            trace = float(shifted.diagonal().sum())
            trajectory = []
            previous = 0.
            for tau in times:
                if tau > previous:
                    dt = float(tau-previous)
                    psi = expm_multiply(-dt*shifted, psi, traceA=-dt*trace)
                    psi /= np.linalg.norm(psi)
                metrics = measure(psi, matrix, (energy, exact), states)
                error = connected_matrix(psi**2, states)-exact_c
                metrics.update(tau=float(tau), exact_chi=float(np.sum(exact_c**2)/12),
                               connected_correlation_rms_error=float(np.sqrt(np.mean(error[offdiag]**2))))
                if trajectory:
                    if metrics['energy'] > trajectory[-1]['energy']+1e-10:
                        raise RuntimeError('exact projection increased energy')
                    if metrics['fidelity'] < trajectory[-1]['fidelity']-1e-10:
                        raise RuntimeError('exact projection decreased fidelity')
                if h == 0 and abs(metrics['parity_expectation']-1) > 1e-10:
                    raise RuntimeError('exact projection broke even parity')
                trajectory.append(metrics)
                previous = tau
            rows.append(dict(sample_id=sample_id, gamma=gamma, h=h, trajectory=trajectory))
            print(sample_id, gamma, h, f'F(tau=0,{times[-1]:g})',
                  trajectory[0]['fidelity'], trajectory[-1]['fidelity'], flush=True)
    return dict(checkpoint=str(checkpoint.resolve()),
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                step=saved['step'], n_sites=12, rows=rows,
                interpretation='Exact normalized exp[-tau(H-E0)] psi on L2; evaluation only. '
                'This establishes finite-size projection-time targets, not walker-population '
                'convergence or larger-system accuracy. Tau is in inverse bond-energy units.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, diagnose(args.checkpoint))


if __name__ == '__main__':
    main()
