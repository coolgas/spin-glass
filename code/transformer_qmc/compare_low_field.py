"""Frozen-checkpoint L2 correlation matrices, gaps and held-out field response."""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch

from kagome_qea.exact import operator_matrix
from .diagnose_shared import measure, reference
from .diagnose_low_field import spectrum
from .experiment import load_checkpoint, write_json
from .physics import basis, make_af_sample
from .symmetric import SymmetricTransformer


def connected_matrix(probability, spins):
    mean = probability @ spins
    return spins.T @ (probability[:, None]*spins)-mean[:, None]*mean[None, :]


@torch.no_grad()
def compare(checkpoints, *, ids=(10000, 10001), field_scan=False):
    """Evaluation only; no parameters or sampling distributions are updated."""
    torch.set_num_threads(4)
    states = basis(12)
    spins = torch.tensor(states, dtype=torch.long)
    fields = [(.5, 0.), (.5, .3), (2., .3)]
    if field_scan:
        fields += [(gamma, h) for gamma, h in ((.4, .15), (.7, .15), (1., .15), (1.5, .15))]
    references = {}
    for sample_id in ids:
        for gamma, h in fields:
            ham = make_af_sample(2, sample_id).hamiltonian(h)
            matrix = operator_matrix(ham, gamma)
            energy, exact, sectors = reference(matrix, h)
            correlation = connected_matrix(exact**2, states)
            if h == 0:
                half = len(states)//2
                direct = matrix[:half, :half]
                cross = matrix[:half, np.arange(len(states)-1, half-1, -1)]
                eigenvalues, vectors, residual = spectrum(direct+cross)
            else:
                eigenvalues, vectors, residual = spectrum(matrix)
            references[sample_id, gamma, h] = (ham, matrix, energy, exact, sectors,
                                              correlation, eigenvalues, vectors, residual)
    runs = []
    for checkpoint in checkpoints:
        saved = load_checkpoint(checkpoint, 'cpu')
        model = SymmetricTransformer(make_af_sample(2).hamiltonian(0.), .5,
                                     **saved['metadata']['architecture']).double().eval()
        model.load_state_dict(saved['model'])
        rows = []
        for (sample_id, gamma, h), ref in references.items():
            ham, matrix, energy, exact, sectors, correlation, values, vectors, residual = ref
            model.set_hamiltonian(ham, gamma)
            psi = model.log_amplitudes(spins).exp().numpy()
            if abs(psi@psi-1) > 1e-10:
                raise RuntimeError('normalization failed')
            measured = measure(psi, matrix, (energy, exact), states)
            actual = connected_matrix(psi**2, states)
            errors = actual-correlation
            offdiag = ~np.eye(12, dtype=bool)
            sector_psi = (psi[:2048]+psi[2048:][::-1])/np.sqrt(2) if h == 0 else psi
            weights = (vectors.T@sector_psi)**2
            gap = float(values[1]-values[0])
            row = dict(sample_id=sample_id, gamma=gamma, h=h, **measured,
                       exact_chi=float(np.sum(correlation**2)/12),
                       connected_correlation_rms_error=float(np.sqrt(np.mean(errors[offdiag]**2))),
                       connected_correlation_max_error=float(np.abs(errors[offdiag]).max()),
                       site_magnetization_rms_error=float(np.sqrt(np.mean(((psi**2-exact**2)@states)**2))),
                       connected_correlation=actual, exact_connected_correlation=correlation,
                       accessible_gap=gap, sectors=sectors, spectral_weights=weights,
                       excitation_energies=values-values[0], first_16_weight=float(weights.sum()),
                       energy_excess_over_gap=(measured['energy']-energy)/gap,
                       max_eigen_residual=residual)
            rows.append(row)
            print(checkpoint.parent.name, checkpoint.name, sample_id, gamma, h,
                  'F', round(row['fidelity'], 5), 'corrRMS', round(row['connected_correlation_rms_error'], 5), flush=True)
        runs.append(dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         step=saved['step'], architecture=saved['metadata']['architecture'], rows=rows))
    return dict(n_sites=12, disorder_ids=list(ids), field_scan=field_scan, runs=runs,
                interpretation='Frozen finite-size evaluation only. At h=0 the accessible gap and '
                'spectral weights use the even sector. All runs retain one parameter set across sizes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--field-scan', action='store_true')
    parser.add_argument('--ids', type=int, nargs='+', default=[10000, 10001])
    args = parser.parse_args()
    write_json(args.output, compare(args.checkpoint, ids=args.ids, field_scan=args.field_scan))


if __name__ == '__main__':
    main()
