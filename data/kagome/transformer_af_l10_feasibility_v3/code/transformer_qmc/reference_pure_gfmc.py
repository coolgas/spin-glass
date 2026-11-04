"""Exact projection of constant/network trial states; evaluation references only."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply

from kagome_qea.exact import operator_matrix
from .experiment import write_json
from .frozen_scan import digest, key
from .physics import basis, correlations, make_af_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.results/'manifest.json').read_text())
    times = [0., .5, 1., 2., 4., 8., 16., 32., 64.]
    states = basis(12)
    rows = []
    finite_forward_rows = []
    for c in manifest['design']['contexts']:
        for sid in manifest['design']['sample_ids']:
            name = key(2, sid, c['gamma'], c['h'], c['K'])
            with np.load(args.scan/'exact'/(name+'.npz')) as cache:
                logs, exact = cache['logs'].copy(), cache['exact'].copy()
            exact_energy = json.loads((args.scan/'exact'/(name+'.json')).read_text())['exact_energy']
            ham = make_af_sample(2, sid, scale=c['K']).hamiltonian(c['h'])
            H = operator_matrix(ham, c['gamma'])
            shifted = H-exact_energy*eye(len(states), format='csr')
            target = correlations(exact**2, states)
            trace = float(shifted.diagonal().sum())
            for kind in manifest['design']['guides']:
                initial = np.ones(len(states)) if kind == 'constant' else np.exp(logs)
                initial /= np.linalg.norm(initial)
                psi = initial.copy()
                previous = 0.
                trajectory = []
                for Ktau in times:
                    tau = Ktau/c['K']
                    if tau > previous:
                        dt = tau-previous
                        psi = expm_multiply(-dt*shifted, psi, traceA=-dt*trace)
                        psi /= np.linalg.norm(psi)
                    current = correlations(psi**2, states)
                    offdiag = ~np.eye(12, dtype=bool)
                    trajectory.append(dict(
                        Ktau=Ktau, fidelity=float((psi@exact)**2),
                        pure_energy_error_per_spin=float((psi@(H@psi)-exact_energy)/12),
                        mixed_energy_error_per_spin=float((initial@(H@psi)/(initial@psi)-exact_energy)/12),
                        chi_connected=current['chi_connected'],
                        exact_chi_connected=target['chi_connected'],
                        connected_correlation_rms_error=float(np.sqrt(np.mean(
                            (current['connected']-target['connected'])[offdiag]**2)))))
                    previous = tau
                rows.append(dict(**c, sample_id=sid, guide=kind, trajectory=trajectory))
                settings = manifest['design']
                start_Ktau = settings['equilibration_Ktau']+settings['measurement_Ktau']
                for forward in settings['forward_Ktau']:
                    # In infinite population, a tag at t followed for f has
                    # ancestor probability proportional to phi_t(s)*phi_f(s),
                    # where phi_t=exp(-t H) psi_T. Four copies then factorize.
                    dt = forward/c['K']
                    right = expm_multiply(-dt*shifted, initial, traceA=-dt*trace)
                    right /= np.linalg.norm(right)
                    dt_start = start_Ktau/c['K']
                    left = expm_multiply(-dt_start*shifted, initial, traceA=-dt_start*trace)
                    left /= np.linalg.norm(left)
                    cycles = []
                    for cycle in range(settings['cycles']):
                        probability = left*right
                        if np.min(probability) < -1e-13:
                            raise RuntimeError('negative exact forward probability')
                        current = correlations(np.maximum(probability, 0), states)
                        cycles.append(dict(
                            cycle=cycle, tag_Ktau=start_Ktau+cycle*forward,
                            chi_connected=current['chi_connected'], q2=current['q2']))
                        left = expm_multiply(-dt*shifted, left, traceA=-dt*trace)
                        left /= np.linalg.norm(left)
                    finite_forward_rows.append(dict(
                        **c, sample_id=sid, guide=kind, forward_Ktau=forward,
                        chi_connected=float(np.mean([v['chi_connected'] for v in cycles])),
                        q2=float(np.mean([v['q2'] for v in cycles])), cycles=cycles))
    write_json(args.results/'projection_reference.json', dict(
        source_sha256=digest(__file__), input_sha256=manifest['input_sha256'],
        checkpoint_sha256=manifest['checkpoint_sha256'], times_Ktau=times,
        interpretation='Deterministic pure projected states, not finite-forward-time stochastic estimators',
        rows=rows, finite_forward_rows=finite_forward_rows,
        finite_forward_definition='p_ancestor(s) proportional to phi_tag_time(s)*phi_forward(s); infinite population',
        completed=True))
    print('Completed', len(rows), 'deterministic projection trajectories', flush=True)


if __name__ == '__main__':
    main()
