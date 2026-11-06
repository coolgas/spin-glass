"""Matched three-size, two-retention replica-overlap scan manifest."""

import argparse
import hashlib
import json
from pathlib import Path

GAMMA_STEP = .025
GAMMAS = [i / 40 for i in range(4, 61)]


def scan_manifest():
    """Requested comparison settings do not imply physical retention convergence."""
    sizes = [4, 6, 8]
    disorders = list(range(60000, 60008))
    return dict(
        schema_version=2,
        state='prepared_for_validated_dispatch',
        scientific_runs_launched=False,
        compute_location='AutoDL',
        sizes=sizes, gammas=GAMMAS, gamma_step=GAMMA_STEP, disorders=disorders,
        physical_parameters=dict(K_distribution='Uniform(0.5,1.5)', h=.15, lambda_rg=1.),
        primary_observables=['q_ea_finite', 'chi_sg_connected_equal_time',
                             'chi_sg_raw_equal_time', 'q2_raw'],
        observables=['q2_raw', 'q_ea_finite', 'chi_sg_connected_equal_time',
                     'chi_sg_connected_diagonal', 'chi_sg_connected_off_diagonal',
                     'magnetization', 'chi_sg_raw_equal_time'],
        secondary_observables=['connected_correlation_length_ratio'],
        matching='Same disorder at every field and retention setting within each size; independent ensembles across sizes',
        uncertainty='Whole-disorder bootstrap, matched across fields and settings within each size',
        retention_settings=[dict(m=8,chi=3),dict(m=8,chi=16)],
        retention_selection='Requested comparison; neither setting is asserted physically converged',
        solver=dict(starts=[0, 1], D_MPS=[32, 64, 128, 256, 512],
                    original_gates='unchanged',
                    observable_checks=dict(final_step_q_absolute=.001,
                        final_step_chi_relative=.005, starts_and_exact_q_absolute=.001,
                         starts_and_exact_chi_relative=.01,
                         final_step_q2_relative=.005, starts_and_exact_q2_relative=.01)),
        scaling=dict(reference='https://arxiv.org/abs/2505.05117',
                     length='ell=sqrt(N)=sqrt(3)*L',
                      candidate_ansatz='[q2_raw]*ell**b_q=f((Gamma-Gamma_c)*ell**(1/nu))',
                      selection='Inspect qualified qEA and connected/raw chiSG alongside q2; use observable-specific forms and polarization backgrounds',
                     fit_parameters=['Gamma_c','b_q','nu'],
                     checks=['whole-disorder bootstrap','fit-window sensitivity',
                             'smallest-size sensitivity','separate chi fits and paired retention shifts'],
                     interpretation='exploratory with three sizes; fixed h=0.15 differs from paper'),
        admission=['Exact retained-space L4 validation for each chi',
                   'Per-size resource profile and qualified pilots before dense expansion',
                   'Resource admission preserving active campaigns and their reservations'],
        analysis=['Complete common-grid curves for qualified sizes',
                    'Compare qEA, connected/raw chiSG, raw replica q2, polarization and off-diagonal susceptibility',
                  'Refine candidate critical-field windows on a common grid across sizes',
                  'Test finite-size scaling with disorder bootstrap and field-window, size and retention sensitivity'],
        point_count_per_retention=len(sizes)*len(GAMMAS)*len(disorders),
        total_case_count=2*len(sizes)*len(GAMMAS)*len(disorders),
        points=[dict(L=L, N=3*L*L, gamma=g, sample_id=s)
                for g in GAMMAS for L in sizes for s in disorders],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = scan_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(dict(output=str(args.output), state=manifest['state'],
                          points=manifest['point_count_per_retention'])))


if __name__ == '__main__':
    main()
