"""Export required production observables from independent population archives.

Input JSON: {"n_sites": 108, "samples": [{"sample_id": 0,
"populations": ["runs/t2_p16.npz", ...]}, ...]}. Paths are relative to
the input JSON. All populations must use the same physical and projector
settings; the input groups must identify independent disorder samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def summarize(specification, directory):
    n_sites = specification['n_sites']
    samples = []
    identities = set()
    paths = set()
    common_settings = None
    for group in specification['samples']:
        identity = group['sample_id']
        if identity in identities:
            raise ValueError('each disorder sample must have exactly one group')
        identities.add(identity)
        moments, energies, complete, forward_times, minimum_kish = [], [], [], [], []
        if len(group['populations']) < 2:
            raise ValueError('projector error needs independent populations')
        for name in group['populations']:
            path = (directory/name).resolve()
            if path in paths:
                raise ValueError('a population archive appears more than once')
            paths.add(path)
            with np.load(path, allow_pickle=False) as part:
                if int(part['n_sites']) != n_sites or int(part['n_replicas']) != 2:
                    raise ValueError('population geometry or replica count disagrees')
                settings = tuple(part[key].item() for key in (
                    'n_walkers', 'n_equilibrate', 'n_measure', 'tau_block',
                    'n_forward_cycles', 'tau_forward_requested', 'floor_unique', 'floor_effective'))
                if common_settings is None:
                    common_settings = settings
                if settings != common_settings:
                    raise ValueError('population projector settings disagree')
                moments.append([float(part[key+'_series'].mean()) for key in ('q', 'q2', 'q3', 'q4')])
                energies.append(float(part['energy_measure_trace'].mean()))
                complete.append(bool(part['projection_complete']))
                forward_times.extend(part['forward_times'].tolist())
                minimum_kish.append(float(part['min_effective_ancestors']))
        matrix = np.asarray(moments)
        mean = matrix.mean(axis=0)
        covariance = np.cov(matrix, rowvar=False, ddof=1)/len(matrix)
        q2, q2_se = float(mean[1]), float(np.sqrt(covariance[1, 1]))
        half = float(t.ppf(.975, len(matrix)-1)*q2_se)
        q2_ci = [max(0, q2-half), min(1, q2+half)]
        samples.append(dict(sample_id=identity, populations=group['populations'],
            population_moments=matrix.tolist(), moments=dict(zip(('q', 'q2', 'q3', 'q4'), mean.tolist())),
            moment_mean_covariance=covariance.tolist(), q2_projector_se=q2_se,
            chi_sg=n_sites*q2, chi_sg_projector_se=n_sites*q2_se,
            q_ea_squared=q2, q_ea=float(np.sqrt(q2)),
            q_ea_projector_ci95=np.sqrt(q2_ci).tolist(),
            energy=float(np.mean(energies)),
            energy_projector_se=float(np.std(energies, ddof=1)/np.sqrt(len(energies))),
            all_projection_complete=all(complete), achieved_forward_times=forward_times,
            min_effective_ancestors=min(minimum_kish)))
    if not samples:
        raise ValueError('at least one disorder sample is required')
    second = np.array([row['q_ea_squared'] for row in samples])
    errors = np.array([row['q2_projector_se'] for row in samples])
    count = len(samples)
    center = float(second.mean())
    if not 0 < center <= 1 or not np.isfinite(second).all():
        raise ValueError('pure second moments must be finite and positive')
    projector_se = float(np.sqrt(np.sum(errors**2))/count)
    disorder_variance = None
    scatter_se = None
    if count > 1:
        # Sample-to-sample scatter contains population measurement noise.
        scatter_se = float(second.std(ddof=1)/np.sqrt(count))
        disorder_variance = max(0, float(second.var(ddof=1)-np.mean(errors**2)))
    return dict(n_sites=n_sites, input_specification=specification,
        disorder_samples=count, samples=samples,
        q_ea_squared=center, q_ea=float(np.sqrt(center)), chi_sg=n_sites*center,
        q2_projector_se=projector_se, chi_sg_projector_se=n_sites*projector_se,
        q_ea_projector_se_delta=projector_se/(2*np.sqrt(center)),
        q2_sample_scatter_se=scatter_se,
        q_ea_sample_scatter_se_delta=None if scatter_se is None else scatter_se/(2*np.sqrt(center)),
        chi_sg_sample_scatter_se=None if scatter_se is None else n_sites*scatter_se,
        q2_disorder_se_estimate=None if disorder_variance is None else float(np.sqrt(disorder_variance/count)),
        all_projection_complete=all(row['all_projection_complete'] for row in samples),
        definitions=dict(q_ea_squared='equal-weight disorder mean of pure q2',
                         q_ea='sqrt(disorder mean of pure q2), finite-size RMS overlap',
                         chi_sg='n_sites * disorder mean of pure q2',
                         projector_se='scatter of independent population means; cycles remain grouped',
                         sample_scatter_se='sample-to-sample standard error including projector noise',
                         disorder_se_estimate='variance-component estimate subtracting mean projector variance; clipped at zero',
                         q_ea_se_delta='linear error propagation through sqrt(mean q2)',
                         systematic_error='forward-time and finite-population biases are assessed separately'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    specification = json.loads(args.input.read_text())
    result = summarize(specification, args.input.resolve().parent)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
