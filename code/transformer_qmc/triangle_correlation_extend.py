"""Continue one unresolved DMRG case from archived checkpoints without replacing them."""
import argparse
import hashlib
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np

from .triangle_effective_mps import optimize, physical_moments
from .triangle_correlation_observables import ratio, spectra


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dimension', type=int, default=1024)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source
    original = json.loads((source/'result.json').read_text())
    if original['solver_passed']:
        raise ValueError('Continuation expects an unresolved case')
    assert (original['L'], original['gamma'], original['sample_id'], original['chi']) == (8, 1., 60000, 3)
    for filename, expected in original['hashes'].items():
        assert sha(source/filename) == expected
    with (source/'model.pkl').open('rb') as stream:
        payload = pickle.load(stream)
    model, order, mpo = payload['model'], payload['order'], payload['mpo']
    from .physics import make_af_sample
    sample = make_af_sample(8, 60000)
    for seed in args.seeds:
        checkpoint = source/f'seed{seed}_D512.pkl'
        output = args.output/f'seed{seed}_D{args.dimension}.json'
        if output.exists():
            print(f'Existing continuation: {output}', flush=True)
            continue
        with checkpoint.open('rb') as stream:
            psi = pickle.load(stream)['psi']
        started = time.monotonic()
        info = optimize(psi, mpo, args.dimension)
        m, zz = physical_moments(psi, model, order)
        s = spectra(m, zz, sample.points, 8)
        energy = float(np.real(mpo.H_MPO.expectation_value(psi)))
        previous = next(r for r in reversed(original['records']) if r['seed'] == seed)
        drift = abs(energy-previous['energy'])/(3*8*8)
        spectral_drift = float(np.max(abs(s-previous['spectra'])/np.maximum(abs(s),1e-10)))
        r = float(ratio(s))
        stable = bool(drift < 1e-7 and spectral_drift < .005 and abs(r-previous['R']) < .002
                      and info['truncation_failure'] is None and info['canonical_error'] < 1e-8
                      and abs(np.diff(info['sweep_statistics']['E'])[-1])/(3*8*8) < 1e-8)
        with (args.output/f'seed{seed}_D{args.dimension}.pkl').open('wb') as stream:
            pickle.dump(dict(psi=psi, info=info), stream)
        np.savez_compressed(args.output/f'seed{seed}_D{args.dimension}.npz', magnetization=m, zz=zz,
                            spectra=s, points=sample.points, bonds=sample.bonds, J=sample.J,
                            disorder_entropy=sample.seed_entropy)
        record = dict(seed=seed, D_MPS=args.dimension, energy=energy, R=r, spectra=s.tolist(),
                      max_bond=max(psi.chi), sweeps=int(info['sweep_statistics']['sweep'][-1]),
                      canonical_error=info['canonical_error'], truncation_failure=info['truncation_failure'],
                      last_energy_change=float(abs(np.diff(info['sweep_statistics']['E'])[-1])),
                      delta_energy_per_spin=drift, spectrum_max_relative_change=spectral_drift,
                      delta_R=abs(r-previous['R']), ladder_stable=stable,
                      seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha(checkpoint),
                      original_result_sha256=sha(source/'result.json'),
                      solver_source_sha256={name:sha(Path(__file__).with_name(name)) for name in
                                            ('triangle_effective_mps.py','triangle_correlation_extend.py')})
        temporary=output.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(output)
        print(json.dumps({k:record[k] for k in ('seed','D_MPS','energy','R','ladder_stable','seconds')}),flush=True)
    completed=[args.output/f'seed{seed}_D{args.dimension}.json' for seed in args.seeds]
    if all(p.exists() for p in completed):
        rows=[json.loads(p.read_text()) for p in completed]
        agreement = abs(rows[0]['energy']-rows[-1]['energy'])/(3*8*8)<1e-6
        agreement &= abs(rows[0]['R']-rows[-1]['R'])<.002
        agreement &= np.max(abs(np.array(rows[0]['spectra'])-rows[-1]['spectra'])/
                            np.maximum(abs(rows[0]['spectra']),1e-10))<.01
        summary=dict(solver_passed=bool(agreement and all(r['ladder_stable'] for r in rows)),
                     starts_agree=bool(agreement), source=str(source), seeds=args.seeds,
                     dimension=args.dimension)
        (args.output/'assessment.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
