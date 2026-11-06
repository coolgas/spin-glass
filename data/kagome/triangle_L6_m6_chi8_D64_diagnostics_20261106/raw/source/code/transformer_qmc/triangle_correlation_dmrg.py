"""Bounded, resumable RG+DMRG validation and four-size correlation pilot."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np
from ising.hamiltonian import Hamiltonian
from .physics import make_af_sample
from .triangle_effective_rg import hierarchy, physical_correlations
from .triangle_effective_mps import build, initial_state, optimize, physical_moments, full_state
from .triangle_correlation_observables import spectra, ratio, overlap_observables
from .triangle_correlation_grid import GAMMAS


class ResourceDeferred(RuntimeError):
    """A saved case can continue when its next solver stage fits resources."""


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case(root, L, gamma, sample_id, chi, validate=False, m=6,
         stage_guard=None, resume=False, eigensolver='cpu', qualify_overlap=False,
         max_bond_dimension=512):
    if max_bond_dimension not in (32, 64, 128, 256, 512):
        raise ValueError('Maximum bond dimension must be a member of the DMRG ladder')
    dimensions = tuple(d for d in (32, 64, 128, 256, 512) if d <= max_bond_dimension)
    key = f'L{L}_g{gamma:g}_s{sample_id}_m{m}_c{chi}'
    folder = root/'cases'/key
    folder.mkdir(parents=True, exist_ok=True)
    final = folder/'result.json'
    if final.exists():
        result = json.loads(final.read_text())
        assert result.get('max_bond_dimension', 512) == max_bond_dimension
        for name, value in result['hashes'].items():
            assert digest(folder/name) == value
        return result
    if shutil.disk_usage(root).free < 3*2**30:
        raise RuntimeError('Less than 3 GiB disk headroom: archive outputs before continuing')
    started = time.monotonic()
    sample = make_af_sample(L, sample_id)
    if resume:
        inputs = dict(L=L, gamma=gamma, sample_id=sample_id, m=m, chi=chi,
                      validate=validate, h=.15, lambda_rg=1., eigensolver=eigensolver)
        if qualify_overlap:
            inputs['qualify_overlap'] = True
        if max_bond_dimension != 512:
            inputs['max_bond_dimension'] = max_bond_dimension
        if (folder/'inputs.json').exists():
            assert json.loads((folder/'inputs.json').read_text()) == inputs
        else:
            assert not (folder/'model.pkl').exists()
            write(folder/'inputs.json', inputs)
        np.savez_compressed(folder/'inputs.npz', bonds=sample.bonds, J=sample.J,
                            points=sample.points, disorder_entropy=sample.seed_entropy)
    if resume and (folder/'model.pkl').exists():
        assert digest(folder/'model.pkl') == json.loads((folder/'model_hash.json').read_text())['sha256']
        with (folder/'model.pkl').open('rb') as stream:
            saved = pickle.load(stream)
        model, mpo, order = saved['model'], saved['mpo'], saved['order']
        construction_seconds = saved['construction_seconds']
    else:
        model = hierarchy(Hamiltonian(3*L*L, sample.bonds, sample.J, h=.15),
                          gamma, m, chi, eigensolver=eigensolver)
        mpo, order = build(model)
        construction_seconds = time.monotonic()-started
        temporary = folder/'model.pkl.tmp'
        with temporary.open('wb') as stream:
            pickle.dump(dict(model=model, order=order, mpo=mpo,
                             construction_seconds=construction_seconds), stream)
        temporary.replace(folder/'model.pkl')
        if resume:
            write(folder/'model_hash.json', dict(sha256=digest(folder/'model.pkl')))
    exact = None
    if validate and resume and (folder/'exact.json').exists():
        exact = json.loads((folder/'exact.json').read_text())
        with np.load(folder/'exact.npz') as archive:
            exact_s = archive['spectra']
    elif validate:
        if stage_guard:
            stage_guard(folder, model, mpo, None, None, 'exact')
        solutions = [model['operator'].solve(seed) for seed in (0, 1)]
        energy, vector, gap, residual = min(solutions, key=lambda row: row[0])
        magnetization, zz = physical_correlations(model, vector)
        exact_s = spectra(magnetization, zz, sample.points, L)
        exact = dict(energy=energy, spectrum=exact_s.tolist(), residual=residual,
                     start_energy_difference=abs(solutions[0][0]-solutions[1][0]))
        np.savez_compressed(folder/'exact.npz', state=vector, second_state=solutions[1][1],
                            magnetization=magnetization, zz=zz, spectra=exact_s)
        # Independently test MPO expectations on full random vectors.
        from tenpy.linalg import np_conserved as npc
        from tenpy.networks.mps import MPS
        checks = []
        for seed in (211, 713):
            if chi <= 4:
                v = np.random.default_rng(seed).normal(size=chi**len(order)); v /= np.linalg.norm(v)
                tensor = v.reshape((chi,)*len(order)).transpose(order)
                labels = [f'p{i}' for i in range(len(order))]
                array = npc.Array.from_ndarray_trivial(tensor, labels=labels)
                p = MPS.from_full(mpo.lat.mps_sites(), array, bc='finite', unit_cell_width=1)
            else:
                # Keep the independent MPO check affordable at chi=16,32.
                p = initial_state(mpo, seed)
                v = full_state(p, order, chi)
            error = abs(float(mpo.H_MPO.expectation_value(p))-v@model['operator'].action(v))
            assert error < 1e-9, error
            checks.append(error)
        exact['mpo_random_expectation_errors'] = checks
        exact['mpo_check_states'] = 'full random vectors' if chi <= 4 else 'random bond-16 MPS'
        write(folder/'exact.json', exact)
    records = []
    deferred_stages = []
    for seed in (0, 1):
        psi = initial_state(mpo, seed)
        prior = None
        for dimension in dimensions:
            record_path = folder/f'seed{seed}_D{dimension}.json'
            checkpoint_path = folder/f'seed{seed}_D{dimension}.pkl'
            if resume and record_path.exists() and checkpoint_path.exists():
                row = json.loads(record_path.read_text())
                for name, expected in row['checkpoint_hashes'].items():
                    assert digest(folder/name) == expected
                with checkpoint_path.open('rb') as stream:
                    psi = pickle.load(stream)['psi']
                records.append(row)
                if row.get('ladder_stable'):
                    break
                prior = row
                continue
            if stage_guard:
                try:
                    stage_guard(folder, model, mpo, psi, dimension, f'seed{seed}_D{dimension}')
                except ResourceDeferred:
                    deferred_stages.append(f'seed{seed}_D{dimension}')
                    break
            info = optimize(psi, mpo, dimension)
            magnetization, zz = physical_moments(psi, model, order)
            s = spectra(magnetization, zz, sample.points, L)
            energy = float(np.real(mpo.H_MPO.expectation_value(psi)))
            rr = float(ratio(s))
            row = dict(seed=seed, D_MPS=dimension, energy=energy,
                       observables=overlap_observables(magnetization, zz),
                       spectra=s.tolist(), R=rr if np.isfinite(rr) else None,
                       max_bond=max(psi.chi), entropy_max=float(max(psi.entanglement_entropy())),
                       sweeps=int(info['sweep_statistics']['sweep'][-1]),
                       truncation_failure=info['truncation_failure'],
                       canonical_error=info['canonical_error'],
                        last_energy_change=float(abs(np.diff(info['sweep_statistics']['E'])[-1])),
                        seconds=time.monotonic()-started,
                        solver_source_sha256={name:digest(Path(__file__).with_name(name)) for name in
                            ('triangle_correlation_dmrg.py','triangle_effective_mps.py','triangle_effective_rg.py')})
            if validate:
                v = full_state(psi, order, chi)
                direct_m, direct_zz = physical_correlations(model, v)
                assert np.max(abs(magnetization-direct_m)) < 1e-9
                assert np.max(abs(zz-direct_zz)) < 1e-9
                row['exact_energy_error_per_spin'] = abs(energy-exact['energy'])/(3*L*L)
                row['exact_spectrum_max_relative_error'] = float(np.max(abs(s-exact_s)/np.maximum(abs(exact_s), 1e-10)))
                row['exact_R_error'] = abs(rr-float(ratio(exact_s))) if np.isfinite(rr) and np.isfinite(ratio(exact_s)) else None
                row['exact_observables'] = overlap_observables(direct_m, direct_zz)
                # Compare against the exact ground state, not the DMRG vector.
                with np.load(folder/'exact.npz') as archive:
                    row['exact_observables'] = overlap_observables(archive['magnetization'],archive['zz'])
            state_path = folder/f'seed{seed}_D{dimension}.pkl'
            temporary = state_path.with_suffix('.pkl.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(dict(psi=psi, info=info), stream)
            temporary.replace(state_path)
            np.savez_compressed(folder/f'seed{seed}_D{dimension}.npz',
                magnetization=magnetization, zz=zz, spectra=s, points=sample.points,
                bonds=sample.bonds, J=sample.J, disorder_entropy=sample.seed_entropy)
            if resume:
                row['checkpoint_hashes'] = {p.name:digest(p) for p in
                    (state_path, folder/f'seed{seed}_D{dimension}.npz')}
            records.append(row)
            stable = prior is not None and abs(energy-prior['energy'])/(3*L*L) < 1e-7
            stable = stable and float(np.max(abs(s-np.array(prior['spectra']))/np.maximum(abs(s), 1e-10))) < .005
            stable = stable and row['last_energy_change']/(3*L*L) < 1e-8
            stable = stable and row['truncation_failure'] is None and row['canonical_error'] < 1e-8
            stable = stable and prior.get('truncation_failure') is None
            stable = stable and prior['R'] is not None and row['R'] is not None and abs(row['R']-prior['R']) < .002
            if validate:
                stable = stable and row['exact_energy_error_per_spin'] < 1e-6 and row['exact_spectrum_max_relative_error'] < .01
                stable = stable and row['exact_R_error'] is not None and row['exact_R_error'] < .002
            if qualify_overlap:
                from .triangle_overlap_report import difference, within
                stable = stable and prior is not None and within(difference(row['observables'],prior['observables']))
                if validate:
                    stable = stable and within(difference(row['observables'],row['exact_observables']),
                                               chi_tolerance=.01,q2_tolerance=.01)
            if stable:
                row['ladder_stable'] = True
                write(record_path, row)
                break
            row['ladder_stable'] = False
            write(record_path, row)
            prior = row
        else:
            row['ladder_stable'] = False
    if deferred_stages:
        write(folder/'resource_unresolved.json', dict(stages=deferred_stages,
              completed_records=records, solver_passed=False))
        raise ResourceDeferred('Next solver stages exceed resource admission; both starts preserved')
    endpoints = [next(r for r in reversed(records) if r['seed'] == seed) for seed in (0, 1)]
    a, b = endpoints
    agreed = abs(a['energy']-b['energy'])/(3*L*L) < 1e-6
    agreed = agreed and np.max(abs(np.array(a['spectra'])-b['spectra'])/np.maximum(abs(np.array(a['spectra'])), 1e-10)) < .01
    agreed = agreed and a['R'] is not None and b['R'] is not None and abs(a['R']-b['R']) < .002
    if qualify_overlap:
        from .triangle_overlap_report import difference, within
        agreed = agreed and within(difference(a['observables'],b['observables']),
                                   chi_tolerance=.01,q2_tolerance=.01)
    passed = bool(agreed and all(r.get('ladder_stable', False) for r in endpoints))
    result = dict(L=L, gamma=gamma, sample_id=sample_id, m=m, chi=chi, h=.15,
                   lambda_rg=1., records=records, exact=exact, solver_passed=passed,
                   max_bond_dimension=max_bond_dimension, D_ladder=list(dimensions),
                   computation_complete=True, start_agreement=bool(agreed),
                   numerical_status='qualified' if passed else 'unqualified_at_bond_cap',
                  retention_qualified=False, construction_seconds=construction_seconds,
                  eigensolver=eigensolver,
                  overlap_gates_enabled=qualify_overlap,
                  seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  mpo_bonds=list(map(int, mpo.H_MPO.chi)),
                  solver_source_sha256={name:digest(Path(__file__).with_name(name)) for name in
                    ('triangle_correlation_dmrg.py', 'triangle_effective_mps.py', 'triangle_effective_rg.py')},
                   hashes={p.name:digest(p) for p in folder.iterdir() if p.suffix in ('.npz', '.pkl')})
    reuse_manifest = root.parent/'checkpoint_reuse.json'
    if reuse_manifest.exists():
        result['checkpoint_reuse'] = dict(path=str(reuse_manifest),sha256=digest(reuse_manifest),
            note='Reused stages retain original elapsed times; result seconds measures this invocation only')
    write(final, result)
    print(json.dumps({k:result[k] for k in ('L','gamma','sample_id','chi','solver_passed','seconds')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--phase', choices=('validation', 'frontier', 'scan'), default='validation')
    args = parser.parse_args(); root = args.output
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parents[1]
    manifest = dict(gammas=GAMMAS, sizes=[4,6,8,10], disorders=list(range(60000,60008)),
                    chi=[2,3], m=6, D_MPS=[32,64,128,256,512], seeds=[0,1], worker_threads=1,
                    sources={str(p.relative_to(source)):digest(p) for p in source.rglob('*.py')})
    if (root/'manifest.json').exists():
        assert json.loads((root/'manifest.json').read_text()) == manifest
    else:
        write(root/'manifest.json', manifest)
    tasks = [(L,g,s,c,True) for L in (4,6) for g in (.1,.5,1.) for s in (60000,60001) for c in (2,3)]
    if args.smoke:
        tasks = [(4,.5,60000,2,True)]
    elif args.phase in ('frontier', 'scan'):
        tasks += [(L,g,s,c,False) for L in (8,10) for g in (.1,.5,1.) for s in (60000,60001) for c in (2,3)]
    if not args.smoke and args.phase == 'scan':
        tasks += [(L,g,s,c,False) for g in GAMMAS for L in (4,6,8,10) for s in range(60000,60008) for c in (2,3)]
    completed = 0
    try:
        for L,g,s,c,validation in tasks:
            write(root/'status.json', dict(state='running',completed=completed,total=len(tasks),
                current=dict(L=L,gamma=g,sample_id=s,chi=c,validation=validation)))
            result = case(root,L,g,s,c,validation)
            if args.phase == 'scan' and s == 60007 and c == 3:
                from .triangle_correlation_report import report
                report(root)
            if not result['solver_passed']:
                raise RuntimeError(f'Unresolved solver sensitivity at L{L} Gamma{g} s{s} chi{c}; review before expanding')
            completed += 1
        write(root/'status.json', dict(state='complete',completed=completed,total=len(tasks)))
    except Exception:
        (root/'failure.txt').write_text(traceback.format_exc())
        write(root/'status.json', dict(state='blocked',completed=completed,total=len(tasks),error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
