"""Finite, resumable CPU benchmark and independent checks for triangle RG."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
from scipy.sparse.linalg import eigsh

from ising.hamiltonian import Hamiltonian
from kagome_qea.exact import operator_matrix
from .physics import make_af_sample, basis, correlations
from .triangle_effective_rg import (Operator, apply_axis, triangles, hierarchy,
                                    physical_correlations, joint_samples)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lift(first, vector, m):
    tensor = vector.reshape((m,)*len(first['sites']))
    for axis, U in enumerate(first['bases']):
        tensor = apply_axis(tensor, U, axis)
    n = 3*len(first['sites'])
    physical = np.arange(1 << n)
    local = [sum(((physical >> site) & 1) << a for a, site in enumerate(block))
             for block in first['sites']]
    return tensor[tuple(local)]


def lift_superblocks(model, vector, m):
    chi = model['retained'][0].shape[1]
    count = len(model['groups'])
    tensor = vector.reshape((chi,)*count)
    for b, W in enumerate(model['retained']):
        tensor = apply_axis(tensor, W, b)
    tensor = tensor.reshape((m,)*(4*count))
    order = [t for group in model['groups'] for t in group]
    return tensor.transpose(np.argsort(order)).ravel()


def validate_hierarchy(output):
    sample = make_af_sample(4, 60000)
    ham = Hamiltonian(48, sample.bonds, sample.J, h=.15)
    rng = np.random.default_rng(381)
    checks = []
    # Two coupled superblocks (eight triangles) in the existing L4 Hamiltonian,
    # keeping exactly every term with support in these triangles. This isolates
    # the second projection independently of full-spin 48-site diagonalization.
    for orientation in (0, 1):
        model = hierarchy(ham, .55, 2, 4, orientation, (0, 0))
        top = model['operator']
        v = rng.normal(size=top.dimension); v /= np.linalg.norm(v)
        w = rng.normal(size=top.dimension); w /= np.linalg.norm(w)
        Lv = lift_superblocks(model, v, 2)
        Lw = lift_superblocks(model, w, 2)
        error = abs(w @ top.action(v)-Lw @ model['first']['operator'].action(Lv))
        assert error < 1e-10, error
        # Independent physical energy from directly transformed original terms.
        _, zz = physical_correlations(model, v)
        state = v.reshape((4,)*4)
        field_energy = 0.
        for b, (D, sites) in enumerate(zip(model['decoders'], model['sites'])):
            moved = np.moveaxis(state, b, 0).reshape(4, -1)
            rho = moved @ moved.T
            for a, i in enumerate(sites):
                z = D.T @ (model['spins'][:, a, None]*D)
                t, local = divmod(a, 3)
                # C-order triangle digit t; little-endian bit local within it.
                flipped = np.arange(4096) ^ (1 << (3*(3-t)+local))
                x = D.T @ D[flipped]
                field_energy -= ham.h*np.sum(rho*z.T)+.55*np.sum(rho*x.T)
        direct = -sum(j*zz[i, k] for (i, k), j in zip(ham.bonds, ham.J))+field_energy
        energy_error = abs(direct-v @ top.action(v))
        assert energy_error < 1e-10, energy_error
        checks.append(dict(orientation=orientation, projection_error=float(error),
                           physical_energy_error=float(energy_error)))
    write_json(output/'hierarchy_validation.json', dict(passed=True, checks=checks))
    return checks


def validate(output):
    """Compare projection with an independently assembled physical Hamiltonian."""
    rng = np.random.default_rng(912)
    sample = make_af_sample(2, 60000)
    ham = Hamiltonian(12, sample.bonds, sample.J, h=.15)
    physical = operator_matrix(ham, .55)
    vals, vecs = eigsh(physical, k=2, which='SA', tol=1e-12,
                       v0=np.ones(4096))
    exact = vecs[:, np.argmin(vals)]
    exact_c = correlations(exact**2, basis(12))
    checks, screens = [], []
    for orientation in (0, 1):
        for m in (2, 4, 6, 8):
            first = triangles(ham, .55, m, orientation)
            v = rng.normal(size=m**4); v /= np.linalg.norm(v)
            w = rng.normal(size=m**4); w /= np.linalg.norm(w)
            Lv, Lw = lift(first, v, m), lift(first, w, m)
            error = abs(w @ first['operator'].action(v) - Lw @ (physical @ Lv))
            assert error < 1e-11, (orientation, m, error)
            if m == 8:
                action_error = np.linalg.norm(lift(first, first['operator'].action(v), m)-physical @ Lv)
                assert action_error < 1e-10
            energy, state, gap, residual = first['operator'].solve()
            psi = lift(first, state, m)
            c = correlations(psi**2, basis(12))
            pq = xor_mass(psi**2, 12)
            exact_pq = xor_mass(exact**2, 12)
            screens.append(dict(orientation=orientation, m=m,
                                energy_error_per_site=float((energy-vals.min())/12),
                                q2_error=float(abs(c['q2']-exact_c['q2'])),
                                tv=float(abs(pq-exact_pq).sum()/2), residual=residual))
            checks.append(dict(orientation=orientation, m=m, projection_error=float(error)))
    # Dense independent Kronecker construction vs tensor-axis action.
    terms = [(1.7, [(0, rng.normal(size=(3, 3))), (2, rng.normal(size=(2, 2)))]),
             (-.3, [(1, rng.normal(size=(4, 4)))])]
    operator = Operator([3, 4, 2], terms)
    v = rng.normal(size=24)
    assert np.max(abs(operator.action(v)-operator.dense() @ v)) < 1e-12
    # Entangled two-block Born distribution: exact 4x4 outcome probabilities.
    # Sparse decoders still use the same physical 4096-outcome sampler as L4.
    decoders = []
    for _ in range(2):
        Q, _ = np.linalg.qr(rng.normal(size=(4, 2)))
        D = np.zeros((4096, 2)); D[:4] = Q
        decoders.append(D)
    digits = np.array(np.unravel_index(np.arange(4096), (8,)*4)).T
    spins = (1-2*((digits[:, :, None] >> np.arange(3)) & 1)).reshape(4096, 12).astype(np.int8)
    model = dict(decoders=decoders, sites=[list(range(12)), list(range(12, 24))], spins=spins)
    v = np.array([1., .3, -.4, .9]); v /= np.linalg.norm(v)
    wave = decoders[0][:4] @ v.reshape(2, 2) @ decoders[1][:4].T
    probability = wave**2
    samples = joint_samples(model, v, 30000, 981)
    indexes = ((1-samples[:, [9, 10, 21, 22]])//2) @ np.array([4, 8, 1, 2])
    empirical = np.bincount(indexes, minlength=16)/len(samples)
    tv = float(abs(empirical-probability.ravel()).sum()/2)
    assert tv < .025, tv
    # Physical one- and two-point products vs direct outcome enumeration,
    # including within-block products, which must not be factorized.
    outcomes = np.array([np.r_[spins[a], spins[b]] for a in range(4) for b in range(4)])
    reference = correlations(probability.ravel(), outcomes)
    magnetization, zz = physical_correlations(model, v)
    assert np.max(abs(zz-reference['zz'])) < 1e-12
    assert np.max(abs(magnetization-reference['site_magnetization'])) < 1e-12
    result = dict(algebra_passed=True, checks=checks, l2_screens=screens,
                  joint_sampling_tv_30000=tv, physical_products_passed=True)
    write_json(output/'validation.json', result)
    print(json.dumps(result), flush=True)
    return result


def xor_mass(probability, n):
    transformed = probability.copy()
    for step in range(n):
        block = transformed.reshape(-1, 2, 1 << step)
        a, b = block[:, 0].copy(), block[:, 1].copy()
        block[:, 0], block[:, 1] = a+b, a-b
    convolution = transformed**2
    for step in range(n):
        block = convolution.reshape(-1, 2, 1 << step)
        a, b = block[:, 0].copy(), block[:, 1].copy()
        block[:, 0], block[:, 1] = (a+b)/2, (a-b)/2
    distance = np.array([i.bit_count() for i in range(1 << n)])
    return np.bincount(n-distance, weights=convolution, minlength=n+1)


def baseline_snapshot(root, output):
    records, files = [], {}
    destination = output/'baseline'
    destination.mkdir(exist_ok=True)
    for path in sorted((root/'projector/runs').glob('*.json')):
        record = json.loads(path.read_text())
        task = record['task']
        if task['gamma'] in (.55, .56) and task['sample_id'] in range(60000, 60008):
            raw = root/record['raw_file']
            actual = digest(raw)
            if actual != record['raw_sha256']:
                raise RuntimeError(f'Baseline raw hash mismatch: {raw}')
            (destination/path.name).write_bytes(path.read_bytes())
            files[str(path)] = digest(path)
            records.append(record)
    # Preserve exactly the archived physical bonds used by existing production.
    for path in (root/'inputs').rglob('*.npz'):
        (destination/path.name).write_bytes(path.read_bytes())
        files[str(path)] = digest(path)
    (destination/'manifest.json').write_bytes((root/'manifest.json').read_bytes())
    # Production regenerates bonds from its archived source rather than saving
    # input arrays. Recreate them with that exact source in an isolated process,
    # then verify against the independent benchmark source before comparison.
    archived_code = root.parent/'code'
    script = '''from transformer_qmc.physics import make_af_sample
import numpy as np
import sys
from pathlib import Path
for sid in range(60000,60008):
 s=make_af_sample(4,sid)
 np.savez_compressed(Path(sys.argv[1])/f"bonds_{sid}.npz",bonds=s.bonds,J=s.J,points=s.points)
'''
    env = dict(os.environ, PYTHONPATH=str(archived_code))
    subprocess.run([os.sys.executable, '-c', script, str(destination.resolve())], env=env, check=True)
    for sid in range(60000, 60008):
        sample = make_af_sample(4, sid)
        with np.load(destination/f'bonds_{sid}.npz') as archived:
            assert np.array_equal(sample.bonds, archived['bonds'])
            assert np.array_equal(sample.J, archived['J'])
            assert np.array_equal(sample.points, archived['points'])
    for name in ('transformer_qmc/physics.py', 'kagome_qea/lattice.py'):
        files[str(archived_code/name)] = digest(archived_code/name)
    write_json(destination/'snapshot.json', dict(created=time.time(), records=len(records),
                                               files=files, raw_hashes_verified=True,
                                               bonds_verified_against_archived_source=True))
    return records


def context(ham, model, vector, sample):
    magnetization, zz = physical_correlations(model, vector)
    connected = zz-np.outer(magnetization, magnetization)
    n, L = ham.n_sites, sample.L
    cell = np.array([[2*L, L], [0, np.sqrt(3)*L]])
    reciprocal = 2*np.pi*np.linalg.inv(cell).T
    momenta = np.stack([reciprocal[:, 0], reciprocal[:, 1], reciprocal[:, 0]+reciprocal[:, 1]])
    phase = np.exp(1j*(sample.points @ momenta.T))
    sk = np.einsum('ik,ij,jk->k', phase.conj(), connected**2, phase).real/n
    # Baseline stores S_connected(0), then the same three reciprocal directions.
    spectrum = np.r_[np.sum(connected**2)/n, sk]
    return magnetization, zz, dict(q=float(magnetization @ magnetization/n),
                                  q2=float(np.sum(zz**2)/n**2), spectrum=spectrum.tolist())


def run_case(output, baseline, sample_id, gamma, m, chi, orientation, origin, draws):
    key = f's{sample_id}_g{gamma:g}_m{m}_c{chi}_o{orientation}_x{origin[0]}y{origin[1]}'
    path = output/'cases'/f'{key}.json'
    if path.exists():
        return json.loads(path.read_text())
    start = time.monotonic()
    sample = make_af_sample(4, sample_id)
    ham = Hamiltonian(48, sample.bonds, sample.J, h=.15)
    model = hierarchy(ham, gamma, m, chi, orientation, origin)
    built = time.monotonic()
    solutions = [model['operator'].solve(seed) for seed in (0, 1)]
    energy, vector, gap, residual = min(solutions, key=lambda x: x[0])
    fidelity = abs(solutions[0][1] @ solutions[1][1])**2
    if abs(solutions[0][0]-solutions[1][0]) > 1e-8 or fidelity < 1-1e-6:
        raise RuntimeError(f'Eigensolver starts disagree: fidelity={fidelity}')
    solved = time.monotonic()
    magnetization, zz, metrics = context(ham, model, vector, sample)
    samples = joint_samples(model, vector, 2*draws, 20261104+sample_id+m*100+chi)
    n_equal = np.sum(samples[::2] == samples[1::2], axis=1)
    counts = np.bincount(n_equal, minlength=49)
    mass = counts/counts.sum()
    comparisons = []
    for kind in ('jastrow', 'rbm_2'):
        rows = [r for r in baseline if r['task']['sample_id'] == sample_id
                and r['task']['gamma'] == gamma and r['task']['kind'] == kind
                and r['task']['audit'] == 'main']
        if rows:
            ref_mass = np.mean([r['mass'] for r in rows], axis=0)
            comparisons.append(dict(kind=kind, repeats=len(rows),
                                    energy_difference_per_site=float((energy-np.mean([r['energy'] for r in rows]))/48),
                                    q2_difference=float(metrics['q2']-np.mean([r['q2'] for r in rows])),
                                    tv=float(abs(mass-ref_mass).sum()/2),
                                    spectrum_difference=(np.array(metrics['spectrum'])-np.mean([r['spectrum'] for r in rows], axis=0)).tolist(),
                                    ancestry_alarms=sum(r['ancestry_alarm'] for r in rows)))
    arrays = dict(bonds=sample.bonds, J=sample.J, points=sample.points, top_state=vector,
                  magnetization=magnetization, zz=zz, spins=samples, counts=counts,
                  sites=np.array(model['sites']), triangle_sites=np.array(model['first']['sites']))
    for b in range(len(model['groups'])):
        arrays[f'W_{b}'] = model['retained'][b]
        arrays[f'decoder_{b}'] = model['decoders'][b]
        arrays[f'spectrum_{b}'] = model['super_spectra'][b]
    arrays['triangle_bases'] = np.array(model['first']['bases'])
    arrays['triangle_spectra'] = np.array(model['first']['spectra'])
    raw_path = path.with_suffix('.npz')
    np.savez_compressed(raw_path, **arrays)
    result = dict(key=key, sample_id=sample_id, gamma=gamma, m=m, chi=chi,
                  orientation=orientation, origin=origin, energy=energy,
                  energy_per_site=energy/48, gap=gap, residual=residual,
                  starts_fidelity=float(fidelity), q_grid=np.linspace(-1, 1, 49).tolist(),
                  mass=mass.tolist(), draws=draws, metrics=metrics,
                  comparisons=comparisons, raw_sha256=digest(raw_path),
                  seconds=dict(build=built-start, solve=solved-built,
                               measurement=time.monotonic()-solved, total=time.monotonic()-start),
                  physical_accuracy_qualified=False)
    write_json(path, result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=16384)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'cases').mkdir(exist_ok=True)
    sources = {p.name: digest(p) for p in Path(__file__).parent.glob('triangle_effective*.py')}
    write_json(args.output/'launch.json', dict(created=time.time(), platform=platform.platform(),
               numpy=np.__version__, sources=sources, threads=os.environ.get('OPENBLAS_NUM_THREADS'),
               scope='validation then diagnostic L4 retention ladder; no L6',
               baseline=str(args.baseline), draws=args.draws))
    try:
        validation = validate(args.output)
        validate_hierarchy(args.output)
        if args.validate_only:
            return
        baseline = baseline_snapshot(args.baseline, args.output)
        # A finite diagnostic pilot is useful even if early truncations fail;
        # expansion to the 320-case grid requires an accuracy assessment.
        rows = []
        for m, chi in ((4, 4), (4, 8), (4, 16), (6, 8), (8, 8)):
            rows.append(run_case(args.output, baseline, 60000, .55, m, chi, 0, (0, 0), args.draws))
            write_json(args.output/'status.json', dict(completed=len(rows), total=5,
                       state='running', rows=[r['key'] for r in rows]))
        write_json(args.output/'status.json', dict(completed=5, total=5,
                   state='pilot_complete_assessment_required', rows=[r['key'] for r in rows],
                   l2_screens=validation['l2_screens']))
    except Exception:
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
