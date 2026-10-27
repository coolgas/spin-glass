"""Resumable Gamma/h/K evaluation and GFMC of the selected frozen shared guide.

No optimizer is constructed. Exact states are references only; the L2 GFMC
lookup contains network amplitudes. Each stage has an exclusive process lock.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply

from kagome_qea.exact import operator_matrix
from .compare_low_field import connected_matrix
from .diagnose_shared import measure, reference
from .experiment import evaluate, load_checkpoint, write_json
from .model import EnumeratedGuide, TransformerGuide
from .physics import basis, correlations, make_af_sample
from .projector import propagate, rates_and_local, project
from .symmetric import SymmetricTransformer


SELECTED_SHA256 = '5f9fa33b04102ddff2e43179440514021090ed3aaf228ce67dce53c2139e2544'
GAMMAS = [.25, .5, .75, 1., 1.5, 2., 3.]
FIELDS = [0., .15, .3, .6, 1.]
SCALES = [.5, 1., 2.]
# Fixed before examining this scan. Includes anchors and field/bond extrapolation.
PROJECTOR_POINTS = [(.5, 0., 1.), (.5, .3, 1.), (2., .3, 1.),
                    (.25, .3, 1.), (.5, .6, 1.), (.5, .3, .5), (.5, .3, 2.)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(L, sample_id, gamma, h, K):
    return f'L{L}_s{sample_id}_g{gamma:g}_h{h:g}_K{K:g}'.replace('.', 'p')


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


@contextmanager
def stage_lock(output, stage):
    output.mkdir(parents=True, exist_ok=True)
    with (output/f'{stage}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def model_from(checkpoint, device):
    saved = load_checkpoint(checkpoint, 'cpu')
    model = SymmetricTransformer(make_af_sample(2).hamiltonian(.3), .5,
                                 **saved['metadata']['architecture']).double().to(device).eval()
    model.load_state_dict(saved['model'])
    model.requires_grad_(False)
    return model


def initialize(args):
    if digest(args.checkpoint) != SELECTED_SHA256:
        raise RuntimeError('checkpoint differs from frozen development winner')
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    manifest = dict(
        checkpoint_sha256=SELECTED_SHA256, checkpoint_name=args.checkpoint.name,
        selection='main seed501; minimum predefined development score 0.291550',
        gamma=GAMMAS, h=FIELDS, K=SCALES, sizes=[2, 4, 6, 8],
        exact_ids=list(range(10000, 10008)), sampled_ids=[10000, 10001],
        reserved_test_ids=list(range(40000, 40008)), reserved_tests_inspected=False,
        disorder='K_ij=K*(1+0.5*u_ij), matched u_ij for each L/sample_id; PBC',
        Hamiltonian='sum K_ij ZiZj - Gamma sum Xi - h sum Zi',
        precision='float64 network inference; no parameter updates',
        sampled_batches=8, sampled_batch=64,
        projector_points=[dict(gamma=g, h=h, K=K) for g, h, K in PROJECTOR_POINTS],
        projector_ids=[10000, 10001], projector_seeds=4, projector_cycles=12,
        projector_walkers=[512, 2048, 8192], projector_forward_Ktau=[2., 8., 16.],
        projector_equilibration_Ktau=[8., 16.], projector_energy_Ktau=4.,
        projector_branch_Ktau=[.1, .05],
        projector_uncertainty='SE across independent populations; every cycle retained',
        training_support='K=1; discrete anchors (Gamma,h)=(0.5,0),(0.5,0.3),(2,0.3). '
                         'Other contexts are transfer evaluations, not certified interpolation.',
        source_sha256={p.name: digest(p) for p in sorted(source.glob('*.py'))})
    path = args.output/'manifest.json'
    with (args.output/'manifest.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError('campaign identity changed; use a new campaign directory')
        if not path.exists():
            write_json(path, manifest)
    return manifest


def status(args, stage, rows, total, complete=False):
    write_json(args.output/f'{stage}_summary.json', dict(
        checkpoint_sha256=SELECTED_SHA256, rows=rows, expected_rows=total,
        completed=complete, updated_unix=time.time()))


def disorder_record(args, L, sample_id, K):
    sample = make_af_sample(L, sample_id, scale=K)
    path = args.output/'disorder'/f'L{L}_s{sample_id}_K{K:g}.npz'
    path.parent.mkdir(exist_ok=True)
    base = make_af_sample(L, sample_id)
    np.testing.assert_array_equal(sample.bonds, base.bonds)
    np.testing.assert_allclose(sample.J, K*base.J, rtol=0, atol=0)
    if not path.exists():
        np.savez_compressed(path, K=-sample.J, J=sample.J, bonds=sample.bonds,
                            points=sample.points, seed_entropy=sample.seed_entropy,
                            relative_K=-base.J, L=L, scale=K)
    return sample


@torch.no_grad()
def exact_scan(args):
    model = model_from(args.checkpoint, args.device)
    states = basis(12)
    spins = torch.tensor(states, device=args.device, dtype=torch.long)
    directory = args.output/'exact'
    directory.mkdir(exist_ok=True)
    rows = []
    # Anchor first so a usable GFMC guide cache is available immediately.
    points = list(dict.fromkeys(PROJECTOR_POINTS + list(itertools.product(GAMMAS, FIELDS, SCALES))))
    total = len(points)*8
    for gamma, h, K in points:
        for sample_id in range(10000, 10008):
            name = key(2, sample_id, gamma, h, K)
            path = directory/f'{name}.json'
            if path.exists():
                row = json.loads(path.read_text())
            else:
                start = time.monotonic()
                sample = disorder_record(args, 2, sample_id, K)
                ham = sample.hamiltonian(h)
                matrix = operator_matrix(ham, gamma)
                energy, exact, sectors = reference(matrix, h)
                model.set_hamiltonian(ham, gamma)
                logs = model.log_amplitudes(spins).cpu().numpy()
                psi = np.exp(logs)
                normalization = float(psi@psi)
                if abs(normalization-1) > 1e-10:
                    raise RuntimeError(f'normalization failed: {normalization}')
                measured = measure(psi, matrix, (energy, exact), states)
                actual = correlations(psi**2, states, sample.points, 2)
                target = correlations(exact**2, states, sample.points, 2)
                offdiag = ~np.eye(12, dtype=bool)
                error = actual['connected']-target['connected']
                residual = float(np.linalg.norm(matrix@exact-energy*exact))
                row = dict(L=2, sample_id=sample_id, gamma=gamma, h=h, K=K,
                           gamma_over_K=gamma/K, h_over_K=h/K,
                           anchor_context=K == 1 and (gamma, h) in ((.5, 0.), (.5, .3), (2., .3)),
                           **measured, exact_energy=energy, exact_chi=target['chi_connected'],
                           exact_q2=target['q2'], normalization=normalization,
                           connected_correlation_rms_error=float(np.sqrt(np.mean(error[offdiag]**2))),
                           connected_correlation=actual['connected'],
                           exact_connected_correlation=target['connected'],
                           magnetization=actual['magnetization'],
                           exact_magnetization=target['magnetization'],
                           sg_kmin=actual['sg_kmin'], exact_sg_kmin=target['sg_kmin'],
                           eigen_residual=residual, sectors=sectors,
                           seconds=time.monotonic()-start)
                # Atomic cache precedes its JSON completion marker.
                temporary = directory/f'{name}.tmp.npz'
                np.savez_compressed(temporary, logs=logs, exact=exact)
                temporary.replace(directory/f'{name}.npz')
                write_json(path, row)
                print('EXACT', name, 'F', measured['fidelity'], 'seconds', row['seconds'], flush=True)
            rows.append(row)
            status(args, 'exact', rows, total)
    status(args, 'exact', rows, total, True)


@torch.no_grad()
def sampled_scan(args):
    model = model_from(args.checkpoint, args.device)
    directory = args.output/'sampled'
    directory.mkdir(exist_ok=True)
    rows = []
    total = 3*2*len(GAMMAS)*len(FIELDS)*len(SCALES)
    # Interleave sizes to get early measurements at every size.
    for gamma, h, K, sample_id, L in itertools.product(GAMMAS, FIELDS, SCALES, [10000, 10001], [4, 6, 8]):
        path = directory/f'{key(L, sample_id, gamma, h, K)}.json'
        if path.exists():
            row = json.loads(path.read_text())
        else:
            start = time.monotonic()
            sample = disorder_record(args, L, sample_id, K)
            ham = sample.hamiltonian(h)
            model.set_hamiltonian(ham, gamma)
            seed = 920000+sample_id*10+L
            result = evaluate(model, ham, gamma, batches=8, batch=64, seed=seed)
            generator = torch.Generator(device=model.device).manual_seed(seed+1)
            c = []
            for _ in range(8):
                s = model.sample(256, generator).cpu().numpy()
                c.append(correlations(np.ones(len(s)), s, sample.points, L))
            # Matrices are descriptive sample estimates; chi above uses unbiased
            # independent four-replica tags, not a squared sample covariance.
            row = dict(L=L, sample_id=sample_id, gamma=gamma, h=h, K=K,
                       gamma_over_K=gamma/K, h_over_K=h/K, **result,
                       site_magnetization=np.mean([v['site_magnetization'] for v in c], axis=0),
                       zz=np.mean([v['zz'] for v in c], axis=0),
                       connected_sample_estimate=np.mean([v['connected'] for v in c], axis=0),
                       seconds=time.monotonic()-start,
                       interpretation='Variational observables only; larger-size ground-state error unknown')
            write_json(path, row)
            print('SAMPLED', path.stem, 'E/N', result['energy']['mean']/ham.n_sites,
                  'seconds', row['seconds'], flush=True)
        rows.append({k: v for k, v in row.items() if k not in ('zz', 'connected_sample_estimate')})
        status(args, 'sampled', rows, total)
    status(args, 'sampled', rows, total, True)


@torch.no_grad()
def benchmark(args):
    model = model_from(args.checkpoint, args.device)
    guide = TransformerGuide(model)
    rows = []
    for L, h in itertools.product([2, 4, 6, 8], [0., .3]):
        ham = make_af_sample(L, 10000).hamiltonian(h)
        model.set_hamiltonian(ham, .5)
        for count in (8, 32, 128):
            start = time.monotonic()
            samples = guide.sample(count, 771+count)
            synchronize(args.device)
            sample_seconds = time.monotonic()-start
            guide.all_flip_log_ratios(samples[:2])
            synchronize(args.device)
            times = []
            for _ in range(2):
                start = time.monotonic()
                ratios = guide.all_flip_log_ratios(samples)
                synchronize(args.device)
                times.append(time.monotonic()-start)
            if not np.isfinite(ratios).all():
                raise FloatingPointError('nonfinite guide ratios')
            row = dict(L=L, h=h, gamma=.5, K=1., states=count,
                       sample_seconds=sample_seconds, flip_ratio_seconds=float(np.median(times)),
                       mean_escape_rate=float(np.mean(.5*np.exp(ratios).sum(1))),
                       max_abs_flip_log_ratio=float(np.abs(ratios).max()))
            if L == 2:
                logs = guide.log_amplitudes(basis(12))
                lookup = EnumeratedGuide(logs, 12)
                np.testing.assert_allclose(ratios, lookup.all_flip_log_ratios(samples), atol=1e-10, rtol=1e-10)
                row['direct_lookup_max_error'] = float(np.abs(ratios-lookup.all_flip_log_ratios(samples)).max())
            rows.append(row)
            status(args, 'benchmark', rows, 24)
            print('COST', row, flush=True)
        # Actual continuous-time propagation including population resampling.
        start = time.monotonic()
        states = guide.sample(64, 887).reshape(16, 4, ham.n_sites)
        _, local = rates_and_local(ham, .5, guide, states)
        result = propagate(ham, .5, guide, states, float(local.mean()), .02, np.random.default_rng(778))
        write_json(args.output/f'direct_step_L{L}_h{h:g}.json', dict(
            L=L, h=h, gamma=.5, K=1., walkers=16, replicas=4, tau=.02,
            seconds=time.monotonic()-start, energy=result[3], energy_growth=result[4],
            weight_ess=result[5], interpretation='Throughput smoke test; no equilibration or accuracy claim'))
    status(args, 'benchmark', rows, 24, True)


def population_summary(records, target):
    result = {}
    for name in ('energy', 'energy_growth', 'q2', 'chi_connected'):
        values = np.array([r[name] for r in records])
        exact = target['energy' if name == 'energy_growth' else name]
        result[name] = dict(mean=float(values.mean()), se=float(values.std(ddof=1)/np.sqrt(len(values))),
                            exact=exact, bias=float(values.mean()-exact), independent_values=values)
    return result


def gfmc_scan(args):
    directory = args.output/'gfmc'
    directory.mkdir(exist_ok=True)
    states = basis(12)
    rows = []
    # Primary grid plus separate equilibration and cadence checks at largest W/time.
    settings = list(itertools.product([512, 2048, 8192], [2., 8., 16.], [.1], [16.]))
    settings += [(8192, 16., .05, 16.), (8192, 16., .1, 8.)]
    total = len(PROJECTOR_POINTS)*2*len(settings)
    for gamma, h, K in PROJECTOR_POINTS:
        for sample_id in (10000, 10001):
            name = key(2, sample_id, gamma, h, K)
            cache = args.output/'exact'/f'{name}.npz'
            if not cache.exists():
                raise RuntimeError(f'run exact scan first: missing {cache}')
            with np.load(cache) as data:
                guide = EnumeratedGuide(data['logs'], 12)
                target = correlations(data['exact']**2, states)
            target['energy'] = json.loads(cache.with_suffix('.json').read_text())['exact_energy']
            ham = make_af_sample(2, sample_id, scale=K).hamiltonian(h)
            for walkers, forward, block, equil in settings:
                records = []
                for seed in range(4):
                    record_path = directory/f'{name}_W{walkers}_f{forward:g}_b{block:g}_e{equil:g}_seed{seed}.json'
                    if record_path.exists():
                        record = json.loads(record_path.read_text())
                    else:
                        start = time.monotonic()
                        record = project(ham, gamma, guide, n_walkers=walkers, n_replicas=4,
                                         tau_block=block/K, n_equilibrate=round(equil/block),
                                         n_measure=round(4/block), tau_forward=forward/K,
                                         n_cycles=12, seed=98200+seed)
                        record.update(seconds=time.monotonic()-start, checkpoint_sha256=SELECTED_SHA256)
                        write_json(record_path, record)
                    records.append(record)
                row = dict(L=2, sample_id=sample_id, gamma=gamma, h=h, K=K, walkers=walkers,
                           forward_Ktau=forward, branch_Ktau=block, equilibration_Ktau=equil,
                           observables=population_summary(records, target), independent_populations=4,
                           min_descendant_ess=min(min(r['descendant_ess']) for r in records),
                           min_unique_ancestors=min(min(r['unique_ancestors']) for r in records),
                           max_largest_lineage=max(max(r['largest_lineage']) for r in records),
                           healthy_populations=sum(r['ancestry_healthy'] for r in records),
                           seconds=sum(r['seconds'] for r in records))
                rows.append(row)
                status(args, 'gfmc', rows, total)
                print('GFMC', name, walkers, forward, block, equil,
                      row['observables']['chi_connected'], flush=True)
    status(args, 'gfmc', rows, total, True)


@torch.no_grad()
def projection_reference(args):
    rows = []
    times = [0., .5, 1., 2., 4., 8., 16., 32.]
    states = basis(12)
    for gamma, h, K in PROJECTOR_POINTS:
        for sample_id in (10000, 10001):
            name = key(2, sample_id, gamma, h, K)
            with np.load(args.output/'exact'/f'{name}.npz') as cache:
                psi, exact = np.exp(cache['logs']), cache['exact']
            ham = make_af_sample(2, sample_id, scale=K).hamiltonian(h)
            matrix = operator_matrix(ham, gamma)
            energy = float(exact@(matrix@exact))
            shifted = matrix-energy*eye(len(psi), format='csr')
            trace = float(shifted.diagonal().sum())
            initial = psi.copy()
            exact_c = connected_matrix(exact**2, states)
            trajectory = []
            previous = 0.
            for Ktau in times:
                tau = Ktau/K
                if tau > previous:
                    dt = tau-previous
                    psi = expm_multiply(-dt*shifted, psi, traceA=-dt*trace)
                    psi /= np.linalg.norm(psi)
                measured = measure(psi, matrix, (energy, exact), states)
                # GFMC mixed energy: <guide|H exp(-tau H)|guide>/<guide|exp(-tau H)|guide>.
                measured['mixed_energy'] = float(initial@(matrix@psi)/(initial@psi))
                error = connected_matrix(psi**2, states)-exact_c
                measured.update(Ktau=Ktau, tau=tau,
                                connected_correlation_rms_error=float(np.sqrt(np.mean(error[~np.eye(12, dtype=bool)]**2))))
                if trajectory and (measured['fidelity'] < trajectory[-1]['fidelity']-1e-10 or
                                   measured['energy'] > trajectory[-1]['energy']+1e-10):
                    raise RuntimeError('exact projection monotonicity failed')
                trajectory.append(measured)
                previous = tau
            rows.append(dict(L=2, sample_id=sample_id, gamma=gamma, h=h, K=K, trajectory=trajectory))
            status(args, 'projection', rows, 14)
            print('PROJECTION', name, trajectory[-1]['fidelity'], flush=True)
    status(args, 'projection', rows, 14, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['exact', 'sampled', 'benchmark', 'gfmc', 'projection'], required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with stage_lock(args.output, args.stage):
        initialize(args)
        {'exact': exact_scan, 'sampled': sampled_scan, 'benchmark': benchmark,
         'gfmc': gfmc_scan, 'projection': projection_reference}[args.stage](args)


if __name__ == '__main__':
    main()
