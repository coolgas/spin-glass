"""Bounded H800 direct-guide feasibility checks; no ground-state qualification."""

import argparse
import hashlib
import json
import platform
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .direct_gfmc_pilot import CachedDirectGuide
from .frozen_scan import SELECTED_SHA256, model_from
from .physics import make_af_sample
from .projector import project, tags


class BudgetExpired(TimeoutError):
    pass


class TimedGuide(CachedDirectGuide):
    def __init__(self, model):
        super().__init__(model)
        self.query_calls = 0
        self.query_seconds = 0.0
        self.sample_seconds = 0.0

    def all_flip_log_ratios(self, states):
        self.query_calls += 1
        start = time.perf_counter()
        result = super().all_flip_log_ratios(states)
        self.query_seconds += time.perf_counter() - start
        return result

    def sample(self, count, seed):
        start = time.perf_counter()
        result = super().sample(count, seed)
        self.sample_seconds += time.perf_counter() - start
        return result


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, default=serializable, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=int, default=600)
    parser.add_argument('--projection-only', action='store_true',
                        help='Longer bounded propagation and larger population cost probes')
    parser.add_argument('--precision-probe', action='store_true',
                        help='Compare float32 inference speed and errors against float64 only')
    args = parser.parse_args()
    if args.projection_only and args.precision_probe:
        parser.error('choose one probe mode')
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'benchmark.json'
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261027)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != SELECTED_SHA256:
        raise ValueError('checkpoint hash mismatch')
    root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(root.rglob('*.py'))}
    record = dict(
        status='running', interpretation='Execution and cost probe only; no accuracy qualification',
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest, source_sha256=source_hashes,
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), cpu_threads=2,
        dtype='float64 versus float32' if args.precision_probe else 'float64',
        sample_id=10000, periodic=True, gamma=0.5, budget_seconds=args.budget_seconds,
        projection_only=args.projection_only,
        precision_probe=args.precision_probe,
        rows=[], started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )
    start_all = time.monotonic()
    write(destination, record)

    def expired(signum, frame):
        raise BudgetExpired('bounded feasibility time budget reached')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.budget_seconds)
    telemetry_file = (args.output / 'gpu.csv').open('w')
    telemetry = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
         '--format=csv,noheader,nounits', '-l', '1'], stdout=telemetry_file,
    )

    def save(row):
        record['rows'].append(row)
        record['elapsed_seconds'] = time.monotonic() - start_all
        write(destination, record)
        print(json.dumps(row, default=serializable), flush=True)

    def measured(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        metrics = dict(seconds=time.perf_counter() - started,
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        return result, metrics

    try:
        model, metrics = measured(lambda: model_from(args.checkpoint, 'cuda'))
        shape_identity = {k: list(v.shape) for k, v in model.state_dict().items()}
        save(dict(stage='load', parameters=sum(p.numel() for p in model.parameters()), **metrics))
        if args.precision_probe:
            torch.set_float32_matmul_precision('highest')
            low = model_from(args.checkpoint, 'cuda').float().eval()
            for h in (.3, 0.):
                ham = make_af_sample(10, 10000, periodic=True).hamiltonian(h)
                model.set_hamiltonian(ham, .5); low.set_hamiltonian(ham, .5)
                generator = torch.Generator(device='cuda').manual_seed(73100)
                samples = model.sample(128, generator)
                random = 1-2*torch.randint(2, (32, 300), device='cuda', generator=generator)
                states = torch.cat([samples, random])
                reference, high_metrics = measured(lambda: model.all_flip_log_ratios(states))
                for chunk in (512, 2048):
                    low.all_flip_log_ratios(states, chunk=chunk)
                    repetitions = []
                    for _ in range(3):
                        ratios, timing = measured(lambda: low.all_flip_log_ratios(states, chunk=chunk))
                        repetitions.append(timing)
                    assert torch.isfinite(ratios).all()
                    error = ratios.double()-reference
                    relative_rates = torch.expm1(error)
                    # Rates and local-energy arithmetic remain double precision in the projector adapter.
                    delta_energy_per_site = (-.5*(ratios.double().exp()-reference.exp()).sum(1)/300)
                    save(dict(stage='precision_probe', L=10, h=h, configurations=len(states),
                              guide_samples=128, independent_random_states=32, chunk=chunk,
                              matmul_precision=torch.get_float32_matmul_precision(),
                              float64_seconds=high_metrics['seconds'],
                              seconds=float(np.median([x['seconds'] for x in repetitions])),
                              peak_allocated_mib=max(x['peak_allocated_mib'] for x in repetitions),
                              max_abs_log_ratio_error=float(error.abs().max()),
                              rms_log_ratio_error=float(error.square().mean().sqrt()),
                              max_abs_relative_rate_error=float(relative_rates.abs().max()),
                              max_abs_local_energy_per_site_error=float(delta_energy_per_site.abs().max()),
                              rms_local_energy_per_site_error=float(delta_energy_per_site.square().mean().sqrt()),
                              repetitions=repetitions, production_qualified=False))
                guide = TimedGuide(low)
                settings = dict(n_walkers=32, n_replicas=4, tau_block=.02,
                                n_equilibrate=2, n_measure=4, tau_forward=.04, n_cycles=2, seed=74010)
                result, metrics = measured(lambda: project(ham, .5, guide, **settings))
                save(dict(stage='precision_project', L=10, h=h, dtype='float32 inference',
                          settings=settings, result=result, total_projection_time=.2,
                          query_calls=guide.query_calls, requested_states=guide.requested,
                          evaluated_states=guide.evaluated, query_seconds=guide.query_seconds,
                          sample_seconds=guide.sample_seconds, production_qualified=False, **metrics))
        for L in (() if args.projection_only or args.precision_probe else (8, 10)):
            for h in (0.3, 0.0):
                sample = make_af_sample(L, 10000, periodic=True)
                ham = sample.hamiltonian(h)
                bonds = np.asarray(ham.bonds)
                degree = np.bincount(bonds.ravel(), minlength=ham.n_sites)
                assert ham.n_sites == 3*L*L
                assert len(bonds) == 2*ham.n_sites and np.all(degree == 4)
                assert len(np.unique(np.sort(bonds, axis=1), axis=0)) == len(bonds)
                _, metrics = measured(lambda: model.set_hamiltonian(ham, .5))
                assert shape_identity == {k: list(v.shape) for k, v in model.state_dict().items()}
                aligned = np.ones((2, 4, ham.n_sites), dtype=np.int8)
                assert np.all(tags(aligned)['q2'] == 1)
                opposed = aligned.copy(); opposed[:, 1] = -1; opposed[:, 3] = -1
                assert np.all(tags(opposed)['chi_connected'] == 4*ham.n_sites)
                save(dict(stage='context', L=L, h=h, n_sites=ham.n_sites,
                          bonds=len(bonds), degree_min=int(degree.min()), degree_max=int(degree.max()),
                          parameter_shapes_unchanged=True, **metrics))
                generator = torch.Generator(device='cuda').manual_seed(70000+L)
                states, metrics = measured(lambda: model.sample(128, generator))
                assert states.shape == (128, ham.n_sites)
                assert torch.isfinite(model.log_prob(states)).all()
                save(dict(stage='sample', L=L, h=h, configurations=128, **metrics))
                for count in (32, 128):
                    values, warm = measured(lambda: model.all_flip_log_ratios(states[:count]))
                    repetitions = []
                    for _ in range(2):
                        values, metrics = measured(lambda: model.all_flip_log_ratios(states[:count]))
                        assert torch.isfinite(values).all()
                        repetitions.append(metrics)
                    reference = .5*model.log_prob(states[:2])
                    error = 0.0
                    for site in (0, ham.n_sites//2, ham.n_sites-1):
                        flip = states[:2].clone(); flip[:, site] *= -1
                        direct = .5*model.log_prob(flip)-reference
                        error = max(error, float((values[:2,site]-direct).abs().max()))
                    assert error < 1e-9, error
                    save(dict(stage='all_flip', L=L, h=h, configurations=count, chunk=2048,
                              cold_seconds=warm['seconds'], seconds=float(np.median([x['seconds'] for x in repetitions])),
                              peak_allocated_mib=max(x['peak_allocated_mib'] for x in repetitions),
                              peak_reserved_mib=max(x['peak_reserved_mib'] for x in repetitions),
                              repeats=repetitions, direct_ratio_max_error=error))
                if L == 10 and h == .3:
                    reference = values.clone()
                    values, metrics = measured(lambda: model.all_flip_log_ratios(states, chunk=512))
                    error = float((values-reference).abs().max())
                    assert error < 1e-9, error
                    save(dict(stage='chunk_probe', L=L, h=h, configurations=128, chunk=512,
                              max_difference=error, **metrics))
        contexts = ((10, .3, 32), (10, 0., 32), (10, .3, 128)) if args.projection_only else (
            (10, .3, 8), (10, 0., 8), (8, .3, 8), (10, .3, 32))
        if args.precision_probe:
            contexts = ()
        for L, h, walkers in contexts:
            ham = make_af_sample(L, 10000, periodic=True).hamiltonian(h)
            model.set_hamiltonian(ham, .5)
            guide = TimedGuide(model)
            settings = dict(n_walkers=walkers, n_replicas=4, tau_block=.005,
                            n_equilibrate=1, n_measure=2, tau_forward=.005, n_cycles=2, seed=74000+L)
            if args.projection_only and walkers == 32:
                settings.update(tau_block=.02, n_equilibrate=2, n_measure=4, tau_forward=.04)
            total_time = ((settings['n_equilibrate'] + settings['n_measure'])*settings['tau_block']
                          + settings['n_cycles']*settings['tau_forward'])
            result, metrics = measured(lambda: project(ham, .5, guide, **settings))
            save(dict(stage='project', L=L, h=h, settings=settings, result=result,
                      total_projection_time=total_time, query_calls=guide.query_calls,
                      requested_states=guide.requested, evaluated_states=guide.evaluated,
                      query_seconds=guide.query_seconds, sample_seconds=guide.sample_seconds, **metrics))
        record['status'] = 'completed'
    except BudgetExpired:
        record['status'] = 'budget_reached'
        record['error'] = traceback.format_exc()
        print(record['error'], flush=True)
    except Exception:
        record['status'] = 'failed'
        record['error'] = traceback.format_exc()
        print(record['error'], flush=True)
        raise
    finally:
        signal.alarm(0)
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_file.close()
        record['elapsed_seconds'] = time.monotonic() - start_all
        record['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write(destination, record)
        print('FINAL_STATUS', record['status'], record['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
