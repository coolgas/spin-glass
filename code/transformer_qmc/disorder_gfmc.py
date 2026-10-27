"""Frozen K=1 matched-disorder production comparison at twelve spins."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from kagome_qea.exact import operator_matrix
from .diagnose_shared import reference
from .experiment import write_json
from .frozen_scan import SELECTED_SHA256, digest, key, model_from, stage_lock
from .physics import basis, correlations, make_af_sample
from .table_projector import RateTable, project


GAMMAS = [.25, .5, .75, 1., 1.5, 2., 3.]
FIELDS = [0., .3]
IDS = list(range(50000, 50032))
SEEDS = [120100, 120101, 120102, 120103]
BASE = dict(walkers=2048, equilibration=32., measurement=8., forward=8., cycles=16, branch=.1)
SETTINGS = dict(main=BASE,
                population=dict(BASE, walkers=8192),
                forward=dict(BASE, forward=16.),
                equilibration=dict(BASE, equilibration=64.),
                cadence=dict(BASE, branch=.05))


def initialize(args):
    if digest(args.checkpoint) != SELECTED_SHA256:
        raise RuntimeError('checkpoint identity mismatch')
    source = Path(__file__).parent
    sources = list(source.glob('*.py')) + [source.parent/'qmc_numerics.py']
    manifest = dict(checkpoint_sha256=SELECTED_SHA256, checkpoint_name=args.checkpoint.name,
                    K=1., L=2, n_sites=12, gamma=GAMMAS, h=FIELDS, sample_ids=IDS,
                    audit_ids=IDS[:8], seeds=SEEDS, settings=SETTINGS,
                    source_sha256={str(p.relative_to(source.parent)):digest(p) for p in sources},
                    disorder='fresh IDs 50000:50032; Kij uniform [0.5,1.5]; matched across Gamma,h,guide',
                    guide='constant psi=1 versus frozen network sqrt(p); exact states references only',
                    q_ea='mean_i <Zi>^2 = <q>; no thermodynamic symmetry breaking at finite N',
                    susceptibility='equal-time chi_SG=N<q^2>; chi_connected=sum_ij Cij^2/N; not response',
                    uncertainty='disorder-level SE; paired disorder errors; independent population repeats',
                    timing='cached-rate propagation including table setup; excludes neural lookup preparation')
    path = args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('campaign identity changed; use a fresh output directory')
    write_json(path, manifest)
    return manifest


@torch.no_grad()
def caches(args):
    model = model_from(args.checkpoint, args.device)
    spins = basis(12)
    tensor = torch.tensor(spins, dtype=torch.long, device=args.device)
    directory = args.output/'cache'
    directory.mkdir(exist_ok=True)
    for sid, h, gamma in itertools.product(IDS, FIELDS, GAMMAS):
        name = key(2, sid, gamma, h, 1.)
        path = directory/(name+'.json')
        if path.exists():
            continue
        start = time.time()
        sample = make_af_sample(2, sid)
        ham = sample.hamiltonian(h)
        matrix = operator_matrix(ham, gamma)
        energy, psi, sectors = reference(matrix, h)
        residual = float(np.linalg.norm(matrix@psi-energy*psi))
        if residual > 1e-9:
            raise RuntimeError('exact reference residual too large')
        model.set_hamiltonian(ham, gamma)
        logs = model.log_amplitudes(tensor).cpu().numpy()
        if abs(np.exp(2*logs).sum()-1) > 1e-10:
            raise RuntimeError('network normalization failed')
        target = correlations(psi**2, spins)
        temporary = directory/(name+'.tmp.npz')
        np.savez_compressed(temporary, logs=logs, exact=psi, bonds=sample.bonds, J=sample.J)
        temporary.replace(directory/(name+'.npz'))
        write_json(path, dict(sample_id=sid, h=h, gamma=gamma, K=1., energy=energy,
                              q=target['q_mean'], q2=target['q2'],
                              chi_connected=target['chi_connected'], sectors=sectors,
                              residual=residual, cache_sha256=digest(directory/(name+'.npz')),
                              seconds=time.time()-start))
        print('CACHE', name, flush=True)


def record_path(output, task):
    sid, h, gamma, setting, kind, seed = task
    return Path(output)/'populations'/f'{key(2,sid,gamma,h,1.)}_{setting}_{kind}_{seed}.json'


def population(output, task):
    torch.set_num_threads(1)
    path = record_path(output, task)
    if path.exists():
        return str(path)
    sid, h, gamma, setting, kind, seed = task
    cache = Path(output)/'cache'/(key(2, sid, gamma, h, 1.)+'.npz')
    with np.load(cache) as data:
        logs = data['logs'].copy() if kind == 'network' else np.zeros(4096)
    started = time.perf_counter()
    table = RateTable(make_af_sample(2, sid).hamiltonian(h), gamma, logs)
    record = project(table, **SETTINGS[setting], seed=seed)
    record.update(sample_id=sid, h=h, gamma=gamma, setting=setting, guide=kind, seed=seed,
                  parameters=SETTINGS[setting], seconds=time.perf_counter()-started,
                  checkpoint_sha256=SELECTED_SHA256, cache_sha256=digest(cache))
    write_json(path, record)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--stage', choices=['cache','populations','all'], default='all')
    parser.add_argument('--hours', type=float, default=6.)
    args = parser.parse_args()
    args.output = args.output.resolve()
    torch.set_num_threads(1)
    with stage_lock(args.output, 'production'):
        manifest = initialize(args)
        started = time.time()
        def status(state, **extra):
            write_json(args.output/'status.json', dict(state=state, pid=os.getpid(),
                        started_unix=started, updated_unix=time.time(), **extra))
        try:
            if args.stage in ('all','cache'):
                status('preparing_caches')
                caches(args)
            if args.stage == 'cache':
                status('caches_completed')
                return
            tasks = [(sid,h,g,setting,kind,seed)
                     for setting in SETTINGS for h,g in itertools.product(FIELDS,GAMMAS)
                     for sid in (IDS if setting == 'main' else IDS[:8])
                     for seed in SEEDS for kind in ('constant','network')]
            (args.output/'populations').mkdir(exist_ok=True)
            pending = iter(t for t in tasks if not record_path(args.output,t).exists())
            done = sum(record_path(args.output,t).exists() for t in tasks)
            status('running', completed=done, expected=len(tasks))
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                active = {pool.submit(population,str(args.output),t) for t in
                          [next(pending,None) for _ in range(args.workers)] if t is not None}
                while active:
                    future = next(as_completed(active))
                    active.remove(future)
                    print('POPULATION',future.result(),flush=True)
                    done += 1
                    status('running',completed=done,expected=len(tasks))
                    if time.time()-started < args.hours*3600:
                        task = next(pending,None)
                        if task is not None:
                            active.add(pool.submit(population,str(args.output),task))
            status('completed' if done == len(tasks) else 'time_cap_reached',
                   completed=done,expected=len(tasks))
        except BaseException as error:
            status('failed',error=repr(error))
            raise


if __name__ == '__main__':
    main()
