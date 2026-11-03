"""Resumable L2 population/forward-time study using a frozen network lookup.

The lookup exactly caches the network, never exact ground-state amplitudes.
Independent populations supply uncertainty; every forward cycle is retained.
"""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from kagome_qea.exact import operator_matrix
from .diagnose_shared import measure, reference
from .experiment import load_checkpoint, write_json
from .model import EnumeratedGuide
from .physics import basis, make_af_sample
from .projector import project
from .symmetric import SymmetricTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--walkers', type=int, nargs='+', default=[512, 2048, 8192])
    parser.add_argument('--forward', type=float, nargs='+', default=[2., 8., 16.])
    parser.add_argument('--blocks', type=float, nargs='+', default=[.1, .05])
    parser.add_argument('--ids', type=int, nargs='+', default=[10000, 10001])
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--cycles', type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'study.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.seeds < 2 or args.cycles < 2:
        raise ValueError('multiple independent populations and cycles required')
    torch.set_num_threads(2)
    saved = load_checkpoint(args.checkpoint, 'cpu')
    identity = dict(checkpoint=str(args.checkpoint.resolve()),
                    sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    walkers=args.walkers, forward=args.forward, blocks=args.blocks,
                    ids=args.ids, seeds=args.seeds, cycles=args.cycles,
                    equilibration_time=16., energy_measurement_time=4.,
                    interpretation='12-spin frozen-network lookup. All cycles included; '
                    'uncertainty across independent populations, not time-series SE.')
    manifest = args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError('projector study identity mismatch')
    write_json(manifest, identity)
    model = SymmetricTransformer(make_af_sample(2).hamiltonian(0.), .5,
                                 **saved['metadata']['architecture']).double().eval()
    model.load_state_dict(saved['model'])
    states = basis(12)
    spins = torch.tensor(states, dtype=torch.long)
    rows = []
    for sample_id in args.ids:
        for h in (0., .3):
            ham = make_af_sample(2, sample_id).hamiltonian(h)
            matrix = operator_matrix(ham, .5)
            energy, exact, _ = reference(matrix, h)
            exact_metrics = measure(exact, matrix, (energy, exact), states)
            model.set_hamiltonian(ham, .5)
            with torch.no_grad():
                logs = model.log_amplitudes(spins).numpy()
            guide = EnumeratedGuide(logs, 12)
            for walkers in args.walkers:
                for forward in args.forward:
                    for block in args.blocks:
                        # Fine cadence is checked at the largest population and time.
                        if block != args.blocks[0] and (walkers != max(args.walkers) or forward != max(args.forward)):
                            continue
                        records = []
                        for seed in range(args.seeds):
                            path = args.output/f's{sample_id}_h{h:g}_W{walkers}_f{forward:g}_b{block:g}_seed{seed}.json'
                            if path.exists():
                                record = json.loads(path.read_text())
                            else:
                                start = time.monotonic()
                                result = project(ham, .5, guide, n_walkers=walkers, n_replicas=4,
                                                 tau_block=block, n_equilibrate=round(16/block),
                                                 n_measure=round(4/block), tau_forward=forward,
                                                 n_cycles=args.cycles, seed=8100+seed)
                                record = dict(result, seconds=time.monotonic()-start)
                                write_json(path, record)
                            records.append(record)
                        summary = {}
                        for observable in ('energy', 'energy_growth', 'q2', 'chi_connected'):
                            values = np.array([r[observable] for r in records])
                            mean = float(values.mean())
                            se = float(values.std(ddof=1)/np.sqrt(len(values)))
                            target = exact_metrics['energy' if observable=='energy_growth' else observable]
                            summary[observable] = dict(mean=mean, se=se, exact=target,
                                                       bias=mean-target, independent_values=values)
                        row = dict(sample_id=sample_id, gamma=.5, h=h, walkers=walkers,
                                   tau_forward=forward, tau_block=block, observables=summary,
                                   min_descendant_ess=min(min(r['descendant_ess']) for r in records),
                                   min_unique_ancestors=min(min(r['unique_ancestors']) for r in records),
                                   max_largest_lineage=max(max(r['largest_lineage']) for r in records),
                                   healthy_populations=sum(r['ancestry_healthy'] for r in records),
                                   independent_populations=len(records),
                                   seconds=sum(r['seconds'] for r in records))
                        rows.append(row)
                        write_json(args.output/'summary.json', dict(manifest=identity, rows=rows, completed=False))
                        print('PROJECTOR',sample_id,h,walkers,forward,block,
                              'chi',summary['chi_connected']['mean'], '±',summary['chi_connected']['se'],
                              'exact',summary['chi_connected']['exact'],
                              'min ancestry ESS',row['min_descendant_ess'],flush=True)
    write_json(args.output/'summary.json', dict(manifest=identity, rows=rows, completed=True))


if __name__ == '__main__':
    main()
