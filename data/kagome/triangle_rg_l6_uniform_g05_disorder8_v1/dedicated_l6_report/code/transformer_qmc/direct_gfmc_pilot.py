"""Bounded larger-size direct-network GFMC pilot; accuracy remains unqualified.

Run on a separate staged copy after the frozen scan releases the GPU. This
records independent populations and deliberately short time convergence probes.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from .experiment import write_json
from .frozen_scan import SELECTED_SHA256, model_from, stage_lock
from .model import TransformerGuide
from .physics import make_af_sample
from .projector import project


class CachedDirectGuide(TransformerGuide):
    """Exact bounded memoization of states for one fixed Hamiltonian only."""

    def __init__(self, model, max_entries=16384):
        super().__init__(model)
        self.max_entries = max_entries
        self.cache = {}
        self.requested = 0
        self.evaluated = 0

    def all_flip_log_ratios(self, states):
        states = np.asarray(states, dtype=np.int8)
        keys = [s.tobytes() for s in states]
        missing = {}
        for i, k in enumerate(keys):
            if k not in self.cache:
                missing.setdefault(k, i)
        if missing:
            indexes = list(missing.values())
            values = super().all_flip_log_ratios(states[indexes])
            self.evaluated += len(indexes)
            # Keep entries needed by this call while bounding memory between calls.
            if len(self.cache)+len(missing) > self.max_entries:
                needed = set(keys)
                self.cache = {k: v for k, v in self.cache.items() if k in needed}
            self.cache.update(zip(missing, values))
        self.requested += len(states)
        return np.stack([self.cache[k] for k in keys])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true',
                        help='Four independent populations at one L4 setting for an immediate smoke benchmark')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != SELECTED_SHA256:
        raise RuntimeError('selected guide hash mismatch')
    with stage_lock(args.output, 'direct_gfmc'):
        settings = [(32, 1., .5, .1), (64, 1., .5, .1),
                    (64, 2., 1., .1), (64, 2., 1., .05)]
        if args.preview:
            settings = settings[:1]
        manifest = dict(checkpoint_sha256=SELECTED_SHA256, L=4,
                        ids=[10000] if args.preview else [10000, 10001],
                        gamma=.5, h=[.3] if args.preview else [0., .3], K=1., settings=settings,
                        independent_populations=4, measurement_time=.5, cycles=4,
                        interpretation='Direct 48-spin network pilot; short times and small populations. '
                                       'No ground-state or finite-size scaling qualification.')
        manifest = json.loads(json.dumps(manifest))
        path = args.output/'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError('pilot identity mismatch')
        write_json(path, manifest)
        model = model_from(args.checkpoint, 'cuda')
        rows = []
        for sample_id, h in itertools.product(manifest['ids'], manifest['h']):
            ham = make_af_sample(4, sample_id).hamiltonian(h)
            model.set_hamiltonian(ham, .5)
            # Fresh cache for each Hamiltonian; no context can share cached rates.
            guide = CachedDirectGuide(model)
            for walkers, equil, forward, block in settings:
                records = []
                for seed in range(4):
                    path = args.output/f's{sample_id}_h{h:g}_W{walkers}_e{equil:g}_f{forward:g}_b{block:g}_seed{seed}.json'
                    if path.exists():
                        record = json.loads(path.read_text())
                    else:
                        start = time.monotonic()
                        requested, evaluated = guide.requested, guide.evaluated
                        record = project(ham, .5, guide, n_walkers=walkers, n_replicas=4,
                                         tau_block=block, n_equilibrate=round(equil/block),
                                         n_measure=round(.5/block), tau_forward=forward,
                                         n_cycles=4, seed=110200+seed)
                        record.update(seconds=time.monotonic()-start,
                                      requested_states=guide.requested-requested,
                                      evaluated_states=guide.evaluated-evaluated,
                                      checkpoint_sha256=SELECTED_SHA256)
                        write_json(path, record)
                        print('DIRECT_GFMC', path.stem, record['energy'], record['chi_connected'],
                              'seconds', record['seconds'], flush=True)
                    records.append(record)
                observables = {}
                for name in ('energy', 'energy_growth', 'q2', 'chi_connected'):
                    values = np.array([r[name] for r in records])
                    observables[name] = dict(mean=float(values.mean()),
                                             se=float(values.std(ddof=1)/np.sqrt(len(values))),
                                             independent_values=values)
                rows.append(dict(L=4, sample_id=sample_id, h=h, gamma=.5, K=1.,
                                 walkers=walkers, equilibration_time=equil, tau_forward=forward,
                                 tau_block=block, observables=observables,
                                 min_descendant_ess=min(min(r['descendant_ess']) for r in records),
                                 max_largest_lineage=max(max(r['largest_lineage']) for r in records),
                                 seconds=sum(r['seconds'] for r in records)))
                write_json(args.output/'summary.json', dict(manifest=manifest, rows=rows, completed=False))
        write_json(args.output/'summary.json', dict(manifest=manifest, rows=rows, completed=True))


if __name__ == '__main__':
    main()
