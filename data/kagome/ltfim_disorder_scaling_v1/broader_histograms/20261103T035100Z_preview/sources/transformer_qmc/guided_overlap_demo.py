"""Bounded L2 guided-GFMC histogram demonstration with finite-time references."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply

from kagome_qea.exact import operator_matrix
from .assess_overlap_onset import overlap_distribution
from .model import EnumeratedGuide
from .physics import make_af_sample
from .projector import project, tags


PARAMETERS = dict(n_walkers=512, n_replicas=4, tau_block=.1,
                  n_equilibrate=40, n_measure=10, tau_forward=2., n_cycles=8)


def histogram_tags(states):
    """One-hot overlap tags; inherited ancestor weights produce probability masses."""
    result = tags(states)
    n = states.shape[-1]
    agreements = (states[:, 0] == states[:, 1]).sum(axis=1)
    result['pq'] = np.eye(n + 1)[agreements]
    return result


def finite_time_distributions(ham, gamma, energy, psi, parameters):
    """Infinite-population forward estimator: p_tag ∝ phi(t_tag) phi(t_FW)."""
    shifted = operator_matrix(ham, gamma) - energy * eye(2**ham.n_sites, format='csr')
    trace = float(shifted.diagonal().sum())
    forward = parameters['tau_forward']
    start = parameters['tau_block'] * (parameters['n_equilibrate'] + parameters['n_measure'])

    def evolve(t, vector):
        return expm_multiply(-t * shifted, vector, traceA=-t * trace)

    left = evolve(start, psi)
    right = evolve(forward, psi)
    distributions = []
    for cycle in range(parameters['n_cycles']):
        p = left * right
        if p.min() < -1e-12:
            raise ValueError('Negative finite-time probability')
        distributions.append(overlap_distribution(np.maximum(p, 0))[1])
        if cycle + 1 < parameters['n_cycles']:
            left = evolve(forward, left)
    return np.asarray(distributions)


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-runs', type=int, default=16)
    parser.add_argument('--full-grid', action='store_true', help='Separate 49-context demonstration')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data = root / 'data/kagome/transformer_af_production_v1'
    out = root / ('data/kagome/guided_overlap_grid_v1' if args.full_grid
                  else 'data/kagome/guided_overlap_demo_v1')
    out.mkdir(exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name('projector.py'),
                    Path(__file__).with_name('model.py'), Path(__file__).with_name('physics.py'),
                    Path(__file__).with_name('assess_overlap_onset.py'),
                    root / 'code/qmc_numerics.py', root / 'code/kagome_qea/exact.py']
    gammas = [.25, .5, .75, 1., 1.5, 2., 3.] if args.full_grid else [.5, 2.]
    fields = [0., .05, .1, .15, .2, .25, .3] if args.full_grid else [0., .05]
    manifest = dict(sample_id=50000, L=2, contexts=[[g, h] for g in gammas for h in fields],
                    parameters=PARAMETERS, repeats=4, ground_state_qualified=False,
                    purpose='One-disorder histogram demonstration; not a convergence qualification',
                    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in source_paths})
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Changed source/settings; use a new version'
    else:
        write(manifest_path, manifest)
        for p in source_paths:
            dest = out / 'source' / p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    count = 0
    for context, (gamma, h) in enumerate(manifest['contexts']):
        stem = f'L2_s50000_g{gamma:g}_h{h:g}_K1'.replace('.', 'p')
        exact_path = data / 'qualification/exact' / (stem + '.json')
        guide_path = data / 'qualification/direct_checks' / (stem + '.json')
        meta = json.loads(exact_path.read_text())
        guide_meta = json.loads(guide_path.read_text())
        inputs = {}
        for p, m in [(exact_path, meta), (guide_path, guide_meta)]:
            assert m['status'] == 'completed'
            assert hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest() == m['npz_sha256']
            for artifact in [p, p.with_suffix('.npz')]:
                inputs[str(artifact.relative_to(root))] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert meta['signature'] == guide_meta['signature']
        sample = make_af_sample(2, 50000)
        ham = sample.hamiltonian(h)
        with np.load(exact_path.with_suffix('.npz')) as a:
            np.testing.assert_array_equal(a['bonds'], sample.bonds)
            np.testing.assert_array_equal(a['J'], sample.J)
            q, exact = overlap_distribution(a['amplitude']**2)
        with np.load(guide_path.with_suffix('.npz')) as a:
            logs = a['logs64'].copy()
        guide = EnumeratedGuide(logs, 12)
        psi = np.exp(logs - logs.max()); psi /= np.linalg.norm(psi)
        ref_path = out / (stem + '_reference.json')
        if not ref_path.exists():
            finite = finite_time_distributions(ham, gamma, meta['energy'], psi, PARAMETERS)
            write(ref_path, dict(gamma=gamma, h=h, sample_id=50000, q=q.tolist(),
                exact=exact.tolist(), guide=overlap_distribution(psi**2)[1].tolist(),
                finite_time_cycles=finite.tolist(), finite_time=finite.mean(0).tolist(),
                input_sha256=inputs, checkpoint_sha256=guide_meta['checkpoint_sha256']))
        else:
            assert json.loads(ref_path.read_text())['input_sha256'] == inputs
        for repeat in range(4):
            path = out / (stem + f'_repeat{repeat}.json')
            if path.exists():
                continue
            if count >= args.max_runs:
                return
            seed = int(np.random.SeedSequence([20261028, 917, context, repeat]).generate_state(1)[0])
            start = time.monotonic()
            result = project(ham, gamma, guide, **PARAMETERS, seed=seed, tagger=histogram_tags)
            hist = np.asarray(result['pq_cycles'])
            np.testing.assert_allclose(hist.sum(axis=1), 1, atol=1e-12)
            for power, name in [(1, 'q'), (2, 'q2'), (4, 'q4')]:
                np.testing.assert_allclose(hist @ q**power, result[name + '_cycles'], atol=1e-12)
            serial = {k: np.asarray(v).tolist() for k, v in result.items()}
            elapsed = time.monotonic() - start
            write(path, dict(gamma=gamma, h=h, sample_id=50000, repeat=repeat, seed=seed,
                parameters=PARAMETERS, seconds=elapsed, result=serial, input_sha256=inputs,
                ancestry_alarm=bool(min(result['descendant_ess']) < 8 or max(result['largest_lineage']) > .5)))
            count += 1
            print(json.dumps(dict(context=[gamma, h], repeat=repeat, seconds=elapsed,
                                  minimum_ess=min(result['descendant_ess']))), flush=True)


if __name__ == '__main__':
    main()
