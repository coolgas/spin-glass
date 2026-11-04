"""Connected overlap lengths and a cached, matched L2 guided-GFMC benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .model import EnumeratedGuide
from .physics import make_af_sample
from .production_protocol import geometry, production_tags
from .projector import project


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/kagome/transformer_af_production_v1'
OUT = ROOT / 'data/kagome/ltfim_correlation_lengths_v2'
GAMMAS = [.5, 1., 2.]
FIELD = .15
IDS = list(range(50000, 50008))
PARAMETERS = dict(n_walkers=1024, n_replicas=4, tau_block=.1,
                  n_equilibrate=80, n_measure=10, tau_forward=8., n_cycles=8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def length(spectra, size):
    """Last axis contains S(0), S(k1), S(k2), S(k3); invalid ratios stay NaN."""
    spectra = np.asarray(spectra, dtype=float)
    zero, shell = spectra[..., 0], spectra[..., 1:].mean(axis=-1)
    ratio = np.divide(zero, shell, out=np.full_like(zero, np.nan), where=shell > 0) - 1
    valid = (zero > 0) & (shell > 0) & (ratio >= 0)
    return np.sqrt(np.where(valid, ratio, np.nan)) * np.sqrt(3) * size / (2 * np.pi)


def bootstrap(spectra, size, draws=3000, seed=20261102):
    """Resample disorder means; the point estimate is a ratio of mean spectra."""
    spectra = np.asarray(spectra, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(spectra), size=(draws, len(spectra)))
    values = length(spectra[indices].mean(axis=1), size)
    finite = np.isfinite(values)
    return dict(xi=float(length(spectra.mean(axis=0), size)),
                se=float(np.std(values[finite], ddof=1)) if finite.sum() > 1 else np.nan,
                invalid_fraction=float(1 - finite.mean()))


def benchmark(max_runs=None):
    """Measure missing L2 momenta with the archived neural-guide lookup table."""
    destination = OUT / 'l2_gfmc'
    destination.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
               ('model.py', 'physics.py', 'production_protocol.py', 'projector.py')],
               ROOT / 'code/qmc_numerics.py']
    manifest = dict(L=2, gamma=GAMMAS, h=FIELD, sample_ids=IDS, repeats=4,
                    parameters=PARAMETERS, ground_state_qualified=False,
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources})
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Changed benchmark identity'
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    count = 0
    lattice = geometry(2)
    for gamma in GAMMAS:
        for sid in IDS:
            stem = f'L2_s{sid}_g{gamma:g}_h{FIELD:g}_K1'.replace('.', 'p')
            guide_path = BASE / 'qualification/direct_checks' / (stem + '.json')
            exact_path = BASE / 'qualification/exact' / (stem + '.json')
            gm, em = json.loads(guide_path.read_text()), json.loads(exact_path.read_text())
            assert gm['status'] == em['status'] == 'completed'
            assert gm['signature'] == em['signature'] and em['residual'] <= 1e-9
            inputs = {}
            for path, meta in ((guide_path, gm), (exact_path, em)):
                assert sha(path.with_suffix('.npz')) == meta['npz_sha256']
                for source in (path, path.with_suffix('.npz')):
                    inputs[str(source.relative_to(ROOT))] = sha(source)
            sample = make_af_sample(2, sid)
            with np.load(exact_path.with_suffix('.npz')) as archive:
                np.testing.assert_array_equal(archive['bonds'], sample.bonds)
                np.testing.assert_array_equal(archive['J'], sample.J)
            with np.load(guide_path.with_suffix('.npz')) as archive:
                guide = EnumeratedGuide(archive['logs64'].copy(), 12)
            for repeat in range(4):
                path = destination / (stem + f'_repeat{repeat}.json')
                if path.exists():
                    previous = json.loads(path.read_text())
                    assert previous['input_sha256'] == inputs and previous['parameters'] == PARAMETERS
                    continue
                if max_runs is not None and count >= max_runs:
                    return
                seed = int(np.random.SeedSequence([20261102, 812, sid,
                           int(100 * gamma), repeat]).generate_state(1)[0])
                started = time.monotonic()
                with threadpool_limits(limits=1):
                    result = project(sample.hamiltonian(FIELD), gamma, guide,
                                     **PARAMETERS, seed=seed,
                                     tagger=lambda states: production_tags(states, lattice))
                record = dict(L=2, sample_id=sid, gamma=gamma, h=FIELD, repeat=repeat,
                              seed=seed, parameters=PARAMETERS, input_sha256=inputs,
                              checkpoint_sha256=gm['checkpoint_sha256'],
                              ground_state_qualified=False,
                              seconds=time.monotonic() - started,
                              ancestry_alarm=bool(min(result['descendant_ess']) < 8 or
                                                  max(result['largest_lineage']) > .5),
                              result={k: np.asarray(v).tolist() for k, v in result.items()})
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
                temporary.replace(path)
                count += 1
                print(json.dumps(dict(sample_id=sid, gamma=gamma, repeat=repeat,
                                      seconds=record['seconds'], ancestry_alarm=record['ancestry_alarm'])),
                      flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-runs', type=int)
    args = parser.parse_args()
    benchmark(args.max_runs)
