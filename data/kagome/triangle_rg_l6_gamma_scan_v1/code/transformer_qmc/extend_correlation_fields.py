"""Fill the four missing fields of the archived eight-disorder GFMC benchmark.

Run: PYTHONPATH=code python -m transformer_qmc.extend_correlation_fields --workers 3
The original 96 runs and their manifest remain the parent of this extension.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .correlation_lengths import BASE, FIELD, IDS, OUT, PARAMETERS, ROOT, sha
from .model import EnumeratedGuide
from .physics import make_af_sample
from .production_protocol import geometry, production_tags
from .projector import project


FIELDS = [.25, .75, 1.5, 3.]
DESTINATION = OUT / 'l2_gfmc_extension'


def atomic_json(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stem(gamma, sid):
    return f'L2_s{sid}_g{gamma:g}_h{FIELD:g}_K1'.replace('.', 'p')


def run_path(gamma, sid, repeat):
    return DESTINATION / (stem(gamma, sid) + f'_repeat{repeat}.json')


def seed_for(gamma, sid, repeat):
    return int(np.random.SeedSequence([20261102, 812, sid,
               int(100 * gamma), repeat]).generate_state(1)[0])


def expected_manifest():
    parent_path = OUT / 'l2_gfmc/manifest.json'
    parent = json.loads(parent_path.read_text())
    assert parent['parameters'] == PARAMETERS
    assert parent['sample_ids'] == IDS and parent['repeats'] == 4
    assert parent['L'] == 2 and parent['h'] == FIELD
    for relative, digest in parent['source_sha256'].items():
        assert sha(ROOT / relative) == digest, f'Changed parent source: {relative}'
    sources = {**parent['source_sha256'],
               str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    return dict(L=2, gamma=FIELDS, h=FIELD, sample_ids=IDS, repeats=4,
                parameters=PARAMETERS, ground_state_qualified=False,
                parent_manifest_sha256=sha(parent_path), source_sha256=sources,
                seed_rule=[20261102, 812, 'sample_id', 'int(100*gamma)', 'repeat'])


def load_inputs(gamma, sid):
    guide_path = BASE / 'qualification/direct_checks' / (stem(gamma, sid) + '.json')
    exact_path = BASE / 'qualification/exact' / (stem(gamma, sid) + '.json')
    gm, em = [json.loads(p.read_text()) for p in (guide_path, exact_path)]
    assert gm['status'] == em['status'] == 'completed'
    assert gm['signature'] == em['signature'] and em['residual'] <= 1e-9
    inputs = {}
    for path, meta in ((guide_path, gm), (exact_path, em)):
        assert (meta['L'], meta['sample_id'], meta['gamma'], meta['h']) == (2, sid, gamma, FIELD)
        assert sha(path.with_suffix('.npz')) == meta['npz_sha256']
        for source in (path, path.with_suffix('.npz')):
            inputs[str(source.relative_to(ROOT))] = sha(source)
    sample = make_af_sample(2, sid)
    with np.load(exact_path.with_suffix('.npz')) as archive:
        np.testing.assert_array_equal(archive['bonds'], sample.bonds)
        np.testing.assert_array_equal(archive['J'], sample.J)
    with np.load(guide_path.with_suffix('.npz')) as archive:
        guide = EnumeratedGuide(archive['logs64'].copy(), 12)
    return sample, guide, inputs, gm['checkpoint_sha256']


def validate_record(record, gamma, sid, repeat, inputs, checkpoint):
    assert (record['L'], record['gamma'], record['h'], record['sample_id'],
            record['repeat']) == (2, gamma, FIELD, sid, repeat)
    assert record['parameters'] == PARAMETERS and record['input_sha256'] == inputs
    assert record['checkpoint_sha256'] == checkpoint
    assert record['seed'] == seed_for(gamma, sid, repeat)
    assert record['ground_state_qualified'] is False
    assert record['extension_manifest_sha256'] == sha(DESTINATION / 'manifest.json')
    result = record['result']
    spectrum = np.asarray(result['connected_spectrum'])
    assert spectrum.shape == (4,) and np.isfinite(spectrum).all()
    np.testing.assert_allclose(spectrum[0], result['chi_connected'], atol=1e-12)
    alarm = min(result['descendant_ess']) < 8 or max(result['largest_lineage']) > .5
    assert record['ancestry_alarm'] == bool(alarm)


def run_context(task):
    gamma, sid = task
    sample, guide, inputs, checkpoint = load_inputs(gamma, sid)
    lattice = geometry(2)
    records = []
    for repeat in range(4):
        path = run_path(gamma, sid, repeat)
        if path.exists():
            record = json.loads(path.read_text())
        else:
            seed = seed_for(gamma, sid, repeat)
            started = time.monotonic()
            with threadpool_limits(limits=1):
                result = project(sample.hamiltonian(FIELD), gamma, guide,
                                 **PARAMETERS, seed=seed,
                                 tagger=lambda states: production_tags(states, lattice))
            record = dict(L=2, sample_id=sid, gamma=gamma, h=FIELD, repeat=repeat,
                          seed=seed, parameters=PARAMETERS, input_sha256=inputs,
                          checkpoint_sha256=checkpoint, ground_state_qualified=False,
                          extension_manifest_sha256=sha(DESTINATION / 'manifest.json'),
                          seconds=time.monotonic() - started,
                          ancestry_alarm=bool(min(result['descendant_ess']) < 8 or
                                              max(result['largest_lineage']) > .5),
                          result={k: np.asarray(v).tolist() for k, v in result.items()})
            validate_record(record, gamma, sid, repeat, inputs, checkpoint)
            atomic_json(path, record)
        validate_record(record, gamma, sid, repeat, inputs, checkpoint)
        records.append(record)
    return dict(gamma=gamma, sample_id=sid, runs=len(records),
                alarms=sum(r['ancestry_alarm'] for r in records),
                seconds=sum(r['seconds'] for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    assert args.workers > 0
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = expected_manifest()
    manifest_path = DESTINATION / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Changed extension identity'
    else:
        atomic_json(manifest_path, manifest)
    tasks = [(gamma, sid) for gamma in FIELDS for sid in IDS]
    expected_paths = {run_path(gamma, sid, repeat)
                      for gamma, sid in tasks for repeat in range(4)}
    assert set(DESTINATION.glob('*_repeat*.json')) <= expected_paths
    total = len(expected_paths)
    def status(state, **extra):
        atomic_json(DESTINATION / 'status.json', dict(
            state=state, completed=sum(p.exists() for p in expected_paths),
            total=total, workers=args.workers, updated_unix=time.time(), **extra))
    status('running')
    completed = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_context, task) for task in tasks]
            for future in as_completed(futures):
                summary = future.result()
                completed.append(summary)
                status('running')
                print(json.dumps(summary), flush=True)
        assert sum(row['runs'] for row in completed) == total
        status('completed', ancestry_alarms=sum(row['alarms'] for row in completed))
    except Exception as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
