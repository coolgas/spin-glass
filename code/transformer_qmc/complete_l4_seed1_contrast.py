"""Complete four missing L4 seed1 projections with frozen production code.

Run on AutoDL only. One single-thread worker leaves the existing campaigns'
worker budgets intact. No guide training or automatic retries are performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    sys.path.insert(0, str(args.production / 'code'))
    from transformer_qmc.projected_overlap_histograms import configure, run_one
    from transformer_qmc.production_protocol import identity
    configure()
    frozen = args.production / 'code/transformer_qmc/projected_overlap_histograms.py'
    assert sha(frozen) == 'c443b3444895740cee975b563dddbee439a2d51a0bdc92675785d1a16de106e3'
    specs = []
    source_hashes = {}
    seed0_records = ['3921146fbe6c832880b0875f', '4dedf474c02be1ad8fd38418',
                     '694b96715c9bbc02afde361d', '25f5bb04de74628f9a9e81f4']
    for key in seed0_records:
        path = args.production / 'results/projector/runs' / f'{key}.json'
        original = json.loads(path.read_text())
        assert sha(path.with_suffix('.npz')) == original['raw_sha256']
        source_hashes[str(path)] = sha(path)
        source_hashes[str(path.with_suffix('.npz'))] = original['raw_sha256']
        task = dict(original['task'], training_seed=1)
        model = args.production / 'results/training' / f"L4_s{task['sample_id']}_g0p6_rbm_2_anneal_seed1.pt"
        training_path = model.with_suffix('.json')
        training = json.loads(training_path.read_text())
        assert training['status'] == 'completed'
        assert training['signature']['learning_rate'] == .01
        assert sha(model) == training['checkpoint_sha256']
        source_hashes[str(model)] = sha(model)
        source_hashes[str(training_path)] = sha(training_path)
        specs.append(dict(task=task, model=str(model)))
    manifest = dict(scope='L4 Gamma .60 seed1, disorders60000/60001, repeats0/1',
                    production=str(args.production), tasks=specs, sources=source_hashes,
                    runner_sha256=sha(Path(__file__)), projector_sha256=sha(frozen),
                    workers=1, threads_per_worker=1, no_deadline=True, automatic_retries=False,
                    original_seed0_artifacts_preserved=True)
    mp = root / 'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest
    else:
        write(mp, manifest)
    completed = []
    for spec in specs:
        key = identity(spec['task'])[:24]
        write(root / 'status.json', dict(state='running', pid=os.getpid(), updated_utc=now(),
                                        completed=completed, active=spec['task'], planned=4))
        print(json.dumps(dict(event='start', time=now(), task=spec['task'])), flush=True)
        try:
            record = run_one(spec, root / 'projector')
            assert record['provenance']['model_sha256'] == source_hashes[spec['model']]
            assert sha(root / 'projector/runs' / f'{key}.npz') == record['raw_sha256']
        except Exception:
            write(root / 'failure.json', dict(task=spec['task'], traceback=traceback.format_exc(), utc=now()))
            write(root / 'status.json', dict(state='failed', completed=completed, active=spec['task'], utc=now()))
            raise
        completed.append(dict(task=spec['task'], path=str(root / 'projector/runs' / f'{key}.json'),
                              raw_sha256=record['raw_sha256'], q2=record['q2'],
                              ancestry_alarm=record['ancestry_alarm'], seconds=record['seconds']))
        print(json.dumps(dict(event='complete', time=now(), result=completed[-1])), flush=True)
    for path, expected in source_hashes.items():
        assert sha(Path(path)) == expected, path
    write(root / 'status.json', dict(state='completed', pid=os.getpid(), updated_utc=now(),
                                    completed=completed, planned=4, source_hashes_preserved=True))
    print(json.dumps(dict(event='all_completed', time=now(), completed=len(completed))), flush=True)


if __name__ == '__main__':
    main()
