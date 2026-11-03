"""Freeze complete ensembles, replay higher-field histograms, and report on AutoDL."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import shutil

from .preliminary_overlap_notebook import digest, dump
from .production_protocol import identity
from .projected_overlap_histograms import configure, run_one, validate_histogram


def freeze(production, reference, folder, fields):
    folder.mkdir(parents=True, exist_ok=False)
    records = [json.loads(p.read_text()) for p in
               sorted((production/'results/projector/runs').glob('*.json'))]
    specs = []; coverage = []; inputs = {}

    def copy(source, relative):
        dest = folder/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        inputs[str(relative)] = digest(dest)
        return dest

    for source in Path(__file__).parent.glob('*.py'):
        copy(source, Path('sources/transformer_qmc')/source.name)
    for name in ['manifest.json', 'parallel_status.json']:
        copy(production/'results'/name, Path('production')/name)
    for source in (production/'results').glob('failure*.json'):
        copy(source, Path('production')/source.name)
    reference_rows = json.loads((reference/'projected_distributions.json').read_text())['rows']
    reused = []
    for gamma in fields:
        for kind in ['jastrow', 'rbm_2']:
            group = [r for r in records if (r['task']['L'], r['task']['gamma'],
                     r['task']['kind'], r['task']['audit']) == (4, gamma, kind, 'main')]
            complete = len(group) == 16 and all(
                sorted(r['task']['repeat'] for r in group if r['task']['sample_id'] == sid)
                == [0, 1] for sid in range(60000, 60008))
            coverage.append(dict(L=4, gamma=gamma, kind=kind, runs=len(group), complete=complete))
            if not complete:
                continue
            for r in group:
                t = r['task']; key = identity(t)[:24]
                original = copy(production/'results/projector/runs'/(key+'.json'),
                                Path('production/runs')/(key+'.json'))
                raw = copy(production/'results/projector/runs'/(key+'.npz'),
                           Path('production/runs')/(key+'.npz'))
                assert digest(raw) == r['npz_sha256']
                cp = production/'results/projector/checkpoints'/(key+'.npz')
                if cp.exists():
                    copy(cp, Path('production/checkpoints')/cp.name)
                old = next((x for x in reference_rows if x['task'] == t), None)
                if old is not None:
                    assert old['replay_verified']
                    raw = copy(reference/old['raw_file'], Path(old['raw_file']))
                    assert digest(raw) == old['raw_sha256']
                    assert old['provenance']['model_sha256'] == r['provenance']['model_sha256']
                    reused.append(old)
                    continue
                name = 'L{L}_s{sample_id}_g{gamma:g}_{kind}_anneal_seed{training_seed}'.format(**t).replace('.', 'p')
                model = copy(production/'results/training'/(name+'.pt'), Path('models')/(name+'.pt'))
                assert digest(model) == r['provenance']['model_sha256']
                specs.append(dict(task=t, model=str(model), original=str(original)))
    dump(folder/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        production=str(production), reference=str(reference), h=.15, fields=fields,
        coverage=coverage, specs=specs, input_sha256=inputs,
        estimator='Six cross-population descendant-weighted pairs; equal cycles; two repeats per disorder; eight equal-weight disorders.',
        replay_tolerance=1e-12, ground_state_qualified=False))
    dump(folder/'projected_distributions.json', dict(rows=reused))
    print('Frozen:', folder, 'reused:', len(reused), 'to replay:', len(specs), flush=True)


def measure(folder, workers):
    validate_histogram()
    protocol = json.loads((folder/'protocol.json').read_text())
    for name, expected in protocol['input_sha256'].items():
        assert digest(folder/name) == expected, name
    rows = json.loads((folder/'projected_distributions.json').read_text())['rows']
    bykey = {identity(r['task']): r for r in rows}
    failures = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=configure) as pool:
        # Deliver inexpensive complete Jastrow ensembles first, then resume RBM
        # checkpoints before starting further RBM fields.
        pending = [s for s in protocol['specs'] if identity(s['task']) not in bykey]
        ordered = sorted(pending, key=lambda s: (
            s['task']['kind'] != 'jastrow', s['task']['gamma'],
            s['task']['sample_id'], s['task']['repeat']))
        futures = {pool.submit(run_one, s, str(folder/'projected_histograms')): s
                   for s in ordered}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                record = future.result()
            except Exception as error:
                failures.append(dict(spec=spec, error=repr(error)))
                dump(folder/'failures.json', dict(rows=failures))
                raise
            bykey[identity(record['task'])] = record
            dump(folder/'projected_distributions.json', dict(rows=list(bykey.values())))
            print('Completed', len(bykey), spec['task'], 'seconds', record['seconds'], flush=True)
    dump(folder/'complete.json', dict(projected=len(bykey), failures=failures))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--fields', type=float, nargs='+', default=[.6, .65, .7, .85, 1.])
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if not (a.folder/'protocol.json').exists():
        freeze(a.production, a.reference, a.folder, a.fields)
    measure(a.folder, a.workers)


if __name__ == '__main__':
    main()
