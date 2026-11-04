"""Longer-time L2 benchmark selected from the deterministic projection screen."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .guided_overlap_demo import PARAMETERS, finite_time_distributions, histogram_tags, write
from .model import EnumeratedGuide
from .overlap_disorder_benchmark import ROOT, load_inputs, validate_run, sha
from .projector import project


OUT = ROOT/'data/kagome/overlap_disorder_long_time_v1'
SETTINGS = {f'w{w}_time32': dict(PARAMETERS, n_walkers=w, n_equilibrate=160, tau_forward=32.)
            for w in (2048, 4096)}


def one(task):
    sid, repeat, setting = task
    stem, ham, em, gm, inputs, q, pq, exact, logs, trial = load_inputs(sid)
    parameters = SETTINGS[setting]
    folder = OUT/setting
    target = folder/f'{stem}_repeat{repeat}.json'
    if target.exists():
        record = json.loads(target.read_text())
        validate_run(record, q, parameters, inputs)
        return dict(sample=sid, repeat=repeat, setting=setting, reused=True)
    seed = int.from_bytes(hashlib.sha256(
        f'overlap-disorder-long-time-v1:{sid}:{setting}:{repeat}'.encode()).digest()[:8], 'little')
    start = time.monotonic()
    result = project(ham, .5, EnumeratedGuide(logs, 12), **parameters, seed=seed, tagger=histogram_tags)
    record = dict(sample_id=sid, gamma=.5, h=.15, repeat=repeat, seed=seed,
                  parameters=parameters, input_sha256=inputs, seconds=time.monotonic()-start,
                  result={k: np.asarray(v).tolist() for k, v in result.items()},
                  ancestry_alarm=bool(min(result['descendant_ess']) < 8 or max(result['largest_lineage']) > .5))
    validate_run(record, q, parameters, inputs)
    write(target, record)
    return dict(sample=sid, repeat=repeat, setting=setting,
                seconds=record['seconds'], alarm=record['ancestry_alarm'])


def main():
    OUT.mkdir(exist_ok=True)
    paths = [ROOT/'code/qmc_numerics.py']
    for package in ['transformer_qmc', 'kagome_qea', 'gfmc', 'ising']:
        paths.extend(sorted((ROOT/'code'/package).glob('*.py')))
    manifest = dict(settings=SETTINGS, sample_ids=list(range(50000, 50008)), repeats=4,
                    gamma=.5, h=.15, L=2, workers=4, ground_state_qualified=False,
                    selection='Time32 chosen using deterministic screen; same times and populations for every disorder',
                    comparison='Population2048 versus4096 at equilibration16/forward32; independent runs',
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})
    mp = OUT/'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest
    else:
        for p in paths:
            target = OUT/'source'/p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
        write(mp, manifest)
    for sid in manifest['sample_ids']:
        stem, ham, em, gm, inputs, q, pq, exact, logs, trial = load_inputs(sid)
        finite = None
        for setting, parameters in SETTINGS.items():
            folder = OUT/setting
            folder.mkdir(exist_ok=True)
            rp = folder/f'{stem}_reference.json'
            if rp.exists():
                r = json.loads(rp.read_text())
                assert r['input_sha256'] == inputs and r['parameters'] == parameters
                continue
            if finite is None:
                finite = finite_time_distributions(ham, .5, em['energy'], trial, parameters)
            write(rp, dict(sample_id=sid, parameters=parameters, input_sha256=inputs,
                           checkpoint_sha256=gm['checkpoint_sha256'], q=q.tolist(), exact=pq.tolist(),
                           exact_observables=exact, finite_time=finite.mean(0).tolist(),
                           finite_time_cycles=finite.tolist()))
    tasks = [(sid, r, setting) for setting in SETTINGS for sid in manifest['sample_ids'] for r in range(4)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for value in pool.map(one, tasks):
            print(json.dumps(value), flush=True)
    write(OUT/'status.json', dict(completed_runs=64, expected_runs=64, ground_state_qualified=False))


if __name__ == '__main__':
    main()
