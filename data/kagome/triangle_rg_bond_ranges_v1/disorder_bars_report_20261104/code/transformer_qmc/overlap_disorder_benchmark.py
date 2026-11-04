"""Matched eight-disorder L2 histogram benchmark and controlled sensitivity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .guided_overlap_demo import PARAMETERS, finite_time_distributions, histogram_tags, write
from .assess_overlap_onset import overlap_distribution
from .model import EnumeratedGuide
from .physics import basis, make_af_sample
from .projector import project


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/kagome/overlap_disorder_benchmark_v1'
SETTINGS = {
    'baseline': dict(PARAMETERS),
    'population2048': dict(PARAMETERS, n_walkers=2048),
    'forward4': dict(PARAMETERS, n_walkers=2048, tau_forward=4.),
    'forward8': dict(PARAMETERS, n_walkers=2048, tau_forward=8.),
    'equilibration8': dict(PARAMETERS, n_walkers=2048, tau_forward=8., n_equilibrate=80),
}
CONTRASTS = [('baseline', 'population2048'), ('population2048', 'forward4'),
             ('forward4', 'forward8'), ('forward8', 'equilibration8')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(sid):
    stem = f'L2_s{sid}_g0p5_h0p15_K1'
    base = ROOT / 'data/kagome/transformer_af_production_v1/qualification'
    ep, gp = base/'exact'/f'{stem}.json', base/'direct_checks'/f'{stem}.json'
    em, gm = json.loads(ep.read_text()), json.loads(gp.read_text())
    assert em['status'] == gm['status'] == 'completed'
    assert em['signature'] == gm['signature']
    assert em['residual'] <= 1e-9
    for p, meta in [(ep, em), (gp, gm)]:
        assert sha(p.with_suffix('.npz')) == meta['npz_sha256']
        assert meta['sample_id'] == sid and meta['gamma'] == .5 and meta['h'] == .15
    inputs = {str(p.relative_to(ROOT)): sha(p)
              for p in (ep, gp, ep.with_suffix('.npz'), gp.with_suffix('.npz'))}
    sample = make_af_sample(2, sid)
    with np.load(ep.with_suffix('.npz')) as a:
        np.testing.assert_array_equal(a['bonds'], sample.bonds)
        np.testing.assert_array_equal(a['J'], sample.J)
        psi = a['amplitude'].copy()
    np.testing.assert_allclose(psi @ psi, 1., atol=1e-10)
    p = psi**2 / (psi @ psi)
    spins = basis(12).astype(float)
    zz = spins.T @ (p[:, None]*spins)
    m = p @ spins
    q, pq = overlap_distribution(p)
    exact = dict(q=float(pq @ q), q2=float(pq @ q**2),
                 chi_connected=float(np.sum((zz-np.outer(m, m))**2)/12))
    np.testing.assert_allclose(exact['q2'], np.sum(zz**2)/144, atol=1e-11)
    np.testing.assert_allclose(exact['q'], m @ m / 12, atol=1e-11)
    for k in exact:
        np.testing.assert_allclose(exact[k], em['observables'][k], atol=1e-10)
    with np.load(gp.with_suffix('.npz')) as a:
        logs = a['logs64'].copy()
    trial = np.exp(logs-logs.max())
    trial /= np.linalg.norm(trial)
    return stem, sample.hamiltonian(.15), em, gm, inputs, q, pq, exact, logs, trial


def validate_run(run, q, parameters, inputs):
    assert run['parameters'] == parameters and run['input_sha256'] == inputs
    result = run['result']
    hist = np.asarray(result['pq_cycles'])
    assert hist.shape == (parameters['n_cycles'], 13)
    assert np.isfinite(hist).all() and (hist >= 0).all()
    np.testing.assert_allclose(hist.sum(1), 1., atol=1e-12)
    np.testing.assert_allclose(hist.mean(0), result['pq'], atol=1e-12)
    for power, name in [(1, 'q'), (2, 'q2'), (4, 'q4')]:
        np.testing.assert_allclose(hist @ q**power, result[name+'_cycles'], atol=1e-12)
        np.testing.assert_allclose(np.mean(result[name+'_cycles']), result[name], atol=1e-12)


def initialize():
    OUT.mkdir(exist_ok=True)
    paths = [ROOT/'code/qmc_numerics.py']
    for package in ['transformer_qmc', 'kagome_qea', 'gfmc', 'ising']:
        paths.extend(sorted((ROOT/'code'/package).glob('*.py')))
    manifest = dict(L=2, gamma=.5, h=.15, sample_ids=list(range(50000, 50008)),
                    repeats=4, settings=SETTINGS, contrasts=CONTRASTS,
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                    ground_state_qualified=False, final_data=False,
                    purpose='Development benchmark; complete matched disorders and one-parameter checks')
    # JSON normalizes tuple-valued contrasts for reproducible comparisons.
    manifest = json.loads(json.dumps(manifest))
    dest = OUT/'manifest.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == manifest, 'Source/settings changed; use a new version'
    else:
        for p in paths:
            target = OUT/'source'/p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
        write(dest, manifest)
    return manifest


def run(settings):
    manifest = initialize()
    for sid in manifest['sample_ids']:
        stem, ham, em, gm, inputs, q, pq, exact, logs, trial = load_inputs(sid)
        guide = EnumeratedGuide(logs, 12)
        for setting in settings:
            parameters = SETTINGS[setting]
            folder = OUT/setting
            folder.mkdir(exist_ok=True)
            rp = folder/f'{stem}_reference.json'
            if not rp.exists():
                finite = finite_time_distributions(ham, .5, em['energy'], trial, parameters)
                write(rp, dict(sample_id=sid, parameters=parameters, input_sha256=inputs,
                               checkpoint_sha256=gm['checkpoint_sha256'], q=q.tolist(),
                               exact=pq.tolist(), exact_observables=exact,
                               finite_time=finite.mean(0).tolist(), finite_time_cycles=finite.tolist()))
            else:
                ref = json.loads(rp.read_text())
                assert ref['parameters'] == parameters and ref['input_sha256'] == inputs
            for repeat in range(4):
                target = folder/f'{stem}_repeat{repeat}.json'
                if target.exists():
                    validate_run(json.loads(target.read_text()), q, parameters, inputs)
                    continue
                old = ROOT/'data/kagome/guided_overlap_grid_v1'/target.name
                if setting == 'baseline' and sid == 50000:
                    old_manifest = json.loads((old.parent/'manifest.json').read_text())
                    for relative, digest in old_manifest['source_sha256'].items():
                        assert sha(ROOT/relative) == digest
                    record = json.loads(old.read_text())
                    assert record['sample_id'] == sid and record['repeat'] == repeat
                    record['reused_from'] = str(old.relative_to(ROOT))
                    record['reused_sha256'] = sha(old)
                else:
                    seed = int.from_bytes(hashlib.sha256(
                        f'overlap-disorder-v1:{sid}:{setting}:{repeat}'.encode()).digest()[:8], 'little')
                    start = time.monotonic()
                    result = project(ham, .5, guide, **parameters, seed=seed, tagger=histogram_tags)
                    record = dict(sample_id=sid, gamma=.5, h=.15, repeat=repeat, seed=seed,
                                  parameters=parameters, input_sha256=inputs,
                                  seconds=time.monotonic()-start,
                                  result={k: np.asarray(v).tolist() for k, v in result.items()},
                                  ancestry_alarm=bool(min(result['descendant_ess']) < 8 or
                                                      max(result['largest_lineage']) > .5))
                validate_run(record, q, parameters, inputs)
                write(target, record)
                print(json.dumps(dict(sample=sid, setting=setting, repeat=repeat,
                                      seconds=record['seconds'], alarm=record['ancestry_alarm'],
                                      reused='reused_from' in record)), flush=True)
    report()


def report():
    manifest = json.loads((OUT/'manifest.json').read_text())
    for relative, digest in manifest['source_sha256'].items():
        assert sha(OUT/'source'/relative) == digest
    rows, bins, seeds = [], [], []
    for setting, parameters in SETTINGS.items():
        for sid in manifest['sample_ids']:
            stem, _, _, _, inputs, q, pq, exact, _, _ = load_inputs(sid)
            folder = OUT/setting
            paths = [folder/f'{stem}_repeat{r}.json' for r in range(4)]
            if not all(p.exists() for p in paths):
                continue
            ref = json.loads((folder/f'{stem}_reference.json').read_text())
            assert ref['input_sha256'] == inputs and ref['parameters'] == parameters
            records = [json.loads(p.read_text()) for p in paths]
            for repeat, record in enumerate(records):
                assert record['sample_id'] == sid and record['repeat'] == repeat
                validate_run(record, q, parameters, inputs)
                seeds.append(record['seed'])
            hist = np.array([r['result']['pq'] for r in records])
            finite = np.array(ref['finite_time'])
            row = dict(setting=setting, sample_id=sid, alarms=sum(r['ancestry_alarm'] for r in records),
                       min_ess=min(min(r['result']['descendant_ess']) for r in records),
                       max_lineage=max(max(r['result']['largest_lineage']) for r in records),
                       seconds=sum(r['seconds'] for r in records),
                       reused_runs=sum('reused_from' in r for r in records),
                       finite_TV=.5*np.abs(finite-pq).sum(),
                       mc_TV=.5*np.abs(hist.mean(0)-pq).sum())
            for name in ['q', 'q2', 'chi_connected']:
                values = np.array([r['result'][name] for r in records])
                row.update({name+'_exact': exact[name], name+'_mc': values.mean(),
                            name+'_mc_se': values.std(ddof=1)/2})
            row['q_finite'], row['q2_finite'] = finite @ q, finite @ q**2
            rows.append(row)
            for k, value in enumerate(q):
                bins.append(dict(setting=setting, sample_id=sid, q=value, exact=pq[k],
                                 mc=hist[:, k].mean(), mc_se=hist[:, k].std(ddof=1)/2,
                                 finite_time=finite[k]))
    assert len(seeds) == len(set(seeds))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'by_disorder.csv', index=False)
    pd.DataFrame(bins).to_csv(OUT/'histograms.csv', index=False)
    summary = []
    for setting, group in frame.groupby('setting', sort=False):
        if len(group) != 8:
            continue
        for name in ['q', 'q2', 'chi_connected']:
            difference = group[name+'_mc']-group[name+'_exact']
            summary.append(dict(setting=setting, observable=name, disorders=8,
                                exact=group[name+'_exact'].mean(),
                                exact_se=group[name+'_exact'].std(ddof=1)/np.sqrt(8),
                                mc=group[name+'_mc'].mean(),
                                mc_se=group[name+'_mc'].std(ddof=1)/np.sqrt(8),
                                difference=difference.mean(),
                                difference_se=difference.std(ddof=1)/np.sqrt(8),
                                conditional_mc_se=np.sqrt(np.sum(group[name+'_mc_se']**2))/8))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv', index=False)
    contrasts = []
    for first, second in CONTRASTS:
        a = frame[frame.setting == first].set_index('sample_id')
        b = frame[frame.setting == second].set_index('sample_id')
        if len(a) != 8 or len(b) != 8:
            continue
        for name in ['q', 'q2', 'chi_connected']:
            d = b[name+'_mc']-a[name+'_mc']
            contrasts.append(dict(first=first, second=second, observable=name,
                                  difference=d.mean(), se=d.std(ddof=1)/np.sqrt(8)))
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv', index=False)
    write(OUT/'status.json', dict(completed_runs=len(seeds), expected_runs=160,
                                  complete_disorder_settings=len(rows), ground_state_qualified=False,
                                  reused_runs=int(frame.reused_runs.sum()),
                                  all_histogram_checks_passed=True))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', nargs='+', choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    report() if args.report_only else run(args.settings)
