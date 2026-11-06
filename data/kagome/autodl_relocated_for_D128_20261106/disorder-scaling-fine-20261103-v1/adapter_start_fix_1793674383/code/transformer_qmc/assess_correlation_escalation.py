"""Audit saved v4 runs and compare matched-disorder populations and forward times.

This offline sidecar does not alter the running campaign or qualify production.
Repeat resampling is conditional on the fixed development disorders; independent
settings use independent repeat draws, with each four-momentum vector kept intact.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .correlation_campaign import digest, name, write
from .correlation_lengths import length
from .production_protocol import identity


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root, parent):
    manifest = json.loads((root/'manifest.json').read_text())
    pm = json.loads((parent/'manifest.json').read_text())
    require(digest(parent/'manifest.json') == manifest['parent_manifest_sha256'],
            'Parent manifest hash mismatch')
    signature, parent_signature = identity(manifest), identity(pm)
    expected = {identity(t)[:24]: t for t in manifest['tasks']}
    require(len(expected) == len(manifest['tasks']), 'Duplicate manifest tasks')
    require(len(expected) == 384, 'Expected the 384-task v4 protocol')
    records, references, seen = [], {}, set()
    for path in sorted((root/'runs').glob('*.json')):
        record = json.loads(path.read_text())
        task = record['task']
        key = identity(task)[:24]
        require(key not in seen, f'Duplicate task: {key}')
        require(path.stem == key and expected.get(key) == task, f'Unexpected task: {path.name}')
        require(record['signature'] == signature and record['status'] == 'completed',
                f'Invalid run identity/status: {key}')
        params = dict(n_walkers=task['walkers'], n_replicas=4, tau_block=manifest['branch'],
                      n_equilibrate=round(manifest['equilibration']/manifest['branch']),
                      n_measure=round(manifest['measurement']/manifest['branch']),
                      tau_forward=task['forward'], n_cycles=manifest['cycles'])
        require(record['parameters'] == params, f'Parameter mismatch: {key}')
        require(record['seed'] == int(identity([signature, task])[:16], 16), f'Seed mismatch: {key}')
        stem = name(task['sample_id'], task['gamma'])
        if stem not in references:
            ep, gp = parent/'exact'/(stem+'.json'), parent/'guides'/(stem+'.json')
            exact, guide = json.loads(ep.read_text()), json.loads(gp.read_text())
            require(exact['signature'] == guide['signature'] == parent_signature,
                    f'Parent record identity mismatch: {stem}')
            require(exact['status'] == guide['status'] == 'completed', f'Incomplete reference: {stem}')
            require(digest(gp.with_suffix('.npz')) == guide['npz_sha256'], f'Guide hash mismatch: {stem}')
            require(digest(ep.with_suffix('.npz')) == exact['npz_sha256'], f'Exact archive hash mismatch: {stem}')
            require(exact['residual'] < 1e-9, f'Exact residual failure: {stem}')
            references[stem] = (exact, guide, digest(ep))
        exact, guide, exact_hash = references[stem]
        require(record['guide_sha256'] == guide['npz_sha256'] and
                record['exact_record_sha256'] == exact_hash, f'Reference hash mismatch: {key}')
        require(record['exact_spectrum'] == exact['spectrum'], f'Exact spectrum mismatch: {key}')
        result = record['result']
        for field in ['seed', 'n_walkers', 'n_replicas', 'n_equilibrate', 'n_measure', 'tau_block', 'tau_forward']:
            require(result[field] == (record['seed'] if field == 'seed' else params[field]),
                    f'Result parameter mismatch: {key}/{field}')
        spectra = np.asarray(result['connected_spectrum'], dtype=float)
        cycles = np.asarray(result['connected_spectrum_cycles'], dtype=float)
        require(spectra.shape == (4,) and cycles.shape == (params['n_cycles'], 4),
                f'Incomplete spectrum/cycle block: {key}')
        require(np.isfinite(cycles).all() and np.isfinite(spectra).all(), f'Nonfinite spectrum: {key}')
        require(np.allclose(cycles.mean(0), spectra, rtol=1e-10, atol=1e-12),
                f'Cycle mean mismatch: {key}')
        for field in ['descendant_ess', 'largest_lineage', 'unique_ancestors', 'forward_times']:
            values = np.asarray(result[field], dtype=float)
            require(values.shape == (params['n_cycles'],) and np.isfinite(values).all(),
                    f'Incomplete ancestry block: {key}/{field}')
        require(np.allclose(result['forward_times'], task['forward']), f'Forward time mismatch: {key}')
        alarm = min(result['descendant_ess']) < 8 or max(result['largest_lineage']) > .5
        require(record['ancestry_alarm'] == alarm, f'Ancestry flag mismatch: {key}')
        seen.add(key)
        records.append(record)
    return manifest, records, [expected[k] for k in sorted(expected.keys()-seen)]


def bootstrap(raw, rng, draws=4000):
    """Resample independent runs within fixed disorders, preserving momentum covariance."""
    disorders, repeats, momenta = raw.shape
    require(momenta == 4 and repeats >= 2, 'Expected repeat spectra with four momenta')
    ix = rng.integers(repeats, size=(draws, disorders, repeats))
    return raw[np.arange(disorders)[None, :, None], ix].mean(axis=(1, 2)), ix


def comparison(mean, boot, target, target_boot=None):
    xb, tb = length(boot, 2)/4, length(target if target_boot is None else target_boot, 2)/4
    delta = float((length(mean, 2)-length(target, 2))/4)
    delta_boot = xb-tb
    finite = np.isfinite(delta_boot)
    se = float(delta_boot[finite].std(ddof=1)) if finite.sum() > 1 else None
    spectral_boot = boot-target if target_boot is None else boot-target_boot
    spectral_delta = mean-target
    bound = abs(delta)+1.96*se if np.isfinite(delta) and se is not None else None
    return dict(xi_difference=delta*4 if np.isfinite(delta) else None,
                xi_difference_mc_se=se*4 if se is not None else None,
                xi_over_side_95_bound=bound, invalid_bootstrap_fraction=float(1-finite.mean()),
                spectral_difference=spectral_delta.tolist(),
                spectral_difference_mc_se=spectral_boot.std(0, ddof=1).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    manifest, records, missing = validate(args.root, args.parent)
    grouped, wanted = defaultdict(list), defaultdict(set)
    group_key = lambda t: (t['gamma'], t['forward'], t['walkers'])
    for task in manifest['tasks']:
        wanted[group_key(task)].add((task['sample_id'], task['repeat']))
    for record in records:
        grouped[group_key(record['task'])].append(record)
    rows, diagnostics, contrasts, data = [], [], [], {}
    for key in sorted(wanted):
        group = sorted(grouped[key], key=lambda r: (r['task']['sample_id'], r['task']['repeat']))
        if {(r['task']['sample_id'], r['task']['repeat']) for r in group} != wanted[key]:
            continue
        ids = sorted({r['task']['sample_id'] for r in group})
        require(len(ids) == 16 and len(group) == 64, 'Incomplete disorder/repeat rectangle')
        raw = np.array([r['result']['connected_spectrum'] for r in group]).reshape(16, 4, 4)
        exact = np.array([r['exact_spectrum'] for r in group]).reshape(16, 4, 4)[:, 0].mean(0)
        seed = int(identity(['v4_assessment', key])[:16], 16)
        boot, ix = bootstrap(raw, np.random.default_rng(seed))
        mean = raw.mean(axis=(0, 1))
        check = comparison(mean, boot, exact)
        sb = (abs(mean-exact)+1.96*boot.std(0, ddof=1))/exact
        alarms = sum(r['ancestry_alarm'] for r in group)
        passes = (check['xi_over_side_95_bound'] is not None and check['xi_over_side_95_bound'] <= .01
                  and check['invalid_bootstrap_fraction'] == 0 and np.all(sb <= .05) and alarms == 0)
        cycles = np.array([r['result']['connected_spectrum_cycles'] for r in group]).reshape(16, 4, 16, 4)
        early, late = cycles[:, :, :8].mean(2), cycles[:, :, 8:].mean(2)
        # Early/late pieces share each run: use paired repeat draws, never independent cycles.
        eb = early[np.arange(16)[None, :, None], ix].mean(axis=(1, 2))
        lb = late[np.arange(16)[None, :, None], ix].mean(axis=(1, 2))
        drift = comparison(late.mean(axis=(0, 1)), lb, early.mean(axis=(0, 1)), eb)
        xi = float(length(mean, 2))
        rows.append(dict(gamma=key[0], forward=key[1], walkers=key[2], disorders=16, repeats=4,
                         xi=xi if np.isfinite(xi) else None, exact_xi=float(length(exact, 2)),
                         mc_se=check['xi_difference_mc_se'], spectrum=mean.tolist(), exact_spectrum=exact.tolist(),
                         spectral_relative_95_bounds=sb.tolist(), ancestry_alarms=alarms,
                         status='within_development_tolerances' if passes else 'unresolved',
                         early_late_diagnostic=drift, ground_state_qualified=False, **check))
        data[key] = (mean, boot, exact)
        for sid in ids:
            subset = [r for r in group if r['task']['sample_id'] == sid]
            diagnostics.append(dict(gamma=key[0], forward=key[1], walkers=key[2], sample_id=sid,
                alarmed_runs=sum(r['ancestry_alarm'] for r in subset),
                minimum_descendant_ess=min(min(r['result']['descendant_ess']) for r in subset),
                maximum_lineage=max(max(r['result']['largest_lineage']) for r in subset),
                minimum_unique_ancestors=min(min(r['result']['unique_ancestors']) for r in subset),
                spectrum_repeat_se=(np.std([r['result']['connected_spectrum'] for r in subset], axis=0, ddof=1)/2).tolist()))
    pairs = [('population', (g, fw, 8192), (g, fw, 16384)) for g, fw in [(.5, 64.), (.5, 128.), (.75, 32.)]]
    pairs += [('forward_time', (.5, 64., w), (.5, 128., w)) for w in [8192, 16384]]
    for kind, first, second in pairs:
        if first not in data or second not in data:
            continue
        a, ab, exact = data[first]
        b, bb, other_exact = data[second]
        require(np.array_equal(exact, other_exact), 'Unmatched contrast references')
        contrast = comparison(b, bb, a, ab)
        spectral_bound = (abs(b-a)+1.96*(bb-ab).std(0, ddof=1))/exact
        within = (contrast['xi_over_side_95_bound'] is not None and contrast['xi_over_side_95_bound'] <= .01
                  and contrast['invalid_bootstrap_fraction'] == 0 and np.all(spectral_bound <= .05))
        contrasts.append(dict(kind=kind, first=first, second=second,
                              spectral_relative_95_bounds=spectral_bound.tolist(),
                              shift_within_development_tolerances=bool(within), **contrast))
    report = dict(completed_runs=len(records), expected_runs=len(manifest['tasks']), missing_tasks=missing,
                  complete_groups=len(rows), rows=rows, contrasts=contrasts, disorder_diagnostics=diagnostics,
                  manifest_sha256=digest(args.root/'manifest.json'), assessor_sha256=digest(Path(__file__)),
                  uncertainty='Fixed-disorder repeat bootstrap; approximate pointwise 95% bounds; four repeats have limited tail resolution. Independent settings resampled independently; cycle halves paired within runs.',
                  qualification='Development only. Exact agreement and small contrasts do not override ancestry alarms; stronger plateaus, alternative starts, held-out checks, and larger-size convergence remain required.')
    write(args.root/'validated_review.json', report)
    with (args.root/'validated_spectra.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'gamma', 'walkers', 'forward', 'repeat', 'S0', 'Sk1', 'Sk2', 'Sk3', 'ancestry_alarm'])
        for r in sorted(records, key=lambda r: (group_key(r['task']), r['task']['sample_id'], r['task']['repeat'])):
            t = r['task']
            writer.writerow([t[k] for k in ['sample_id', 'gamma', 'walkers', 'forward', 'repeat']]+r['result']['connected_spectrum']+[r['ancestry_alarm']])
    lines = ['# Validated v4 convergence review', '', report['qualification'], '', report['uncertainty'], '',
             f'Validated {len(records)}/{len(manifest["tasks"])} runs; {len(rows)}/6 complete groups.', '',
             '| Γ | Forward | Walkers | GFMC ξ | Exact ξ | MC SE | Alarmed runs | Status |',
             '|---:|---:|---:|---:|---:|---:|---:|---|']
    fmt = lambda v: 'unresolved' if v is None else f'{v:.5f}'
    for r in rows:
        lines.append(f'| {r["gamma"]} | {r["forward"]:g} | {r["walkers"]} | {fmt(r["xi"])} | {fmt(r["exact_xi"])} | {fmt(r["mc_se"])} | {r["ancestry_alarms"]}/64 | {r["status"]} |')
    lines += ['', '| Contrast (second − first) | Δξ | MC SE | Shift within tolerance |', '|---|---:|---:|---|']
    for c in contrasts:
        lines.append(f'| {c["kind"]}: {c["second"]} − {c["first"]} | {fmt(c["xi_difference"])} | {fmt(c["xi_difference_mc_se"])} | {c["shift_within_development_tolerances"]} |')
    lines += ['', 'Per-disorder ancestry and paired early/late-cycle diagnostics are in `validated_review.json`. All runs are retained.']
    (args.root/'VALIDATED_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(validated_runs=len(records), complete_groups=len(rows), contrasts=len(contrasts), missing_runs=len(missing))))


if __name__ == '__main__':
    main()
