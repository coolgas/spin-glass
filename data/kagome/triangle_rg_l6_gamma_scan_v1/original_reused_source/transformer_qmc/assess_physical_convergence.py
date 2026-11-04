"""Pointwise population/time stability from verified independent GFMC repeats."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .assess_stage_b import assess, write_csv
from .experiment import write_json
from .production_protocol import TOLERANCES, identity


SCALARS = ('energy_per_site', 'q2', 'q4', 'q_ea_finite', 'chi_connected_per_site')
LEVELS = ('baseline', 'population256', 'population1024', 'equilibration4', 'forward2')
CONTRASTS = (('baseline', 'population256'), ('population256', 'population1024'),
             ('population1024', 'equilibration4'), ('population1024', 'forward2'))


def difference_stats(x, y, tolerance):
    """Independent settings on the same disorder; four whole-population repeats."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != (4,) or y.shape != (4,) or not np.isfinite(np.r_[x, y]).all():
        raise ValueError('Need four finite independent repeats at each setting')
    vx, vy = x.var(ddof=1)/len(x), y.var(ddof=1)/len(y)
    se = float(np.sqrt(vx+vy))
    delta = float(y.mean()-x.mean())
    df = float((vx+vy)**2/(vx**2/3+vy**2/3)) if vx+vy else None
    half = float(student_t.ppf(.975, df)*se) if df else 0.
    bound = abs(delta)+1.96*se
    if tolerance is None:
        status = 'no_prespecified_tolerance'
    elif bound <= tolerance:
        status = 'within_pointwise_tolerance'
    elif abs(delta)-half > tolerance:
        status = 'resolved_shift_beyond_tolerance'
    else:
        status = 'unresolved_precision_or_shift'
    return dict(lower_mean=float(x.mean()), higher_mean=float(y.mean()),
        difference=delta, independent_run_se=se, welch_df=df,
        welch95_low=delta-half, welch95_high=delta+half,
        normal95_absolute_bound=bound, welch95_absolute_bound=abs(delta)+half,
        tolerance=tolerance, status=status,
        welch_within_tolerance=abs(delta)+half<=tolerance if tolerance is not None else None)


def values(record):
    result = record['result']
    out = {o: result[o] for o in SCALARS}
    N = 3*record['task']['L']**2
    out.update({f'connected_spectrum_per_site_{i}': float(v)/N
                for i, v in enumerate(result['connected_spectrum'])})
    return out


def analyze(root):
    stage_b = assess(root)
    groups = defaultdict(dict)
    hashes = {}
    for stage in ('notebook_stage_b', 'notebook_stage_b_escalation'):
        folder = root/stage
        manifest = json.loads((folder/'manifest.json').read_text())
        tasks = json.loads((folder/'tasks.json').read_text())
        if identity(tasks) != manifest['tasks_sha256']:
            raise RuntimeError('Task identity mismatch')
        for task in tasks:
            if task['sample_id'] != 60000 or task['gamma'] not in (.5, 2.) or task['h'] not in (0., .05):
                continue
            path = folder/'results'/(task['id']+'.json')
            if not path.exists():
                continue
            raw = path.read_bytes()
            record = json.loads(raw)
            if record['task'] != task or record['signature'] != manifest['signature'] or record['state'] != 'completed':
                raise RuntimeError('Result identity mismatch: '+str(path))
            hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            level = task.get('ladder', 'base')
            if level in ('base', 'baseline_h005'):
                level = 'baseline'
            key = (task['gamma'], task['h'], task['method'], level)
            if task['repeat'] in groups[key]:
                raise RuntimeError('Duplicate independent repeat')
            groups[key][task['repeat']] = record
    summaries, contrasts, coverage = [], [], []
    for g in (.5, 2.):
        for h in (0., .05):
            for method in ('network', 'constant'):
                for level in LEVELS:
                    runs = groups[g, h, method, level]
                    coverage.append(dict(gamma=g, h=h, method=method, level=level,
                        completed=len(runs), expected=4,
                        alarms=sum(r['ancestry_alarm'] for r in runs.values()),
                        minimum_ess=min((r['minimum_descendant_ess'] for r in runs.values()), default=None),
                        minimum_ess_fraction=min((r['minimum_descendant_ess']/r['task']['parameters']['n_walkers']
                            for r in runs.values()), default=None),
                        maximum_lineage=max((max(r['result']['largest_lineage']) for r in runs.values()), default=None)))
                    if set(runs) != set(range(4)):
                        continue
                    samples = [values(runs[r]) for r in range(4)]
                    for obs in samples[0]:
                        x = np.array([s[obs] for s in samples])
                        summaries.append(dict(gamma=g, h=h, method=method, level=level,
                            observable=obs, mean=float(x.mean()), mc_se=float(x.std(ddof=1)/2),
                            minimum=float(x.min()), maximum=float(x.max()), repeats=4, disorders=1))
                for low, high in CONTRASTS:
                    a, b = groups[g, h, method, low], groups[g, h, method, high]
                    if set(a) != set(range(4)) or set(b) != set(range(4)):
                        continue
                    x, y = [values(a[r]) for r in range(4)], [values(b[r]) for r in range(4)]
                    for obs in x[0]:
                        stats = difference_stats([s[obs] for s in x], [s[obs] for s in y], TOLERANCES.get(obs))
                        contrasts.append(dict(gamma=g, h=h, method=method, lower=low, higher=high,
                            observable=obs, **stats, disorders=1, repeats_per_setting=4,
                            lower_alarms=sum(r['ancestry_alarm'] for r in a.values()),
                            higher_alarms=sum(r['ancestry_alarm'] for r in b.values()),
                            qualification_passed=False))
    destination = root/'physical_convergence'
    destination.mkdir(exist_ok=True)
    for name, data in [('coverage', coverage), ('estimates', summaries), ('contrasts', contrasts)]:
        write_csv(destination/(name+'.csv'), data)
    guided = [r for r in contrasts if r['method']=='network' and r['observable'] in TOLERANCES]
    counts = {status: sum(r['status']==status for r in guided) for status in
              ('within_pointwise_tolerance', 'resolved_shift_beyond_tolerance', 'unresolved_precision_or_shift')}
    write_json(destination/'summary.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_signatures=stage_b['input_signatures'], result_sha256=hashes,
        guided_toleranced_comparison_counts=counts, coverage=coverage, contrasts=contrasts,
        ground_state_qualified=False, final_data_started=False,
        estimator='q_ea_finite=q=<N^-1 sum_i s_i^(1) s_i^(2)>_pure, targeting N^-1 sum_i <Zi>^2',
        caveat='Single disorder. Approximate pointwise normal gate and Welch sensitivity; no simultaneous coverage. Incomplete repeat sets omitted from estimates.'))
    lines = ['# Physical-observable convergence review', '', datetime.now(timezone.utc).isoformat(), '',
        '**Development evidence; ground-state qualification remains pending.**', '',
        'The matched comparisons use L=4 (48 spins), PBC, disorder 60000 and four independent whole-population repeats at each setting. All ancestry alarms are retained. A repeat is not a new disorder; four forward cycles within a run are correlated and are not used as independent error samples.', '',
        '`q_ea_finite` is the signed two-replica overlap mean, targeting N^-1 sum_i <Zi>^2 after pure projection. It is not sqrt(q2). The unique finite-size symmetric ground state has zero site magnetization at h=0; at h>0 this estimator includes field-induced polarization. No clipping or absolute-value conversion is applied.', '',
        'The notebook prescribes |difference| + 1.96 SE <= tolerance for E/N (0.001), q2 (0.002), and chi_connected/N (0.002). Tables also retain Welch t intervals because four repeats give uncertain variance estimates. A large interval or insignificant difference does not demonstrate equivalence. No tolerance has been prespecified for q4, q_ea_finite or nonzero-momentum spectra; those remain descriptive.', '',
        'Only complete sets of four repeats enter means and contrasts. Differences compare independent runs at two settings on the same Hamiltonian; no trajectory pairing is assumed. These pointwise comparisons measure MC uncertainty only, and do not estimate disorder variability.', '',
        '## Guided coverage', '', '| Gamma | h | Setting | Completed repeats | Alarms | Worst ESS | Worst ESS/W | Largest lineage share |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'pending' if v is None else f'{v:.6g}'
    for row in coverage:
        if row['method']=='network':
            lines.append(f"| {row['gamma']} | {row['h']} | {row['level']} | {row['completed']}/4 | {row['alarms']} | {fmt(row['minimum_ess'])} | {fmt(row['minimum_ess_fraction'])} | {fmt(row['maximum_lineage'])} |")
    lines += ['', '## Guided physical contrasts', '',
        '| Gamma | h | Change | Observable | Stronger minus candidate | MC SE | Normal 95% absolute bound | Tolerance | Assessment |',
        '|---:|---:|---|---|---:|---:|---:|---:|---|']
    for row in contrasts:
        if row['method']=='network' and row['observable'] in SCALARS:
            lines.append('| '+' | '.join(map(str, (row['gamma'], row['h'], row['lower']+' → '+row['higher'],
                row['observable'], fmt(row['difference']), fmt(row['independent_run_se']),
                fmt(row['normal95_absolute_bound']), fmt(row['tolerance']), row['status'])))+' |')
    lines += ['', '## Next convergence work', '',
        f"Available guided contrasts with declared tolerances: {counts['within_pointwise_tolerance']} within the approximate pointwise bound, {counts['resolved_shift_beyond_tolerance']} with a resolved shift beyond tolerance, and {counts['unresolved_precision_or_shift']} unresolved.", '',
        'Finish incomplete time contrasts before choosing settings. Where a shift is resolved, strengthen that parameter and obtain a third level before claiming a plateau. Where uncertainty is too broad, add independent repeats or longer validated measurements. Longer forward time must be judged together with ESS/W and lineage concentration.', '',
        'Formal qualification also needs the notebook development-disorder policy, third time levels, branching-cadence, alternative-start and combined checks, and a measured feasible budget. Current one-disorder data cannot establish robustness or qualify the full field grid. Define q4 and q_ea_finite tolerances prospectively before using them in an acceptance gate.', '',
        'CSV exports include every connected-spectrum momentum divided by N, with original momentum ordering (my,mx); k=0 duplicates chi_connected/N. Nonzero-momentum shifts remain descriptive. Ratios and finite-size scaling need qualified disorder averages and covariance propagation.', '',
        '## Continuation cost', '']
    pointer = root/'active_continuation.json'
    if pointer.exists():
        launch = Path(json.loads(pointer.read_text())['launch_path'])
        # Mirrored pointers name remote paths; resolve the relative campaign suffix locally.
        if not launch.exists():
            launch = root/'continuations'/launch.parent.name/'launch.json'
        if launch.exists():
            data = json.loads(launch.read_text())
            pending = set(data['pending_task_ids'])
            completed = []
            for path in (root/'notebook_stage_b_escalation/results').glob('*.json'):
                r = json.loads(path.read_text())
                if r['task']['id'] in pending:
                    completed.append(r)
            lines += [f"Window: {data['window_hours']} hours. Initial work estimate: {data['estimated_remaining_gpu_hours']} GPU-hours. Newly completed pending tasks: {len(completed)}/{len(pending)}.", '',
                f"Completed continuation task segments total {sum(r['metrics']['segment_seconds'] for r in completed)/3600:.3f} GPU-worker hours. This excludes unfinished segments and operational/analysis overhead; resumed task totals also include time before this window.", '']
    (root/'PHYSICAL_CONVERGENCE.md').write_text('\n'.join(lines))
    plot_estimates(destination, summaries)
    print(json.dumps(dict(report=str(root/'PHYSICAL_CONVERGENCE.md'), guided_contrast_counts=counts)))
    return contrasts


def plot_estimates(destination, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(SCALARS), 4, figsize=(16, 14), squeeze=False)
    for col, (g, h) in enumerate(((.5, 0.), (.5, .05), (2., 0.), (2., .05))):
        for row, obs in enumerate(SCALARS):
            ax = axes[row, col]
            points = [r for r in summaries if (r['gamma'], r['h'], r['method'], r['observable']) == (g, h, 'network', obs)]
            points.sort(key=lambda r: LEVELS.index(r['level']))
            ax.errorbar([LEVELS.index(r['level']) for r in points], [r['mean'] for r in points],
                        yerr=[r['mc_se'] for r in points], fmt='o', capsize=3)
            ax.set_xticks(range(5), ['W64', 'W256', 'W1024', 'eq4', 'FW2'], rotation=45)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(f'Γ={g}, h={h}')
            if col == 0:
                ax.set_ylabel(obs)
    fig.suptitle('Guided GFMC: L=4, one disorder, four independent repeats; error bars ±1 MC SE\n'
                 'eq4 and FW2 each change one parameter relative to W1024; missing points are incomplete')
    fig.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ('png', 'pdf'):
        fig.savefig(destination/('guided_estimates.'+suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/kagome/transformer_af_production_v1'))
    args = parser.parse_args()
    analyze(args.root)


if __name__ == '__main__':
    main()
