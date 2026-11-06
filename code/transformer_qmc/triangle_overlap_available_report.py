"""Remote snapshot of available qualified overlap results while dispatch is held."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_overlap_report import collect, difference, digest, figure_cell, within, write

OBS = ('q_ea_finite', 'chi_sg_connected_equal_time', 'q2_raw')
LABELS = (r'$q_{\rm EA}^{(N)}$', r'$\chi_{\rm SG,conn}^{\rm eq}$', r'$\langle q^2\rangle_{\rm raw}$')
FIELDS = (.1, .5, 1.)
IDS = (60000, 60001)
COLORS = {3: '#0072B2', 16: '#D55E00'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--previous-cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    # Re-read and hash-check archived moments rather than plotting stale values.
    previous = json.loads(args.previous_cases.read_text())
    index = []
    for row in previous:
        if row['kind'] == 'archived_exact_retained_space':
            index.append(dict({k: row[k] for k in ('L', 'gamma', 'sample_id', 'm', 'chi')},
                              source=row['result_path'], raw_sha256=row['provenance'][0]['sha256']))
    write(out/'archive_index.json', index)
    rows, pending, statuses = collect(args.roots, out/'archive_index.json')
    unique = {}
    for row in rows:
        key = tuple(row[k] for k in ('L', 'gamma', 'sample_id', 'm', 'chi', 'kind'))
        if key in unique:
            assert unique[key]['result_sha256'] == row['result_sha256'], key
        unique[key] = row
    rows = list(unique.values())
    good = [r for r in rows if r['analysis_qualified']]
    requested = [r for r in good if r['kind'] == 'dmrg' and r['L'] in (4, 6, 8)
                 and r['m'] == 8 and r['chi'] in COLORS
                 and r['sample_id'] in range(60000, 60008) and .1 <= r['gamma'] <= 1.5]
    lookup = {(r['L'], r['chi'], r['gamma'], r['sample_id']): r for r in requested}
    archives = [r for r in good if r['kind'] == 'archived_exact_retained_space']
    now = datetime.now(timezone.utc).isoformat()
    cells = [nbformat.v4.new_markdown_cell(r'''# Available results: overlap and spin-glass susceptibility

Physical parameters: $K_{ij}\sim U(0.5,1.5)$, $\lambda=1$, $h=0.15$.
The requested comparison is $m=8$, $\chi=3,16$, $L=4,6,8$.
New scan dispatch is held pending a resource upgrade; active solves may finish.
$m=8$ retains the full triangle space; $\chi=16$ is partial superblock
retention (the full four-triangle space has 4096 states).

All figures use qualified completed results. Moment hashes and matched disorder
identities are checked remotely. DMRG points must pass the original solver gates
and the additional observable drift, two-start and available exact checks.
Archived points are exact solutions of their retained Hamiltonians.
Additional observable checks require final-D drift <=0.001 for qEA and
<=0.5% for connected susceptibility and raw q2, separately for both starts;
start and available exact agreement require <=0.001 and <=1%, respectively.

$q_{\rm EA}^{(N)}=N^{-1}\sum_i\langle Z_i\rangle^2$;
$\chi_{\rm SG,conn}^{\rm eq}=N^{-1}\sum_{ij}(\langle Z_iZ_j\rangle-
\langle Z_i\rangle\langle Z_j\rangle)^2$;
$\langle q^2\rangle_{\rm raw}=N^{-2}\sum_{ij}\langle Z_iZ_j\rangle^2$.
The raw susceptibility is $N\langle q^2\rangle_{\rm raw}$, with $N=3L^2$.
Susceptibility includes the diagonal terms and is equal-time.

At nonzero longitudinal field, finite $q_{\rm EA}$ and raw overlap include
polarization backgrounds. Retained-space solver convergence does not establish
physical retention convergence. Current coverage supports a sensitivity study;
it does not support a critical-point or exponent fit.
''' + f'\nSnapshot: **{now}**. Extraction and plotting: **AutoDL**.')]
    figures = []

    def save(fig, name, caption):
        path = out/f'{name}.png'
        fig.savefig(path, dpi=135)
        fig.savefig(path.with_suffix('.pdf'))
        plt.close(fig)
        figures.append(path)
        cells.extend([nbformat.v4.new_markdown_cell(caption), figure_cell(path)])

    # Missing qualified pilot fields break lines, rather than bridging gaps.
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout='constrained')
    for i, L in enumerate((4, 6, 8)):
        for j, obs in enumerate(OBS):
            ax = axes[i, j]
            for chi in COLORS:
                for sid, style in zip(IDS, ('-', '--')):
                    values = [lookup[(L, chi, g, sid)]['observables'][obs]
                              if (L, chi, g, sid) in lookup else np.nan for g in FIELDS]
                    if not np.isfinite(values).any():
                        continue
                    ax.plot(FIELDS, values, 'o'+style, color=COLORS[chi], markersize=4,
                            label=f'chi={chi}, disorder {sid}')
            ax.set(title=f'L={L}, m=8', xlabel=r'$\Gamma$', ylabel=LABELS[j], xticks=FIELDS)
            ax.grid(alpha=.2)
            if i == j == 0:
                ax.legend(fontsize=7)
    save(fig, 'requested_individual_curves', '## Requested matched pilot curves\n\n'
         'Each line is one disorder, with no error bars. Colors identify retention; '
         'line styles identify disorders. Missing qualified fields break lines. '
         'Lines are visual guides across the three pilot fields.')

    comparisons = []
    for L in (4, 6, 8):
        for g in FIELDS:
            for sid in IDS:
                a, b = (lookup.get((L, chi, g, sid)) for chi in (3, 16))
                if a and b:
                    d = difference(a['observables'], b['observables'])
                    comparisons.append(dict(L=L, gamma=g, sample_id=sid, **d,
                        within_retention_tolerance=within(d, .01, .02, .02)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for L, marker in zip((4, 6, 8), ('o', 's', '^')):
        for sid, style in zip(IDS, ('-', '--')):
            data = [next((r for r in comparisons if (r['L'], r['gamma'], r['sample_id']) == (L, g, sid)), None) for g in FIELDS]
            if not any(r is not None for r in data):
                continue
            for ax, key in zip(axes, ('q_absolute', 'chi_relative', 'q2_relative')):
                ax.plot(FIELDS, [r[key] if r else np.nan for r in data], marker+style,
                        label=f'L={L}, {sid}')
    for ax, label, tol in zip(axes, (r'$|\Delta q_{\rm EA}|$', 'relative connected susceptibility shift', 'relative raw overlap shift'), (.01, .02, .02)):
        ax.axhline(tol, color='black', linestyle=':', label=f'tolerance {tol:g}')
        ax.set(xlabel=r'$\Gamma$', ylabel=label, xticks=FIELDS)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, 'paired_retention_shifts', '## Paired retention sensitivity\n\n'
         'Compare chi=3 and 16 within each identical disorder. Relative shifts use '
         'the larger absolute value as denominator. Dotted lines show prospective '
         'tolerances (0.01 absolute for qEA; 2% relative for the other quantities). '
         'Unavailable pairs are omitted.')

    means = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for L, style in zip((4, 6, 8), ('-', '--', ':')):
        for chi in COLORS:
            values = []
            for g in FIELDS:
                group = [lookup.get((L, chi, g, sid)) for sid in IDS]
                row = None
                if all(r is not None for r in group):
                    row = dict(L=L, chi=chi, gamma=g, disorders=list(IDS),
                               observables={obs:float(np.mean([r['observables'][obs] for r in group])) for obs in OBS})
                    means.append(row)
                values.append(row)
            if not any(r is not None for r in values):
                continue
            for ax, obs in zip(axes, OBS):
                ax.plot(FIELDS, [r['observables'][obs] if r else np.nan for r in values],
                        'o'+style, color=COLORS[chi], label=f'L={L}, chi={chi}')
    for ax, label in zip(axes, LABELS):
        ax.set(xlabel=r'$\Gamma$', ylabel=label, xticks=FIELDS)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, 'requested_two_disorder_means', '## Fixed two-disorder means by size\n\n'
         'Every displayed mean requires both disorders 60000 and 60001 at that size, '
         'field and retention. No changing-disorder averages are used. Two disorders '
         'give limited sensitivity information, so uncertainty bars are omitted here. '
         'The disorder realizations are matched across fields and retention within each size. '
         'Size ensembles are independent even though their numerical disorder IDs coincide.')

    bootstrap = np.random.default_rng(20261105).integers(8, size=(5000, 8))
    np.save(out/'archive_bootstrap_indices.npy', bootstrap)
    archive_means = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for m, chi in sorted({(r['m'], r['chi']) for r in archives if r['L'] == 6}):
        groups = []
        for g in sorted({r['gamma'] for r in archives if r['L'] == 6}):
            data = sorted([r for r in archives if (r['L'], r['m'], r['chi'], r['gamma']) == (6, m, chi, g)], key=lambda r:r['sample_id'])
            if [r['sample_id'] for r in data] != list(range(60000, 60008)):
                continue
            row = dict(L=6, m=m, chi=chi, gamma=g, disorders=list(range(60000, 60008)), observables={})
            for obs in OBS:
                values = np.array([r['observables'][obs] for r in data])
                row['observables'][obs] = dict(mean=float(values.mean()), interval95=np.quantile(values[bootstrap].mean(axis=1), [.025, .975]).tolist())
            groups.append(row)
            archive_means.append(row)
        for ax, obs in zip(axes, OBS):
            line, = ax.plot([r['gamma'] for r in groups], [r['observables'][obs]['mean'] for r in groups], 'o-', markersize=3, label=f'm={m}, chi={chi}')
            for r in groups:
                ax.vlines(r['gamma'], *r['observables'][obs]['interval95'], color=line.get_color(), alpha=.5)
    for ax, label in zip(axes, LABELS):
        ax.set(xlabel=r'$\Gamma$', ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    save(fig, 'existing_L6_eight_disorder_scan', '## Existing L6 field scan: eight disorders\n\n'
         'Earlier exact retained-space solutions provide broader field coverage. '
         'Bars are 95% disorder bootstrap intervals (5,000 resamples); the same whole-disorder '
         'resampling indices are used across fields, settings and observables. '
         'These earlier retention settings carry physical truncation bias.')

    l8 = [r for r in good if r['kind'] == 'dmrg' and r['L'] == 8]
    chis = [2, 3, 4, 8, 16, 32]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout='constrained')
    for i, g in enumerate(FIELDS):
        for j, obs in enumerate(OBS):
            ax = axes[i, j]
            for m, color in zip((4, 6, 8), ('#0072B2', '#D55E00', '#009E73')):
                for sid, style in zip(IDS, ('-', '--')):
                    values = []
                    for chi in chis:
                        r = next((r for r in l8 if (r['m'], r['chi'], r['gamma'], r['sample_id']) == (m, chi, g, sid)), None)
                        values.append(r['observables'][obs] if r else np.nan)
                    if not np.isfinite(values).any():
                        continue
                    ax.plot(range(len(chis)), values, 'o'+style, color=color, markersize=4, label=f'm={m}, {sid}')
            ax.set(title=f'L=8, Gamma={g:g}', xlabel='retained superblock states chi', ylabel=LABELS[j], xticks=range(len(chis)), xticklabels=chis)
            ax.grid(alpha=.2)
            if i == j == 0:
                ax.legend(fontsize=7, ncol=2)
    save(fig, 'existing_L8_retention', '## Existing L8 retention ladder\n\n'
         'Each curve is an individual disorder, with no error bars. This uses the '
         'qualified completed retention campaign, including settings beyond the requested '
         'm=8 comparison. Missing or failed solves break lines.')

    coverage = [dict(L=L, chi=chi,
                     count=sum(r['L'] == L and r['chi'] == chi for r in requested),
                     qualified_pilots=sum((L, chi, g, sid) in lookup for g in FIELDS for sid in IDS))
                for L in (4, 6, 8) for chi in COLORS]
    table = '| L | chi | qualified matched pilots / 6 |\n|---|---|---|\n'
    for L in (4, 6, 8):
        for chi in COLORS:
            n = sum((L, chi, g, sid) in lookup for g in FIELDS for sid in IDS)
            table += f'| {L} | {chi} | {n} / 6 |\n'
    sensitive = sum(not r['within_retention_tolerance'] for r in comparisons)
    example = [lookup.get((4, chi, .5, 60000)) for chi in (3, 16)]
    if all(r is not None for r in example):
        values = [r['observables'] for r in example]
        cells.append(nbformat.v4.new_markdown_cell('## What the available curves show\n\n'
            'The earlier L6 eight-disorder scan shows decreasing qEA and raw overlap, '
            'and increasing connected susceptibility over Gamma=0.1 to 1.0 at all three '
            'displayed retention settings. This limited field range does not locate a '
            'susceptibility maximum or critical field.\n\n'
            'Retention shifts are substantial even when the retained Hamiltonian is solved accurately. '
            'For L4, Gamma=0.5, disorder 60000, chi=3 to 16 gives '
            f"qEA {values[0][OBS[0]]:.6f} to {values[1][OBS[0]]:.6f}, "
            f"connected susceptibility {values[0][OBS[1]]:.6f} to {values[1][OBS[1]]:.6f}, "
            f"and raw q2 {values[0][OBS[2]]:.6f} to {values[1][OBS[2]]:.6f}. "
            'L4 has all six matched pilot pairs. Larger-size retention comparisons '
            'remain incomplete; the coverage table records their availability.'))
    cells.append(nbformat.v4.new_markdown_cell('## Coverage and assessment\n\n'+table+
        f'\n{len(requested)} qualified requested cases; {len(l8)} qualified L8 retention cases; '
        f'{len(archives)} archived exact retained-space cases. Overlapping requested and L8 counts '
        'describe different views of the same data.\n\n'
        f'{sensitive} of {len(comparisons)} available paired chi=3 to 16 comparisons exceed at least '
        'one prospective retention tolerance. Physical retention convergence is not established. '
        'The incomplete field and size coverage does not support finite-size scaling fits yet. '
        'Next: incorporate finishing active solves, then reassess the field grid and larger retention '
        'after the resource upgrade. Large states and raw artifacts remain on AutoDL.'))
    excluded = [{k:r[k] for k in ('L', 'gamma', 'sample_id', 'm', 'chi', 'solver_passed', 'analysis_qualified')} for r in rows if not r['analysis_qualified']]
    cells.append(nbformat.v4.new_markdown_cell('## Exclusions\n\n'
        f'{len(excluded)} completed cases fail at least one required gate; '
        f'{len(pending)} unfinished case entries were found across the input roots. '
        'They contribute no plotted scientific points. Detailed gates, paths and moment hashes '
        'remain in the remote cases.json and summary.json. Held controller statuses are historical; '
        'coverage is recomputed directly from completed result files.'))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata=dict(
        execution_location='AutoDL', snapshot_utc=now, error_bar_policy='disorder only',
        kernelspec=dict(display_name='Python 3', language='python', name='python3')))
    nbformat.validate(notebook)
    nbformat.write(notebook, out/'triangle_overlap_available.ipynb')
    counts = dict(distinct_extracted=len(rows), qualified=len(good), requested=len(requested),
                  L8_retention=len(l8), archived=len(archives), excluded=len(excluded),
                  paired_retention=len(comparisons), sensitive_pairs=sensitive)
    summary = dict(utc=now, counts=counts, paired_retention=comparisons, two_disorder_means=means,
                   archive_means=archive_means, excluded=excluded, pending=pending,
                   statuses=statuses, retention_qualified=False, scaling_fit_attempted=False)
    write(out/'cases.json', rows)
    write(out/'summary.json', summary)
    write(out/'brief.json', dict(utc=now, counts=counts, requested_coverage=coverage, paired_retention=comparisons))
    write(out/'delivery.json', dict(utc=now, source_sha256=digest(Path(__file__)),
        dependency_sha256={name:digest(Path(__file__).with_name(name)) for name in
                           ('triangle_overlap_report.py', 'triangle_correlation_observables.py')},
        bootstrap_sha256=digest(out/'archive_bootstrap_indices.npy'),
        files={p.name:dict(bytes=p.stat().st_size, sha256=digest(p)) for p in [out/'triangle_overlap_available.ipynb', out/'brief.json', *figures]},
        roots=list(map(str,args.roots)), previous_cases_sha256=digest(args.previous_cases)))
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
