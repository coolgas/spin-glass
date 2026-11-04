"""Export main disorder-averaged overlap scans for presentation slides."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path('data/kagome/transformer_af_disorder_gfmc_v1'))
    parser.add_argument('--observable', choices=('q', 'q2'), default='q')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path('figures/q2_vs_gamma' if args.observable == 'q2'
                           else 'figures/qea_vs_gamma')
    squared = args.observable == 'q2'
    column = 'q2' if squared else 'qea'
    manifest = json.loads((args.results/'manifest.json').read_text())
    ids, seeds = manifest['sample_ids'], manifest['seeds']
    gamma = manifest['gamma']
    records = {}
    for path in (args.results/'populations').glob('*_main_*.json'):
        row = json.loads(path.read_text())
        key = (row['h'], row['gamma'], row['sample_id'], row['guide'], row['seed'])
        if key in records:
            raise ValueError(f'Duplicate population: {key}')
        records[key] = row[args.observable]
    exact = {}
    for path in (args.results/'cache').glob('*.json'):
        row = json.loads(path.read_text())
        exact[(row['h'], row['gamma'], row['sample_id'])] = row[args.observable]

    series, table = {}, []
    for h in manifest['h']:
        for kind in ('constant', 'network', 'exact'):
            means, errors = [], []
            for g in gamma:
                values = np.array([
                    exact[(h, g, sample_id)] if kind == 'exact' else
                    np.mean([records[(h, g, sample_id, kind, seed)] for seed in seeds])
                    for sample_id in ids
                ])
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite values at {(h, g, kind)}')
                mean = values.mean()
                error = values.std(ddof=1)/np.sqrt(len(values))
                means.append(mean)
                errors.append(error)
                table.append(dict(h=h, gamma=g, method=kind, disorders=len(ids),
                                  **{column+'_mean': mean, column+'_disorder_se': error}))
            series[(h, kind)] = (np.array(means), np.array(errors))

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 16,
        'axes.labelsize': 22, 'axes.titlesize': 20,
        'xtick.labelsize': 15, 'ytick.labelsize': 15,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 1.1, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    })
    fig, axes = plt.subplots(1, len(manifest['h']), figsize=(13.6, 7.65),
                             sharex=True, sharey=squared)
    fig.subplots_adjust(left=.085, right=.975, bottom=.22, top=.76, wspace=.25)
    styles = {
        'constant': dict(color='#D55E00', marker='s', label='Unguided GFMC'),
        'network': dict(color='#0072B2', marker='o', label='Transformer-guided GFMC'),
        'exact': dict(color='#222222', marker='D', label='Exact ground state'),
    }
    for ax, h in zip(np.atleast_1d(axes), manifest['h']):
        if squared:
            baseline = 1/manifest['n_sites']
            ax.axhline(baseline, color='#777777', lw=1.2, ls=':', zorder=1)
            ax.text(2.05, baseline+.025, r'$1/N$ baseline', color='#666666', fontsize=13)
        else:
            ax.axhline(0, color='#8A8A8A', lw=.9, zorder=0)
        for kind in styles:
            means, errors = series[(h, kind)]
            ax.errorbar(gamma, means, yerr=errors,
                        linestyle='--' if kind == 'exact' else '-',
                        lw=2.1, markersize=6.5, capsize=4, elinewidth=1.4,
                        markerfacecolor='white' if kind == 'exact' else styles[kind]['color'],
                        zorder=4 if kind == 'exact' else 3, **styles[kind])
        ax.set(title=rf'$h={h:g}$', xlabel=r'$\Gamma/K$', xlim=(.14, 3.12))
        ax.set_xticks([.25, .5, 1, 1.5, 2, 3])
        ax.set_xticklabels(['0.25', '0.5', '1', '1.5', '2', '3'])
        ax.grid(axis='y', color='#D8D8D8', linewidth=.8, alpha=.7)
        ax.tick_params(direction='out', length=5)
        if squared:
            ax.set_ylim(0, 1.04)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
            if ax is np.atleast_1d(axes)[0]:
                ax.set_ylabel(r'$[\langle q^2\rangle]_{D}$')
        elif h == 0:
            ax.set_ylabel(r'$[q_{\mathrm{EA}}]_{D}$')
            ax.set_ylim(-.13, .15)
            ax.set_yticks([-.1, -.05, 0, .05, .1, .15])
        else:
            ax.set_ylim(-.065, 1.04)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
    fig.suptitle('Squared replica overlap' if squared else 'Edwards–Anderson overlap',
                 fontsize=27, y=.965)
    fig.text(.5, .885,
             rf'$N={manifest["n_sites"]}$ spins  ·  $K=1$  ·  $K_{{ij}}\sim\mathcal{{U}}[0.5,1.5]$  ·  $D={len(ids)}$ disorders',
             ha='center', fontsize=16, color='#444444')
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .85),
               ncol=3, frameon=False, fontsize=15, columnspacing=1.7, handlelength=2.2)
    fig.text(.5, .085,
             rf'Error bars: $\pm1$ SE across {len(ids)} disorders; {len(seeds)} population runs averaged per disorder.',
             ha='center', fontsize=13, color='#444444')
    fig.text(.5, .045, 'GFMC: 2,048 walkers; finite population and projection time.',
             ha='center', fontsize=12, color='#666666')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        target = args.output.with_suffix('.'+extension)
        fig.savefig(target, dpi=300)
        print(target.resolve())
    plt.close(fig)
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == '__main__':
    main()
