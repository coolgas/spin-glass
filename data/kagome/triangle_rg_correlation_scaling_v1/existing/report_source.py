"""Extract physical correlation lengths from immutable saved RG moments."""
import argparse
import base64
import json
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_correlation_observables import spectra, disorder_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for root in args.archives:
        for path in sorted((root/'results/cases').glob('*.json')):
            row = json.loads(path.read_text())
            if row['lambda_rg'] != 1 or row['h'] != .15:
                continue
            raw = path.with_suffix('.npz')
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == row['raw_sha256']
            with np.load(raw) as a:
                s = spectra(a['magnetization'], a['zz'], a['points'], row['L'])
            key = (row['L'], row['gamma'], row['m'], row['chi'], row['sample_id'])
            item = dict(L=key[0], gamma=key[1], m=key[2], chi=key[3], sample_id=key[4],
                        spectra=s.tolist(), q2=row['metrics']['q2'],
                        Q_conn=float(s[0]/row['n']), source=str(path), raw_sha256=row['raw_sha256'])
            if key in rows:
                assert rows[key]['raw_sha256'] == item['raw_sha256']
            rows[key] = item
    ids = sorted({k[-1] for k in rows})
    indices = np.random.default_rng(20261104).integers(len(ids), size=(5000, len(ids)))
    np.save(args.output/'bootstrap_indices.npy', indices)
    groups = sorted({k[:4] for k in rows})
    summary = []
    for L, g, m, chi in groups:
        data = [rows[L, g, m, chi, sid] for sid in ids]
        r = disorder_summary([d['spectra'] for d in data], indices)
        r.update(L=L, gamma=g, m=m, chi=chi, disorders=ids)
        summary.append(r)
    (args.output/'cases.json').write_text(json.dumps(list(rows.values()), indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for (m, chi), color in zip(((4, 2), (6, 2), (6, 3)), ('#0072B2', '#D55E00', '#009E73')):
        group = [r for r in summary if (r['L'], r['m'], r['chi']) == (6, m, chi)]
        for r in group:
            if r['R'] is not None:
                axes[0].plot(r['gamma'], r['R'], 'o', color=color, label=None)
                if r['R_interval95']:
                    axes[0].vlines(r['gamma'], *r['R_interval95'], color=color, alpha=.6)
        axes[0].plot([r['gamma'] for r in group], [r['R'] for r in group], color=color, label=f'm={m}, chi={chi}')
        group = [r for r in summary if (r['gamma'], r['m'], r['chi']) == (.5, m, chi)]
        axes[1].plot([r['L'] for r in group], [r['R'] for r in group], 'o-', color=color, label=f'm={m}, chi={chi}')
        for r in group:
            if r['R_interval95']:
                axes[1].vlines(r['L'], *r['R_interval95'], color=color, alpha=.6)
    axes[0].set(xlabel='Gamma', ylabel='R = xi_R / L', title='L6: existing sparse field scan')
    axes[1].set(xlabel='L', ylabel='R = xi_R / L', title='Gamma = 0.5')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Original bonds, h=0.15 | 8 disorders | bars: disorder bootstrap 95% intervals')
    image = args.output/'correlation_archive.png'
    fig.savefig(image, dpi=160); fig.savefig(image.with_suffix('.pdf')); plt.close(fig)
    intro = '# Physical correlation lengths: existing RG archives\n\n'
    intro += f'{len(rows)} unique saved cases. No new eigensolves. Physical connected correlations; '
    intro += 'ratio of disorder-averaged spectra, three shortest reciprocal directions. '
    intro += 'R=xi_R/L, with xi_R in nearest-neighbor units. Bars are disorder-only 95% bootstrap intervals. '
    intro += 'Invalid bootstrap draws leave the interval unresolved. Retention convergence remains unestablished. '
    intro += 'The upcoming common field grid is Gamma=0.1,0.2,...,1.5.\n\n'
    cells = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_markdown_cell(
        '![Correlation lengths](attachment:correlation_archive.png)',
        attachments={'correlation_archive.png': {'image/png': base64.b64encode(image.read_bytes()).decode()}})]
    table = ['| L | Gamma | m | chi | R | 95% interval | invalid fraction |',
             '| ---: | ---: | ---: | ---: | ---: | :--- | ---: |']
    for r in summary:
        table.append(f"| {r['L']} | {r['gamma']} | {r['m']} | {r['chi']} | {r['R']} | {r['R_interval95']} | {r['invalid_bootstrap_fraction']} |")
    cells.append(nbformat.v4.new_markdown_cell('\n'.join(table)))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'execution_location': 'AutoDL',
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    nbformat.validate(nb); nbformat.write(nb, args.output/'triangle_correlation_existing.ipynb')
    print('Extracted', len(rows), 'cases', flush=True)


if __name__ == '__main__':
    main()
