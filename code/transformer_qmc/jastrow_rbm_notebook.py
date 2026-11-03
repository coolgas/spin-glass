"""Consolidate saved L=2 and L=4 overlap results; execute on AutoDL only."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nbformat as nb

if __package__:
    from .overlap_polarization import notebook_cells as polarization_cells
else:
    from overlap_polarization import notebook_cells as polarization_cells


MARKER = 'consolidated-disorder-results'
SIZE_NOTE = r'''**System sizes:** periodic kagome, three spins per unit cell,
$N=3L^2$. The original exact/variational sections use **$L=2$, $2\times2$
unit cells, $N=12$ spins**. The measured larger-system sections use
**$L=4$, $4\times4$ unit cells, $N=48$ spins**. All results use **$h=0.15$**.

**Where are the $L=4$ results?** Scroll to **L=4, N=48: before and after
projection**, followed by **L=4, N=48: higher-field distributions**.
The $L=2$ exact-guide, exact-ground-state, guide-MCMC and GFMC comparison
appears immediately before them. Plot-panel counts are not lattice sizes.

'''


def label_l2_cells(cells):
    """Keep the original small-system analysis and make its scope explicit."""
    for cell in cells:
        source = cell.source
        if cell.cell_type == 'markdown':
            source = source.replace(
                'All plots here are **unprojected**.',
                'The original exact/variational plots in this first part are **unprojected**.')
            if source.startswith('## ') and 'L=2, N=12' not in source.splitlines()[0]:
                source = source.replace('## ', '## L=2, N=12: ', 1)
        else:
            source = source.replace('root=Path.cwd()',
                "root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data/kagome').is_dir())")
            source = source.replace(
                "fig.suptitle('Exact and energy-selected unprojected Jastrow/RBM guides, L=2')",
                "fig.suptitle('Exact and unprojected Jastrow/RBM guides | L=2, N=12 spins | PBC | h=0.15')")
            source = source.replace(
                "fig.suptitle('Matched difficult disorder 50043: exact and unprojected guides')",
                "fig.suptitle('Difficult disorder 50043 | L=2, N=12 spins | PBC | h=0.15')")
            for filename, title in [
                ('all_seeds_pq', 'All optimization seeds'),
                ('sampled_guides_pq', 'Sampled-VMC guides'),
                ('field_scan_pq', 'Equal-weight eight-disorder averages'),
            ]:
                anchor = f"fig.tight_layout();fig.savefig(figures/'{filename}.png'"
                replacement = (f"fig.suptitle('{title} | L=2, N=12 spins | PBC | h=0.15')\n" + anchor)
                if replacement not in source:
                    source = source.replace(anchor, replacement)
            if 'Field-scan diagnostics | L=2, N=12' not in source:
                source = source.replace(
                    'axes[0].legend();axes[1].legend();fig.tight_layout()',
                    "axes[0].legend();axes[1].legend()\nfig.suptitle('Field-scan diagnostics | L=2, N=12 spins | PBC | h=0.15')\nfig.tight_layout()")
            source = source.replace("fmt='ko-',label='Exact')", "fmt='ko-',capsize=2,label='Exact')")
            source = source.replace("fmt='o--',label=r['kind'])", "fmt='o--',capsize=2,label=r['kind'])")
        cell.source = source
    return cells


def unique_cell(notebook, anchor):
    matches = [cell for cell in notebook.cells if anchor in cell.source]
    if len(matches) != 1:
        raise ValueError(f'Expected one companion cell containing {anchor!r}, found {len(matches)}')
    return deepcopy(matches[0])


def consolidate(notebook, root):
    compact_path = root/'notebook/preliminary_disorder_overlap.ipynb'
    broader_path = root/'notebook/broader_overlap_distributions.ipynb'
    compact = nb.read(compact_path, as_version=4)
    broader = nb.read(broader_path, as_version=4)
    original = []
    for cell in notebook.cells:
        if MARKER in cell.metadata.get('tags', []):
            break
        original.append(deepcopy(cell))
    cells = label_l2_cells(original)
    title, body = cells[0].source.split('\n', 1)
    if SIZE_NOTE not in body:
        cells[0].source = title + '\n\n' + SIZE_NOTE + body.lstrip()
    marker = nb.v4.new_markdown_cell(
        '## Saved disorder averages: sizes, Hamiltonian and provenance\n\n'
        'The following results are frozen measurements from the completed companion reports. '
        'Snapshot dates and ensemble coverage are printed by the loading cells. '
        'They are not a live campaign status.')
    marker.metadata.tags = [MARKER]
    cells.append(marker)
    protocol = unique_cell(compact, '**Hamiltonian and two-body disorder.**')
    protocol.source = '### Shared protocol\n\n' + protocol.source.split('**Hamiltonian and two-body disorder.**', 1)[1]
    protocol.source = '**Sizes: L=2, N=12 and L=4, N=48; periodic boundaries; h=0.15.**\n\n' + protocol.source
    protocol.source = protocol.source.replace('There are two figures: $L=2$ with exact and Monte Carlo comparisons, and $L=4$.',
        'The next two figures show $L=2$ with exact and Monte Carlo comparisons, then $L=4$.')
    cells.append(protocol)
    setup = unique_cell(compact, "guides=load('guide_distributions.json')")
    setup.source = setup.source.replace("figures=folder/'figures'", "figures=root/'figures/simple_overlap'")
    cells.append(setup)
    l2_text = unique_cell(compact, '**$L=2$: all comparisons on the same axes.**')
    l2_text.source = '## L=2, N=12: exact guides, ground state, MCMC and GFMC\n\n' + l2_text.source
    cells.append(l2_text)
    l2_plot = unique_cell(compact, "finish(fig,'disorder_mean_l2_overlay')")
    l2_plot.source = l2_plot.source.replace('L=2, Gamma=', 'L=2, N=12, Gamma=').replace(
        'L=2 disorder averages |', 'L=2, N=12 spins | PBC |')
    cells.append(l2_plot)
    l4_text = unique_cell(compact, '**$L=4$: before and after projection.**')
    l4_text.source = ('## L=4, N=48: before and after projection\n\n'
        '**4 × 4 unit cells, three spins per cell, periodic boundaries in both directions.** '
        'There are 49 allowed overlaps with spacing 1/24.\n\n' + l4_text.source)
    cells.append(l4_text)
    l4_plot = unique_cell(compact, "finish(fig,'disorder_mean_l4_before_after')")
    l4_plot.source = l4_plot.source.replace('L=4, Gamma=', 'L=4, N=48, Gamma=').replace(
        'L=4 disorder averages |', 'L=4, N=48 spins | PBC |')
    cells.append(l4_plot)
    cells.append(unique_cell(compact, '# Integrity checks and a compact numerical audit'))
    cells.append(unique_cell(compact, 'All disorder samples, seeds, starts, repeats and ancestry alarms'))
    for cell in deepcopy(broader.cells):
        source = cell.source
        source = source.replace('# Broader overlap distributions at fixed $h=0.15$',
            '## L=4, N=48: higher-field distributions')
        source = source.replace('The earlier [exact $L=2$ and guide-MCMC comparisons](preliminary_disorder_overlap.html)\nremain available in the companion report.',
            'The exact $L=2$ and guide-MCMC comparisons are included earlier in this notebook.')
        source = source.replace("figures=folder/'figures'", "figures=root/'figures/simple_overlap'")
        source = source.replace("folder/'summary.json'", "figures/'l4_broader_summary.json'")
        source = source.replace('Measured L=4 overlap distributions |', 'Measured L=4, N=48 spins | PBC |')
        source = source.replace('L=4, fixed h=0.15 |', 'L=4, N=48 spins | PBC | h=0.15 |')
        source = source.replace('ax.set(title=names[kind]+', "ax.set(title='L=4, N=48 | '+names[kind]+")
        cell.source = source
        cells.append(cell)
    cells.extend(polarization_cells())
    cells.append(nb.v4.new_markdown_cell(r'''### Interpretation and pending size coverage

These $L=4$ histograms are measured finite-population, finite-forward-time
estimates, not a ground-state convergence certificate. Replay agreement checks
reproducibility, not equilibration or sector exploration. Larger width by itself
does not establish greater glassiness.

The preserved fine scan is $L=4$, $N=48$, $\Gamma=0.55,0.56,\ldots,0.70$.
The original $L=6$, $N=108$ coverage is $0.60,0.65,0.70,0.85,1.00$, supplemented
by the lower-field fine grid. These frozen snapshots contain no complete
fine-grid or $L=6$ histogram ensemble. Pending curves are explicitly marked;
no incomplete disorder ensemble is plotted. Seeds, failures, invalid ratios,
ancestry alarms, raw measurements and checkpoints remain in the source bundles.
'''))
    result = nb.v4.new_notebook(cells=cells, metadata=deepcopy(notebook.metadata))
    result.metadata['consolidated_sources'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [compact_path, broader_path]}
    for cell in result.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Execute on AutoDL and export HTML')
    args = parser.parse_args()
    root = args.root.resolve()
    path = root/'notebook/jastrow_rbm_overlap_distributions.ipynb'
    notebook = consolidate(nb.read(path, as_version=4), root)
    nb.write(notebook, path)
    if args.execute:
        from nbclient import NotebookClient
        from nbconvert import HTMLExporter
        NotebookClient(notebook, timeout=600, kernel_name='python3',
            resources={'metadata': {'path': str(root)}}).execute()
        notebook.metadata['executed_utc'] = datetime.now(timezone.utc).isoformat()
        notebook.metadata['execution_location'] = 'AutoDL'
        nb.validate(notebook)
        nb.write(notebook, path)
        html, _ = HTMLExporter().from_notebook_node(notebook)
        path.with_suffix('.html').write_text(html)
        outputs = [path, path.with_suffix('.html'), Path(__file__).resolve(),
                   Path(__file__).with_name('overlap_polarization.py').resolve()]
        outputs.extend(sorted((root/'figures/simple_overlap').glob('*')))
        manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in outputs if p.is_file()}
        (root/'jastrow_rbm_delivery_sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Executed' if args.execute else 'Edited sources:', path, flush=True)


if __name__ == '__main__':
    main()
