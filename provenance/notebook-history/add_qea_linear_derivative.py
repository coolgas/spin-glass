from pathlib import Path
import base64, datetime, hashlib, json, os, shutil
import nbformat
from nbclient import NotebookClient

target = Path('/Users/tianyiyan/Dropbox/spin-glasses/notebook/rydberg_kagome_L4_Pq_progress.ipynb')
root = Path.home() / 'ResearchLocal/spin-glasses/staging'
assert root.resolve() == root
assert shutil.disk_usage(root).free > 100_000_000
stage = root / ('rydberg-L4-both-derivatives-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
stage.mkdir()
shutil.copy2(target, stage / 'before-update.ipynb')
original_hash = hashlib.sha256(target.read_bytes()).hexdigest()
nb = nbformat.read(target, as_version=4)
sources = [c.source for c in nb.cells]
nb.cells[8].source = r'''## 4. Mean qEA and both drive derivatives

The same figure compares three quantities, with one column per detuning:

1. $Q_L=[q_{\mathrm{EA}}]_{\mathrm{dis}}$;
2. $R_{\ln g}=-dQ_L/d\ln g$, the response to a fractional increase in drive;
3. $R_g=-dQ_L/dg$, the response to an absolute increase in drive, where $g=\Omega/V_0$.

All horizontal axes are logarithmic; all vertical axes are linear. **No logarithm of the derivative is taken.** In the continuum $R_{\ln g}=gR_g$, so their maxima generally differ.

**Finite differences, no smoothing:** differentiate each geometry first, then average the same 72 matched geometries. The logarithmic derivative uses centered differences on the uniformly spaced natural-log grid. The linear-drive derivative uses `np.gradient(qea, drives, axis=-1, edge_order=1)`: interior three-point, second-order differences with the actual unequal drive spacings. Both use first-order one-sided differences at endpoints (open circles). These independently discretized derivatives obey the chain rule only up to finite-grid error. Adjacent drives differ by about 22%; peak markers and tables use interior grid maxima without sub-grid fitting.

Shading in all rows gives **pointwise 95% percentile bootstrap intervals for the disorder mean**: 2,000 common whole-curve geometry resamples, seed 20261117, shared across models, detunings, and observables. Whole-curve resampling preserves drive correlations. These are distinct from disorder standard deviations in Section 3 and exclude RG, solver, and finite-difference systematic errors. The tables give central 95% bootstrap ranges of the discrete interior peak drive, not sub-grid confidence limits.

**Diamonds:** derivative maxima. **Dotted lines:** the corresponding peak drive within each derivative row. **Red crosses:** flagged qEA inputs and every derivative stencil that uses one; the nonuniform linear-drive stencil can also depend on its central input. Both flagged geometries remain included. The stronger response locates a finite-size crossover/candidate-transition region, not proof of a second-order transition or glass order. qEA contains longitudinal polarization; RG sensitivity at chi12/D128 remains unresolved.
'''
source = nb.cells[9].source
prefix = source[:source.index('# Column k gives')]
summary = source[source.index('# Quantitative snapshot'):]
nb.cells[9].source = prefix + r'''
# Direct derivative on the nonuniform linear-drive grid.
linear_by_geometry = -np.gradient(qea, drives, axis=-1, edge_order=1)
mean_linear = linear_by_geometry.mean(axis=0)
bootstrap_linear = np.concatenate([linear_by_geometry[batch].mean(axis=1)
                                  for batch in np.array_split(resamples, 20)], axis=0)
np.testing.assert_allclose(mean_linear, -np.gradient(mean_qea, drives, axis=-1), atol=1e-13)
np.testing.assert_allclose(bootstrap_linear, -np.gradient(bootstrap, drives, axis=-1), atol=1e-13)
linear_lo, linear_hi = np.quantile(bootstrap_linear, [.025, .975], axis=0)
linear_peaks = np.argmax(mean_linear[..., 1:-1], axis=-1) + 1
linear_bootstrap_peaks = np.argmax(bootstrap_linear[..., 1:-1], axis=-1) + 1
linear_peak_lo, linear_peak_hi = np.sort(drives[linear_bootstrap_peaks], axis=0)[quantile_rows]

diagnostics = [
    dict(name='Logarithmic-drive derivative', mean=mean_derivative,
         low=derivative_lo, high=derivative_hi, peaks=peak_indices,
         peak_low=peak_drive_lo, peak_high=peak_drive_hi,
         operator=-np.gradient(np.eye(len(drives)), h, axis=0),
         ylabel=r'$-dQ_L/d\ln g$'),
    dict(name='Linear-drive derivative', mean=mean_linear,
         low=linear_lo, high=linear_hi, peaks=linear_peaks,
         peak_low=linear_peak_lo, peak_high=linear_peak_hi,
         operator=-np.gradient(np.eye(len(drives)), drives, axis=0),
         ylabel=r'$-dQ_L/dg$')]

fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.5), sharex='col', sharey='row',
                         constrained_layout=True)
for d in range(len(deltas)):
    upper = axes[0,d]
    for m in range(len(labels)):
        upper.plot(drives, mean_qea[m,d], color=colors[m], label=labels[m])
        upper.fill_between(drives, lo[m,d], hi[m,d], color=colors[m], alpha=.17, lw=0)
    upper.set_title(f'Δ/V₀ = {deltas[d]:g}')
    upper.set_ylim(0, 1.04)
    for row, diagnostic in enumerate(diagnostics, start=1):
        ax = axes[row,d]
        for m in range(len(labels)):
            values = diagnostic['mean'][m,d]
            ax.plot(drives, values, color=colors[m], lw=1.5)
            ax.fill_between(drives, diagnostic['low'][m,d], diagnostic['high'][m,d],
                            color=colors[m], alpha=.17, lw=0)
            ax.plot(drives[[0,-1]], values[[0,-1]], 'o', color=colors[m], mfc='white', ms=4)
            k = diagnostic['peaks'][m,d]
            ax.plot(drives[k], values[k], 'D', color=colors[m], mec='black', mew=.6, ms=5, zorder=5)
            ax.axvline(drives[k], color=colors[m], ls=':', lw=.8, alpha=.55)
        ax.axhline(0, color='.5', lw=.6)
    axes[-1,d].set_xscale('log')
    axes[-1,d].set_xlabel(r'Drive $g=\Omega/V_0$')
for m,d,k,sid in flag_points:
    axes[0,d].plot(drives[k], mean_qea[m,d,k], 'x', color='red', ms=7, mew=1.5)
    for row, diagnostic in enumerate(diagnostics, start=1):
        affected = np.flatnonzero(abs(diagnostic['operator'][:,k]) > 1e-14)
        axes[row,d].plot(drives[affected], diagnostic['mean'][m,d,affected],
                         'x', color='red', ms=7, mew=1.5)
axes[0,0].set_ylabel(r'$Q_L=[q_{\mathrm{EA}}]_{\mathrm{dis}}$')
for row, diagnostic in enumerate(diagnostics, start=1):
    axes[row,0].set_ylabel(diagnostic['ylabel'])
axes[0,1].legend(frameon=False)
fig.suptitle(f'{len(sample_ids)} matched geometries · shading: pointwise 95% disorder-bootstrap intervals', fontsize=11)
plt.show()

for diagnostic in diagnostics:
    table = [f"### {diagnostic['name']}", '',
             '| Model | Detuning | Peak drive (grid) | 95% bootstrap grid-peak range | Peak height |',
             '|---|---:|---:|---:|---:|']
    for m in range(len(labels)):
        for d in range(len(deltas)):
            k = diagnostic['peaks'][m,d]
            table.append(f'| {labels[m]} | {deltas[d]:g} | {drives[k]:.4g} | '
                         f"{diagnostic['peak_low'][m,d]:.4g}–{diagnostic['peak_high'][m,d]:.4g} | "
                         f"{diagnostic['mean'][m,d,k]:.4f} |")
    display(Markdown('\n'.join(table)))
print(f'Adjacent drives differ by a factor {np.exp(h):.4f}; grid-peak locations have finite resolution.')
print('Bootstrap ranges describe disorder resampling only, not numerical systematic errors.')

''' + summary
nb.cells[9].source = nb.cells[9].source.replace("Markdown('\\\\n'.join(table))", "Markdown('\\n'.join(table))")
os.environ['MPLCONFIGDIR'] = str(stage / 'mpl-cache')
NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(stage)}}).execute()
nbformat.validate(nb)
for i, cell in enumerate(nb.cells):
    if i not in (8,9): assert cell.source == sources[i]
    for output in cell.get('outputs', []):
        assert output.output_type != 'error'
        if i == 9 and 'image/png' in output.get('data', {}):
            (stage / 'both-derivatives.png').write_bytes(base64.b64decode(output.data['image/png']))
        if i == 9 and 'text/markdown' in output.get('data', {}): print(output.data['text/markdown'])
local = stage / target.name
nbformat.write(nb, local)
(stage / 'publication-ready.json').write_text(json.dumps(dict(target=str(target), original_sha256=original_hash,
    notebook_sha256=hashlib.sha256(local.read_bytes()).hexdigest(), executed=True), indent=2)+'\n')
print('REVIEW', stage)
