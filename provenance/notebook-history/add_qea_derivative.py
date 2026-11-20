"""Build and validate the derivative notebook locally before publication."""
from pathlib import Path
import base64
import datetime
import hashlib
import json
import os
import shutil

import nbformat
from nbclient import NotebookClient

target = Path('/Users/tianyiyan/Dropbox/spin-glasses/notebook/rydberg_kagome_L4_Pq_progress.ipynb')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
stage = Path.home() / 'ResearchLocal/spin-glasses/staging' / f'rydberg-L4-qea-derivative-{stamp}'
stage.mkdir(exist_ok=False)
shutil.copy2(target, stage / 'before-derivative.ipynb')
original_hash = hashlib.sha256(target.read_bytes()).hexdigest()
nb = nbformat.read(target, as_version=4)
original_sources = [c.source for c in nb.cells]
nb.cells[8].source = r'''## 4. Mean qEA and its logarithmic-drive derivative

The **same figure** shows $Q_L=[q_{\mathrm{EA}}]_{\mathrm{dis}}$ above and
$$R_L(g)=-\frac{dQ_L}{d\ln g}=-g\frac{dQ_L}{dg},\qquad g=\Omega/V_0,$$
below, with one column per detuning and consistent model colors. Positive $R_L$ means qEA decreases with drive. All curves use the same **72 matched geometries**, including both flagged cases.

**Finite differences, no smoothing:** differentiate each geometry on the existing uniformly spaced natural-log grid ($h=\ln(g_{k+1}/g_k)$). Interior values use $-(q_{k+1}-q_{k-1})/(2h)$; the two endpoints use first-order one-sided differences, shown as open circles. Averaging these derivatives is equivalent to differentiating the mean. Peak markers and the table use only interior grid points; no sub-grid fit is made. Grid spacing is approximately 22% in drive, and interior derivatives span two intervals, so peak location/width are grid-limited.

Shading in **both rows** gives **pointwise 95% percentile bootstrap intervals for the disorder mean**, using 2,000 common whole-curve geometry resamples (seed 20261117), shared across models, detunings, and both observables. These are not disorder standard deviations (shown separately in Section 3), simultaneous confidence bands, or bounds on RG, solver, or finite-difference errors. Resampling whole curves preserves correlations between neighboring drive points. The table also reports the central 95% bootstrap range of the *interior grid argmax*; it is discrete and does not establish sub-grid precision.

Red crosses mark flagged qEA points and every derivative stencil that depends on a flagged point. Black-edged model-colored diamonds mark the derivative maxima, with dotted lines aligning their drives to the qEA panels. A derivative peak locates a **finite-size crossover/candidate-transition region**; it does not establish a second-order transition or glass order. qEA includes longitudinal polarization, and the chi12/D128 approximation retains unresolved RG sensitivity.
'''
summary = nb.cells[9].source[nb.cells[9].source.index('# Quantitative snapshot'):]
nb.cells[9].source = r'''log_drive = np.log(drives)
h = float(np.diff(log_drive).mean())
np.testing.assert_allclose(np.diff(log_drive), h, rtol=1e-12, atol=1e-14)
derivative_by_geometry = -np.gradient(qea, h, axis=-1, edge_order=1)
mean_derivative = derivative_by_geometry.mean(axis=0)
np.testing.assert_allclose(mean_derivative,
                           -np.gradient(mean_qea, h, axis=-1, edge_order=1), atol=2e-14)

rng = np.random.default_rng(20261117)
resamples = rng.integers(0, len(sample_ids), size=(2000, len(sample_ids)))
# Bounded batches keep intermediate resampled arrays small.
bootstrap = np.concatenate([qea[batch].mean(axis=1)
                            for batch in np.array_split(resamples, 20)], axis=0)
bootstrap_derivative = np.concatenate([derivative_by_geometry[batch].mean(axis=1)
                                       for batch in np.array_split(resamples, 20)], axis=0)
np.testing.assert_allclose(bootstrap_derivative,
                           -np.gradient(bootstrap, h, axis=-1, edge_order=1), atol=2e-14)
lo, hi = np.quantile(bootstrap, [.025, .975], axis=0)
derivative_lo, derivative_hi = np.quantile(bootstrap_derivative, [.025, .975], axis=0)
peak_indices = np.argmax(mean_derivative[..., 1:-1], axis=-1) + 1
bootstrap_peak_indices = np.argmax(bootstrap_derivative[..., 1:-1], axis=-1) + 1
# Nearest-rank quantiles stay on the sampled drive grid (NumPy-version compatible).
ordered_peak_drives = np.sort(drives[bootstrap_peak_indices], axis=0)
quantile_rows = np.rint(np.array([.025, .975]) * (len(resamples)-1)).astype(int)
peak_drive_lo, peak_drive_hi = ordered_peak_drives[quantile_rows]

# Column k gives the response of every derivative output to input qEA[k].
derivative_operator = -np.gradient(np.eye(len(drives)), h, axis=0, edge_order=1)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex='col', sharey='row',
                         constrained_layout=True, gridspec_kw={'height_ratios': [1, 1.1]})
for d in range(len(deltas)):
    upper, lower = axes[:, d]
    for m in range(len(labels)):
        upper.plot(drives, mean_qea[m,d], color=colors[m], label=labels[m])
        upper.fill_between(drives, lo[m,d], hi[m,d], color=colors[m], alpha=.17, lw=0)
        lower.plot(drives, mean_derivative[m,d], color=colors[m], lw=1.5)
        lower.fill_between(drives, derivative_lo[m,d], derivative_hi[m,d],
                           color=colors[m], alpha=.17, lw=0)
        lower.plot(drives[[0,-1]], mean_derivative[m,d,[0,-1]], 'o',
                   color=colors[m], mfc='white', ms=4)
        k = peak_indices[m,d]
        lower.plot(drives[k], mean_derivative[m,d,k], 'D', color=colors[m],
                   mec='black', mew=.6, ms=5, zorder=5)
        for ax in (upper, lower):
            ax.axvline(drives[k], color=colors[m], ls=':', lw=.8, alpha=.55)
    upper.set_title(f'Δ/V₀ = {deltas[d]:g}')
    upper.set_ylim(0, 1.04)
    lower.axhline(0, color='.5', lw=.6)
    lower.set_xlabel(r'Drive $g=\Omega/V_0$')
    lower.set_xscale('log')
for m,d,k,sid in flag_points:
    axes[0,d].plot(drives[k], mean_qea[m,d,k], 'x', color='red', ms=7, mew=1.5)
    affected = np.flatnonzero(abs(derivative_operator[:,k]) > 1e-14)
    axes[1,d].plot(drives[affected], mean_derivative[m,d,affected], 'x',
                   color='red', ms=7, mew=1.5)
axes[0,0].set_ylabel(r'$Q_L=[q_{\mathrm{EA}}]_{\mathrm{dis}}$')
axes[1,0].set_ylabel(r'$-dQ_L/d\ln g$')
axes[0,1].legend(frameon=False)
fig.suptitle('72 matched geometries · shading: pointwise 95% disorder-bootstrap intervals', fontsize=11)
plt.show()

peak_rows = []
table = ['| Model | Detuning | Peak drive (grid) | 95% bootstrap grid-peak range | Peak height |',
         '|---|---:|---:|---:|---:|']
for m in range(len(labels)):
    for d in range(len(deltas)):
        k = peak_indices[m,d]
        table.append(f'| {labels[m]} | {deltas[d]:g} | {drives[k]:.4g} | '
                     f'{peak_drive_lo[m,d]:.4g}–{peak_drive_hi[m,d]:.4g} | '
                     f'{mean_derivative[m,d,k]:.4f} |')
        peak_rows.append(dict(model=labels[m], delta=float(deltas[d]), index=int(k),
                              drive=float(drives[k]), height=float(mean_derivative[m,d,k]),
                              bootstrap_grid_peak_range=[float(peak_drive_lo[m,d]),
                                                         float(peak_drive_hi[m,d])]))
display(Markdown('\n'.join(table)))
print(f'Log-grid spacing h={h:.6f}; adjacent drives differ by a factor {np.exp(h):.4f}.')
print('Peak ranges quantify disorder resampling only; finite-grid and numerical systematic errors are separate.')

''' + summary
# The raw code string needs one backslash in its newline literal.
nb.cells[9].source = nb.cells[9].source.replace("Markdown('\\\\n'.join(table))", "Markdown('\\n'.join(table))")
os.environ['MPLCONFIGDIR'] = str(stage / 'mpl-cache')
NotebookClient(nb, timeout=180, kernel_name='python3',
               resources={'metadata': {'path': str(stage)}}).execute()
nbformat.validate(nb)
for i, cell in enumerate(nb.cells):
    if i not in (8, 9):
        assert cell.source == original_sources[i]
    assert not any(ord(ch) < 32 and ch not in '\n\t' for ch in cell.source)
    for output in cell.get('outputs', []):
        assert output.output_type != 'error'
        if i == 9 and 'image/png' in output.get('data', {}):
            (stage / 'qea-and-derivative.png').write_bytes(base64.b64decode(output.data['image/png']))
        if i == 9 and 'text/markdown' in output.get('data', {}):
            print(output.data['text/markdown'])
        if output.output_type == 'stream':
            print(output.text)
local = stage / target.name
nbformat.write(nb, local)
(stage / 'publication-ready.json').write_text(json.dumps(dict(
    target=str(target), original_sha256=original_hash,
    notebook_sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
    snapshot_utc=nb.metadata['snapshot_utc'], modified_cells=[8,9], executed=True), indent=2)+'\n')
print('READY_FOR_VISUAL_REVIEW', stage)
