"""Add and validate overlap Binder analysis using only embedded local data."""
from pathlib import Path
import ast
import base64
import datetime
import hashlib
import io
import json
import os
import shutil

import nbformat
import numpy as np
from nbclient import NotebookClient

target = Path('/Users/tianyiyan/Dropbox/spin-glasses/notebook/rydberg_kagome_L4_Pq_progress.ipynb')
root = Path('/Users/tianyiyan/ResearchLocal/spin-glasses/staging')
assert root.resolve() == root
assert shutil.disk_usage(root).free > 100_000_000
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
stage = root / f'rydberg-L4-binder-{stamp}'
stage.mkdir(exist_ok=False)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
original_sha = sha(target)
shutil.copy2(target, stage / 'before-binder.ipynb')
nb = nbformat.read(stage / 'before-binder.ipynb', as_version=4)
assert len(nb.cells) == 11
assert nb.metadata['production_ensemble_complete']
original_sources = [cell.source for cell in nb.cells]
payload_line = next(line for line in nb.cells[1].source.splitlines() if line.startswith('PAYLOAD = '))
payload = base64.b64decode(ast.literal_eval(payload_line.split(' = ', 1)[1]))
payload_sha = hashlib.sha256(payload).hexdigest()

markdown = r'''## 5. Raw overlap Binder cumulant at L=4

For each geometry $s$, recover the **sampled** overlap moments directly from its full, signed histogram:

$$m_{k,s}(g)=\sum_q P_s(q;g)q^k,\qquad
B_L(g)=\frac12\left(3-\frac{[m_{4,s}(g)]_{\rm dis}}
{[m_{2,s}(g)]_{\rm dis}^{\,2}}\right),\qquad L=4\;(N=48).$$

The brackets give equal weight to all **128 matched geometries**. This is the **ratio of disorder-averaged moments**, not the disorder average of per-geometry ratios. Both numerator and denominator come from the same saved 20,000-pair histograms; the deterministic `qea` and `q2` arrays are not substituted into this estimator. No centering, folding or smoothing is applied. The alternative normalization $U=1-[m_4]/(3[m_2]^2)$ satisfies $B_L=3U/2$.

One panel per detuning compares Full, NN+NNN and NN at all 30 production drives. Shading gives **pointwise 95% percentile disorder-bootstrap intervals**, recomputing the complete nonlinear ratio in each of 2,000 whole-curve geometry resamples (seed 20261117; the same resamples as Section 4, shared across models and detunings). These are intervals for the disorder estimator, not geometry-to-geometry standard deviations or simultaneous bands. They do not separately quantify finite Born-sampling error or bound solver/RG systematic errors. All geometries remain included; **red crosses** mark curve points containing one or more of the six flagged cases.

**Interpretation:** longitudinal polarization gives $\langle q\rangle_s=q_{{\rm EA},s}>0$ even without glass order. A narrow distribution at any nonzero overlap has $B_L\to1$, so a large raw Binder alone is not evidence of a glass. These six curves characterize one finite size; they cannot provide a crossing between sizes or establish a critical drive. RG sensitivity at chi12/D128 and the six solver flags remain unresolved.

For quantum overlap moments and an alternative disorder-averaging convention, see [Mukherjee, Rajak and Chakrabarti, PRE 92, 042107 (2015), Eqs. (4) and (6)](https://arxiv.org/html/1412.2973v2). For the additional issues with Binder diagnostics in a field, see [Feng et al., arXiv:1403.4560, Eq. (3)](https://arxiv.org/html/1403.4560v1); their centered definition differs from the raw definition plotted here.
'''

code = r'''# Moments of the physical two-replica overlap, estimated from saved histograms.
binder_m2 = np.einsum('smdkq,q->smdk', p, q**2)
binder_m4 = np.einsum('smdkq,q->smdk', p, q**4)
assert np.all(binder_m2 > 0)
assert np.all(binder_m4 >= binder_m2**2 - 1e-14)

def overlap_binder(mean_second, mean_fourth):
    return 0.5 * (3.0 - mean_fourth / mean_second**2)

binder = overlap_binder(binder_m2.mean(axis=0), binder_m4.mean(axis=0))
# Each row resamples complete geometry curves, keeping all comparisons paired.
binder_seed = 20261117
binder_resamples = np.random.default_rng(binder_seed).integers(
    0, len(sample_ids), size=(2000, len(sample_ids)))
np.testing.assert_array_equal(binder_resamples, resamples)
binder_bootstrap = np.concatenate([
    overlap_binder(binder_m2[batch].mean(axis=1), binder_m4[batch].mean(axis=1))
    for batch in np.array_split(binder_resamples, 20)
], axis=0)
binder_lo, binder_hi = np.quantile(binder_bootstrap, [.025, .975], axis=0)
assert binder.shape == (3, 2, 30)
assert np.isfinite(binder).all() and np.isfinite(binder_bootstrap).all()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True,
                         constrained_layout=True)
for d, ax in enumerate(axes):
    for m in range(len(labels)):
        ax.plot(drives, binder[m,d], color=colors[m], lw=1.5,
                marker='.', ms=3, label=labels[m])
        ax.fill_between(drives, binder_lo[m,d], binder_hi[m,d],
                        color=colors[m], alpha=.17, lw=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'Drive $g=\Omega/V_0$')
    ax.set_title(f'Δ/V₀ = {deltas[d]:g}')
    ax.axhline(0, color='.5', lw=.7, ls=':')
    ax.grid(axis='y', alpha=.15)
for m, d, k in sorted({(m, d, k) for m, d, k, sid in flag_points}):
    axes[d].plot(drives[k], binder[m,d,k], 'x', color='red', ms=7,
                 mew=1.5, zorder=5)
axes[0].set_ylabel(r'Raw overlap Binder $B_{L=4}$')
handles, legend_labels = axes[1].get_legend_handles_labels()
handles.append(Line2D([], [], color='red', marker='x', linestyle='none',
                      markersize=7, markeredgewidth=1.5))
axes[1].legend(handles, legend_labels + ['Contains flagged case'],
               frameon=False, fontsize=9, loc='lower left')
fig.suptitle('L=4 (48 spins) · 128 matched geometries · 95% disorder-bootstrap intervals',
             fontsize=11)
plt.show()
print('Ratio of disorder-averaged histogram moments; all 128 geometries retained.')
print('Bootstrap intervals exclude solver/RG systematics; Born-sampling error is not separately estimated.')
'''

nb.cells[10:10] = [nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(code)]
nb.metadata['binder_analysis'] = dict(
    definition='0.5*(3-mean_disorder(m4)/mean_disorder(m2)**2)',
    moments='sampled signed-overlap histograms', centered=False,
    geometry_count=128, bootstrap_resamples=2000, bootstrap_seed=20261117,
    uncertainty='pointwise 95% percentile whole-geometry bootstrap',
    payload_sha256=payload_sha, created_utc=stamp)
for i in range(10):
    assert nb.cells[i].source == original_sources[i]
assert nb.cells[12].source == original_sources[10]

# Independent formula checks: deterministic overlap and exactly enumerated
# unbiased independent-spin replicas have known finite-size Binder values.
from math import comb
n = 48
q_ref = 1 - 2 * np.arange(n + 1) / n
p_ref = np.array([comb(n, k) for k in range(n + 1)], dtype=float) / 2**n
m2_ref, m4_ref = p_ref @ q_ref**2, p_ref @ q_ref**4
b_ref = .5 * (3 - m4_ref / m2_ref**2)
np.testing.assert_allclose(b_ref, 1/n, atol=2e-15)
np.testing.assert_allclose(.5 * (3 - .5**4 / (.5**2)**2), 1.)

os.environ['MPLCONFIGDIR'] = str(stage / 'mpl-cache')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
NotebookClient(nb, timeout=180, kernel_name='python3',
               resources={'metadata': {'path': str(stage)}}).execute()
nbformat.validate(nb)
figures = []
for i, cell in enumerate(nb.cells):
    for output in cell.get('outputs', []):
        assert output.output_type != 'error'
        if 'image/png' in output.get('data', {}):
            figures.append(i)
            if i == 11:
                (stage / 'binder.png').write_bytes(base64.b64decode(output.data['image/png']))
assert len(figures) == 5 and figures[-1] == 11

# Verify the actual analysis cell numerically against independent pooled
# histogram reduction and a direct first-bootstrap calculation.
scope = {'np': np}
with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
    scope['p'] = arrays['counts'] / arrays['counts'].sum(axis=-1, keepdims=True)
    scope['q'] = arrays['q']
    scope['sample_ids'] = arrays['sample_ids']
scope['resamples'] = np.random.default_rng(20261117).integers(0, 128, (2000, 128))
exec(code[:code.index('fig, axes =')], scope)
pooled = scope['p'].mean(axis=0)
direct = .5 * (3 - (pooled @ scope['q']**4) / (pooled @ scope['q']**2)**2)
np.testing.assert_allclose(scope['binder'], direct, atol=2e-14)
first = scope['p'][scope['resamples'][0]].mean(axis=0)
direct_first = .5 * (3 - (first @ scope['q']**4) / (first @ scope['q']**2)**2)
np.testing.assert_allclose(scope['binder_bootstrap'][0], direct_first, atol=2e-14)
np.savez_compressed(stage / 'binder-analysis.npz',
                    binder=scope['binder'], low=scope['binder_lo'], high=scope['binder_hi'],
                    m2=scope['binder_m2'], m4=scope['binder_m4'])
candidate = stage / target.name
nbformat.write(nb, candidate)
readback = nbformat.read(candidate, as_version=4)
nbformat.validate(readback)
assert readback.cells[1].source == original_sources[1]
assert readback.metadata['snapshot_utc'] == nb.metadata['snapshot_utc']
record = dict(target=str(target), candidate=str(candidate), stage=str(stage),
              original_sha256=original_sha, candidate_sha256=sha(candidate),
              embedded_payload_sha256=payload_sha, verified=True, figures=5,
              independent_spin_baseline=b_ref,
              binder_min=float(scope['binder'].min()), binder_max=float(scope['binder'].max()),
              original_cell_sources_preserved=True, local_only=True,
              original_backup=str(stage / 'before-binder.ipynb'))
(stage / 'publication-ready.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
