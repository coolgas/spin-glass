from pathlib import Path
import base64,json,os
import nbformat
from nbclient import NotebookClient

stage=Path('/Users/tianyiyan/ResearchLocal/spin-glasses/staging/rydberg-L4-pq-preview-20261117T124125Z')
root=Path('/Users/tianyiyan/Dropbox/spin-glasses')
target=root/'rydberg_kagome_L4_Pq_progress.ipynb'
assert not target.exists(), 'Preserve an existing notebook rather than overwrite it'
os.environ['MPLCONFIGDIR']=str(stage/'mpl-cache')
meta=json.load(open(stage/'metadata.json'))
payload=base64.b64encode((stage/'histograms.npz').read_bytes()).decode()
cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):cells.append(nbformat.v4.new_code_cell(s))
md('''# Rydberg Kagome L4: preliminary P(q) distributions

**Snapshot: 17 November 2026, 12:41 UTC.** These plots use the same **29 completed geometries (IDs 0–28)** for all three models, both detunings and all 30 drives: 5,220 cases. The intended ensemble has 128 geometries; these are preliminary results at **chi_RG=12, D_MPS=128**, not RG-converged predictions.

Each histogram contains 20,000 independent replica pairs from the lower-energy of two optimized states. P(q) below is **probability mass per allowed overlap**, averaged equally over geometries, without smoothing or symmetry folding. At N=48 there are 49 overlaps spaced by 1/24. The physical identity is

$$\langle q\rangle_{P_s}=q_{\mathrm{EA},s}=\frac1{48}\sum_i\langle Z_i\rangle_s^2.$$

**One flagged case is included:** geometry 25, NN, Delta/V0=2, drive=10. One start failed the sweep-stability criterion at 120 sweeps; the two final energies and qEA agree closely and sampling passes. Its affected point is marked below. Keeping it avoids a passing-only disorder selection.

All plotting data are embedded in this notebook. Re-execution requires only NumPy and Matplotlib, with no cluster access or checkpoint downloads.''')
code('''import base64, hashlib, io, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from IPython.display import display, Markdown

METADATA = json.loads('''+repr(json.dumps(meta))+''')
PAYLOAD = '''+repr(payload)+'''
raw = base64.b64decode(PAYLOAD)
assert hashlib.sha256(raw).hexdigest() == METADATA['files']['histograms.npz']
with np.load(io.BytesIO(raw), allow_pickle=False) as data:
    arrays = {key: data[key].copy() for key in data.files}
counts, qea, passed = arrays['counts'], arrays['qea'], arrays['passed']
q, drives, deltas = arrays['q'], arrays['drives'], arrays['deltas']
sample_ids = arrays['sample_ids']
# Axes: geometry, model, detuning, drive, overlap (last axis for histograms).
p = counts / counts.sum(axis=-1, keepdims=True)
mean_p = p.mean(axis=0)
mean_qea = qea.mean(axis=0)
assert counts.shape == (29, 3, 2, 30, 49)
assert np.all(counts.sum(axis=-1) == 20000)
np.testing.assert_allclose(mean_p.sum(axis=-1), 1)
labels = ['Full', 'NN+NNN', 'NN']
colors = ['#24578a', '#cc7924', '#26856c']
plt.rcParams.update({'font.size': 10, 'figure.dpi': 140})
print(f'{len(sample_ids)} geometries; {passed.size} cases; {(~passed).sum()} flagged case retained.')
''')
md('''## 1. Evolution of P(q) across the drive range

Rows correspond to detuning; columns to interaction model. A common logarithmic color scale shows the probability mass in each overlap bin. Zero-probability bins are white; positive masses below 1e-4 use the bottom color. The dashed black line is the deterministic disorder-averaged qEA, not a fitted peak position. The red cross identifies the drive containing the flagged case.''')
code('''fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True, constrained_layout=True)
log_g = np.log(drives)
g_edges = np.exp(np.r_[log_g[0]-(log_g[1]-log_g[0])/2,
                       (log_g[:-1]+log_g[1:])/2,
                       log_g[-1]+(log_g[-1]-log_g[-2])/2])
dq = q[1]-q[0]
q_edges = np.r_[q-dq/2, q[-1]+dq/2]
cmap = plt.get_cmap('viridis').copy()
cmap.set_bad('white')
for d in range(2):
    for m in range(3):
        ax = axes[d,m]
        image = ax.pcolormesh(g_edges, q_edges, np.ma.masked_equal(mean_p[m,d].T, 0),
                             cmap=cmap, norm=LogNorm(vmin=1e-4, vmax=1), shading='flat')
        ax.plot(drives, mean_qea[m,d], 'k--', lw=1.2)
        ax.set_xscale('log')
        ax.set_ylim(-1-dq/2, 1+dq/2)
        ax.text(.03,.04, f'{labels[m]}, Δ/V₀ = {deltas[d]:g}', transform=ax.transAxes,
                color='black', bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        if m==0: ax.set_ylabel('Overlap q')
        if d==1: ax.set_xlabel('Drive Ω/V₀')
axes[0,2].plot(10, mean_qea[2,0,-1], 'x', color='red', ms=8, mew=2)
fig.colorbar(image, ax=axes, label='Disorder-averaged probability mass', shrink=.9)
plt.show()
''')
md('''## 2. Direct comparison at selected drives

Each panel uses the nearest production-grid point to the indicated representative scale (0.03, 0.3, 1, 10); annotations give the actual drive. Lines connect discrete masses and do not represent smoothing. All curves use the same 29 geometries. The high-drive NN curve at detuning 2 includes the flagged case. These panels display shapes; their model differences have not yet been established as RG-converged.''')
code('''indices = [int(np.argmin(abs(np.log(drives/x)))) for x in [.03,.3,1.,10.]]
fig, axes = plt.subplots(2, 4, figsize=(12, 5.8), sharex=True, constrained_layout=True)
for d in range(2):
    for col,k in enumerate(indices):
        ax = axes[d,col]
        for m in range(3):
            ax.plot(q, mean_p[m,d,k], color=colors[m], lw=1.5, label=labels[m])
        ax.text(.04,.96, f'Δ/V₀={deltas[d]:g}\\nΩ/V₀={drives[k]:.4g}',
                transform=ax.transAxes, va='top', fontsize=9)
        ax.set_xlim(-1.03,1.03)
        ax.set_ylim(bottom=0)
        if col==0: ax.set_ylabel('Probability mass')
        if d==1: ax.set_xlabel('Overlap q')
axes[0,0].legend(loc='center left', fontsize=8, frameon=False)
plt.show()
''')
md('''## 3. Mean overlap and disorder uncertainty

Lines show deterministic qEA averaged over the 29 geometries. Shading gives **pointwise 95% percentile bootstrap intervals for the disorder mean** (2,000 resamples); the same geometry resamples are shared across models, detunings and whole drive curves. These intervals do not quantify RG error, are not simultaneous confidence bands, and do not include the unresolved solver effect. The red cross marks the affected high-drive point. The signed overlap is retained: nonzero qEA alone is not a glass-phase diagnosis.''')
code('''rng = np.random.default_rng(20261117)
resamples = rng.integers(0,len(sample_ids),size=(2000,len(sample_ids)))
bootstrap = qea[resamples].mean(axis=1)
lo, hi = np.quantile(bootstrap,[.025,.975],axis=0)
fig, axes = plt.subplots(1,2,figsize=(9,3.5),sharey=True,constrained_layout=True)
for d,ax in enumerate(axes):
    for m in range(3):
        ax.plot(drives, mean_qea[m,d], color=colors[m], label=labels[m])
        ax.fill_between(drives,lo[m,d],hi[m,d],color=colors[m],alpha=.16,lw=0)
    ax.set_xscale('log')
    ax.set_xlabel('Drive Ω/V₀')
    ax.text(.03,.04,f'Δ/V₀={deltas[d]:g}',transform=ax.transAxes)
    ax.set_ylim(0,1.04)
axes[0].set_ylabel('Disorder-averaged qEA')
axes[0].plot(10,mean_qea[2,0,-1],'x',color='red',ms=8,mew=2)
axes[1].legend(frameon=False)
plt.show()

# Quantitative snapshot and sensitivity to the single flagged geometry.
table=['| Model | Detuning | qEA at lowest drive | qEA at highest drive |',
       '|---|---:|---:|---:|']
for m in range(3):
    for d in range(2):
        table.append(f'| {labels[m]} | {deltas[d]:g} | {mean_qea[m,d,0]:.5f} | {mean_qea[m,d,-1]:.5f} |')
display(Markdown('\\n'.join(table)))
keep = sample_ids != 25
without25 = p[keep].mean(axis=0)
cdf_sensitivity = np.max(abs(np.cumsum(mean_p[2,0,-1])-np.cumsum(without25[2,0,-1])))
print(f'High-drive NN Δ=2: omitting geometry 25 changes the mean CDF by {cdf_sensitivity:.6f}.')
print('This omission sensitivity changes the cohort and is not a correction or a bound on solver error.')
''')
md('''## Provenance and interpretation

- Source: `rydberg-L4-production-20261117-v5`, source release `rydberg-kagome-20261117-v5`.
- Snapshot archive SHA-256: `660113ea84c5354b972160b752beaef035a90fb8805604c50af724f332c0d675`.
- Histogram payload SHA-256 is verified in the loading cell; source case/histogram hashes are retained in the locally verified snapshot manifest.
- Archive, manifest and transfer record: `~/ResearchLocal/spin-glasses/staging/rydberg-L4-pq-preview-20261117T124125Z/`.
- Each remote histogram was checked against its result-record hash, normalized probabilities and 20,000-pair count before export. This snapshot contains reduced observables, not wavefunction checkpoints.
- Use the common completed cohort for these preliminary comparisons. The final target remains all 128 geometries. Sampling uncertainty, disorder uncertainty, solver checks and RG sensitivity are distinct.

This is a frozen snapshot. Opening or executing the notebook does not refresh data, contact AutoDL, or modify the production run.''')
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'snapshot_utc':meta['snapshot_utc']})
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(stage)}}).execute()
nbformat.validate(nb)
nbformat.write(nb,stage/target.name)
nbformat.write(nb,target)
index=0
for c in nb.cells:
 for output in c.get('outputs',[]):
  if 'image/png' in output.get('data',{}):
   index+=1;(stage/f'plot-{index}.png').write_bytes(base64.b64decode(output['data']['image/png']))
  if output.output_type=='stream':print(output.text)
  if 'text/markdown' in output.get('data',{}):print(output['data']['text/markdown'])
print(target, target.stat().st_size, 'bytes;',index,'embedded figures')
