"""Document and recompute the completed forward-time study in the notebook."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
path = ROOT.parents[3]/'notebook'/'gfmc_pqmc_imaginary_time.ipynb'
notebook = nbformat.read(path, as_version=4)
heading = '## 10. Susceptibility and Edwards–Anderson overlap at Gamma/sigma=5'
assert not any(heading in cell.source for cell in notebook.cells)
intro = r'''## 10. Susceptibility and Edwards–Anderson overlap at Gamma/sigma=5

Final production retains the pure moments for every disorder sample $J$:
$$m_{2,J}=\langle q^2\rangle_J,\qquad
\chi_{\mathrm{SG},J}=N m_{2,J},\qquad
q_{\mathrm{EA}}^2=[m_{2,J}],\qquad q_{\mathrm{EA}}=\sqrt{[m_{2,J}]}.$$
The square root is taken **after** averaging over disorder. This follows the convention in `kagome_qea_scan.ipynb`. At finite size it is an RMS overlap proxy: uncoupled spins have $\chi_{\mathrm{SG}}=1$ and $q_{\mathrm{EA}}=1/\sqrt{N}$, so a positive value alone does not establish thermodynamic spin-glass order. Here $N=108$ and the comparison uses one disorder sample; its intervals describe projector sampling uncertainty.

The unchanged flat-guide solver uses 10,000 walkers, 32 equilibration and 32 measurement blocks of duration 0.2, followed by eight tagging/forward cycles. Eight independent populations run at each requested forward time 0.1, 0.2, 0.4 and 0.8. Seeds are disjoint children of `SeedSequence(2026102603).spawn(32)`. The uncertainty unit is the mean of the eight cycles within one population. Adjacent-duration comparisons use independent-population Welch intervals. Every cycle is retained, including any cycle stopped by the ancestry floor.

The following cells verify file hashes, display the actual harness and recompute the results from all 32 archives. They launch no simulations. The energy comparison uses only `energy_measure_trace`; the scalar `energy` also includes accepted forward steps.
'''
code = '''overlap_dir = campaign_dir.parent/'issue9-overlap-highfield-20261026'
overlap_campaign = json.loads((overlap_dir/'campaign.json').read_text())
assert overlap_campaign['source_sha256'] == campaign['source_sha256']
for name, digest in overlap_campaign['study_sha256'].items():
    assert hashlib.sha256((overlap_dir/name).read_bytes()).hexdigest() == digest
overlap_manifest = json.loads((overlap_dir/'download_manifest.json').read_text())
for name, entry in overlap_manifest.items():
    content = (overlap_dir/name).read_bytes()
    assert len(content) == entry['bytes']
    assert hashlib.sha256(content).hexdigest() == entry['sha256']
overlap_saved = json.loads((overlap_dir/'final.json').read_text())
assert overlap_saved['settings'] == overlap_campaign['settings']
harness = (overlap_dir/'overlap_study.py').read_text()
node = next(n for n in ast.parse(harness).body if isinstance(n, ast.FunctionDef) and n.name == 'project')
display(Markdown('```python\\n'+ast.get_source_segment(harness, node)+'\\n```'))
overlap_parts, overlap_chi = {}, []
rows = ['| Forward time | chi SG ± SE | chi SG 95% CI | q EA (RMS) | q EA 95% CI | Truncated / 64 | Min descendant Kish |',
        '|---:|---:|---|---:|---|---:|---:|']
for i, duration in enumerate(overlap_campaign['settings']['forward_times']):
    parts = []
    for child in range(8*i, 8*(i+1)):
        with np.load(overlap_dir/'runs'/f't{i}_p{child}.npz', allow_pickle=False) as archive:
            parts.append({key: archive[key] for key in archive.files})
    overlap_parts[duration] = parts
    second = np.array([part['q2_series'].mean() for part in parts])
    saved = overlap_saved['summaries'][str(duration)]
    population_estimate(second, saved['q2'])
    chi = N*second
    overlap_chi.append(chi)
    mean, se, ci = population_estimate(chi, saved['chi_sg'])
    rms = np.sqrt(second.mean())
    rms_ci = np.sqrt(np.maximum(0, saved['q2']['ci95']))
    np.testing.assert_allclose(rms, saved['q_ea_overlap_rms'])
    rows.append(f"| {duration:g} | {mean:.6f} ± {se:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {rms:.6f} | [{rms_ci[0]:.6f}, {rms_ci[1]:.6f}] | {saved['truncated_cycles']} | {saved['min_effective_ancestors']:.1f} |")
display(Markdown('\\n'.join(rows)))
rows = ['| Forward comparison | chi SG difference | Welch 95% CI |', '|---|---:|---|']
durations = overlap_campaign['settings']['forward_times']
for i in range(1, len(durations)):
    a, b = overlap_chi[i-1:i+1]
    variances = np.array([a.var(ddof=1), b.var(ddof=1)])/8
    df = variances.sum()**2/np.sum(variances**2/7)
    delta = b.mean()-a.mean()
    interval = delta+np.array([-1, 1])*student_t.ppf(.975, df)*np.sqrt(variances.sum())
    name = f'{durations[i]}_minus_{durations[i-1]}'
    np.testing.assert_allclose(interval, overlap_saved['chi_forward_contrasts'][name]['ci95'], atol=1e-10)
    rows.append(f'| {name} | {delta:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] |')
display(Markdown('\\n'.join(rows)))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for i, duration in enumerate(durations):
    chi = overlap_chi[i]
    axes[0].scatter(np.full(8, duration), chi, s=15, alpha=.5)
    axes[0].errorbar(duration, chi.mean(), yerr=chi.std(ddof=1)/np.sqrt(8), color='black', marker='o', capsize=4)
    for part in overlap_parts[duration]:
        axes[1].scatter(part['forward_times'], part['cycle_effective'], s=10, alpha=.4)
axes[0].axhline(1, ls='--', color='gray')
axes[0].set(xlabel='Requested forward time', ylabel='Spin-glass susceptibility', title='Population means and mean ± SE')
axes[1].axhline(8, ls='--', color='gray')
axes[1].set(xlabel='Achieved forward time', ylabel='Descendant Kish size', yscale='log', title='All 256 forward cycles')
for ax in axes:
    ax.grid(alpha=.2)
plt.show()
'''
notebook.cells.extend([nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(code)])
for cell in notebook.cells:
    cell.source = cell.source.replace(
        '**Current evidence:** extending through time 4096 leaves large population offsets and all late means above a valid variational upper bound. The matched-time walker comparison through 512 is inconclusive. Ground-state energy and susceptibility convergence remain unestablished.',
        '**Current evidence:** the weak-field run through time 4096 leaves population offsets above a valid variational upper bound. At Gamma/sigma=5, energy meets the tested stability criteria. The next section measures pure-overlap forward-time dependence and sampling precision for susceptibility and the finite-size q EA proxy.')
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(path)
