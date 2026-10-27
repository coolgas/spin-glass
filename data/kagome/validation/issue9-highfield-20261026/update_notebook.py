"""Insert the completed stronger-field evidence into the implementation notebook."""
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
notebook_path = ROOT.parents[3]/'notebook'/'gfmc_pqmc_imaginary_time.ipynb'
saved = json.loads((ROOT/'final.json').read_text())
notebook = nbformat.read(notebook_path, as_version=4)
heading = '## 8. Stronger transverse fields on the same 6×6 sample'
assert not any(heading in c.source for c in notebook.cells)
intro = r'''## 8. Stronger transverse fields on the same 6×6 sample

The weak-field studies used $\Gamma=0.05$ and $\sigma=1$. Following the suggestion to investigate the paramagnetic regime, the next screen uses **$\Gamma/\sigma=3$ and $5$**, with the same disorder sample and the same flat-guide projector. These are candidate paramagnetic controls; the field values alone do not establish the phase.

For each field, eight independent populations at each of 5,000 and 10,000 walkers run 64 blocks of duration 0.2, reaching $\tau=12.8$. All initial transients remain in the archive and no forward cycles run. The late comparison is $\tau=6.4$–9.6 versus 9.6–12.8. Disjoint children of `SeedSequence(20261026).spawn(32)` supply the populations. The walker contrast uses an independent-sample Welch interval.

The code below verifies the downloaded artifacts, displays the exact frozen harness call, and recomputes population statistics from the saved traces. It does not launch simulations. The all-X product state gives the field-specific upper bound $E_0\leq -108\Gamma$; the saved low-field product state is also reevaluated at each new field. The second-order high-field expression $-108\Gamma-\sum_b J_b^2/(4\Gamma)$ is an approximation, not a bound.
'''
code = '''high_dir = campaign_dir.parent / 'issue9-highfield-20261026'
high_campaign = json.loads((high_dir / 'campaign.json').read_text())
assert high_campaign['source_sha256'] == campaign['source_sha256']
assert high_campaign['solver_patch_sha256'] == campaign['solver_patch_sha256']
for name, digest in high_campaign['study_sha256'].items():
    assert hashlib.sha256((high_dir/name).read_bytes()).hexdigest() == digest
high_manifest = json.loads((high_dir/'download_manifest.json').read_text())
for name, entry in high_manifest.items():
    content = (high_dir/name).read_bytes()
    assert len(content) == entry['bytes']
    assert hashlib.sha256(content).hexdigest() == entry['sha256']
high_saved = json.loads((high_dir/'final.json').read_text())
assert high_saved['settings'] == high_campaign['settings']
assert not high_saved['production_allowed'] and not high_saved['phase_established']
high_harness = (high_dir/'field_study.py').read_text()
project_node = next(node for node in ast.parse(high_harness).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'project')
display(Markdown('```python\\n'+ast.get_source_segment(high_harness, project_node)+'\\n```'))
high_matrices = {}
rows = ['| Gamma/sigma | Walkers | Late energy | SE | Late drift | Drift 95% interval |',
        '|---|---:|---:|---:|---:|---|']
for field_index, field in enumerate((3.0, 5.0)):
    for walker_index, walkers in enumerate((5000, 10000)):
        traces_high = []
        for child in range(8*(2*field_index+walker_index), 8*(2*field_index+walker_index+1)):
            with np.load(high_dir/'runs'/f'g{field:g}_m{walkers}_p{child}.npz', allow_pickle=False) as archive:
                np.testing.assert_allclose(archive['energy_measure_time'], np.arange(1,65)*0.2)
                traces_high.append(archive['energy_measure_trace'].copy())
        matrix = np.stack(traces_high)
        high_matrices[field, walkers] = matrix
        row = high_saved['summaries'][f'g{field:g}_m{walkers}']
        mean, se, ci = population_estimate(matrix[:,48:].mean(axis=1), row['late_energy'])
        dm, ds, dc = population_estimate(matrix[:,48:].mean(axis=1)-matrix[:,32:48].mean(axis=1), row['late_drift'])
        rows.append(f'| {field:g} | {walkers:,} | {mean:.6f} | {se:.6f} | {dm:.6f} | [{dc[0]:.6f}, {dc[1]:.6f}] |')
display(Markdown('\\n'.join(rows)))
rows = ['| Gamma/sigma | 10,000 minus 5,000 | Welch 95% interval | Within ±0.108? |', '|---|---:|---|---|']
for field in (3.0, 5.0):
    a, b = [high_matrices[field, walkers][:,48:].mean(axis=1) for walkers in (5000,10000)]
    variance = np.array([a.var(ddof=1), b.var(ddof=1)])/8
    df = variance.sum()**2 / np.sum(variance**2/7)
    mean = b.mean()-a.mean()
    ci = mean + np.array([-1,1])*student_t.ppf(.975, df)*np.sqrt(variance.sum())
    np.testing.assert_allclose(ci, high_saved['walker_contrasts'][str(field)]['ci95'])
    rows.append(f'| {field:g} | {mean:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {max(abs(ci)) <= .108} |')
display(Markdown('\\n'.join(rows)))
fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for i, field in enumerate((3.0,5.0)):
    for j, walkers in enumerate((5000,10000)):
        ax = axes[i,j]
        matrix = high_matrices[field,walkers]/N
        for trace in matrix:
            ax.plot(np.arange(1,65)*.2, trace, lw=.7, alpha=.6)
        ax.plot(np.arange(1,65)*.2, matrix.mean(axis=0), color='black', lw=2)
        ax.axvspan(9.6,12.8,color='gray',alpha=.1)
        ax.set(title=f'Gamma/sigma={field:g}; {walkers:,} walkers', xlabel='Imaginary time τ', ylabel='Energy per spin')
        ax.grid(alpha=.2)
plt.show()
'''
lines = ['**Completed stronger-field screen.** The table uses total energies per replica.']
for field in (3.0, 5.0):
    row = saved['summaries'][f'g{field:g}_m10000']
    drift = row['late_drift']['ci95']
    contrast = saved['walker_contrasts'][str(field)]['ci95']
    lines.append(f"At Gamma/sigma={field:g}, the 10,000-walker late energy is {row['late_energy']['mean']:.6f} ± {row['late_energy']['se']:.6f} SE. Its drift interval is [{drift[0]:.6f}, {drift[1]:.6f}]; the walker-contrast interval is [{contrast[0]:.6f}, {contrast[1]:.6f}].")
lines.append('These comparisons use the working ±0.108 total-energy tolerance. Gamma/sigma=5 with 10,000 walkers meets both the tested drift and walker-count criteria; there is no demonstrated energy failure at this accuracy. Energy production is justified at this tested L6/sample0/field setting with 32 equilibration blocks and at least 32 measurement blocks of duration 0.2 and eight independent populations. With no forward cycles, this partition preserves the tested dynamics while excluding the initial transient from the energy estimator. Longer projection and larger-population checks would add confidence. The archived production_allowed=false records the original screen protocol, rather than a failed Gamma=5 energy criterion. These energy-only runs contain no q2 or susceptibility measurements, so pure overlap and correlations require the separate forward-time validation below. Energy stability does not establish a paramagnetic phase. The earlier weak-field convergence failure remains unresolved.')
index = next(i for i,c in enumerate(notebook.cells) if c.source.startswith('## 8. Why susceptibility'))
notebook.cells[index].source = notebook.cells[index].source.replace('## 8. Why susceptibility', '## 9. Why susceptibility')
notebook.cells[index:index] = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(code), nbformat.v4.new_markdown_cell('\n\n'.join(lines))]
for cell in notebook.cells:
    cell.source = cell.source.replace('The next useful diagnostic is dependence on independently prepared initial populations while retaining the flat guide, followed by longer matched-time comparisons where warranted.', 'The stronger-field screen below tests whether convergence improves when the transverse field increases, retaining the flat guide.')
nbformat.validate(notebook)
nbformat.write(notebook, notebook_path)
print(notebook_path)
