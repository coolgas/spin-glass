"""Reconstruct and export the L6 auxiliary pilot results on AutoDL."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import nbformat
import numpy as np
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from .collective_campaign import write
from .correlation_lengths import sha


def born_overlap_mass(states):
    """Cross independent chain columns; time-pair within each pair of columns."""
    draws, chains, n = states.shape
    mass = np.zeros(n+1)
    for a in range(0, chains, 2):
        left = states[:, a].astype(np.int64)
        right = states[:, a+1].astype(np.int64)
        index = (left @ right.T+n)//2
        mass += np.bincount(index.ravel(), minlength=n+1)
    return (mass/mass.sum()).tolist()


def summarize(root):
    import torch
    from .physics import diagonal, make_af_sample
    from .simple_guides import RBM
    torch.set_num_threads(1)
    manifest = json.loads((root/'manifest.json').read_text())
    bank = json.loads((root/'candidate_bank.json').read_text())
    assert sha(root/'candidate_bank.npz') == bank['raw_sha256']
    with np.load(root/'candidate_bank.npz') as raw:
        references = raw['references'].astype(float)
        sample = make_af_sample(6, manifest['sample_id'])
        np.testing.assert_array_equal(raw['J'], sample.J)
        np.testing.assert_array_equal(raw['bonds'], sample.bonds)
        np.testing.assert_array_equal(raw['points'], sample.points)
        ham = sample.hamiltonian(.15)
        np.testing.assert_allclose(diagonal(ham, references), bank['reference_energies'], atol=1e-12)
    frozen = []
    for path in sorted((root/'frozen').glob('*/*.json')):
        record = json.loads(path.read_text())
        assert sha(path.with_suffix('.npz')) == record['raw_sha256']
        checkpoint = root/'training'/record['branch']/f"update_{record['update']:04d}.pt"
        assert sha(checkpoint) == record['model_sha256']
        model = RBM(108, 2).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
        n = 108
        with np.load(path.with_suffix('.npz')) as raw:
            details = []
            for row in record['rows']:
                start = row['start']
                energy = raw[start+'_energy']
                spins = raw[start+'_spins'].astype(float)
                # Independently evaluate explicit flipped configurations. This
                # bypasses the cached-field flip-ratio implementation.
                indices = np.linspace(0, spins.shape[0]*spins.shape[1]-1, 8, dtype=int)
                states = spins.reshape(-1, n)[indices]
                with torch.no_grad():
                    x = torch.tensor(states, dtype=torch.float64, device='cuda')
                    flips = x[:, None, :].expand(-1, n, -1).clone()
                    sites = torch.arange(n, device='cuda')
                    flips[:, sites, sites] *= -1
                    ratios = model(flips.reshape(-1, n)).reshape(-1, n)-model(x)[:, None]
                    independent = diagonal(ham, states)-.6*torch.exp(ratios).sum(1).cpu().numpy()
                np.testing.assert_allclose(independent, energy.ravel()[indices], atol=1e-9, rtol=1e-10)
                np.testing.assert_allclose(energy.mean()/n, row['energy_per_site'], atol=1e-12)
                np.testing.assert_allclose(energy.var()/n, row['variance_per_site'], atol=1e-12)
                flat = spins.reshape(-1, n)
                pair = flat.T @ flat / len(flat)
                connected = pair - np.outer(flat.mean(0), flat.mean(0))
                np.testing.assert_allclose(connected, raw[start+'_connected'], atol=1e-12)
                np.testing.assert_allclose((pair**2).mean(), row['q2_born'], atol=1e-12)
                np.testing.assert_allclose((connected**2).sum()/n, row['chi_connected_born'], atol=1e-12)
                # Chain columns never exchange with one another. Keep all time
                # correlation within a column in this conditional error estimate.
                means = energy.mean(0)/n
                overlap = spins @ references.T / n
                nearest = np.argmax(overlap, axis=-1)
                weights = np.bincount(nearest.ravel(), minlength=3)/nearest.size
                mass = born_overlap_mass(spins)
                centered = energy-energy.mean(0, keepdims=True)
                denominator = np.sum(centered*centered)
                autocorrelation = [float(np.sum(centered[:-lag]*centered[lag:])/denominator)
                                   for lag in [1, 4, 16, 64]] if denominator else [0.]*4
                details.append(dict(**row,
                    conditional_energy_se=float(means.std(ddof=1)/np.sqrt(len(means))),
                    nearest_reference_mass=weights.tolist(),
                    born_overlap_mass=mass,
                    energy_autocorrelation_lags_1_4_16_64=autocorrelation))
        start_masses = np.array([r['born_overlap_mass'] for r in details])
        start_tv = max(float(np.abs(a-b).sum()/2) for a in start_masses for b in start_masses)
        frozen.append(dict(branch=record['branch'], update=record['update'],
            rows=details, energy_per_site=float(np.mean([r['energy_per_site'] for r in details])),
            variance_per_site=float(np.mean([r['variance_per_site'] for r in details])),
            energy_start_range=record['energy_start_range'], raw_sha256=record['raw_sha256'],
            model_sha256=record['model_sha256'], explicit_flip_energy_audited=True,
            born_histogram_max_start_tv=start_tv))
    comparisons = []
    for record in frozen:
        baseline = next((r for r in frozen if r['branch']=='unpinned' and r['update']==record['update']), None)
        if baseline and record['branch'] != 'unpinned':
            branch_raw = root/'frozen'/record['branch']/f"update_{record['update']:04d}.npz"
            baseline_raw = root/'frozen'/'unpinned'/f"update_{record['update']:04d}.npz"
            variances = []
            with np.load(branch_raw) as a, np.load(baseline_raw) as b:
                for start in ['random', 'reference', 'inverse']:
                    paired = (a[start+'_energy']-b[start+'_energy']).mean(0)/108
                    variances.append(paired.var(ddof=1)/len(paired))
            comparisons.append(dict(branch=record['branch'], update=record['update'],
                delta_energy_per_site=record['energy_per_site']-baseline['energy_per_site'],
                conditional_paired_energy_se=float(np.sqrt(sum(variances))/3),
                variance_ratio=record['variance_per_site']/baseline['variance_per_site'],
                energy_start_range=record['energy_start_range']))
    reference_dependence = []
    initial_path_checks = []
    for index in range(3):
        a = root/'training'/f'release_r{index}'/'update_0400.pt'
        b = root/'training'/f'fixed_r{index}'/'update_0400.pt'
        if a.exists() and b.exists():
            state_a = torch.load(a, map_location='cpu', weights_only=False)['model']
            state_b = torch.load(b, map_location='cpu', weights_only=False)['model']
            for key in state_a:
                torch.testing.assert_close(state_a[key], state_b[key], atol=0, rtol=0)
            initial_path_checks.append(dict(reference=index, update=400, bitwise_equal=True))
    for update in sorted(set(r['update'] for r in frozen)):
        records = [r for r in frozen if r['update']==update and r['branch'].startswith('release_')]
        for index, a in enumerate(records):
            for b in records[index+1:]:
                mass_a = np.mean([r['born_overlap_mass'] for r in a['rows']], axis=0)
                mass_b = np.mean([r['born_overlap_mass'] for r in b['rows']], axis=0)
                reference_dependence.append(dict(update=update, branches=[a['branch'], b['branch']],
                    energy_difference_per_site=abs(a['energy_per_site']-b['energy_per_site']),
                    born_histogram_tv=float(np.abs(mass_a-mass_b).sum()/2),
                    q2_difference=abs(float(np.mean([r['q2_born'] for r in a['rows']]))
                        -float(np.mean([r['q2_born'] for r in b['rows']])))))
    projections = []
    from .feedback_frontier import verify_measurement
    for path in sorted((root/'projection').glob('*/runs/*.json')):
        record = json.loads(path.read_text())
        checkpoint = root/'training'/record['task']['branch']/'update_2400.pt'
        assert record['provenance']['model_sha256'] == sha(checkpoint)
        verification = verify_measurement(record, path.parent.parent)
        projections.append(dict(**record, raw_reconstructed=verification['reconstructed']))
    repeats = []
    for branch in sorted(set(r['task']['branch'] for r in projections)):
        records = sorted((r for r in projections if r['task']['branch']==branch), key=lambda r:r['task']['repeat'])
        if len(records) != 2:
            continue
        a, b = records
        spectrum_a, spectrum_b = np.array(a['spectrum']), np.array(b['spectrum'])
        relative = np.abs(spectrum_a-spectrum_b)/np.maximum(np.maximum(np.abs(spectrum_a),np.abs(spectrum_b)), .001)
        row = dict(branch=branch, energy_difference_per_site=abs(a['energy']-b['energy'])/108,
            q2_difference=abs(a['q2']-b['q2']),
            histogram_tv=float(np.abs(np.array(a['mass'])-b['mass']).sum()/2),
            spectrum_max_relative_difference=float(relative.max()),
            ancestry_alarm=bool(a['ancestry_alarm'] or b['ancestry_alarm']))
        row['repeat_thresholds_passed'] = bool(row['energy_difference_per_site']<=.001
            and row['q2_difference']<=.01 and row['histogram_tv']<=.05
            and row['spectrum_max_relative_difference']<=.1)
        repeats.append(row)
    projection_comparisons = []
    for r in projections:
        baseline = next((b for b in projections if b['task']['branch']=='unpinned'
            and b['task']['repeat']==r['task']['repeat']), None)
        if baseline and r['task']['branch'] != 'unpinned':
            projection_comparisons.append(dict(branch=r['task']['branch'], repeat=r['task']['repeat'],
                delta_energy_per_site=(r['energy']-baseline['energy'])/108,
                q2_difference=r['q2']-baseline['q2'],
                histogram_tv=float(np.abs(np.array(r['mass'])-baseline['mass']).sum()/2),
                minimum_descendant_ess_ratio=r['minimum_descendant_ess']/baseline['minimum_descendant_ess'],
                ancestry_alarm=bool(r['ancestry_alarm'] or baseline['ancestry_alarm'])))
    data = dict(updated_utc=datetime.now(timezone.utc).isoformat(), manifest=manifest, bank=bank,
        analysis_source_sha256=sha(Path(__file__)),
        status=json.loads((root/'status.json').read_text()), frozen=frozen, comparisons=comparisons,
        projections=projections, repeats=repeats, ground_state_qualified=False,
        released_reference_dependence=reference_dependence,
        identical_schedule_checks=initial_path_checks,
        projection_comparisons=projection_comparisons,
        failures=json.loads((root/'failures.json').read_text()) if (root/'failures.json').exists() else [])
    write(root/'analysis_results.json', data)
    return data


def build(root, output):
    data = summarize(root)
    baseline_updates = {r['update'] for r in data['frozen'] if r['branch']=='unpinned'}
    completed_updates = [update for update in baseline_updates if
        sum(r['update']==update for r in data['frozen'])==7]
    latest = max(completed_updates) if completed_updates else None
    lines = ['# L6 auxiliary-field pilot', '',
        f"Verified snapshot: {data['updated_utc']}", '',
        'One periodic disorder (60000), one training seed (0), N=108, Gamma=0.6, h=0.15.',
        'Every listed energy and variance uses the original Hamiltonian.', '']
    if latest is not None:
        lines.extend([f'Latest checkpoint completed by all seven branches: {latest} updates.', '',
            '| Branch | Target energy/site | Variance/N | Energy difference from baseline | Conditional paired SE | Variance ratio |',
            '|---|---:|---:|---:|---:|---:|'])
        for r in sorted((r for r in data['frozen'] if r['update']==latest), key=lambda r:r['branch']):
            c = next((c for c in data['comparisons'] if c['branch']==r['branch'] and c['update']==latest), None)
            tail = (f"{c['delta_energy_per_site']:+.7f} | {c['conditional_paired_energy_se']:.7f} | {c['variance_ratio']:.3f}"
                    if c else '0 | — | 1')
            lines.append(f"| {r['branch']} | {r['energy_per_site']:.7f} | {r['variance_per_site']:.6f} | {tail} |")
        lines.extend(['', 'Negative energy differences favor auxiliary training; variance ratios below one favor it.',
            'Conditional SEs preserve temporal correlation within independent chain columns and paired streams.',
            'They assume equilibration and do not measure disorder or training-seed variability.',
            'Update 400 is fully pinned; update 1600 follows epsilon=0.02 training;',
            'updates 2000/2400 have 400/800 zero-pinning refinement updates.', ''])
        released = [c for c in data['comparisons'] if c['update']==latest and c['branch'].startswith('release_')]
        if latest == 2400 and len(released) == 3:
            better_energy = sum(c['delta_energy_per_site'] < 0 for c in released)
            better_variance = sum(c['variance_ratio'] < 1 for c in released)
            lines.extend([f'After 800 updates with pinning removed, {better_energy}/3 release branches have lower sampled target energy '
                f'and {better_variance}/3 have lower local-energy variance than the baseline.',
                'This pilot does not show a reliable benefit across the three classical references.',
                'The references were selected before training; choosing the best branch afterward adds selection uncertainty.', ''])
    if data['projections']:
        lines.extend(['| Projection branch | Repeat | Energy/site | Descendant ESS minimum | Largest lineage | Alarm |',
            '|---|---:|---:|---:|---:|---|'])
        for r in data['projections']:
            lines.append(f"| {r['task']['branch']} | {r['task']['repeat']} | {r['energy']/108:.7f} | "
                f"{r['minimum_descendant_ess']:.2f} | {r['largest_lineage']:.3f} | {r['ancestry_alarm']} |")
        lines.append('')
    lines.extend([f"Completed projection repeats: {len(data['projections'])}/14; "
        f"genealogy alarms: {sum(r['ancestry_alarm'] for r in data['projections'])}.",
        f"Recorded failed branches/tasks: {len(data['failures'])}.", '',
        'Projection records are independently reconstructed from raw arrays before inclusion.',
        'Reference dependence and start checks are retained in analysis_results.json and the executed results notebook.',
        'Ground-state convergence is unestablished.'])
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    intro = r'''# L6 auxiliary-field VMC: measured pilot

This is one periodic kagome sample (disorder 60000, 108 spins), one training
seed, and three classical references. The target Hamiltonian has positive
antiferromagnetic couplings $K\sim U(0.5,1.5)$, longitudinal field $h=0.15$,
and transverse field $\Gamma=0.6$.

All branches start from the same saved high-field RBM and get 2400 VMC
updates. The baseline stays unpinned. Three release branches use 400 updates
each at $\epsilon=0.2,0.1,0.05,0.02$, then 800 at zero; three fixed controls
stay at $\epsilon=0.2$. The added energy is $\epsilon d_H(s,s^\star)$.
The fixed and release branches are identical through update 400, providing
a reproducibility check. An update-1600 release checkpoint still has the
last nonzero pinning field; updates 2000 and 2400 have 400 and 800 zero-field
refinement updates. Every energy below is evaluated with the original target
Hamiltonian, including measurements of guides trained with fixed pinning.

The implemented sampler is blocked two-copy Gibbs tempering: its cold
marginal is exactly $|\psi|^2$, while its intermediate distributions differ
from the initial power-tempering design. Classical search uses 128 vectorized
starts in one fixed RNG stream with shared sweep permutations. These choices
were fixed before examining results. No branch or reference is selected away.

Frozen diagnostics use random, reference, and inverse-reference starts,
128 chains at each of 16 temperatures, 400 warmup sweeps, and 256 saved
measurements separated by four sweeps. The energy SE is computed across
independent chain-column means, retaining time correlation within columns.
It assumes equilibration and does not cover missed configurations or training
seed variability. The displayed start spread and autocorrelation checks are
additional necessary checks. Nearest-reference regions use maximum overlap,
breaking ties by reference index; they are descriptive regions.
Guide Born overlap histograms cross distinct chain columns and all pairs
of saved times. They describe the trained wave function before projection.

Final projection screens use four independent 512-walker populations, time
16 for equilibration, time 16 per forward cycle, eight cycles, and two repeats.
Raw tags, descendant counts, overlap histograms, spectra and energies are
independently reconstructed before inclusion. A genealogy alarm means any
cycle/population has descendant ESS below 8 or a largest lineage above 0.5.
Neither a lower guide energy nor this finite projection screen establishes
ground-state convergence. The reference search does not identify known
classical ground states or demonstrate a larger quantum gap.
'''
    code = '''import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
data = json.loads(DATA_LITERAL)
print("Updated UTC:", data["updated_utc"])
print("Campaign status:", data["status"])
print("Classical reference energies:", data["bank"]["reference_energies"])
print("Reference Hamming distances:", data["bank"]["reference_distances"])
print("Identical first-stage model checks:", data["identical_schedule_checks"])
print("Frozen checkpoints:", len(data["frozen"]), "Projection repeats:", len(data["projections"]), "/ 14")
if data["failures"]: display(data["failures"])
frozen = pd.DataFrame([{k:v for k,v in r.items() if k!="rows"} for r in data["frozen"]])
comparison = pd.DataFrame(data["comparisons"])
if len(frozen):
    display(frozen[["branch","update","energy_per_site","variance_per_site","energy_start_range","born_histogram_max_start_tv"]])
if len(comparison):
    display(comparison)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for branch, group in comparison.groupby("branch"):
        axes[0].plot(group["update"], group.delta_energy_per_site, "o-", label=branch)
        axes[1].plot(group["update"], group.variance_ratio, "o-", label=branch)
    axes[0].axhline(0, color="gray", lw=1)
    axes[1].axhline(1, color="gray", lw=1)
    axes[0].set(xlabel="VMC updates", ylabel="Target energy/site minus baseline")
    axes[1].set(xlabel="VMC updates", ylabel="Target variance / baseline variance")
    for ax in axes: ax.axvline(1600, color="gray", ls=":", lw=1)
    axes[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig("auxiliary_vmc_comparison.png", dpi=180); plt.show()
rows = [dict(branch=g["branch"],update=g["update"],**r) for g in data["frozen"] for r in g["rows"]]
if rows:
    display(pd.DataFrame(rows)[["branch","update","start","energy_per_site","conditional_energy_se","variance_per_site","q2_born","reference_overlap","nearest_reference_mass","round_trips","energy_autocorrelation_lags_1_4_16_64"]])
projections = [dict(branch=r["task"]["branch"],repeat=r["task"]["repeat"],energy_per_site=r["energy"]/108,q2=r["q2"],ess=r["minimum_descendant_ess"],largest_lineage=r["largest_lineage"],alarm=r["ancestry_alarm"],raw_reconstructed=r["raw_reconstructed"]) for r in data["projections"]]
display(pd.DataFrame(projections))
display(pd.DataFrame(data["repeats"]))
display(pd.DataFrame(data["projection_comparisons"]))
display(pd.DataFrame(data["released_reference_dependence"]))
if len(data["frozen"]):
    update = max(r["update"] for r in data["frozen"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for record in data["frozen"]:
        if record["update"] != update or record["branch"].startswith("fixed_"): continue
        mass = np.mean([r["born_overlap_mass"] for r in record["rows"]], axis=0)
        ax.plot(np.linspace(-1, 1, 109), mass, label=record["branch"])
    ax.set(xlabel="Overlap q", ylabel="Probability mass", title=f"Guide Born overlap, update {update}; no projection")
    ax.legend(); fig.tight_layout()
    fig.savefig("auxiliary_vmc_born_overlap.png", dpi=180); plt.show()
if not projections: print("Original-Hamiltonian projection screens pending.")
'''.replace('DATA_LITERAL', repr(json.dumps(data)))
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('\n'.join(lines)),
        nbformat.v4.new_markdown_cell(intro),
        nbformat.v4.new_code_cell(code)], metadata=dict(kernelspec=dict(name='python3',
        language='python', display_name='Python 3')))
    output.mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=180, resources={'metadata':{'path':str(output)}}).execute()
    nbformat.validate(nb)
    dest = output/'auxiliary_field_vmc_results.ipynb'
    nbformat.write(nb, dest)
    html, _ = HTMLExporter().from_notebook_node(nb)
    dest.with_suffix('.html').write_text(html)
    print(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)
