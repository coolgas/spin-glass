"""Assemble the measured L4 RG pilot into a concise results notebook on AutoDL."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_effective_benchmark import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    order = [(4, 4), (4, 8), (4, 16), (6, 8), (8, 8)]
    rows = sorted([json.loads(p.read_text()) for p in (root/'cases').glob('*.json')],
                  key=lambda d: order.index((d['m'], d['chi'])))
    baseline = [json.loads(p.read_text()) for p in (root/'baseline').glob('*.json')
                if p.name not in ('snapshot.json', 'manifest.json')]
    reference = {}
    for kind in ('jastrow', 'rbm_2'):
        selected = [d for d in baseline if d['task']['sample_id'] == 60000
                    and d['task']['gamma'] == .55 and d['task']['kind'] == kind
                    and d['task']['audit'] == 'main']
        reference[kind] = dict(energy_per_site=float(np.mean([d['energy'] for d in selected])/48),
                               q2=float(np.mean([d['q2'] for d in selected])),
                               mass=np.mean([d['mass'] for d in selected], axis=0))
    uncertainties = []
    for row in rows:
        raw = root/'cases'/f"{row['key']}.npz"
        assert digest(raw) == row['raw_sha256']
        with np.load(raw) as data:
            states = data['spins']
            q = np.mean(states[::2]*states[1::2], axis=1)
            q2_se = float(np.std(q*q, ddof=1)/np.sqrt(len(q)))
            q2_sampling_error = float(np.mean(q*q)-row['metrics']['q2'])
            rng = np.random.default_rng(91234)
            bootstrap = rng.multinomial(len(q), np.asarray(row['mass']), size=1000)/len(q)
            tv_intervals = {kind: np.quantile(abs(bootstrap-ref['mass']).sum(1)/2, [.025, .975]).tolist()
                            for kind, ref in reference.items()}
        uncertainties.append(dict(key=row['key'], q2_sampling_error=q2_sampling_error,
                                  q2_sampling_se=q2_se, tv_bootstrap_conditional=tv_intervals))
    table = ['| (m, chi) | E/N | ΔE/N vs Jastrow | q² | TV vs Jastrow | TV vs RBM | Total seconds |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        comparison = {d['kind']: d for d in row['comparisons']}
        table.append(f"| ({row['m']}, {row['chi']}) | {row['energy_per_site']:.6f} | "
                     f"{comparison['jastrow']['energy_difference_per_site']:+.6f} | {row['metrics']['q2']:.6f} | "
                     f"{comparison['jastrow']['tv']:.4f} | {comparison['rbm_2']['tv']:.4f} | {row['seconds']['total']:.2f} |")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for kind, ref in reference.items():
        axes[0].plot(rows[0]['q_grid'], ref['mass'], label=kind+' GFMC', linewidth=2)
    for row in rows:
        axes[0].plot(row['q_grid'], row['mass'], label=f"RG ({row['m']},{row['chi']})", alpha=.8)
    axes[0].set(xlabel='Physical-spin overlap q', ylabel='Probability mass', xlim=(0, 1))
    axes[0].legend(fontsize=7)
    axes[1].bar([f"({d['m']},{d['chi']})" for d in rows],
                [d['energy_per_site']-reference['jastrow']['energy_per_site'] for d in rows])
    axes[1].axhline(.001, color='red', linestyle='--', label='Screening limit .001')
    axes[1].set(xlabel='Retained states (m, chi)', ylabel='Energy/spin difference from Jastrow GFMC')
    axes[1].legend(fontsize=8)
    figure = root/'l4_rg_pilot.png'
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    assessment = dict(status='fixed_low_energy_schedule_fails_pilot', physical_accuracy_qualified=False,
                      scope=dict(L=4, N=48, sample_id=60000, gamma=.55, h=.15,
                                 orientation=0, origin=[0, 0]),
                      cases=len(rows), reference={k: {x: v for x, v in ref.items() if x != 'mass'}
                                                  for k, ref in reference.items()},
                      uncertainties=uncertainties,
                      decision='Do not expand this fixed schedule to the full L4 grid or L6. Reassess state selection.',
                      limitations=['One disorder, field and partition; not an ensemble conclusion.',
                                   'GFMC is comparison data, not exact N48 ground truth.',
                                   'TV bootstrap conditions on fixed GFMC baseline and is not a convergence certificate.',
                                   'Reported timings exclude baseline hashing, imports and validation; no speedup claim.',
                                   'The full Stage B/C/D grid was not run after the initial screen failed.'])
    write_json(root/'assessment.json', assessment)
    text = '\n'.join(table)
    body = ("# L4 triangle RG: first benchmark\n\n"
            "**The tested fixed low-energy schedule fails the pilot accuracy limits.**\n\n"
            "Periodic N=48, disorder60000, Gamma=.55, h=.15; upward triangles, origin (0,0). "
            "48 spins → 16 triangles (m states each) → 4 superblocks (chi states each). "
            "Five schedules; two eigensolver starts each; 16,384 independent physical-spin replica pairs per schedule.\n\n"
            + text + '\n\n'
            f"Full-spin reference E/N: Jastrow {reference['jastrow']['energy_per_site']:.6f}, "
            f"RBM {reference['rbm_2']['energy_per_site']:.6f}. "
            f"Reference q²: {reference['jastrow']['q2']:.6f}, {reference['rbm_2']['q2']:.6f}, respectively. "
            "Each reference averages two main projection repeats. GFMC is comparison data, not exact ground truth.\n\n"
            "![Physical overlaps and energy discrepancies](attachment:l4_rg_pilot.png)\n\n"
            "Screening limits: |ΔE|/N ≤ .001, |Δq²| ≤ .01, TV ≤ .05. "
            "All five schedules fail the energy comparison. At fixed chi=8, increasing m improves the overlap; "
            "m=8 removes triangle truncation but leaves a large superblock error. "
            "The larger-m top subspaces are not necessarily nested.\n\n"
            "**Checks passed:** first-level projection for both orientations; full m=8 recovery on exact L2; "
            "second-level projection and direct physical-energy reconstruction; same-block two-point products; "
            "joint Born sampling of an entangled test state (TV=.00462 with 30,000 draws). "
            "L4 bond arrays match regeneration from the archived baseline source; baseline raw hashes verified.\n\n"
            "**Interpretation:** low internal superblock energy alone does not select enough states to describe "
            "coupling across its boundary. The final eigensolvers converge, but physical truncation bias remains. "
            "The isolated-block basis needs a separate accuracy study with more states or an environment-adapted basis.\n\n"
            "**Decision:** stop expansion of this schedule. No L6 run. The result does not rule out RG with a different basis. "
            "Only one physical context and partition were tested; the full validation grid remains unexecuted.\n\n"
            "Raw isometries, states, physical samples, comparisons, conditional sampling uncertainties, "
            "failures and source provenance accompany this notebook. Timings exclude baseline verification and validation; "
            "they do not establish a speed advantage at matched accuracy.")
    cell = nbformat.v4.new_markdown_cell(body)
    cell.id = 'l4-rg-first-benchmark'
    cell.attachments = {'l4_rg_pilot.png': {'image/png': __import__('base64').b64encode(figure.read_bytes()).decode()}}
    notebook = nbformat.v4.new_notebook(cells=[cell], metadata={'research_results': assessment['status'],
                                                              'scientific_compute_platform': 'AutoDL'})
    nbformat.validate(notebook)
    nbformat.write(notebook, root/'triangle_effective_l4_results.ipynb')
    (root/'ASSESSMENT.md').write_text(body.replace('attachment:l4_rg_pilot.png', 'l4_rg_pilot.png')+'\n')
    print(json.dumps(assessment), flush=True)


if __name__ == '__main__':
    main()
