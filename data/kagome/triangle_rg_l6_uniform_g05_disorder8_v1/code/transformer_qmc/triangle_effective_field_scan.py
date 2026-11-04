"""Matched physical P(q) ensemble for the frozen L4 higher-field figure."""
import argparse
import base64
import json
from pathlib import Path
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_effective_benchmark import digest, run_case, write_json


FIELDS = (.6, .65, .7, .85, 1.)
IDS = tuple(range(60000, 60008))


def report(root, reference):
    rows = [json.loads(p.read_text()) for p in (root/'cases').glob('*.json')]
    sampling_checks = []
    for row in rows:
        raw = root/'cases'/f"{row['key']}.npz"
        assert digest(raw) == row['raw_sha256']
        with np.load(raw) as data:
            samples = data['spins']
            overlaps = np.mean(samples[::2]*samples[1::2], axis=1)
            counts = np.bincount(np.sum(samples[::2] == samples[1::2], axis=1), minlength=49)
            np.testing.assert_array_equal(counts, data['counts'])
            np.testing.assert_allclose(counts/counts.sum(), row['mass'], atol=1e-15)
            moments = []
            for power, expected in ((1, row['metrics']['q']), (2, row['metrics']['q2'])):
                values = overlaps**power
                se = float(values.std(ddof=1)/np.sqrt(len(values)))
                difference = float(values.mean()-expected)
                assert abs(difference) < 6*se+1e-12, (row['key'], power, difference, se)
                moments.append(dict(moment=power, difference=difference, se=se))
            sampling_checks.append(dict(key=row['key'], moments=moments))
    curves, comparisons = [], []
    q = np.linspace(-1, 1, 49)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    colors = ['#440154', '#3b528b', '#21918c', '#e08b20', '#cc3311']
    for g, color in zip(FIELDS, colors):
        rg = sorted([r for r in rows if r['gamma'] == g and r['m'] == 6],
                    key=lambda r: r['sample_id'])
        assert [r['sample_id'] for r in rg] == list(IDS)
        masses = np.array([r['mass'] for r in rg])
        mean, se = masses.mean(0), masses.std(0, ddof=1)/np.sqrt(8)
        axes[0].errorbar(q, mean, yerr=se, color=color, fmt='o-', ms=2.5,
                         capsize=2, lw=1, label=f'{g:.2f}')
        curves.append(dict(gamma=g, mass=mean.tolist(), disorder_se=se.tolist(),
                           mean_q=float(q@mean), q2=float(q*q@mean),
                           width=float(np.sqrt(q*q@mean-(q@mean)**2))))
        for ax, kind in zip(axes[1:], ('jastrow', 'rbm_2')):
            selected = [r for r in reference if r['task']['gamma'] == g
                        and r['task']['kind'] == kind]
            if not selected:
                continue
            assert len(selected) == 16
            values = np.array([np.mean([r['mass'] for r in selected
                                        if r['task']['sample_id'] == sid], axis=0) for sid in IDS])
            ref, err = values.mean(0), values.std(0, ddof=1)/np.sqrt(8)
            ax.errorbar(q, ref, yerr=err, color=color, fmt='o-', ms=2.5,
                        capsize=2, lw=1, label=f'{g:.2f} GFMC')
            ax.plot(q, mean, color=color, ls='--', lw=1.5)
            comparisons.append(dict(gamma=g, kind=kind, tv=float(abs(mean-ref).sum()/2),
                                    mean_paired_tv=float(abs(masses-values).sum(1).mean()/2),
                                    delta_mean_q=float(q@(mean-ref)),
                                    delta_q2=float(q*q@(mean-ref)),
                                    reference_mean_q=float(q@ref),
                                    reference_width=float(np.sqrt(q*q@ref-(q@ref)**2)),
                                    ancestry_alarms=sum(r['ancestry_alarm'] for r in selected)))
    for ax, title in zip(axes, ('Spatial RG (m=6, chi=8)',
                               'Jastrow GFMC; dashed = RG', 'RBM GFMC; dashed = RG')):
        ax.set(title=title, xlabel='Physical-spin overlap q', xlim=(-1.02, 1.02))
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    axes[0].set_ylabel('Disorder mean P(q), probability mass')
    fig.suptitle('L=4, N=48 | periodic kagome | h=0.15 | eight matched disorders | bars: ±1 disorder SE')
    figure = root/'higher_field_rg_comparison.png'
    fig.savefig(figure, dpi=180); fig.savefig(figure.with_suffix('.pdf')); plt.close(fig)
    sensitivity = []
    for g in FIELDS:
        a = next(r for r in rows if (r['gamma'], r['sample_id'], r['m']) == (g, 60000, 6))
        b = next(r for r in rows if (r['gamma'], r['sample_id'], r['m']) == (g, 60000, 8))
        sensitivity.append(dict(gamma=g, sample_id=60000,
                                tv=float(abs(np.array(a['mass'])-b['mass']).sum()/2),
                                delta_energy_per_site=b['energy_per_site']-a['energy_per_site'],
                                delta_q2=b['metrics']['q2']-a['metrics']['q2']))
    table = ['| Gamma | Reference | TV of disorder means | Mean paired TV | Delta mean q |',
             '| ---: | --- | ---: | ---: | ---: |']
    for c in comparisons:
        table.append(f"| {c['gamma']:.2f} | {c['kind']} | {c['tv']:.4f} | {c['mean_paired_tv']:.4f} | {c['delta_mean_q']:+.4f} |")
    sensitivity_table = ['| Gamma | TV: m=6 versus m=8, chi=8 | Delta energy/spin |',
                         '| ---: | ---: | ---: |']
    for c in sensitivity:
        sensitivity_table.append(f"| {c['gamma']:.2f} | {c['tv']:.4f} | {c['delta_energy_per_site']:+.6f} |")
    shape_table = ['| Gamma | RG mean q | Jastrow mean q | RG width | Jastrow width |',
                   '| ---: | ---: | ---: | ---: | ---: |']
    for c in curves:
        ref = next(r for r in comparisons if r['gamma'] == c['gamma'] and r['kind'] == 'jastrow')
        shape_table.append(f"| {c['gamma']:.2f} | {c['mean_q']:.4f} | {ref['reference_mean_q']:.4f} | {c['width']:.4f} | {ref['reference_width']:.4f} |")
    body = (r'''# L4 higher-field physical overlap distributions: spatial RG

Matched to the section “L=4, N=48: higher-field distributions” in
`jastrow_rbm_overlap_distributions.ipynb`. Same five fields, eight disorder IDs
60000–60007, periodic bonds, h=.15 and 49 physical-overlap bins.

48 spins → 16 triangles with m=6 states → 4 superblocks with chi=8 states.
Each superblock basis retains its lowest internal energy states. Two solver
starts and 16,384 independent physical replica pairs per case. Physical joint
Born sampling through the saved isometries gives
$q=N^{-1}\sum_i z_i^{(a)}z_i^{(b)}$. All disorders have equal weight.

![Matched higher-field curves](attachment:higher_field_rg_comparison.png)

Solid curves in the right panels are the frozen GFMC figure inputs; dashed
curves are spatial RG. RBM has complete frozen curves only at .60 and .65.
Bars describe variation across eight disorders; they exclude truncation and
GFMC projection bias. The frozen references retain all ancestry alarms.

**Measured behavior:** the RG curves capture the shift toward lower overlap
and broadening with increasing transverse field, with a systematic shift
toward lower q relative to the frozen GFMC curves. Reproduction is qualitative;
the retained basis remains a significant source of bias. At the highest field
the RG width decreases slightly, whereas the reference width continues to grow.

''' + '\n'.join(shape_table) + '\n\n' + '\n'.join(table) + '\n\n' + '\n'.join(sensitivity_table) + '''

TV of disorder means measures agreement of the plotted curves; mean paired TV
also exposes errors that cancel under disorder averaging. Full triangle
retention (m=8, chi=8) is checked for disorder60000 at every field in
`assessment.json`. This is a limited sensitivity check; chi convergence,
partition stability and ground-state qualification remain unestablished.

Raw physical samples, states, isometries, spectra, residuals, source hashes and
failures accompany the report. Scientific computation and figure generation
ran on AutoDL. This comparison does not establish finite-size scaling.

[Raw results and provenance](../data/kagome/triangle_rg_higher_fields_v1/README.md).
''')
    assessment = dict(curves=curves, comparisons=comparisons, sensitivity=sensitivity,
                      sampling_checks=sampling_checks,
                      physical_accuracy_qualified=False, cases=len(rows))
    write_json(root/'assessment.json', assessment)
    (root/'REPORT.md').write_text(body.replace('attachment:', '')+'\n')
    cell = nbformat.v4.new_markdown_cell(body)
    cell.id = 'l4-higher-field-rg'
    cell.attachments = {figure.name: {'image/png': base64.b64encode(figure.read_bytes()).decode()}}
    nb = nbformat.v4.new_notebook(cells=[cell], metadata={'execution_location': 'AutoDL'})
    nbformat.validate(nb); nbformat.write(nb, root/'triangle_effective_higher_field_distributions.ipynb')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    (root/'cases').mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.reference.read_text())['rows']
    assert sorted({r['task']['gamma'] for r in reference}) == list(FIELDS)
    write_json(root/'launch.json', dict(created=time.time(), reference_sha256=digest(args.reference),
               fields=FIELDS, disorders=IDS, schedules='m6 chi8 all 40 contexts; m8 chi8 disorder60000 all five fields',
               threads=1, sources={p.name:digest(p) for p in Path(__file__).parent.glob('triangle_effective*.py')}))
    tasks = [(g, sid, 6) for g in FIELDS for sid in IDS]+[(g, 60000, 8) for g in FIELDS]
    try:
        for index, (g, sid, m) in enumerate(tasks):
            result = run_case(root, [], sid, g, m, 8, 0, (0, 0), 16384)
            assert digest(root/'cases'/f"{result['key']}.npz") == result['raw_sha256']
            write_json(root/'status.json', dict(completed=index+1, total=len(tasks), state='running'))
        report(root, reference)
        write_json(root/'status.json', dict(completed=len(tasks), total=len(tasks), state='complete'))
    except Exception:
        (root/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
