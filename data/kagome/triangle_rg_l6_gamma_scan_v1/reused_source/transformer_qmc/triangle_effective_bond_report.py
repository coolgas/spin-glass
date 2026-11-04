"""Supplementary saved-data field overlays for the all-bond screen on AutoDL."""
import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_effective_bond_scan import SCALES, GAMMAS, FIELDS, report
from .triangle_effective_benchmark import write_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    root = parser.parse_args().output
    report(root)
    assessment = json.loads((root/'assessment.json').read_text())
    rows = [json.loads(p.read_text()) for p in (root/'cases').glob('*.json')]
    lookup = {(r['scale'],r['gamma'],r['h']):r for r in rows if (r['m'],r['chi'])==(6,16)}
    all_lookup = {(r['scale'],r['gamma'],r['h'],r['m'],r['chi']):r for r in rows}
    q = np.linspace(-1,1,49)
    for shift in assessment['shifts']:
        endpoint_rows = [lookup[shift['scale'],shift[f'gamma_{end}'],shift[f'h_{end}']]
                         for end in ('from','to')]
        variances = [max(0,float(np.array(r['mass'])@q**4)-r['metrics']['q2']**2)/r['pairs']
                     for r in endpoint_rows]
        shift['q2_sampling_se'] = float(np.sqrt(sum(variances)))
        sensitivities = []
        for r in endpoint_rows:
            a,b,c = [all_lookup[r['scale'],r['gamma'],r['h'],m,chi]
                     for m,chi in ((4,8),(6,8),(6,16))]
            sensitivities.append(max(abs(a['metrics']['q2']-b['metrics']['q2']),
                                     abs(b['metrics']['q2']-c['metrics']['q2'])))
        shift['q2_endpoint_retention_sensitivities'] = sensitivities
        shift['resolved_q2_candidate_only'] = bool(shift['both_candidates'] and
            abs(shift['delta_q2']) > 3*shift['q2_sampling_se']+sum(sensitivities))
    write_json(root/'assessment.json',assessment)
    images = []
    for variable,fixed_values,varying in [('Gamma',FIELDS,GAMMAS),('h',GAMMAS,FIELDS)]:
        fig,axs = plt.subplots(4,4,figsize=(17,13),sharex=True,layout='constrained')
        for i,scale in enumerate(SCALES):
            for j,fixed in enumerate(fixed_values):
                ax = axs[i,j]
                for value in varying:
                    key = (scale,value,fixed) if variable=='Gamma' else (scale,fixed,value)
                    r = lookup[key]
                    ax.plot(q,r['mass'],'.-',ms=2,lw=1,label=f'{variable}={value:g}')
                fixed_name = 'h' if variable=='Gamma' else 'Gamma'
                ax.set_title(f'K~U({.5*scale:g},{1.5*scale:g}), {fixed_name}={fixed:g}',fontsize=10)
                ax.set(xlim=(-1.02,1.02),ylim=(0,None),xticks=[-1,-.5,0,.5,1])
                ax.grid(alpha=.2)
                if i == 3: ax.set_xlabel('Physical overlap q')
                if j == 0: ax.set_ylabel('Probability mass')
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=4,
                   title=f'L4 field overlays | m=6, chi=16 | varying {variable} | one disorder')
        name = f'l4_{variable.lower()}_overlays'
        for ext in ('png','pdf'): fig.savefig(root/f'{name}.{ext}',dpi=160)
        plt.close(fig)
        images.append(name+'.png')
    nb = nbformat.read(root/'triangle_effective_bond_scan.ipynb',as_version=4)
    note = r'''

## Zero-field convention and direct field overlays

At $h=0$, every exact and retained eigensolve explicitly restricts global
spin-flip parity to the even sector. This avoids arbitrary asymmetric mixtures
when the finite-system tunnelling splitting lies below numerical resolution.
The archived exact gap is the unrestricted gap, whereas retained gaps at zero
field refer to the even sector. They should not be compared as the same gap.
The unrestricted pilot is preserved separately. At finite longitudinal field,
solves use the full retained space. Near-degenerate cutoff subspaces and solver
start disagreements are retained as alarms.

The overlays use the same saved $P(q)$ measurements as the preceding panels.
They show motion directly; they do not include truncation uncertainty. Consult
the retention tables before using a shift to select a candidate window.

![Gamma overlays](attachment:l4_gamma_overlays.png)

![h overlays](attachment:l4_h_overlays.png)
'''
    candidates = [r for r in assessment['panels'] if r['candidate_only']]
    shifts = [r for r in assessment['shifts'] if r['both_candidates']]
    note += f'\nThe provisional screen passes {len(candidates)} of 64 contexts; '
    note += f'{len(shifts)} adjacent field comparisons have both endpoints passing.\n'
    if not candidates:
        note += '\nNo field window is selected by this finite screen. Further retention checks are necessary before finite-size scaling.\n'
    note += '\n| Mean K | Gamma from/to | h from/to | Change in q2 | Sampling SE | Endpoint retention sensitivity sum | Resolved candidate |\n'
    note += '| ---: | :--- | :--- | ---: | ---: | ---: | :---: |\n'
    for r in shifts:
        note += f"| {r['scale']:g} | {r['gamma_from']:g}/{r['gamma_to']:g} | {r['h_from']:g}/{r['h_to']:g} | {r['delta_q2']:+.5f} | {r['q2_sampling_se']:.5f} | {sum(r['q2_endpoint_retention_sensitivities']):.5f} | {'yes' if r['resolved_q2_candidate_only'] else 'no'} |\n"
    note += '\nA resolved candidate has a q2 shift exceeding three sampling SE plus the sum of the measured q2 retention sensitivities at its endpoints. This is a descriptive screen, not an error bound. Histogram TV also contains a sampling noise floor.\n'
    attachments = {name:{'image/png':base64.b64encode((root/name).read_bytes()).decode()}
                   for name in images}
    nb.cells.append(nbformat.v4.new_markdown_cell(note,attachments=attachments))
    nbformat.validate(nb)
    nbformat.write(nb,root/'triangle_effective_bond_scan.ipynb')
    with (root/'REPORT.md').open('a') as stream:
        stream.write(note.replace('attachment:',''))
    from .triangle_effective_bond_clarity import clarify
    clarify(root)
    provenance = dict(report_source_sha256=digest(Path(__file__)),
        calculation_source_sha256=digest(Path(__file__).with_name('triangle_effective_bond_scan.py')),
        notebook_sha256=digest(root/'triangle_effective_bond_scan.ipynb'),
        note='Saved-data overlays; original immutable campaign sources preserved separately')
    write_json(root/'report_provenance.json',provenance)
    print(json.dumps(dict(candidates=len(candidates),adjacent_passing_pairs=len(shifts),
                         report_provenance=provenance)),flush=True)


if __name__ == '__main__':
    main()
