"""Reader-facing summary of saved physical distributions and retention checks."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import nbformat
import numpy as np

from .triangle_effective_bond_scan import SCALES, GAMMAS, FIELDS, SCHEDULES


def clarify(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root
    rows = [json.loads(p.read_text()) for p in (root/'cases').glob('*.json')]
    small = [json.loads(p.read_text()) for p in (root/'exact').glob('*.json')]
    lookup = {(r['scale'],r['gamma'],r['h'],r['m'],r['chi']):r for r in rows}
    refs = {(r['scale'],r['gamma'],r['h']):r for r in small}
    panels = []
    for scale in SCALES:
        for g in GAMMAS:
            for h in FIELDS:
                selected = [lookup.get((scale,g,h,m,c)) for m,c in SCHEDULES]
                ref = refs.get((scale,g,h))
                if any(r is None for r in selected) or ref is None:
                    continue
                a,b,c = selected
                tvm = float(abs(np.array(a['mass'])-b['mass']).sum()/2)
                tvc = float(abs(np.array(b['mass'])-c['mass']).sum()/2)
                de = abs(b['energy_per_site']-c['energy_per_site'])/scale
                dq = abs(b['metrics']['q2']-c['metrics']['q2'])
                check = next(t for t in ref['checks'] if t['m']==6)
                checks = {'m histogram':tvm<.05,'chi histogram':tvc<.05,
                    'chi energy':de<.001,'chi q2':dq<.01,
                    'L2 exact':check['tv']<.05 and check['energy_error_per_site']/scale<.001 and check['q2_error']<.01,
                    'alarms':not any(r['start_alarm'] or r['unresolved_cutoff'] or
                        any(t['alarm'] for t in r['sampling_checks']) for r in selected)
                        and not any(t['full_recovery_alarm'] for t in ref['checks'])}
                panels.append(dict(scale=scale,gamma=g,h=h,tv_m=tvm,tv_chi=tvc,
                    de=de,dq=dq,checks=checks,passed=all(checks.values())))
    panel_lookup = {(p['scale'],p['gamma'],p['h']):p for p in panels}
    def image_cell(text,name):
        return nbformat.v4.new_markdown_cell(text,attachments={name:{'image/png':
            base64.b64encode((output/name).read_bytes()).decode()}})
    complete = len(rows)==192 and len(small)==64
    text = r'''# What was calculated, and what the results mean

We sample **all bonds** from the distributions below. Within each size we reuse
one base disorder realization and multiply every bond by the stated factor.
Internal triangle bonds and boundary bonds follow the same distribution.

| Factor / mean K | Distribution of each positive antiferromagnetic K |
| ---: | :--- |
| 1.0 | Uniform(0.5, 1.5) |
| 0.5 | Uniform(0.25, 0.75) |
| 0.2 | Uniform(0.10, 0.30) |
| 0.1 | Uniform(0.05, 0.15) |

The Hamiltonian is $H=\sum K_{ij}Z_iZ_j-h\sum Z_i-\Gamma\sum X_i$.
The code stores $J_{ij}=-K_{ij}$. All results here use disorder60000,
periodic kagome geometry, triangle orientation0 and superblock origin(0,0).
This is the **all-bond** study; the intertriangle-only study has its own notebook.

**The plotted quantity:** draw two independent physical spin configurations
$z^{(a)},z^{(b)}$ from the reconstructed RG ground-state probability, then compute

$$q=\frac1{48}\sum_{i=1}^{48}z_i^{(a)}z_i^{(b)}.$$

At q=1 all spins agree; at q=-1 all oppose; q near zero means roughly half
agree. The vertical axis is **probability mass in a discrete q bin**, not a
continuous probability density. L4 has 48 physical spins, so q spacing is 1/24.
Every curve uses 8192 independent replica pairs. Lines join bins as a visual aid.

**Two distinct comparisons answer our research question:** changing fields at
fixed RG retention shows whether P(q) moves; changing retention at fixed fields
shows how sensitive that result is to the approximation. The main L4 plots
contain RG curves. Exact curves appear only in the separate L2 checks.
'''
    text += f'\n**Snapshot:** {len(rows)}/192 L4 models and {len(small)}/64 exact L2 contexts saved. '
    text += ('The scan is complete.' if complete else 'The scan is still running; gray cells and missing curves mean pending results.')
    text += f' Of {len(panels)} fully comparable field contexts, **{sum(p["passed"] for p in panels)} pass the provisional retention screen**.\n'
    text += '\n| K range | Complete contexts | Passing contexts | Most frequent failed checks |\n| :--- | ---: | ---: | :--- |\n'
    for scale in SCALES:
        pp=[p for p in panels if p['scale']==scale]
        failures={name:sum(not p['checks'][name] for p in pp) for name in
                  ('m histogram','chi histogram','chi energy','chi q2','L2 exact','alarms')}
        failed=', '.join(f'{name}: {n}/{len(pp)}' for name,n in sorted(failures.items(),key=lambda x:-x[1]) if n)
        text+=f'| U({.5*scale:g}, {1.5*scale:g}) | {len(pp)}/16 | {sum(p["passed"] for p in pp)} | {failed or ("none" if pp else "pending")} |\n'
    text+='\nA field-dependent curve can still have substantial truncation bias. The figures below show motion first, then the retention checks. One disorder and retention agreement cannot certify a phase transition or full-spin L4 accuracy.\n'
    cells=[nbformat.v4.new_markdown_cell(text)]
    # Compact physical response maps, all with the same color range.
    fig,axs=plt.subplots(1,4,figsize=(15,4.8),layout='constrained')
    for ax,scale in zip(axs,SCALES):
        values=np.full((4,4),np.nan)
        for i,g in enumerate(GAMMAS):
            for j,h in enumerate(FIELDS):
                r=lookup.get((scale,g,h,6,16))
                if r is not None: values[i,j]=r['metrics']['q2']
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dddddd')
        im=ax.imshow(values,vmin=0,vmax=1,cmap=cmap,aspect='equal')
        for i in range(4):
            for j in range(4):
                value=values[i,j]
                p=panel_lookup.get((scale,GAMMAS[i],FIELDS[j]))
                label='pending' if np.isnan(value) else f'{value:.3f}'+(' *' if p and p['passed'] else '')
                ax.text(j,i,label,ha='center',va='center',fontsize=9,
                        color='white' if np.isfinite(value) and value<.55 else 'black')
        ax.set(title=f'K ~ U({.5*scale:g}, {1.5*scale:g})',xlabel='Longitudinal field h',
               xticks=range(4),xticklabels=[f'{h:g}' for h in FIELDS],
               yticks=range(4),yticklabels=[f'{g:g}' for g in GAMMAS])
    axs[0].set_ylabel('Transverse field Gamma')
    fig.colorbar(im,ax=axs,label='Physical <q²>  (RG m=6, chi=16)',shrink=.8)
    name='clear_q2_maps.png';fig.savefig(output/name,dpi=160);plt.close(fig)
    cells.append(image_cell('## 1. Does the physical overlap change with field?\n\n'
        'Each square gives **physical ⟨q²⟩**, evaluated directly in the retained state. '
        'Higher values mean independent snapshots have a larger overlap magnitude. '
        'Read across to vary h; read down to vary Gamma. All four maps use the same color scale. '
        'An asterisk marks a provisional passing context. Gray means pending. '
        'These are finite-system moments, not fitted q_EA values.\n\n'
        f'![Physical overlap maps](attachment:{name})',name))
    # Direct field changes on one common representative h, showing actual curves.
    h=.075
    fig,axs=plt.subplots(1,4,figsize=(16,4.8),sharex=True,layout='constrained')
    colors=('#0072B2','#009E73','#E69F00','#CC79A7')
    for ax,scale in zip(axs,SCALES):
        count=0
        for g,color in zip(GAMMAS,colors):
            r=lookup.get((scale,g,h,6,16))
            if r is not None:
                ax.plot(np.linspace(-1,1,49),r['mass'],'.-',color=color,ms=3,
                        label=f'Gamma={g:g}; <q²>={r["metrics"]["q2"]:.3f}')
                count+=1
        ax.set(title=f'K ~ U({.5*scale:g}, {1.5*scale:g})',xlabel='Physical overlap q',xlim=(-1,1),ylim=(0,None))
        ax.grid(alpha=.2)
        if count: ax.legend(fontsize=8)
        if count<4: ax.text(.03,.97,f'{count}/4 fields available',transform=ax.transAxes,va='top',fontsize=9)
    axs[0].set_ylabel('Probability mass per q bin')
    name='clear_field_motion.png';fig.savefig(output/name,dpi=160);plt.close(fig)
    cells.append(image_cell('## 2. What does that change look like in P(q)?\n\n'
        '**Hold h=0.075 and retention (m,chi)=(6,16) fixed.** Each color is a different '
        'Gamma; each panel uses a different K range. Compare these curves to see field-driven '
        'motion directly. This slice is a common reference field, not a selected optimal window. '
        'Sampling bars are omitted here for readability; the detailed grids retain them.\n\n'
        f'![Field-driven distribution motion](attachment:{name})',name))
    fig,axs=plt.subplots(2,4,figsize=(15,8),layout='constrained')
    cmap=plt.get_cmap('RdYlGn_r').copy();cmap.set_bad('#dddddd')
    for j,scale in enumerate(SCALES):
        for row in range(2):
            ax=axs[row,j];values=np.full((4,4),np.nan)
            for p in panels:
                if p['scale']==scale:
                    value=max(p['tv_m'],p['tv_chi'])/.05 if row==0 else p['de']/.001
                    values[GAMMAS.index(p['gamma']),FIELDS.index(p['h'])]=max(value,.01)
            im=ax.imshow(values,cmap=cmap,norm=LogNorm(vmin=.1,vmax=10))
            for i in range(4):
                for k in range(4):
                    v=values[i,k]
                    label='pending' if np.isnan(v) else f'{v:.1f}x'
                    ax.text(k,i,label,ha='center',va='center',fontsize=9)
            ax.set(title=f'Mean K={scale:g}',xlabel='h',xticks=range(4),
                   xticklabels=[f'{h:g}' for h in FIELDS],yticks=range(4),
                   yticklabels=[f'{g:g}' for g in GAMMAS])
    axs[0,0].set_ylabel('Gamma\nHistogram change / tolerance')
    axs[1,0].set_ylabel('Gamma\nEnergy change / tolerance')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Measured change / allowed change (1 = threshold)',shrink=.8)
    name='clear_retention_checks.png';fig.savefig(output/name,dpi=160);plt.close(fig)
    cells.append(image_cell('## 3. Is that field response stable under RG retention?\n\n'
        '**Values below 1 pass the displayed check; values above 1 fail.** '
        'Top: the larger of the two histogram changes, divided by the TV tolerance 0.05. '
        'Bottom: the chi=8 to 16 energy change per spin, divided by mean K and then by 0.001. '
        'Green indicates a smaller change; red indicates a larger change. Color saturates at '
        '0.1 and 10; labels give the measured ratio. Gray means the comparison is incomplete. '
        'The overall screen also checks q2, exact L2 errors and numerical alarms.\n\n'
        'TV(P1,P2)=½ Σq |P1(q)−P2(q)|; TV=0 means identical histograms. Independent sampling '
        'adds a noise floor. Energy sensitivity alone does not control P(q) error.\n\n'
        f'![Retention check maps](attachment:{name})\n\n'
        'The detailed grids that follow use **colors for retention**, whereas the field-motion '
        'figure above uses **colors for Gamma**. In each detailed L4 panel, blue is (4,8), '
        'orange is (6,8), and green is (6,16). The m change is blue→orange; the chi change '
        'is orange→green. Here m is the states kept per triangle and chi the states kept '
        'per four-triangle superblock. L2 black curves are exact; no L4 curve is exact.',name))
    path=output/'triangle_effective_bond_scan.ipynb'
    nb=nbformat.read(path,as_version=4)
    # Idempotent: remove an earlier reader summary if rerun.
    nb.cells=[c for c in nb.cells if not c.get('metadata',{}).get('reader_summary')]
    for c in cells: c.metadata['reader_summary']=True
    nb.cells=cells+nb.cells
    nbformat.validate(nb);nbformat.write(nb,path)
    (output/'reader_summary.json').write_text(json.dumps(dict(l4_models=len(rows),l2_contexts=len(small),
        completed_contexts=len(panels),passing_contexts=sum(p['passed'] for p in panels)),indent=2)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args();clarify(args.results,args.output)
