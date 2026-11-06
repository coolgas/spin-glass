"""Executed exact/Jastrow/RBM P(q) notebook following the existing overlap template."""
import argparse
import json
from pathlib import Path

import numpy as np
import nbformat as nb
from nbclient import NotebookClient

from .collective_campaign import CONTEXTS,label,write
from .correlation_lengths import sha
from .simple_overlap_scan import checked_pq


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scan',type=Path)
    p.add_argument('--sampled',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    fits=json.loads((args.screen/'review.json').read_text())['rows'];rows=[]
    for gamma,sid in CONTEXTS:
        gp=args.inputs/'guides'/(label(gamma,sid)+'.npz')
        with np.load(gp) as data:exact=data['exact_amplitude'].copy()
        q,truth=checked_pq(exact**2)
        for kind in ['jastrow','rbm_1','rbm_2']:
            subset=[r for r in fits if (r['gamma'],r['sample_id'],r['kind'])==(gamma,sid,kind)]
            best=min(subset,key=lambda r:r['energy'])['seed']
            for fit in subset:
                name=f"{label(gamma,sid)}_{kind}_seed{fit['seed']}"
                path=args.screen/(name+'.npz');assert sha(path)==fit['npz_sha256']
                with np.load(path) as data:logs=data['logs'].copy()
                _,mass=checked_pq(np.exp(2*(logs-logs.max())))
                rows.append(dict(gamma=gamma,sample_id=sid,kind=kind,seed=fit['seed'],selected=fit['seed']==best,
                    q=q.tolist(),exact_pq=truth.tolist(),guide_pq=mass.tolist(),
                    tv=float(abs(mass-truth).sum()/2),energy=fit['energy'],
                    guide_sha256=sha(path),exact_input_sha256=sha(gp)))
    write(args.output/'distributions.json',dict(rows=rows,scope='Unprojected guide probabilities, selected L2 contexts; no disorder averaging across different fields'))
    if args.sampled:
        sampled=[]
        for record in json.loads((args.sampled/'review.json').read_text())['rows']:
            t=record['task']
            if t['L']!=2:continue
            name='L{L}_s{sample_id}_g{gamma:g}_{kind}_{mode}_seed{seed}'.format(**t).replace('.','p')
            path=args.sampled/(name+'.npz');assert sha(path)==record['npz_sha256']
            gp=args.inputs/'guides'/(label(t['gamma'],t['sample_id'])+'.npz')
            with np.load(gp) as data:q,truth=checked_pq(data['exact_amplitude']**2)
            with np.load(path) as data:_,mass=checked_pq(np.exp(2*data['logs']))
            sampled.append(dict(**t,q=q.tolist(),exact_pq=truth.tolist(),guide_pq=mass.tolist(),
                tv=float(abs(mass-truth).sum()/2),source_sha256=sha(path)))
        write(args.output/'sampled_distributions.json',dict(rows=sampled,
            scope='Frozen sampled-trained L2 guides, enumerated only for post-fit evaluation; original unequal-update schedules'))
    if args.scan:
        scan=json.loads((args.scan/'review.json').read_text())
        for row in scan['rows']:
            path=args.scan/(label(row['gamma'],row['sample_id'])+'.npz')
            assert sha(path)==row['npz_sha256']
        write(args.output/'scan_snapshot.json',scan)
    relative=args.output.resolve().relative_to(root)
    cells=[nb.v4.new_markdown_cell(r"""# Exact, Jastrow and RBM overlap distributions

This follows `exact_guide_overlap_distributions.ipynb`: two independent spin
configurations of the **same disorder** give $q=N^{-1}\sum_i s_i t_i$.
For $L=2$, $N=12$, there are 13 allowed values from -1 to 1, spaced by 1/6.
The plotted ordinate is **probability mass per bin**, whose heights sum to one.
Connecting lines are visual aids, not a continuous density.

$$P_J(q)=\sum_{s,t}p_J(s)p_J(t)\,\mathbf1[N^{-1}s\cdot t=q].$$

Exact uses $p=\psi_0^2$; each unprojected guide uses normalized $p=\psi_T^2$.
The XOR/Walsh convolution matches the original notebook. Normalization and
first/second moments are independently checked from magnetizations and pairs.
All plots here are **unprojected**. Saved GFMC moment spectra cannot reconstruct
the full $P(q)$; a separate histogram estimator would be required.

The first comparison shows the four existing diagnostic contexts at $h=0.15$.
Guides were fitted using full-basis variational energy, without ground-state
labels. Minimum-energy seed selection precedes evaluation. This is an L2
ansatz/optimization test, not scalable sampled-training validation.
"""),nb.v4.new_code_cell(f"""from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
root=Path.cwd()
folder=root/{str(relative)!r}
rows=json.loads((folder/'distributions.json').read_text())['rows']
contexts=[(.25,50025),(.25,50059),(.5,50043),(.75,50004)]
kinds=['jastrow','rbm_1','rbm_2']
figures=root/'figures/simple_overlap';figures.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for ax,(g,s) in zip(axes.flat,contexts):
    group=[r for r in rows if (r['gamma'],r['sample_id'])==(g,s)]
    q=np.array(group[0]['q']);ax.plot(q,group[0]['exact_pq'],'ko-',label='Exact',lw=2)
    for kind in kinds:
        r=next(r for r in group if r['kind']==kind and r['selected'])
        ax.plot(q,r['guide_pq'],'o--',label=f"{{kind}}, TV={{r['tv']:.4f}}")
    ax.set_title(f'Gamma={{g}}, disorder {{s}}, h=0.15');ax.set_xlabel('q');ax.set_ylabel('P(q) mass')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Exact and energy-selected unprojected Jastrow/RBM guides, L=2')
fig.tight_layout();fig.savefig(figures/'selected_guides_pq.png',dpi=180);fig.savefig(figures/'selected_guides_pq.pdf');plt.show()
"""),nb.v4.new_markdown_cell("""## All optimization seeds

Faint curves retain every seed, including fits trapped in another sector. A good
selected curve does not imply reliable optimization. These panels show individual
disorders, so no disorder error bars are attached. TV is half the sum of absolute
bin-probability differences from exact and lies between zero and one.
"""),nb.v4.new_code_cell("""fig,axes=plt.subplots(3,4,figsize=(15,9),sharex=True)
for i,kind in enumerate(kinds):
    for j,(g,s) in enumerate(contexts):
        ax=axes[i,j];group=[r for r in rows if (r['gamma'],r['sample_id'],r['kind'])==(g,s,kind)]
        q=np.array(group[0]['q']);ax.plot(q,group[0]['exact_pq'],'k-',lw=2,label='Exact')
        for r in group:
            ax.plot(q,r['guide_pq'],'o--',alpha=1 if r['selected'] else .35,
                    label=f"seed {r['seed']}, TV={r['tv']:.3f}"+(' selected' if r['selected'] else ''))
        ax.set_title(f'{kind}: Gamma={g}, s={s}');ax.set_xlabel('q');ax.set_ylabel('P(q)');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(figures/'all_seeds_pq.png',dpi=180);fig.savefig(figures/'all_seeds_pq.pdf');plt.show()
""")]
    if args.sampled:
        cells.extend([nb.v4.new_markdown_cell("""## Guides learned with sampled VMC

These fits used parallel-tempering samples during training. Each panel shows
both seeds and direct/continuation schedules; no seed is selected. The curves
are computed from frozen guide probabilities after training, using L2-only
enumeration to remove histogram sampling noise. In this original campaign,
continuation used more updates than direct training. A separate matched-update
field campaign addresses that confound.
"""),nb.v4.new_code_cell("""sampled=json.loads((folder/'sampled_distributions.json').read_text())['rows']
fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True)
for i,kind in enumerate(['jastrow','rbm_2']):
    for j,(g,s) in enumerate(contexts):
        ax=axes[i,j];group=[r for r in sampled if (r['kind'],r['gamma'],r['sample_id'])==(kind,g,s)]
        q=np.array(group[0]['q']);ax.plot(q,group[0]['exact_pq'],'k-',lw=2,label='Exact')
        for r in group:
            ax.plot(q,r['guide_pq'],ls='--' if r['mode']=='direct' else ':',
                label=f"{r['mode']} seed {r['seed']}, TV={r['tv']:.3f}")
        ax.set_title(f'{kind}: Gamma={g}, s={s}');ax.set_xlabel('q');ax.set_ylabel('P(q) mass');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(figures/'sampled_guides_pq.png',dpi=180);fig.savefig(figures/'sampled_guides_pq.pdf');plt.show()
""")])
    if args.scan:
        cells.extend([nb.v4.new_markdown_cell(r"""## Gradually increasing Gamma: physical structure and numerical reliability

Hold $h=0.15$ and each disorder fixed. Fields are 0.50, 0.55, 0.60, 0.65, 0.70,
0.75, 0.85 and 1.00. The eight IDs 50000–50007 form the matched disorder set;
50043 is shown separately as a difficult-case diagnostic. Compute $P_J$ first,
then average across disorders; never pair states from different disorders.
Error bars are one descriptive standard error across disorders, not training
uncertainty or Monte Carlo error.

We track exact overlap variance and gap separately from guide TV, spectral error
and seed reliability. Broad or changing P(q) in L2 is evidence of a finite-size
crossover, not a thermodynamic phase. Convergence failure cannot define a phase.
A candidate field needs both retained physical structure and stable algorithms;
this full-basis fit screen alone cannot qualify GFMC or larger L.
"""),nb.v4.new_code_cell("""scan=json.loads((folder/'scan_snapshot.json').read_text())
print('Completed field/disorder contexts:',len(scan['rows']),'of 72')
summary=scan['summary'];gammas=sorted(set(r['gamma'] for r in summary))
fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True)
for ax,g in zip(axes.flat,gammas):
    group=[r for r in summary if r['gamma']==g];q=np.linspace(-1,1,13)
    ax.errorbar(q,group[0]['exact_pq'],yerr=group[0]['exact_disorder_se'],fmt='ko-',label='Exact')
    for r in group:
        ax.errorbar(q,r['guide_pq'],yerr=r['guide_disorder_se'],fmt='o--',label=r['kind'])
    ax.set_title(f"Gamma={g}, D={group[0]['disorders']}");ax.set_xlabel('q');ax.set_ylabel('mean P(q)');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(figures/'field_scan_pq.png',dpi=180);fig.savefig(figures/'field_scan_pq.pdf');plt.show()
fig,axes=plt.subplots(1,3,figsize=(14,4))
for kind in ['jastrow','rbm_2']:
    group=sorted([r for r in summary if r['kind']==kind],key=lambda r:r['gamma'])
    x=[r['gamma'] for r in group]
    axes[0].plot(x,[r['mean_paired_tv'] for r in group],'o-',label=kind)
    axes[1].plot(x,[r['seed_fraction_pq_tv_below_005'] for r in group],'o-',label=kind)
group=sorted([r for r in summary if r['kind']=='jastrow'],key=lambda r:r['gamma'])
axes[2].plot([r['gamma'] for r in group],[r['mean_overlap_variance'] for r in group],'ko-')
axes[0].set_ylabel('Mean paired P(q) TV error');axes[1].set_ylabel('Seed fraction with P(q) TV <= 0.05')
axes[2].set_ylabel('Mean exact overlap variance')
for ax in axes:ax.set_xlabel('Gamma');ax.grid(alpha=.2)
axes[0].legend();axes[1].legend();fig.tight_layout()
fig.savefig(figures/'field_scan_tradeoff.png',dpi=180);fig.savefig(figures/'field_scan_tradeoff.pdf');plt.show()
print('Difficult disorder 50043:')
for r in sorted([r for r in scan['rows'] if r['sample_id']==50043],key=lambda r:r['gamma']):
    print('Gamma',r['gamma'],'gap',r['exact']['gap'],'overlap variance',r['exact']['variance'],
          [(f['kind'],f['pq_tv'],f['max_spectral_error']) for f in r['fits'] if f['selected']])
fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True)
hard=sorted([r for r in scan['rows'] if r['sample_id']==50043],key=lambda r:r['gamma'])
for ax,r in zip(axes.flat,hard):
    q=np.array(r['exact']['q']);ax.plot(q,r['exact']['pq'],'ko-',label='Exact')
    for f in r['fits']:
        if f['selected']:ax.plot(q,f['pq'],'o--',label=f"{f['kind']}, TV={f['pq_tv']:.3f}")
    ax.set_title(f"Gamma={r['gamma']}, gap={r['exact']['gap']:.3g}")
    ax.set_xlabel('q');ax.set_ylabel('P(q) mass');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.suptitle('Matched difficult disorder 50043: exact and unprojected guides')
fig.tight_layout();fig.savefig(figures/'difficult_disorder_scan.png',dpi=180)
fig.savefig(figures/'difficult_disorder_scan.pdf');plt.show()
"""),nb.v4.new_markdown_cell(r"""## Candidate field region

The completed full-basis screen identifies **$\Gamma=0.60$–$0.70$ as an RBM
candidate window**. On disorder 50043 the exact negative-overlap shoulder grows
through this interval while the selected RBM TV error falls from 0.0060 to
0.0016. At $\Gamma=0.65$, TV is 0.0022 and connected-spectrum error is 0.94%.
All three RBM seeds on this disorder have TV below 0.01 from 0.60 onward. Across
the separate eight-disorder ensemble, all 24 RBM seeds have TV below 0.05 from
0.60 onward. These descriptive thresholds do not certify ground-state sampling.

Jastrow remains unreliable here: its selected TV on 50043 rises to 0.243 at
0.65. The apparent window is therefore ansatz-dependent. Negative-overlap weight
alone is not a phase identifier; it can also rise as a distribution approaches
a symmetric paramagnet. The shoulder and two maxima in this small system are
features to follow with size, not evidence of thermodynamic replica symmetry
breaking.

Two independent follow-ups have been launched on AutoDL: equal-update sampled
VMC at $L=2,4$ (1200 updates per fit, direct versus continuation), and CT GFMC
population/time sensitivity over all eight fields. Their live records are under
`data/kagome/ltfim_simple_overlap_v1/sampled/` and `projector/`. This notebook's
field recommendation is based on the completed full-basis screen; it does not
claim those follow-ups have converged.
""")])
    notebook=nb.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
    path=root/'notebook/jastrow_rbm_overlap_distributions.ipynb';nb.write(notebook,path)
    print('Executed',path,flush=True)


if __name__=='__main__':main()
