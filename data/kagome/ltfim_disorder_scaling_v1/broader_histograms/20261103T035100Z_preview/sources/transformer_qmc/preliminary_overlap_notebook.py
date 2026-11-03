"""Freeze a pilot snapshot, sample frozen-guide P(q), and execute a review notebook.

Run on AutoDL only. No production scientific source or checkpoint is modified.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import nbformat as nb
from nbclient import NotebookClient
import numpy as np
import torch

from .disorder_scaling import reduce_results
from .report_disorder_scaling import report
from .parallel_disorder_scaling import fit_name
from .sampled_simple_guides import TemperedChains
from .simple_guides import Jastrow, RBM


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_sampler():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@torch.no_grad()
def sample_one(record, production, snapshot, selected):
    """Independent chain columns form pairs; temporal samples remain correlated."""
    t=record['task']; name=fit_name(t)
    destination=snapshot/'guide_samples'/(name+'.npz')
    checkpoint=production/'training'/(name+'.pt')
    assert digest(checkpoint)==record['checkpoint_sha256']
    model=(Jastrow(48) if t['kind']=='jastrow' else RBM(48,2)).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False)['model'])
    model.eval(); arrays={}; diagnostics=[]; start_time=time.monotonic()
    for start in ['random','up','down']:
        seed=int(hashlib.sha256((name+start).encode()).hexdigest()[:8],16)
        sampler=TemperedChains(model,128,8,seed,start)
        values=[]
        for step in range(300):
            states=sampler.sweep()
            if step>=100 and (step-100)%4==0:
                # Disjoint columns have independent RNG histories. Never pair
                # two configurations taken from a single chain trajectory.
                mismatch=(states[:64]!=states[64:]).sum(1)
                values.append(mismatch.cpu().numpy())
        counts=np.array(values,dtype=np.int16)
        arrays[start+'_mismatches']=counts
        mass=np.bincount(counts.ravel(),minlength=49)[::-1]/counts.size
        grid=np.linspace(-1,1,49)
        assert np.isclose(mass.sum(),1)
        np.testing.assert_allclose(grid@mass,(1-2*counts/48).mean(),atol=1e-14)
        diagnostics.append(dict(start=start,mass=mass.tolist(),mean=float(grid@mass),
            second_moment=float(grid**2@mass),negative_mass=float(mass[grid<0].sum()),
            pairs=int(counts.size),**sampler.diagnostics()))
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**arrays)
    return dict(task=t,selected=name in selected,q_grid=grid.tolist(),
        starts=diagnostics,checkpoint_sha256=digest(checkpoint),
        raw_file=str(destination.relative_to(snapshot)),raw_sha256=digest(destination),
        seconds=time.monotonic()-start_time)


def sample_guides(production, snapshot):
    fits=json.loads((snapshot/'training_records.json').read_text())['rows']
    selections=json.loads((snapshot/'selection_records.json').read_text())['rows']
    selected={fit_name(r['selected']) for r in selections}
    path=snapshot/'guide_distributions.json'
    rows=json.loads(path.read_text())['rows'] if path.exists() else []
    done={fit_name(r['task']) for r in rows}
    pending=[r for r in fits if r['task']['L']==4 and r['task']['gamma'] in [.6,.65] and fit_name(r['task']) not in done]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=configure_sampler) as pool:
        futures=[pool.submit(sample_one,r,production,snapshot,selected) for r in pending]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            dump(path,dict(rows=rows,
                definition='Unprojected frozen guide psi_T squared; finite parallel-tempering MCMC',
                protocol='128 independent chain columns, 8 temperatures, 100 burn-in sweeps, 200 retained-window sweeps, stride 4; pair disjoint 64-column halves; random/up/down starts',
                uncertainty='Temporal dependence and finite burn-in remain; bootstrap bands cover disorder variation only'))
            print('sampled',fit_name(row['task']),round(row['seconds'],2),flush=True)


def make_notebook(snapshot, output):
    relative=snapshot.relative_to(output.parent.parent)
    cells=[]
    def md(text): cells.append(nb.v4.new_markdown_cell(text))
    def code(text): cells.append(nb.v4.new_code_cell(text))
    md(r'''# Preliminary overlap distributions and disorder scaling at $h=0.15$

This notebook freezes the running $L=4,6$ pilot at the timestamp below. Outputs are
executed on AutoDL and can be viewed locally without running any computation.

**Can we still see peaks at $\Gamma=0.60$ and $0.65$?** The first plots show
the exact $L=2$ reference, followed by newly sampled, unprojected $L=4$ guide
distributions. The GFMC campaign stores $q$, $q^2$, and connected spectra; these
moments do not determine the full projected $P(q)$. No projected peak claim is made.

For two independent configurations of the same disorder,
$q_{st}=N^{-1}\sum_i s_i t_i$. Curves show **probability mass per allowed value**,
not a continuous density. Connecting lines do not imply interpolation physics.
At nonzero field the mean overlap includes polarization. Multiple finite-size
maxima alone do not establish a thermodynamic glass phase.''')
    code(f'''%matplotlib inline
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data/kagome').is_dir())
folder=root/{str(relative)!r}
load=lambda name: json.loads((folder/name).read_text())
meta=load('snapshot.json'); review=load('review.json'); sensitivity=load('sensitivity.json')
runs=load('run_records.json')['rows']; fits=load('training_records.json')['rows']
guides=load('guide_distributions.json')['rows']; exact=load('exact_l2_reference.json')
expected={{(kind,g,sid,seed) for kind in ['jastrow','rbm_2'] for g in [.6,.65]
          for sid in range(60000,60008) for seed in [0,1]}}
assert len(guides)==64
assert {{tuple(r['task'][k] for k in ['kind','gamma','sample_id','seed']) for r in guides}}==expected
assert sum(r['selected'] for r in guides)==32
figures=folder/'figures'; figures.mkdir(exist_ok=True)
def save(fig,name):
    fig.tight_layout(); fig.savefig(figures/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(figures/(name+'.pdf'),bbox_inches='tight'); plt.show()
def table(headers, rows):
    display(Markdown('| '+' | '.join(headers)+' |\\n| '+' | '.join(['---']*len(headers))+' |\\n'+
        '\\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)))
print('Snapshot UTC:',meta['created_utc'])
print('Completed fits:',len(fits),'/ 320; completed GFMC runs:',len(runs),'/ 560')
print('Live campaign status at capture:',meta['parallel_status']['state'])
print('Captured execution failures:',len(meta['failures']))
print('All GFMC points remain ground_state_qualified=False.')
''')
    md('''## Exact small-system reference: peaks and disorder variation

The eight-disorder average uses IDs 50000–50007. The separate difficult disorder
50043 illustrates structure that an ensemble average can obscure. These $L=2$
data use a different disorder ensemble from the current $L=4$ pilot. Full-basis
guide fits in this reference are also distinct from the current sampled fits.
Bands are descriptive 95% whole-disorder bootstrap intervals, pointwise in $q$.''')
    code('''fig,axes=plt.subplots(1,3,figsize=(16,4.5),sharex=True)
rng=np.random.default_rng(817)
for g,color in [(.6,'tab:blue'),(.65,'tab:orange')]:
    group=[r for r in exact['rows'] if r['gamma']==g and r['sample_id']!=50043]
    grid=np.array(group[0]['exact']['q']); a=np.array([r['exact']['pq'] for r in group])
    boot=a[rng.integers(len(a),size=(2000,len(a)))].mean(1)
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    axes[0].plot(grid,a.mean(0),'o-',color=color,label=f'Gamma={g}, D={len(a)}')
    axes[0].fill_between(grid,lo,hi,color=color,alpha=.15)
    hard=next(r for r in exact['rows'] if r['gamma']==g and r['sample_id']==50043)
    axes[1].plot(grid,hard['exact']['pq'],'o-',color=color,label=f'Gamma={g}')
    for r in group: axes[2].plot(grid,r['exact']['pq'],color=color,alpha=.25)
axes[0].set_title('Exact L=2 disorder average'); axes[1].set_title('Exact L=2 difficult disorder 50043')
axes[2].set_title('Every reference disorder')
for ax in axes: ax.set(xlabel='q',ylabel='P(q) mass'); ax.grid(alpha=.2)
axes[0].legend(); axes[1].legend(); save(fig,'exact_l2_peaks')
peak_rows=[]
for r in exact['rows']:
    p=np.array(r['exact']['pq']); q=np.array(r['exact']['q'])
    peaks=[round(float(q[i]),3) for i in range(len(p))
           if (i==0 or p[i]>p[i-1]) and (i==len(p)-1 or p[i]>p[i+1])]
    peak_rows.append([r['gamma'],r['sample_id'],peaks,f"{p[q<0].sum():.4f}"])
table(['Gamma','L=2 disorder','Discrete local maxima in q','P(q<0)'],peak_rows)
''')
    md(r'''## Current L=4 frozen-guide distributions near Gamma = 0.60 and 0.65

These curves sample $|\psi_T|^2$ after training, **before GFMC projection**.
The lower-energy seed is selected using the existing frozen energy diagnostics.
Every seed and random/up/down start remains in the saved data. Each histogram
uses independent chain columns for the two physical replicas; successive draws
are correlated in Monte Carlo time. Each start has only 100 burn-in sweeps and
50 retained draws of 64 pairs. Start disagreement flags unresolved mixing.

The main curves average the three starts, then average the eight disorders
equally. Bands resample disorders and do not certify MCMC convergence. Peak
counts in noisy 49-bin histograms are not used as a physical diagnostic.''')
    code('''def mass(r): return np.mean([s['mass'] for s in r['starts']],axis=0)
colors={.6:'tab:blue',.65:'tab:orange'}
fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharex=True)
guide_stats=[]
for ax,kind in zip(axes,['jastrow','rbm_2']):
    for g in [.6,.65]:
        group=sorted([r for r in guides if r['selected'] and r['task']['kind']==kind and r['task']['gamma']==g],key=lambda r:r['task']['sample_id'])
        a=np.array([mass(r) for r in group]); grid=np.array(group[0]['q_grid'])
        boot=a[np.random.default_rng(817).integers(len(a),size=(2000,len(a)))].mean(1)
        lo,hi=np.quantile(boot,[.025,.975],axis=0); mean=a.mean(0)
        ax.plot(grid,mean,'o-',ms=3,color=colors[g],label=f'Gamma={g}, D={len(a)}')
        ax.fill_between(grid,lo,hi,color=colors[g],alpha=.15)
        guide_stats.append([kind,g,len(a),f'{grid[np.argmax(mean)]:.3f}',f'{grid@mean:.4f}',f'{mean[grid<0].sum():.4f}'])
    ax.set(title=f'L=4 {kind}: unprojected guide',xlabel='q',ylabel='P(q) mass'); ax.legend(); ax.grid(alpha=.2)
save(fig,'l4_guide_pq'); table(['Guide','Gamma','D','Largest bin at q','Mean q','P(q<0)'],guide_stats)
display(Markdown('The sampled L=4 guide averages retain a positive-overlap peak. '
    'Inspect the individual-disorder and start panels below before interpreting '
    'a small shoulder. These curves describe the frozen guide probability; '
    'the projected GFMC peak shape remains unmeasured.'))
''')
    code('''for kind in ['jastrow','rbm_2']:
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True)
    for ax,sid in zip(axes.flat,range(60000,60008)):
        for r in guides:
            t=r['task']
            if t['kind']!=kind or t['sample_id']!=sid: continue
            ax.plot(r['q_grid'],mass(r),color=colors[t['gamma']],
                ls='-' if r['selected'] else '--',alpha=1 if r['selected'] else .4,
                label=f"g={t['gamma']}, seed={t['seed']}"+(' selected' if r['selected'] else ''))
        ax.set(title=f'Disorder {sid}',xlabel='q',ylabel='P(q) mass'); ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.suptitle(f'L=4 {kind}: all disorders and both training seeds; unprojected')
    save(fig,kind+'_all_seeds_pq')
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
for i,kind in enumerate(['jastrow','rbm_2']):
    for j,g in enumerate([.6,.65]):
        ax=axes[i,j]; group=[r for r in guides if r['selected'] and r['task']['kind']==kind and r['task']['gamma']==g]
        for start in ['random','up','down']:
            a=np.array([next(s['mass'] for s in r['starts'] if s['start']==start) for r in group])
            ax.plot(group[0]['q_grid'],a.mean(0),label=start)
        tv=max(sum(abs(np.array(a['mass'])-np.array(b['mass'])))/2 for r in group for a in r['starts'] for b in r['starts'])
        ax.set(title=f'{kind}, Gamma={g}; max within-disorder start TV={tv:.3f}',xlabel='q',ylabel='P(q) mass')
        ax.legend(); ax.grid(alpha=.2)
save(fig,'guide_start_sensitivity')
mixing=[]
for r in sorted([r for r in guides if r['selected']],key=lambda r:tuple(r['task'][k] for k in ['kind','gamma','sample_id'])):
    t=r['task']; starts=r['starts']
    tv=max(np.abs(np.array(a['mass'])-np.array(b['mass'])).sum()/2 for a in starts for b in starts)
    mixing.append([t['kind'],t['gamma'],t['sample_id'],t['seed'],f'{tv:.4f}',
        f"{min(s['mean'] for s in starts):.4f} to {max(s['mean'] for s in starts):.4f}",
        min(s['round_trips'] for s in starts),min(min(s['exchange_fraction']) for s in starts)])
table(['Guide','Gamma','Disorder','Selected seed','Max start TV','Start mean q range',
       'Minimum temperature round trips','Minimum exchange fraction'],mixing)
''')
    md('''## Guide moments compared with GFMC moments

The histogram mean and second moment can be compared with the independently
projected moments for exactly the same disorders. Agreement of two moments
would still leave the peak structure undetermined. Differences can reflect
guide error, MCMC error, or finite-population and projection effects.''')
    code('''comparison=[]
for kind in ['jastrow','rbm_2']:
    for g in [.6,.65]:
        rows=[r for r in review['rows'] if (r['L'],r['gamma'],r['kind'],r['audit'])==(4,g,kind,'main')]
        if not rows: continue
        r=rows[0]; ids=r['disorders']
        group=[x for x in guides if x['selected'] and x['task']['kind']==kind and x['task']['gamma']==g and x['task']['sample_id'] in ids]
        a=np.mean([mass(x) for x in group],axis=0); grid=np.array(group[0]['q_grid'])
        comparison.append([kind,g,len(ids),f'{grid@a:.5f}',f"{r['q']:.5f}",f'{grid**2@a:.5f}',f"{r['q2']:.5f}"])
table(['Guide','Gamma','Matched D','Guide mean q','GFMC mean q','Guide mean q²','GFMC mean q²'],comparison)
''')
    md(r'''## GFMC moments and connected correlation lengths: every available field

The mean overlap is $[N^{-1}\sum_i\langle Z_i\rangle^2]_J$ and the second
moment is $[N^{-2}\sum_{ij}\langle Z_i Z_j\rangle^2]_J$. Products use distinct
physical populations. Connected replicon lengths use the **ratio of disorder
mean spectra**, with $|k_{\min}|=2\pi/(\sqrt3 L)$:
$\xi_R/L=\frac{\sqrt3}{2\pi}\sqrt{\overline S(0)/\overline S(k_{\min})-1}$.
Invalid ratios remain unresolved; no clipping is applied.

Only disorders with both repeats complete enter a displayed mean. Open markers
denote incomplete disorder coverage and are not connected into a field curve.
Whole-disorder intervals describe the pilot ensemble; they do not include a
numerical-bias bound. Alarmed samples retain their equal disorder weight.''')
    code('''main=[r for r in review['rows'] if r['audit']=='main']
def fmt(x): return 'unresolved' if x is None else f'{x:.5f}'
table(['L','Gamma','Guide','D/8','q','q²','xi_R/L','xi 95% interval','Alarmed runs','Invalid bootstrap fraction'],
    [[r['L'],r['gamma'],r['kind'],f"{len(r['disorders'])}/8",fmt(r['q']),fmt(r['q2']),fmt(r['xi_over_L']),
      r['xi_interval'],f"{r['ancestry_alarms']}/{2*len(r['disorders'])}",r['invalid_bootstrap_fraction']] for r in main])
for kind in ['jastrow','rbm_2']:
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for L,color in [(4,'tab:blue'),(6,'tab:orange')]:
        group=sorted([r for r in main if r['kind']==kind and r['L']==L],key=lambda r:r['gamma'])
        for ax,key,ci in zip(axes,['q','q2','xi_over_L'],['q_interval',None,'xi_interval']):
            full=[r for r in group if len(r['disorders'])==8 and r[key] is not None]
            if full: ax.plot([r['gamma'] for r in full],[r[key] for r in full],'-',color=color,label=f'L={L}, D=8')
            for r in group:
                if r[key] is None: continue
                ax.plot(r['gamma'],r[key],'o',color=color,mfc=color if len(r['disorders'])==8 else 'none')
                if ci and r[ci]: ax.vlines(r['gamma'],*r[ci],color=color)
                ax.annotate(f"D={len(r['disorders'])}",(r['gamma'],r[key]),xytext=(3,5),textcoords='offset points',fontsize=7)
            ax.set(xlabel='Gamma',ylabel=key); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle(f'{kind}: unqualified finite-projection GFMC')
    save(fig,kind+'_gfmc_moments')
''')
    code('''fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,kind in zip(axes,['jastrow','rbm_2']):
    for r in main:
        if r['L']==4 and r['kind']==kind and r['gamma'] in [.6,.65]:
            ax.plot(range(4),r['spectrum'],'o-',label=f"Gamma={r['gamma']}, D={len(r['disorders'])}")
    ax.set_xticks(range(4),['S(0)','S(k1)','S(k2)','S(k3)'])
    ax.set(title=f'{kind}: disorder-mean replicon spectrum',ylabel='S_R'); ax.legend(); ax.grid(alpha=.2)
save(fig,'connected_spectra')
coverage=[]
for L in [4,6]:
    for kind in ['jastrow','rbm_2']:
        for g in [.6,.65,.7,.85,1.]:
            rs=[r for r in runs if (r['task']['L'],r['task']['kind'],r['task']['gamma'],r['task']['audit'])==(L,kind,g,'main')]
            counts={sid:sum(r['task']['sample_id']==sid for r in rs) for sid in range(60000,60008)}
            coverage.append([L,kind,g,len(rs),[sid for sid,c in counts.items() if c==2],[sid for sid,c in counts.items() if c==1]])
table(['L','Guide','Gamma','Completed main runs / 16','Two-repeat IDs','One-repeat IDs'],coverage)
''')
    md('''## Disorder variation and paired changes between the two fields

Each point averages the two GFMC repeats of a single disorder. Equal IDs pair
fields within a size; equal IDs across sizes do not describe the same physical
disorder. Paired whole-disorder bootstrap changes use only common completed IDs.
These panels show variation of the **estimated mean overlap across
disorders**. This is a different quantity from the configuration overlap $P(q)$.''')
    code('''def disorder_groups(L,g,kind,audit='main'):
    result={}
    for r in runs:
        t=r['task']
        if (t['L'],t['gamma'],t['kind'],t['audit'])==(L,g,kind,audit): result.setdefault(t['sample_id'],[]).append(r)
    return {sid:rs for sid,rs in result.items() if len(rs)==2}
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
paired=[]
for ax,kind in zip(axes,['jastrow','rbm_2']):
    a=disorder_groups(4,.6,kind); b=disorder_groups(4,.65,kind); ids=sorted(set(a)&set(b))
    for sid in ids:
        y=[np.mean([r['q'] for r in group[sid]]) for group in [a,b]]
        ax.plot([.6,.65],y,'o-',alpha=.65,label=str(sid))
    d=np.array([np.mean([r['q'] for r in b[sid]])-np.mean([r['q'] for r in a[sid]]) for sid in ids])
    if len(d):
        boot=d[np.random.default_rng(817).integers(len(d),size=(2000,len(d)))].mean(1)
        paired.append([kind,len(ids),float(d.mean()),np.quantile(boot,[.025,.975]).tolist()])
    ax.set(title=f'L=4 {kind}',xlabel='Gamma',ylabel='Single-disorder mean overlap'); ax.grid(alpha=.2); ax.legend(fontsize=6,ncol=2)
save(fig,'paired_disorder_overlap'); table(['Guide','Paired D','q(0.65)-q(0.60)','Exploratory 95% interval'],paired)
''')
    md('''## Population, projection-time, and initialization checks

Audits compare the same disorders against the main setting: 1024 versus 512
walkers per population; doubled equilibration and forward times; and uniform
versus guide initialization. Each setting uses four independently branching
physical populations. The audited subset contains at most two disorders.
Small differences cannot establish a bias bound, particularly when ancestry
collapses. All available fields and sizes are included in the table.''')
    code('''table(['L','Gamma','Guide','Audit','Paired IDs','Delta q','Delta xi/L'],
    [[r['L'],r['gamma'],r['kind'],r['audit'],r['paired_disorders'],fmt(r['delta_q']),fmt(r['delta_xi_over_L'])] for r in sensitivity['rows']])
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for kind,marker in [('jastrow','o'),('rbm_2','s')]:
    for audit,color in [('population','tab:blue'),('time','tab:orange'),('start','tab:green')]:
        group=[r for r in sensitivity['rows'] if r['L']==4 and r['kind']==kind and r['audit']==audit]
        for ax,key in zip(axes,['delta_q','delta_xi_over_L']):
            valid=[r for r in group if r[key] is not None]
            ax.scatter([r['gamma'] for r in valid],[r[key] for r in valid],marker=marker,color=color,label=f'{kind}: {audit}')
for ax,key in zip(axes,['Delta q','Delta xi_R/L']):
    ax.axhline(0,color='k',lw=.7); ax.set(xlabel='Gamma',ylabel=key); ax.grid(alpha=.2)
axes[1].legend(fontsize=7); save(fig,'paired_gfmc_audits')
alarm_rows=[]
for r in review['rows']:
    rs=[x for x in runs if all(x['task'][k]==r[k] for k in ['L','gamma','kind','audit']) and x['task']['sample_id'] in r['disorders']]
    alarm_rows.append([r['L'],r['gamma'],r['kind'],r['audit'],len(r['disorders']),f"{sum(x['ancestry_alarm'] for x in rs)}/{len(rs)}",f"{min(x['minimum_descendant_ess'] for x in rs):.2f}"])
table(['L','Gamma','Guide','Setting','D','Alarmed runs','Minimum descendant ESS'],alarm_rows)
''')
    md('''## Raw GFMC cycle diagnostics at the two focus fields

These panels keep all completed main runs, including ancestry alarms. A cycle
overlap uses distinct physical populations at the same cycle; the reported
campaign overlap instead contracts the cycle-averaged moments. They need not
have identical Monte Carlo noise. The minimum ESS is over four populations at
each cycle. Lines connect measurements for visual inspection only.''')
    code('''from itertools import combinations
fig,axes=plt.subplots(2,2,figsize=(13,8))
for col,kind in enumerate(['jastrow','rbm_2']):
    for r in runs:
        t=r['task']
        if t['L']!=4 or t['kind']!=kind or t['gamma'] not in [.6,.65] or t['audit']!='main': continue
        with np.load(folder/r['snapshot_raw_file']) as data:
            m=data['magnetization_cycles']; ess=data['descendant_ess']
            qcycle=np.mean([np.mean(m[:,a]*m[:,b],axis=1) for a,b in combinations(range(4),2)],axis=0)
        x=np.arange(1,len(qcycle)+1)
        axes[0,col].plot(x,qcycle,color=colors[t['gamma']],alpha=.25)
        axes[1,col].plot(x,ess.min(1),color=colors[t['gamma']],alpha=.25)
    axes[0,col].set(title=f'L=4 {kind}',ylabel='Cycle mean overlap')
    axes[1,col].set(xlabel='Measurement cycle',ylabel='Minimum descendant ESS',yscale='log')
    axes[1,col].axhline(8,color='k',ls='--',lw=.8,label='ESS alarm threshold')
    for ax in axes[:,col]: ax.grid(alpha=.2)
axes[1,0].legend(); save(fig,'gfmc_raw_cycles')
''')
    md('''## Training diagnostics and snapshot integrity

Both optimization seeds are retained, including unselected seeds. Frozen energies
come from finite MCMC diagnostics with random/up/down starts. Energy selection
does not establish reliable peak weights. Raw GFMC cycles, run records, frozen
training diagnostics and guide overlap samples accompany this notebook.''')
    code('''fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,kind in zip(axes,['jastrow','rbm_2']):
    for L,marker in [(4,'o'),(6,'s')]:
        for seed,color in [(0,'tab:blue'),(1,'tab:orange')]:
            group=[r for r in fits if r['task']['kind']==kind and r['task']['L']==L and r['task']['seed']==seed]
            ax.scatter([r['task']['gamma'] for r in group],
                [np.mean([d['energy'] for d in r['diagnostics']])/(3*L*L) for r in group],
                marker=marker,color=color,alpha=.4,label=f'L={L}, seed={seed}')
    ax.set(title=kind,xlabel='Gamma',ylabel='Frozen diagnostic energy / site'); ax.legend(fontsize=7); ax.grid(alpha=.2)
save(fig,'training_all_seeds')
for r in runs:
    path=folder/r['snapshot_raw_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['npz_sha256']
    assert not r['ground_state_qualified']
for r in guides:
    assert hashlib.sha256((folder/r['raw_file']).read_bytes()).hexdigest()==r['raw_sha256']
    with np.load(folder/r['raw_file']) as data:
        for start in r['starts']:
            counts=data[start['start']+'_mismatches']
            assert counts.shape==(50,64)
            assert np.all((counts>=0)&(counts<=48))
            reconstructed=np.bincount(counts.ravel(),minlength=49)[::-1]/counts.size
            np.testing.assert_allclose(reconstructed,start['mass'],atol=1e-14)
            assert np.isclose(np.sum(start['mass']),1)
print('Verified raw GFMC and guide sample hashes; all plotted histogram masses normalize.')
print('Snapshot files:',folder)
''')
    md('''## Interpretation and remaining measurements

Use the first panels to distinguish an exact small-system peak from an
unprojected guide peak. Use the disorder panels to see whether an ensemble
feature is shared or carried by particular disorders. Start and seed sensitivity
limit the interpretation of newly sampled guide histograms.

The current GFMC moments can track overlap and connected lengths, but a dedicated
pure-estimator overlap histogram is still required to establish projected
$P(q)$ at $L=4,6$. Ancestry alarms, two-disorder audits, and the eight-disorder
pilot precision remain unresolved. No transition, critical point, or collapse is
inferred. This is a dated snapshot; newly completed campaign tasks require a new
snapshot to appear here.''')
    notebook=nb.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    output.parent.mkdir(parents=True,exist_ok=True)
    nb.write(notebook,output)
    NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(output.parent.parent)}}).execute()
    nb.write(notebook,output)
    print('Executed',output,flush=True)
    try:
        from nbconvert import HTMLExporter
        html,_=HTMLExporter().from_notebook_node(notebook)
        output.with_suffix('.html').write_text(html)
        print('Exported',output.with_suffix('.html'),flush=True)
    except ImportError:
        print('HTML export unavailable; executed ipynb contains all outputs.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production',type=Path,required=True)
    p.add_argument('--bundle-root',type=Path,required=True)
    p.add_argument('--exact-scan',type=Path,required=True)
    p.add_argument('--resume-snapshot',type=Path)
    a=p.parse_args()
    if a.resume_snapshot:
        sample_guides(a.production,a.resume_snapshot)
        output=a.bundle_root/'notebook/preliminary_disorder_overlap.ipynb'
        make_notebook(a.resume_snapshot,output)
        dump(a.bundle_root/'latest_notebook.json',dict(snapshot=str(a.resume_snapshot),notebook=str(output)))
        return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snapshot=a.bundle_root/'data/kagome/ltfim_disorder_scaling_v1/notebook_snapshot'/stamp
    snapshot.mkdir(parents=True)
    shutil.copy2(a.production/'manifest.json',snapshot/'manifest.json')
    runs=[]
    for path in sorted((a.production/'projector/runs').glob('*.json')):
        r=json.loads(path.read_text()); raw=path.with_suffix('.npz')
        assert digest(raw)==r['npz_sha256']
        dest=snapshot/'projector/runs'/raw.name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(raw,dest);shutil.copy2(path,dest.with_suffix('.json'))
        r['snapshot_raw_file']=str(dest.relative_to(snapshot));runs.append(r)
    fits=[]
    for path in sorted((a.production/'training').glob('L*.json')):
        r=json.loads(path.read_text()); raw=path.with_suffix('.npz')
        assert digest(raw)==r['npz_sha256']
        dest=snapshot/'training'/raw.name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(raw,dest); shutil.copy2(path,dest.with_suffix('.json')); fits.append(r)
    selections=[json.loads(p.read_text()) for p in sorted((a.production/'selections').glob('*.json'))]
    dump(snapshot/'run_records.json',dict(rows=runs))
    dump(snapshot/'training_records.json',dict(rows=fits))
    dump(snapshot/'selection_records.json',dict(rows=selections))
    status=json.loads((a.production/'parallel_status.json').read_text())
    dump(snapshot/'snapshot.json',dict(created_utc=stamp,source=str(a.production),parallel_status=status,
        failures=[dict(file=p.name,record=json.loads(p.read_text())) for p in a.production.glob('failure*.json')]))
    reduce_results(snapshot);report(snapshot)
    exact=[]
    for path in sorted(a.exact_scan.glob('L2*.json')):
        r=json.loads(path.read_text())
        if r.get('gamma') in [.6,.65]:exact.append(r)
    assert len(exact)==18
    dump(snapshot/'exact_l2_reference.json',dict(rows=exact,scope='Existing exact L2 reference; different ensemble from current pilot'))
    sample_guides(a.production,snapshot)
    output=a.bundle_root/'notebook/preliminary_disorder_overlap.ipynb'
    make_notebook(snapshot,output)
    dump(a.bundle_root/'latest_notebook.json',dict(snapshot=str(snapshot),notebook=str(output)))


if __name__=='__main__':main()
