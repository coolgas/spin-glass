"""Generate a self-contained interim notebook from saved AutoDL measurements."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from transformer_qmc.triangle_effective_bond_scan import SCALES, GAMMAS, FIELDS, SCHEDULES
from transformer_qmc.triangle_effective_benchmark import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root,out = args.results,args.output
    out.mkdir(parents=True,exist_ok=True)
    records = {}
    for folder in ('cases','exact'):
        (out/folder).mkdir(exist_ok=True)
        records[folder] = []
        for path in sorted((root/folder).glob('*.json')):
            row = json.loads(path.read_text())
            assert digest(path.with_suffix('.npz')) == row['raw_sha256']
            shutil.copy2(path,out/folder/path.name)
            records[folder].append(row)
    rows,small = records['cases'],records['exact']
    timestamp = datetime.now(timezone.utc).isoformat()
    status = json.loads((root/'status.json').read_text())
    write_json(out/'snapshot.json',dict(timestamp_utc=timestamp,status=status,
        l4_models=len(rows),l2_contexts=len(small),raw_hashes_verified=len(rows)+len(small)))
    lookup = {(r['scale'],r['gamma'],r['h'],r['m'],r['chi']):r for r in rows}
    refs = {(r['scale'],r['gamma'],r['h']):r for r in small}
    panels = []
    q = np.linspace(-1,1,49)

    def cell_with_images(text,names):
        return nbformat.v4.new_markdown_cell(text,attachments={name:{'image/png':
            base64.b64encode((out/name).read_bytes()).decode()} for name in names})

    cells = [nbformat.v4.new_markdown_cell(r'''# Lowering all bond strengths: L4 overlap distributions

**Interim results — the finite scan is still running.**

We lower both endpoints of every antiferromagnetic bond distribution and scan
the transverse field $\Gamma$ and longitudinal field $h$. The goal is a field
window where physical $P(q)$ shifts while RG remains stable under retention
changes, before finite-size scaling of $q_{\rm EA}$ and correlation length.

The model is $H=\sum_{ij}K_{ij}Z_iZ_j-h\sum_i Z_i-\Gamma\sum_i X_i$.
All bonds use $K=sK^{(0)}$, $K^{(0)}\sim U(0.5,1.5)$, with
$s=1,0.5,0.2,0.1$. Within each size, the same disorder60000 base draws are
used throughout. Internal and boundary bonds receive the same factor.

Each L4 system has 48 physical spins. Rows are $\Gamma=0.05,0.10,0.25,0.50$;
columns are $h=0,0.025,0.075,0.15$. Curves compare $(m,\chi)=(4,8),(6,8),(6,16)$,
using two Lanczos starts and 8192 independent physical replica pairs per curve.
Error bars are pointwise ±1 sampling SE; they exclude truncation uncertainty.
At zero field the solvers restrict to even spin-flip parity, avoiding arbitrary
mixtures when the tunnelling splitting is below numerical resolution.

All calculations and plots were generated on AutoDL. Images and tables are
embedded here; opening this notebook requires no cluster connection or rerun.
''' + f'\nSnapshot: **{timestamp}**. Saved results: **{len(rows)}/192 L4 models** and '
        f'**{len(small)}/64 exact L2 contexts**. Missing curves and panels are labeled.\n')]
    for scale in SCALES:
        fig,axs = plt.subplots(4,4,figsize=(17,14),sharex=True,layout='constrained')
        fig2,bxs = plt.subplots(4,4,figsize=(17,14),sharex=True,layout='constrained')
        table = ['| Gamma | h | Curves | TV(m) | TV(chi) | chi energy/spin/mean K | chi q2 change | Screen |',
                 '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |']
        for i,g in enumerate(GAMMAS):
            for j,h in enumerate(FIELDS):
                ax,bx = axs[i,j],bxs[i,j]
                selected = [lookup.get((scale,g,h,m,c)) for m,c in SCHEDULES]
                available = [r for r in selected if r is not None]
                for r,color in zip(selected,('#0072B2','#D55E00','#009E73')):
                    if r is None: continue
                    p = np.array(r['mass'])
                    ax.errorbar(q,p,yerr=np.sqrt(p*(1-p)/r['pairs']),color=color,
                        marker='.',ms=2,lw=1,capsize=1,label=f"m={r['m']}, chi={r['chi']}")
                if len(available)<3:
                    ax.text(.04,.94,f'{len(available)}/3 curves completed',transform=ax.transAxes,
                            va='top',fontsize=9,color='#666666')
                ref = refs.get((scale,g,h))
                if ref is not None:
                    bx.plot(np.linspace(-1,1,13),ref['mass'],'.-',color='black',label='Exact L2')
                    for check,color in zip(ref['checks'][:2],('#0072B2','#D55E00')):
                        # Read only the archived companion probabilities for illustration.
                        with np.load(root/'exact'/(ref['key']+'.npz')) as raw:
                            bx.plot(np.linspace(-1,1,13),raw[f"mass_m{check['m']}"],'.-',
                                color=color,label=f"L2 m={check['m']}")
                else:
                    bx.text(.5,.5,'Pending',ha='center',transform=bx.transAxes,color='#666666')
                for axis in (ax,bx):
                    axis.set_title(f'Gamma={g:g}, h={h:g}',fontsize=10)
                    axis.set(xlim=(-1.02,1.02),ylim=(0,None),xticks=[-1,-.5,0,.5,1])
                    axis.grid(alpha=.2)
                    if i==3: axis.set_xlabel('Physical overlap q')
                    if j==0: axis.set_ylabel('Probability mass')
                if len(available)==3 and ref is not None:
                    a,b,c=selected
                    tvm=float(abs(np.array(a['mass'])-b['mass']).sum()/2)
                    tvc=float(abs(np.array(b['mass'])-c['mass']).sum()/2)
                    de=abs(b['energy_per_site']-c['energy_per_site'])/scale
                    dq=abs(b['metrics']['q2']-c['metrics']['q2'])
                    check=next(t for t in ref['checks'] if t['m']==6)
                    alarm=any(r['start_alarm'] or r['unresolved_cutoff'] or
                        any(t['alarm'] for t in r['sampling_checks']) for r in selected)
                    alarm |= any(t['full_recovery_alarm'] for t in ref['checks'])
                    passed=bool(tvm<.05 and tvc<.05 and de<.001 and dq<.01 and
                        check['tv']<.05 and check['energy_error_per_site']/scale<.001 and
                        check['q2_error']<.01 and not alarm)
                    panels.append(dict(scale=scale,gamma=g,h=h,candidate_only=passed,
                        tv_m=tvm,tv_chi=tvc,energy_change_normalized=de,q2_change=dq))
                    table.append(f"| {g:g} | {h:g} | 3/3 | {tvm:.4f} | {tvc:.4f} | {de:.6f} | {dq:.5f} | {'candidate' if passed else 'fails'} |")
                else:
                    table.append(f'| {g:g} | {h:g} | {len(available)}/3 | — | — | — | — | pending |')
        names=[]
        for figure,axes,prefix,title in [(fig,axs,'l4','L4, N=48 retention comparison'),
                                        (fig2,bxs,'l2','L2, N=12 exact companion')]:
            handles,labels=[],[]
            for axis in axes.flat:
                hh,ll=axis.get_legend_handles_labels()
                for handle,label in zip(hh,ll):
                    if label not in labels: handles.append(handle);labels.append(label)
            figure.suptitle(f'{title} | K~U({.5*scale:g},{1.5*scale:g}) | INTERIM',fontsize=14)
            if handles: figure.legend(handles,labels,loc='outside upper center',ncol=3)
            name=f'{prefix}_k{scale:g}_interim.png'
            figure.savefig(out/name,dpi=140);plt.close(figure);names.append(name)
        text=f'## K sampled from U({.5*scale:g}, {1.5*scale:g})\n\n'
        text+=f'![L4 grid](attachment:{names[0]})\n\n![L2 companion](attachment:{names[1]})\n\n'
        text+='L2 and L4 use different bonds because the disorder seed includes size. '
        text+='L2 agreement checks the first projection; it does not certify L4 superblock truncation.\n\n'
        text+='\n'.join(table)+'\n'
        cells.append(cell_with_images(text,names))
    fig,axs=plt.subplots(4,4,figsize=(17,13),sharex=True,layout='constrained')
    for i,scale in enumerate(SCALES):
        for j,h in enumerate(FIELDS):
            ax=axs[i,j];count=0
            for g in GAMMAS:
                r=lookup.get((scale,g,h,6,16))
                if r is not None:
                    ax.plot(q,r['mass'],'.-',ms=2,label=f'Gamma={g:g}');count+=1
            ax.set_title(f'Mean K={scale:g}, h={h:g} ({count}/4 fields)',fontsize=10)
            ax.set(xlim=(-1.02,1.02),ylim=(0,None));ax.grid(alpha=.2)
            if i==3: ax.set_xlabel('Physical overlap q')
            if j==0: ax.set_ylabel('Probability mass')
            if count: ax.legend(fontsize=8)
            else: ax.text(.5,.5,'Pending',ha='center',transform=ax.transAxes,color='#666666')
    fig.suptitle('Field-driven P(q) shifts | L4 m=6, chi=16 | INTERIM')
    name='l4_gamma_interim_overlays.png';fig.savefig(out/name,dpi=140);plt.close(fig)
    cells.append(cell_with_images('## Field overlays\n\n'
        f'![Field overlays](attachment:{name})\n\n'
        'These overlays expose changes hidden by separate panels. At h=0, mean q vanishes '
        'by spin-flip symmetry; distribution shape and q2 can still change.\n\n'
        '**Interpretation:** a candidate requires TV(m), TV(chi)<0.05, chi energy '
        'change/spin/mean K<0.001, chi q2 change<0.01, matching exact L2 m6 screens, '
        'and no solver, cutoff or sampling alarms. Normalizing by mean K prevents a '
        'smaller energy unit from creating an artificial pass. Histogram TV contains sampling noise. '
        'Passing is a retention sensitivity screen, not a bound on physical error. '
        'This one-disorder snapshot does not establish a transition or validate finite-size scaling. '
        'The field background of q_EA and physical connected correlations must be treated explicitly.',[name]))
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'execution_location':'AutoDL',
        'snapshot_utc':timestamp,'interim':True,'kernelspec':{
            'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.validate(nb)
    nbformat.write(nb,out/'triangle_effective_bond_scan.ipynb')
    write_json(out/'interim_assessment.json',dict(panels=panels,
        completed_contexts=len(panels),candidates=sum(r['candidate_only'] for r in panels)))
    shutil.copy2(Path(__file__),out/'interim_report_source.py')
    write_json(out/'provenance.json',dict(source_sha256=digest(Path(__file__)),
        notebook_sha256=digest(out/'triangle_effective_bond_scan.ipynb'),
        figures={p.name:digest(p) for p in out.glob('*.png')}))
    print(json.dumps(dict(snapshot=timestamp,l4=len(rows),l2=len(small),
        completed_contexts=len(panels),candidates=sum(r['candidate_only'] for r in panels))),flush=True)


if __name__=='__main__': main()
