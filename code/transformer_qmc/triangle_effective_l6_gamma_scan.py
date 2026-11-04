"""Matched-disorder L6 transverse-field scan of the original bond ensemble."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_effective_benchmark import digest, write_json
from .triangle_effective_l6_adaptive import run_case, SCHEDULES, PAIRS
from .triangle_effective_l6_uniform import SAMPLES, mean_se

GAMMAS = (.1, .25, .5, .75, 1.)
TOTAL = len(GAMMAS)*len(SAMPLES)*len(SCHEDULES)


def alarm(row):
    return bool(row['start_alarm'] or row['unresolved_cutoff'] or
                any(c['alarm'] for c in row['sampling_checks']))


def report(root):
    rows=[json.loads(p.read_text()) for p in (root/'cases').glob('*.json')]
    lookup={(r['gamma'],r['sample_id'],r['m'],r['chi']):r for r in rows}
    complete=[s for s in SAMPLES if all((g,s,m,c) in lookup
              for g in GAMMAS for m,c in SCHEDULES)]
    if len(complete)<2:
        raise RuntimeError('At least two fully matched disorders required for a report')
    averages=[];retention=[];field_changes=[]
    for g in GAMMAS:
        for m,c in SCHEDULES:
            group=[lookup[g,s,m,c] for s in complete]
            mass,se=mean_se([r['mass'] for r in group])
            avg,err=mean_se([[r['metrics']['q'],r['metrics']['q2'],r['energy_per_spin']] for r in group])
            averages.append(dict(gamma=g,m=m,chi=c,mean=avg.tolist(),se=err.tolist(),
                mass=mass.tolist(),mass_disorder_se=se.tolist(),alarms=sum(map(alarm,group))))
    def differences(axis,first,second,fixed):
        values=[]
        for s in complete:
            if axis=='retention':
                x,y=lookup[(fixed,s)+first],lookup[(fixed,s)+second]
            else:
                x,y=lookup[(first,s)+fixed],lookup[(second,s)+fixed]
            values.append([y['metrics']['q']-x['metrics']['q'],
                y['metrics']['q2']-x['metrics']['q2'],y['energy_per_spin']-x['energy_per_spin']])
        avg,se=mean_se(values)
        return dict(source=first,destination=second,fixed=fixed,mean=avg.tolist(),se=se.tolist())
    for g in GAMMAS:
        for a,b in zip(SCHEDULES,SCHEDULES[1:]): retention.append(differences('retention',a,b,g))
    for schedule in SCHEDULES:
        for a,b in zip(GAMMAS,GAMMAS[1:]): field_changes.append(differences('field',a,b,schedule))
    fig,axs=plt.subplots(1,3,figsize=(16,4.8),sharex=True,sharey=True,layout='constrained')
    colors=plt.cm.viridis(np.linspace(.05,.9,len(GAMMAS)))
    for ax,(m,c) in zip(axs,SCHEDULES):
        for g,color in zip(GAMMAS,colors):
            r=next(a for a in averages if (a['gamma'],a['m'],a['chi'])==(g,m,c))
            ax.errorbar(np.linspace(-1,1,109),r['mass'],yerr=r['mass_disorder_se'],
                color=color,ls='-',lw=1.2,capsize=1.5,label=f'Gamma={g:g}')
        ax.set(title=f'm={m}, chi={c}',xlabel='Physical overlap q',xlim=(-1,1),ylim=(0,None))
        ax.grid(alpha=.2)
    axs[0].set_ylabel('Probability mass per bin');axs[-1].legend(fontsize=9)
    fig.suptitle(f'L6, N=108 | h=0.15 | original bonds | {len(complete)} matched disorders | bars: disorder SE')
    for ext in ('png','pdf'):fig.savefig(root/f'l6_gamma_pq.{ext}',dpi=160)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for (m,c),color in zip(SCHEDULES,('#0072B2','#D55E00','#009E73')):
        group=[next(a for a in averages if (a['gamma'],a['m'],a['chi'])==(g,m,c)) for g in GAMMAS]
        for j,ax in enumerate(axs):
            ax.errorbar(GAMMAS,[r['mean'][j] for r in group],yerr=[r['se'][j] for r in group],
                color=color,fmt='o-',capsize=3,label=f'm={m}, chi={c}')
    for ax,label in zip(axs,('Direct mean overlap','Direct second moment','Energy per spin')):
        ax.set(xlabel='Gamma',ylabel=label);ax.grid(alpha=.2)
    axs[-1].legend(fontsize=9)
    fig.suptitle(f'L6 | h=0.15 | {len(complete)} matched disorders | bars: disorder SE')
    for ext in ('png','pdf'):fig.savefig(root/f'l6_gamma_moments.{ext}',dpi=160)
    plt.close(fig)
    intro=r'''# L6 kagome RG: transverse-field scan

**108 spins; h=0.15; every K drawn independently from Uniform(0.5,1.5); lambda=1.**
The Hamiltonian is
$$H=\sum_{\langle ij\rangle}K_{ij}Z_iZ_j-0.15\sum_iZ_i-\Gamma\sum_iX_i.$$

Scan Gamma=0.1,0.25,0.5,0.75,1.0, using the same eight disorders60000–60007
at every field and retention setting (m,chi)=(4,2),(6,2),(6,3).
The hierarchy is 108 physical spins → 36 triangles → 9 superblocks;
final dimensions are 512 and 19683. Bond seeds are SeedSequence([20261027,6,sample_id,1]).

**All plotted bars are ±1 disorder SE.** Average normalized histograms with
equal disorder weights; bars are the sample standard deviation across disorder
histograms divided by sqrt(8). Each histogram uses 4096 independent physical
replica pairs. No replica-sampling bars are drawn. Finite sampling affects
the histograms; disorder bars do not include truncation bias. Direct moments
are evaluated from physical Z and ZZ expectations without replica sampling.
The completed Gamma=0.5 cases are reused with their original seeds and sources.
'''
    cells=[nbformat.v4.new_markdown_cell(intro)]
    for name,text in (
        ('l6_gamma_pq.png',r'''## Physical overlap distributions

$$q=\frac1{108}\sum_i z_i^{(a)}z_i^{(b)}.$$

Bin spacing is 1/54; the vertical axis is probability mass per bin.
Each panel fixes retention and overlays fields. At h>0 polarization contributes
to the raw overlap; field-dependent histogram shifts alone do not establish a transition.'''),
        ('l6_gamma_moments.png','## Direct moments and energy versus Gamma\n\nEach point averages eight disorders. Compare retention shifts with field changes before interpreting the physical trend.')):
        cells.append(nbformat.v4.new_markdown_cell(text+f'\n\n![L6 field scan](attachment:{name})',
            attachments={name:{'image/png':base64.b64encode((root/name).read_bytes()).decode()}}))
    def fmt(values,errors):return ' | '.join(f'{a:.6f} ± {b:.6f}' for a,b in zip(values,errors))
    table=['## Disorder averages and paired comparisons','',
        'All ± entries are disorder SE. Paired differences use the same disorder at both endpoints.','',
        '| Gamma | m | chi | mean q | q² | energy/spin | alarms |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in averages:table.append(f"| {a['gamma']:g} | {a['m']} | {a['chi']} | {fmt(a['mean'],a['se'])} | {a['alarms']} |")
    table+=['','| Gamma | retention change | delta q | delta q² | delta energy/spin |',
        '| ---: | :--- | ---: | ---: | ---: |']
    for a in retention:table.append(f"| {a['fixed']:g} | {tuple(a['source'])} → {tuple(a['destination'])} | {fmt(a['mean'],a['se'])} |")
    table+=['','Paired field changes at (m,chi)=(6,3):','',
        '| Gamma change | delta q | delta q² | delta energy/spin |',
        '| :--- | ---: | ---: | ---: |']
    for a in field_changes:
        if a['fixed']==(6,3):table.append(f"| {a['source']:g} → {a['destination']:g} | {fmt(a['mean'],a['se'])} |")
    alarms=[r['key'] for r in rows if alarm(r)]
    table+=['',f"Completed {len(rows)}/{TOTAL} cases; {len(complete)} fully matched disorders. "
        f"Diagnostic alarms: {len(alarms)}. Maximum eigenpair residual: {max(r['residual'] for r in rows):.3g}; "
        f"maximum physical-energy consistency error: {max(r['physical_energy_check_error'] for r in rows):.3g}.",'',
        'These diagnostics test the reduced solver, projection consistency and sampling. '
        'They do not establish physical accuracy. Tiny chi remains a retention diagnostic; '
        'no exact full-spin L6 reference or finite-size scaling conclusion is supplied. '
        'Inspect the paired retention changes at each field.','',
        'Calculations and figures were generated on AutoDL. Raw disorder, both solver starts, '
        'seeds, states, isometries, spectra, physical measurements, sampling diagnostics and '
        'source snapshots are retained locally in '
        '`data/kagome/triangle_rg_l6_gamma_scan_v1/`.']
    cells.append(nbformat.v4.new_markdown_cell('\n'.join(table)))
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(execution_location='AutoDL',L=6,
        gammas=list(GAMMAS),h=.15,lambda_rg=1.,disorders=complete,error_bars='disorder SE only',
        kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbformat.validate(nb);nbformat.write(nb,root/'triangle_effective_l6_gamma_scan.ipynb')
    write_json(root/'summary.json',dict(columns=['q','q2','energy_per_spin'],complete_samples=complete,
        averages=averages,paired_retention_changes=retention,paired_field_changes=field_changes,alarms=alarms))
    write_json(root/'report_provenance.json',dict(source_sha256=digest(Path(__file__)),
        files={p.name:digest(p) for p in root.iterdir() if p.suffix in ('.png','.pdf','.ipynb') or p.name=='summary.json'}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse',type=Path)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();root=args.output
    for name in ('cases','failures'):(root/name).mkdir(parents=True,exist_ok=True)
    if args.report_only:return report(root)
    reused=[]
    if args.reuse:
        for path in sorted((args.reuse/'cases').glob('*.json')):
            r=json.loads(path.read_text())
            if r['L']!=6 or r['gamma'] not in GAMMAS or r['lambda_rg']!=1 or r['h']!=.15:continue
            assert digest(path.with_suffix('.npz'))==r['raw_sha256']
            for p in (path,path.with_suffix('.npz')):
                destination=root/'cases'/p.name
                if not destination.exists():shutil.copy2(p,destination)
                assert digest(p)==digest(destination)
            reused.append(r['key'])
    write_json(root/'launch.json',dict(L=6,gammas=GAMMAS,schedules=SCHEDULES,sample_ids=SAMPLES,
        h=.15,lambda_rg=1.,ensemble='K ~ Uniform(0.5,1.5) on every bond',pairs=PAIRS,
        total=TOTAL,worker_threads=1,reused_cases=reused,
        sources={str(p.relative_to(Path(__file__).parents[1])):digest(p)
                 for p in Path(__file__).parents[1].rglob('*.py')}))
    completed=0;failures=[]
    for g in GAMMAS:
        for s in SAMPLES:
            for m,c in SCHEDULES:
                key=f's{s}_l6_g{g:g}_l1_m{m}_c{c}'
                write_json(root/'status.json',dict(state='running',completed=completed,total=TOTAL,current=key,failures=failures))
                try:run_case(root,6,g,1.,m,c,s);completed+=1
                except Exception:
                    (root/'failures'/(key+'.txt')).write_text(traceback.format_exc())
                    failures.append(key);traceback.print_exc()
    try:report(root)
    except Exception:
        (root/'failures/report.txt').write_text(traceback.format_exc());failures.append('report');traceback.print_exc()
    write_json(root/'status.json',dict(state='complete' if not failures else 'complete_with_failures',
        completed=completed,total=TOTAL,failures=failures))


if __name__=='__main__':main()
