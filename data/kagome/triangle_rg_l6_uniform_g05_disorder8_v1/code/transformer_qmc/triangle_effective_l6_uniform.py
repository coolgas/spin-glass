"""Bounded L4/L6 retention pilot for the original uniform AF bond ensemble."""
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

GAMMAS = (.5,)
SAMPLES = tuple(range(60000,60008))
TOTAL = 2*len(GAMMAS)*len(SCHEDULES)*len(SAMPLES)


def mean_se(values):
    values=np.asarray(values)
    return values.mean(axis=0),values.std(axis=0,ddof=1)/np.sqrt(len(values))

def report(root):
    rows = sorted((json.loads(p.read_text()) for p in (root/'cases').glob('*.json')),
                  key=lambda r:(r['L'],r['gamma'],r['sample_id'],r['m'],r['chi']))
    assert all(r['lambda_rg'] == 1 and r['gamma']==.5 and r['h']==.15 for r in rows)
    complete={L: sorted(s for s in SAMPLES if all(any(
        (r['L'],r['sample_id'],r['m'],r['chi'])==(L,s,m,c) for r in rows)
        for m,c in SCHEDULES)) for L in (4,6)}
    intro = r'''# Original-bond L4/L6 disorder-averaged retention pilot

Every physical bond is drawn independently from $K_{ij}\sim U(0.5,1.5)$.
The same bond array for each disorder is used across all retained spaces.
All interactions enter with their original strength: **lambda=1 throughout**.
The Hamiltonian is $H=\sum_{ij}K_{ij}Z_iZ_j-0.15\sum_i Z_i-\Gamma\sum_i X_i$;
stored couplings use J=-K. Different sizes use different disorder draws
from SeedSequence([20261027,L,sample_id,1]), sample IDs 60000–60007.

L4: 48 spins → 16 triangle sites → 4 superblocks.
L6: 108 spins → 36 triangle sites → 9 superblocks.
We test (m,chi)=(4,2),(6,2),(6,3) at Gamma=0.5, h=0.15, orientation0 and origin(0,0).
The L6 top dimensions are 512 and 19683. Each reduced Hamiltonian is solved
from two starts; reconstructed physical energies check the composed projection.
These are approximate physical states with deliberately small retained spaces.

P(q) uses 4096 independent pairs of physical Born samples per disorder and schedule,
with $q=N^{-1}\sum_i z_i^{(a)}z_i^{(b)}$. Each disorder's normalized histogram
receives equal weight. Curves use only disorders complete at all three
retention settings, so retention comparisons are paired. Error bars show one
SE across disorder histograms (including their sampling noise); they exclude
truncation uncertainty. Bin spacing is 2/N. Eight disorders are a first pilot,
not a precise ensemble estimate. Direct physical moments are evaluated without
replica sampling noise and then averaged over disorder.
At finite h, polarization contributes to the overlap.
'''
    cells=[nbformat.v4.new_markdown_cell(intro+f'\nSaved models: **{len(rows)}/{TOTAL}**.\n')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True,layout='constrained')
    for i,L in enumerate((4,6)):
        for gamma in GAMMAS:
            ax=axes[i]
            for (m,chi),color in zip(SCHEDULES,('#0072B2','#D55E00','#009E73')):
                group=[r for r in rows if (r['L'],r['gamma'],r['m'],r['chi'])==(L,gamma,m,chi)
                       and r['sample_id'] in complete[L]]
                if not group: continue
                p,se=mean_se([r['mass'] for r in group]);q=np.linspace(-1,1,group[0]['n']+1)
                ax.errorbar(q,p,yerr=se,color=color,
                            ls='-',marker='.',ms=3,lw=1.2,capsize=1,label=f'm={m}, chi={chi}')
            ax.set(title=f'L={L}, Gamma={gamma:g}, {len(complete[L])} disorders',xlabel='Physical overlap q',
                   ylabel='Probability mass per bin',xlim=(-1,1),ylim=(0,None))
            ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=9)
    fig.suptitle('Original K ~ Uniform(0.5, 1.5) on all bonds | h=0.15')
    name='uniform_retention_pq.png'
    for ext in ('png','pdf'): fig.savefig(root/f'uniform_retention_pq.{ext}',dpi=150)
    plt.close(fig)
    cells.append(nbformat.v4.new_markdown_cell('## Physical distributions and retention sensitivity\n\n'
        'All curves are RG results at original bond strength. Each panel fixes L and Gamma; '
        'colors change only the retained space.\n\n'+f'![Physical distributions](attachment:{name})',
        attachments={name:{'image/png':base64.b64encode((root/name).read_bytes()).decode()}}))
    table=['## Disorder averages and paired retention changes','',
           'Entries are mean ± one SE across disorders. Direct moment evaluation has no replica '
           'sampling error. Delta SE uses per-disorder paired differences, preserving covariance.','',
           '| L | disorders | m | chi | mean q | q² | energy/spin | alarms |',
           '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |']
    summaries=[];differences=[]
    def fmt(values):
        avg,se=mean_se(values)
        return f'{avg:.6f} ± {se:.6f}'
    for L in (4,6):
        for m,c in SCHEDULES:
            group=[r for r in rows if (r['L'],r['m'],r['chi'])==(L,m,c) and r['sample_id'] in complete[L]]
            if not group: continue
            alarms=sum(r['start_alarm'] or r['unresolved_cutoff'] or any(x['alarm'] for x in r['sampling_checks']) for r in group)
            values=[[r['metrics']['q'],r['metrics']['q2'],r['energy_per_spin']] for r in group]
            avg,se=mean_se(values)
            summaries.append(dict(L=L,m=m,chi=c,samples=complete[L],mean=avg.tolist(),se=se.tolist(),alarms=alarms))
            table.append(f"| {L} | {len(group)} | {m} | {c} | {fmt([v[0] for v in values])} | {fmt([v[1] for v in values])} | {fmt([v[2] for v in values])} | {alarms}/{len(group)} |")
    table+=['','| L | retention change | delta mean q | delta q² | delta energy/spin |',
            '| ---: | :--- | ---: | ---: | ---: |']
    for L in (4,6):
        for gamma in GAMMAS:
            for a,b in zip(SCHEDULES,SCHEDULES[1:]):
                values=[]
                for s in complete[L]:
                    group={(r['m'],r['chi']):r for r in rows if (r['L'],r['gamma'],r['sample_id'])==(L,gamma,s)}
                    x,y=group[a],group[b]
                    values.append([y['metrics']['q']-x['metrics']['q'],y['metrics']['q2']-x['metrics']['q2'],y['energy_per_spin']-x['energy_per_spin']])
                if not values: continue
                avg,se=mean_se(values)
                differences.append(dict(L=L,source=a,destination=b,samples=complete[L],mean=avg.tolist(),se=se.tolist()))
                table.append(f"| {L} | {a} → {b} | {fmt([v[0] for v in values])} | {fmt([v[1] for v in values])} | {fmt([v[2] for v in values])} |")
    table+=['','Energy is evaluated under the original Hamiltonian; no exact full-spin L4/L6 '
        'reference is available here. Both solver agreement and the physical-energy reconstruction '
        'test numerical consistency, rather than physical convergence. Changing m also changes the '
        'superblock basis, so top spaces at fixed chi need not be nested. At fixed m, increasing chi '
        'does give a nested variational space. Small changes over this short ladder alone cannot '
        'establish convergence.', '',
        'Next: increase chi at fixed m and original bonds, then expand disorder and check partitions '
        'before q_EA or correlation-length scaling. All raw disorder, states from both starts, '
        'decoders, spectra, physical samples, seeds, residuals and alarms are preserved. '
        'Scientific calculations and figures were generated on AutoDL.']
    cells.append(nbformat.v4.new_markdown_cell('\n'.join(table)))
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(execution_location='AutoDL',
        lambda_rg=1.,preliminary=True,kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbformat.validate(nb);nbformat.write(nb,root/'triangle_effective_l6_uniform.ipynb')
    write_json(root/'summary.json',dict(columns=['q','q2','energy_per_spin'],complete_samples=complete,
        averages=summaries,paired_changes=differences,cases=len(rows),total=TOTAL))
    write_json(root/'report_provenance.json',dict(cases=len(rows),files={p.name:digest(p)
        for p in root.iterdir() if p.suffix in ('.png','.pdf','.ipynb')}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse',type=Path)
    args=parser.parse_args();root=args.output
    for name in ('cases','failures'): (root/name).mkdir(parents=True,exist_ok=True)
    reused=[]
    if args.reuse:
        for path in sorted((args.reuse/'cases').glob('*.json')):
            row=json.loads(path.read_text())
            if row['lambda_rg']!=1 or row['gamma'] not in GAMMAS: continue
            assert digest(path.with_suffix('.npz'))==row['raw_sha256']
            for p in (path,path.with_suffix('.npz')):
                destination=root/'cases'/p.name
                if not destination.exists(): shutil.copy2(p,destination)
                assert digest(destination)==digest(p)
            reused.append(row['key'])
    manifest=dict(lambda_rg=1.,ensemble='K ~ Uniform(0.5,1.5) on every bond',
        sizes=[4,6],gammas=GAMMAS,schedules=SCHEDULES,sample_ids=SAMPLES,h=.15,
        pairs=PAIRS,total=TOTAL,worker_threads=1,reused_cases=reused,
        sources={str(p.relative_to(Path(__file__).parents[1])):digest(p)
                 for p in Path(__file__).parents[1].rglob('*.py')})
    write_json(root/'launch.json',manifest)
    completed=0;failures=[]
    for L in (4,6):
        for gamma in GAMMAS:
            for sample_id in SAMPLES:
                for m,chi in SCHEDULES:
                    key=f's{sample_id}_l{L}_g{gamma:g}_l1_m{m}_c{chi}'
                    write_json(root/'status.json',dict(state='running',completed=completed,total=TOTAL,
                        current=key,failures=failures))
                    try:
                        run_case(root,L,gamma,1.,m,chi,sample_id);completed+=1
                    except Exception:
                        (root/'failures'/(key+'.txt')).write_text(traceback.format_exc())
                        failures.append(key);traceback.print_exc()
    report(root)
    write_json(root/'status.json',dict(state='complete' if not failures else 'complete_with_failures',
        completed=completed,total=TOTAL,failures=failures))


if __name__=='__main__': main()
