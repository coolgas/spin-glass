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
TOTAL = 2*len(GAMMAS)*len(SCHEDULES)

def report(root):
    rows = sorted((json.loads(p.read_text()) for p in (root/'cases').glob('*.json')),
                  key=lambda r:(r['L'],r['gamma'],r['m'],r['chi']))
    assert all(r['lambda_rg'] == 1 for r in rows)
    intro = r'''# Original-bond L4/L6 retention pilot

Every physical bond is drawn independently from $K_{ij}\sim U(0.5,1.5)$.
The same bond array at each size is used for every field and retained space.
All interactions enter with their original strength: **lambda=1 throughout**.
The Hamiltonian is $H=\sum_{ij}K_{ij}Z_iZ_j-0.15\sum_i Z_i-\Gamma\sum_i X_i$;
stored couplings use J=-K. Different sizes use different disorder draws
from SeedSequence([20261027,L,60000,1]).

L4: 48 spins → 16 triangle sites → 4 superblocks.
L6: 108 spins → 36 triangle sites → 9 superblocks.
We test (m,chi)=(4,2),(6,2),(6,3) at Gamma=0.5, h=0.15, orientation0 and origin(0,0).
The L6 top dimensions are 512 and 19683. Each reduced Hamiltonian is solved
from two starts; reconstructed physical energies check the composed projection.
These are approximate physical states with deliberately small retained spaces.

P(q) uses 4096 independent pairs of physical Born samples per curve,
with $q=N^{-1}\sum_i z_i^{(a)}z_i^{(b)}$. Error bars show one sampling SE;
they exclude truncation and disorder uncertainty. Bin spacing is 2/N.
Direct physical moments are evaluated without replica sampling noise.
At finite h, polarization contributes to the overlap.
'''
    cells=[nbformat.v4.new_markdown_cell(intro+f'\nSaved models: **{len(rows)}/{TOTAL}**.\n')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True,layout='constrained')
    for i,L in enumerate((4,6)):
        for gamma in GAMMAS:
            ax=axes[i]
            for (m,chi),color in zip(SCHEDULES,('#0072B2','#D55E00','#009E73')):
                r=next((r for r in rows if (r['L'],r['gamma'],r['m'],r['chi'])==(L,gamma,m,chi)),None)
                if r is None: continue
                p=np.array(r['mass']);q=np.linspace(-1,1,r['n']+1)
                ax.errorbar(q,p,yerr=np.sqrt(p*(1-p)/PAIRS),color=color,
                            ls='-',marker='.',ms=3,lw=1.2,capsize=1,label=f'm={m}, chi={chi}')
            ax.set(title=f'L={L}, Gamma={gamma:g}',xlabel='Physical overlap q',
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
    table=['## Direct moments and numerical checks','','| L | Gamma | m | chi | mean q | q² | energy/spin | alarms |',
           '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |']
    for r in rows:
        alarm=r['start_alarm'] or r['unresolved_cutoff'] or any(x['alarm'] for x in r['sampling_checks'])
        table.append(f"| {r['L']} | {r['gamma']:g} | {r['m']} | {r['chi']} | {r['metrics']['q']:.6f} | {r['metrics']['q2']:.6f} | {r['energy_per_spin']:.7f} | {'yes' if alarm else 'no'} |")
    table+=['','| L | Gamma | retention change | delta mean q | delta q² | delta energy/spin |',
            '| ---: | ---: | :--- | ---: | ---: | ---: |']
    for L in (4,6):
        for gamma in GAMMAS:
            group={(r['m'],r['chi']):r for r in rows if (r['L'],r['gamma'])==(L,gamma)}
            for a,b in zip(SCHEDULES,SCHEDULES[1:]):
                if a not in group or b not in group: continue
                x,y=group[a],group[b]
                table.append(f"| {L} | {gamma:g} | {a} → {b} | {y['metrics']['q']-x['metrics']['q']:+.6f} | {y['metrics']['q2']-x['metrics']['q2']:+.6f} | {y['energy_per_spin']-x['energy_per_spin']:+.7f} |")
    table+=['','Energy is evaluated under the original Hamiltonian; no exact full-spin L4/L6 '
        'reference is available here. Both solver agreement and the physical-energy reconstruction '
        'test numerical consistency, rather than physical convergence. Changing m also changes the '
        'superblock basis, so top spaces at fixed chi need not be nested. At fixed m, increasing chi '
        'does give a nested variational space. Small changes over this short ladder alone cannot '
        'establish convergence.', '',
        'Next: increase chi at fixed m and original bonds, then check disorder and partitions '
        'before q_EA or correlation-length scaling. All raw disorder, states from both starts, '
        'decoders, spectra, physical samples, seeds, residuals and alarms are preserved. '
        'Scientific calculations and figures were generated on AutoDL.']
    cells.append(nbformat.v4.new_markdown_cell('\n'.join(table)))
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(execution_location='AutoDL',
        lambda_rg=1.,preliminary=True,kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbformat.validate(nb);nbformat.write(nb,root/'triangle_effective_l6_uniform.ipynb')
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
        sizes=[4,6],gammas=GAMMAS,schedules=SCHEDULES,sample_id=60000,h=.15,
        pairs=PAIRS,total=TOTAL,worker_threads=1,reused_cases=reused,
        sources={str(p.relative_to(Path(__file__).parents[1])):digest(p)
                 for p in Path(__file__).parents[1].rglob('*.py')})
    write_json(root/'launch.json',manifest)
    completed=0;failures=[]
    for L in (4,6):
        for gamma in GAMMAS:
            for m,chi in SCHEDULES:
                key=f'l{L}_g{gamma:g}_l1_m{m}_c{chi}'
                write_json(root/'status.json',dict(state='running',completed=completed,total=TOTAL,
                    current=key,failures=failures))
                try:
                    run_case(root,L,gamma,1.,m,chi);completed+=1
                except Exception:
                    (root/'failures'/(key+'.txt')).write_text(traceback.format_exc())
                    failures.append(key);traceback.print_exc()
    report(root)
    write_json(root/'status.json',dict(state='complete' if not failures else 'complete_with_failures',
        completed=completed,total=TOTAL,failures=failures))


if __name__=='__main__': main()
