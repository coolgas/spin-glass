"""Dedicated L6 saved-data report with disorder standard errors only."""
import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_effective_benchmark import digest, write_json
from .triangle_effective_l6_uniform import mean_se, SCHEDULES, SAMPLES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.results;out=args.output
    out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(p.read_text()) for p in (root/'cases').glob('*.json')]
    rows=[r for r in rows if r['L']==6]
    assert len(rows)==24
    for r in rows:
        assert r['gamma']==.5 and r['h']==.15 and r['lambda_rg']==1
        assert digest(root/'cases'/(r['key']+'.npz'))==r['raw_sha256']
    lookup={(r['sample_id'],r['m'],r['chi']):r for r in rows}
    assert len(lookup)==24
    cells=[nbformat.v4.new_markdown_cell(r'''# L6 kagome RG: disorder averages

**108 physical spins, Gamma=0.5, h=0.15.** Every antiferromagnetic bond is
drawn from $K_{ij}\sim U(0.5,1.5)$ and kept at its original strength:

$$H=\sum_{\langle ij\rangle}K_{ij}Z_iZ_j-0.15\sum_iZ_i-0.5\sum_iX_i.$$

The code stores J=-K. There is no intertriangle rescaling (lambda=1).
Eight independent disorders, sample IDs 60000–60007, use
SeedSequence([20261027,6,sample_id,1]). Each disorder's bond array is reused
across all retention settings. The hierarchy is 108 physical spins → 36
triangle sites → 9 superblocks, orientation0 and origin(0,0).
We compare (m,chi)=(4,2),(6,2),(6,3); final dimensions are 512 and 19683.

**Every plotted error bar is ±1 standard error across disorders.** No
sampling error bars are plotted. For each bin, average the eight normalized
histograms with equal weights, then compute their sample standard deviation
divided by sqrt(8). This estimates disorder variation from eight sampled
histograms; finite replica sampling can still affect those histograms.
It does not include RG truncation bias. These eight disorders form an initial
pilot, and the small retained spaces have not established physical convergence.
''')]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    averages=[]
    for (m,c),color in zip(SCHEDULES,('#0072B2','#D55E00','#009E73')):
        group=[lookup[s,m,c] for s in SAMPLES]
        p,se=mean_se([r['mass'] for r in group])
        ax.errorbar(np.linspace(-1,1,109),p,yerr=se,color=color,ls='-',marker='.',
                    ms=3,lw=1.3,capsize=2,label=f'm={m}, chi={c}')
        values=np.array([[r['metrics']['q'],r['metrics']['q2'],r['energy_per_spin']] for r in group])
        avg,err=mean_se(values)
        averages.append(dict(m=m,chi=c,mean=avg.tolist(),se=err.tolist(),
                             mass=p.tolist(),mass_disorder_se=se.tolist()))
    ax.set(title='L6 | Gamma=0.5, h=0.15 | 8 disorders | bars: disorder SE',
           xlabel='Physical overlap q',ylabel='Probability mass per bin',xlim=(-1,1),ylim=(0,None))
    ax.grid(alpha=.2);ax.legend()
    for ext in ('png','pdf'): fig.savefig(out/f'l6_disorder_pq.{ext}',dpi=160)
    plt.close(fig)
    name='l6_disorder_pq.png'
    cells.append(nbformat.v4.new_markdown_cell(r'''## Physical overlap distribution

$$q=\frac1{108}\sum_i z_i^{(a)}z_i^{(b)}.$$

For each disorder and retention setting, draw 4096 independent physical replica
pairs from the reconstructed RG state. Bin spacing is 1/54; the vertical axis
is probability mass, not density. Colored solid lines compare retained spaces.
At nonzero h, polarization contributes to the raw overlap.

![Disorder-averaged L6 distributions](attachment:l6_disorder_pq.png)
''',attachments={name:{'image/png':base64.b64encode((out/name).read_bytes()).decode()}}))
    def fmt(avg,se): return f'{avg:.6f} ± {se:.6f}'
    table=['## Direct physical moments and retention sensitivity','',
        'The moments below are evaluated directly from physical Z and ZZ expectations '
        'in each retained state, then averaged over disorders. They have no replica '
        'sampling error. All ± entries are disorder SE.','',
        '| m | chi | disorder mean of ⟨q⟩ | disorder mean of ⟨q²⟩ | energy/spin |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for r in averages:
        table.append(f"| {r['m']} | {r['chi']} | "+' | '.join(fmt(a,e) for a,e in zip(r['mean'],r['se']))+' |')
    table+=['','Retention differences use the same disorders at both settings. '
        'Their SE is calculated from the eight paired differences.','',
        '| change in (m,chi) | delta ⟨q⟩ | delta ⟨q²⟩ | delta energy/spin |',
        '| :--- | ---: | ---: | ---: |']
    changes=[]
    for a,b in zip(SCHEDULES,SCHEDULES[1:]):
        values=[]
        for s in SAMPLES:
            x,y=lookup[(s,)+a],lookup[(s,)+b]
            values.append([y['metrics']['q']-x['metrics']['q'],
                           y['metrics']['q2']-x['metrics']['q2'],y['energy_per_spin']-x['energy_per_spin']])
        avg,se=mean_se(values)
        changes.append(dict(source=a,destination=b,mean=avg.tolist(),se=se.tolist()))
        table.append(f'| {a} → {b} | '+' | '.join(fmt(v,e) for v,e in zip(avg,se))+' |')
    alarms=[r['key'] for r in rows if r['start_alarm'] or r['unresolved_cutoff'] or any(t['alarm'] for t in r['sampling_checks'])]
    table+=['',f"All 24 L6 cases completed. Diagnostic alarms: **{len(alarms)}**. "
        f"Maximum eigenpair residual: {max(r['residual'] for r in rows):.3g}; "
        f"maximum physical-energy reconstruction error: {max(r['physical_energy_check_error'] for r in rows):.3g}.",'',
        'The physical-energy check reconstructs the expectation under the original '
        'Hamiltonian and compares it with the reduced eigenvalue. It tests projection '
        'consistency. No exact full-spin L6 reference is available here.','',
        '**Retention is still consequential:** increasing chi from 2 to 3 at m=6 '
        'changes disorder-averaged q² by about +0.02976 and lowers energy by '
        '0.01510 per spin. Increase chi at fixed m before interpreting these results '
        'as converged L6 physics or using them for finite-size scaling.','',
        'Raw disorder, both solver starts, seeds, spectra, isometries, decoders, '
        'physical samples and diagnostics accompany the local data delivery at '
        '`data/kagome/triangle_rg_l6_uniform_g05_disorder8_v1/`. '
        'Scientific calculations and this figure were generated on AutoDL.']
    cells.append(nbformat.v4.new_markdown_cell('\n'.join(table)))
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(execution_location='AutoDL',
        L=6,gamma=.5,h=.15,lambda_rg=1.,disorders=list(SAMPLES),error_bars='disorder SE only',
        kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbformat.validate(nb);nbformat.write(nb,out/'triangle_effective_l6_results.ipynb')
    write_json(out/'summary.json',dict(averages=averages,paired_changes=changes,alarms=alarms))
    write_json(out/'provenance.json',dict(source_sha256=digest(Path(__file__)),
        cases={r['key']:r['raw_sha256'] for r in rows},
        files={p.name:digest(p) for p in out.iterdir() if p.name!='provenance.json'}))


if __name__=='__main__': main()
