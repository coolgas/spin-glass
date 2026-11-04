"""Insert the prioritized L6/L8/L10 campaign checkpoint and execute on AutoDL."""
import argparse
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
from nbconvert import HTMLExporter


TAG='feedback-scaling-20261103'


def update(root,archive,execute=False,progress=None):
    path=root/'notebook/jastrow_rbm_overlap_distributions.ipynb'
    notebook=nb.read(path,as_version=4)
    notebook.cells=[c for c in notebook.cells if TAG not in c.metadata.get('tags',[])]
    cells=[]
    def md(text):cells.append(nb.v4.new_markdown_cell(text,metadata=dict(tags=[TAG])))
    def code(text):cells.append(nb.v4.new_code_cell(text,metadata=dict(tags=[TAG])))
    md(r'''## Priority: iterative feedback at $L=6,8,10$

**User-directed compute priority, 2026-11-03.** New research compute now targets
$L=6,8,10$ ($N=108,192,300$), not an extended small-system campaign.
The five-arm $L=2$ continuation proposal was never launched. Nineteen short
implementation tests passed on AutoDL, including CPU/CUDA rates at the requested
sizes, exact tiny-state kernels, restart/dispatch equality and independent
raw-data reconstruction. These are implementation checks, not physical convergence.

Three detached size workers launched at **2026-11-03T15:12:24.982677Z**.
Each size covers disorders60000/60001, seeds0/1, $h=.15$, $\Gamma=.60$,
and two paired arms from the same starting checkpoint:

- **Baseline:**1024 walkers; persistent two-copy Gibbs negatives.
- **Robust:**four independent1024-walker populations;16-temperature two-copy
  replica exchange; bounded empirical log-density steps.

Every context receives12 feedback stints. Each stint uses equil8/collect12,
delta.1,64 target samples/block,1000 likelihood updates,lr.001,batch256 and
10 negative sweeps/update. Two-stint round-robin dispatch exposes contexts
early. L6 starts from archived VMC400 controls; L8/L10 create repaired VMC
warm starts at Gamma1.5,.75,.60 (400 updates/stage). Existing L4 production
and the finite original L6 pilot continue.

Frozen guides at six and twelve stints and each starting guide receive new
four-population GFMC measurements:512 walkers/population,equil16,forward16,
eight cycles,two repeats. Same measurement task/RNG identities within each
context; separate model/source/device provenance. All sampling and ancestry
alarms are retained. Training is prioritized before these frozen measurements.

Finite scope:24 training contexts,288 stints,112 new GFMC runs plus eight reused
L6 fixed-guide controls. One worker/size, no deadline or monitoring loop.
The robust arm costs more compute and combines interventions; this is not
a compute-matched attribution of each ingredient. Physical results are pending.''')
    code('''%matplotlib inline
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown,display
scaling_root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data/kagome').is_dir())
scaling_archive=scaling_root/'''+repr(str(archive))+'''
scaling_hashes=json.loads((scaling_archive/'sha256.json').read_text())
for relative,digest in scaling_hashes.items():
    assert hashlib.sha256((scaling_archive/relative).read_bytes()).hexdigest()==digest,relative
scaling_snapshot=json.loads((scaling_archive/'snapshot.json').read_text())
scaling_tests=json.loads((scaling_archive/'validation/scaling_regressions.json').read_text())
assert scaling_tests['passed'] and scaling_tests['tests_run']==19 and scaling_tests['skipped']==0
display(Markdown('**Saved status: '+scaling_snapshot['utc']+'** — dated checkpoint, not a live monitor.'))
table=['| Size | Alive | Phase | Current context | Completed stints | Failures |',
       '|---|---|---|---|---:|---:|']
for r in scaling_snapshot['scaling']:
    s=r['status'];c=s.get('current') or {}
    detail=' / '.join(str(c.get(k,'')) for k in ['arm','sample_id','seed'])
    done=sum(x.get('completed_stints',0) for x in r['training_status'])
    table.append(f"| L{r['L']} | {r['alive']} | {s.get('state','pending')} | {detail} | {done}/96 | {len(s.get('failures',[]))} |")
display(Markdown(chr(10).join(table)))
print('Verified archive files:',len(scaling_hashes))
print('Implementation tests:',scaling_tests['tests_run'],'passed; none skipped')
''')
    code('''scaling_benchmark=json.loads((scaling_archive/'validation/kernel_benchmark.json').read_text())
fig,ax=plt.subplots(figsize=(8,4))
positions=np.arange(3)
for device,offset in [('cpu',-.18),('cuda',.18)]:
    values=[next(r['seconds_per_1024_rates'] for r in scaling_benchmark if r['L']==L and r['device']==device) for L in [6,8,10]]
    ax.bar(positions+offset,values,width=.36,label=device.upper())
ax.set(xticks=positions,xticklabels=['L6/N108','L8/N192','L10/N300'],yscale='log',
       ylabel='Seconds per 1024-walker rate/local-energy call',
       title='Measured kernel throughput on AutoDL; three-call averages')
ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();plt.show()
print('Kernel speedups:',{L:next(r['seconds_per_1024_rates'] for r in scaling_benchmark if r['L']==L and r['device']=='cpu')/next(r['seconds_per_1024_rates'] for r in scaling_benchmark if r['L']==L and r['device']=='cuda') for L in [6,8,10]})
''')
    md(r'''**Algorithm details.** The tempered marginal is
$q_eta(s)\propto\exp(2eta a\cdot s)\prod_j[2\cosh(eta(sW+b)_j)]^2$.
It is uniform at beta0 and exactly $\psi^2$ at beta1; at intermediate beta
it is not $\psi^{2eta}$. Exchanges use the marginal-density ratio.
Backtracking limits each empirical centered change in $2\log\psi$ to RMS.025
and max.15 on the current positive/negative minibatch; this is not a global KL
or unvisited-state bound. Four population islands retain separate ancestors,
normalizations and RNGs; they pool equal sample counts without ancestry weighting.

CUDA changes only batched guide evaluation, not the continuous-time event or
log-resampling mathematics. CPU/CUDA roundoff can alter realized trajectories.
The measured26–57x speedups are kernel benchmarks, not total-run speedups.

### Original L6 pilot: genealogy motivates the stronger arm

High one-step weight ESS and thousands of distinct configurations did not
prevent severe ancestry concentration. The following raw-data diagnostic is
training evidence only; no guide-update checkpoint is a pure-observable result.''')
    code('''pilot_rows=[]
for path in sorted((scaling_archive/'original_l6/training').glob('*/summary.json')):
    for row in json.loads(path.read_text())['rows']:
        folder=path.parent/f"stint_{row['stint']:03d}"
        raw=np.load(folder/'walkers.npz')
        collection=raw['lineages'][80:]
        np.testing.assert_allclose(collection[:,0].min(),row['minimum_training_lineage_ess'])
        np.testing.assert_allclose(collection[:,1].max(),row['largest_training_lineage'])
        pilot_rows.append(dict(context=path.parent.name,**row,weight_ess_min=float(raw['weight_ess'].min())))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for name in sorted({r['context'] for r in pilot_rows}):
    rows=[r for r in pilot_rows if r['context']==name];x=[r['stint']+1 for r in rows]
    axes[0].plot(x,[r['minimum_training_lineage_ess'] for r in rows],'o-',label=name)
    axes[1].plot(x,[r['largest_training_lineage'] for r in rows],'o-',label=name)
axes[0].set(yscale='log',xlabel='Completed stint',ylabel='Minimum collection lineage ESS')
axes[0].axhline(8,color='red',ls=':')
axes[1].set(xlabel='Completed stint',ylabel='Largest lineage fraction');axes[1].axhline(.5,color='red',ls=':')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.tight_layout();plt.show()
print('Archived original-pilot stints:',len(pilot_rows),'training ancestry alarms:',sum(r['training_ancestry_alarm'] for r in pilot_rows))
''')
    if progress is not None:
        md('''### First completed large-size feedback stints

This later checkpoint retains the first robust-arm training results and all
alarms. The first target was collected using the starting guide, before any
new likelihood fit; its genealogy therefore cannot measure the effect of the
subsequent tempered fit. Improvements between two training stints are not
independent physical-observable validation.''')
        code('''scaling_progress=scaling_root/'''+repr(str(progress))+'''
progress_hashes=json.loads((scaling_progress/'sha256.json').read_text())
for relative,digest in progress_hashes.items():
    assert hashlib.sha256((scaling_progress/relative).read_bytes()).hexdigest()==digest,relative
progress_snapshot=json.loads((scaling_progress/'snapshot.json').read_text())
display(Markdown('**Later checkpoint: '+progress_snapshot['utc']+'**'))
table=['| Size / arm / context | Stint | Worst island ESS | Largest lineage | Ancestry alarm | Tempering round trips | Min exchange | Bounded / rejected steps |',
       '|---|---:|---:|---:|---|---:|---:|---:|']
for entry in progress_snapshot['rows']:
    for group in entry['training']:
        for r in group['rows']:
            sampler=r.get('negative_sampler') or {}
            exchange=min(sampler.get('exchange_fraction',[float('nan')]))
            table.append(f"| L{entry['L']} / {group['arm']} / {group['context']} | {r['stint']+1} | {r['minimum_training_lineage_ess']:.3f} | {r['largest_training_lineage']:.3f} | {r['training_ancestry_alarm']} | {sampler.get('round_trips','—')} | {exchange:.3f} | {r.get('bounded_steps',0)} / {r.get('rejected_steps',0)} |")
    for r in entry['initial']:
        print('Warm start:',entry['L'],'stage field',r['gamma'],'update',r['step'],'of 400; no feedback/physical result yet')
display(Markdown(chr(10).join(table)))
print('Verified later-checkpoint files:',len(progress_hashes))
''')
    md('''**Next scientific output:** independently measured P(q),q2,connected spectra,
energy and ancestry at L6/L8/L10, compared across six/twelve stints and fixed
guides. Population/time/start dependence and uncertainty remain necessary
before calling any result converged. No critical-field or exponent claim is
supported by this launch or the kernel speedups.

The earlier dated notebook sections remain historical checkpoints. The full
new method and finite scope are documented in `code/transformer_qmc/FEEDBACK_SCALING.md`.''')
    notebook.cells[1:1]=cells;nb.write(notebook,path)
    if execute:
        NotebookClient(notebook,timeout=1200,kernel_name='python3',
            resources={'metadata':{'path':str(root)}},allow_errors=False).execute()
        nb.write(notebook,path);html,_=HTMLExporter().from_notebook_node(notebook)
        path.with_suffix('.html').write_text(html)
        errors=[o for c in notebook.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
        assert not errors
        print('Executed',len(notebook.cells),'cells;',sum(c.cell_type=='code' for c in notebook.cells),'code cells; zero errors',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--progress',type=Path)
    a=p.parse_args();update(a.root,a.archive,a.execute,a.progress)
