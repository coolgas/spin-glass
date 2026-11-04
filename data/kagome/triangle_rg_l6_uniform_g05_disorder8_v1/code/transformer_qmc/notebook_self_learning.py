"""Add the completed self-learning validation to the research notebook.

Execute the full notebook on AutoDL; this editor preserves existing sections.
"""
import argparse
from pathlib import Path

import nbformat as nb
from nbclient import NotebookClient
from nbconvert import HTMLExporter


TAG = 'self-learning-validation-20261103'


def update(root, execute=False):
    path = root/'notebook/jastrow_rbm_overlap_distributions.ipynb'
    notebook = nb.read(path, as_version=4)
    notebook.cells = [c for c in notebook.cells if TAG not in c.metadata.get('tags', [])]
    cells = []
    def md(text):
        cells.append(nb.v4.new_markdown_cell(text, metadata=dict(tags=[TAG])))
    def code(text):
        cells.append(nb.v4.new_code_cell(text, metadata=dict(tags=[TAG])))
    md(r'''## Iterative PQMC–RBM feedback: completed $L=2$ validation

**New evidence, 2026-11-03:** both seeds completed all six feedback stints,
followed by independent final-guide sampling and four-population GFMC.
This diagnostic is periodic kagome **$L=2$, $N=12$, $h=0.15$, $\Gamma=0.60$,
disorder 60000**, not a disorder ensemble. Every seed and stint is retained.

**Method correction.** Earlier VMC-budget extensions fit energy and then
project with a fixed guide; they omit projected-walker feedback. Here each
fixed-guide stint produces mixed walkers $f_s\propto\psi_s\psi_0$; normalized
$\psi_\theta^2$ is fitted by likelihood, and $\psi_\theta$ guides the next stint.
Ideally $\psi_{s+1}\propto\sqrt{\psi_s\psi_0}$. The two-hidden-copy sampler
matches our log-amplitude convention; this differs from the single-RBM density
architecture in the papers. Exact information is used only for evaluation.
See Pilati, Inack & Pieri, *Phys. Rev. E* **100**, 043301 (2019),
arXiv:1907.00907, and Brodoloni & Pilati, arXiv:2407.05978, Sec. II.2.

Each stint uses 1024 walkers, equilibration8/collection12, branching interval0.1,
64 walker configurations/block, 1000 Adam updates at 0.001, 256 persistent
negative chains and ten Gibbs sweeps/update. New populations equilibrate at
every guide change. Forward tags and physical measurement averages never cross
an adaptive update. The frozen final-guide measurements use new populations
and RNGs: 512 walkers per independent population, equilibration16/forward16,
eight cycles, two repeats per seed.

**Scope:** this is an implementation/finite-observable screen. Neither it nor
the historical figures certify ground-state convergence. The finite matched
$L=6$ pilot has launched; its physical results are pending. Earlier dated
campaign snapshots remain below as historical records.''')
    code('''%matplotlib inline
from pathlib import Path
import hashlib, json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
feedback_root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data/kagome').is_dir())
feedback_folder=feedback_root/'data/kagome/ltfim_disorder_scaling_v1/repaired_campaign/20261103_v2/self_learning'
feedback_hashes=json.loads((feedback_folder/'sha256.json').read_text())
for relative,digest in feedback_hashes.items():
    assert hashlib.sha256((feedback_folder/relative).read_bytes()).hexdigest()==digest,relative
feedback_load=lambda name:json.loads((feedback_folder/name).read_text())
feedback_training=feedback_load('validation/training_report.json')['rows']
feedback_measured=feedback_load('validation/summary.json')
feedback_guides=feedback_load('validation/guides.json')['rows']
feedback_runs=feedback_load('validation/measurements.json')['rows']
assert len(feedback_training)==12 and len(feedback_runs)==4
for r in feedback_runs:
    grid=np.asarray(r['q_grid']);mass=np.asarray(r['mass'])
    np.testing.assert_allclose([mass.sum(),grid@mass,grid**2@mass],[1,r['q'],r['q2']],atol=1e-12)
    assert not r['ancestry_alarm']
print('Verified feedback archive files:',len(feedback_hashes))
print('Completed stints:',len(feedback_training),'independent final-guide GFMC runs:',len(feedback_runs))
print('Exact eigensolver residual:',feedback_measured['exact']['residual'])
''')
    code('''fig,axes=plt.subplots(2,2,figsize=(12,8))
for seed,color in [(0,'tab:blue'),(1,'tab:orange')]:
    group=sorted([r for r in feedback_training if r['seed']==seed],key=lambda r:r['stint'])
    evaluations=[group[0]['before']]+[r['after'] for r in group]
    for ax,key,label in [(axes[0,0],'fidelity','Ground-state fidelity'),
                         (axes[0,1],'energy_error_per_site','Variational energy error/site'),
                         (axes[1,0],'ground_probability_tv','Configuration probability TV')]:
        ax.plot(range(7),[r[key] for r in evaluations],'o-',color=color,label=f'seed {seed}')
        ax.set(xlabel='Completed feedback stints',ylabel=label)
    axes[1,1].plot(range(1,7),[r['minimum_training_lineage_ess'] for r in group],
                   'o-',color=color,label=f'seed {seed}')
    alarmed=[r for r in group if r['training_ancestry_alarm']]
    axes[1,1].scatter([r['stint']+1 for r in alarmed],[r['minimum_training_lineage_ess'] for r in alarmed],
                      marker='x',s=100,color='red')
axes[0,1].set_yscale('log');axes[1,0].set_yscale('log');axes[1,1].set_yscale('log')
axes[0,1].axhline(.001,color='gray',ls=':',label='energy screening tolerance')
axes[1,1].axhline(8,color='red',ls=':',label='ESS alarm threshold')
axes[1,1].set(xlabel='Feedback stint',ylabel='Minimum training lineage ESS')
for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('L=2, N=12 | h=0.15 | Gamma=0.60 | disorder60000 | both seeds retained')
fig.tight_layout();plt.show()
''')
    code('''feedback_table=['| Seed | Final fidelity | Guide energy error/site | Configuration TV | Final training ESS | GFMC P(q) TV | GFMC q2 error | GFMC max spectrum error | GFMC minimum ESS |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for seed in [0,1]:
    t=next(r for r in feedback_training if r['seed']==seed and r['stint']==5)
    r=next(r for r in feedback_measured['rows'] if r['seed']==seed)
    relative=np.max(np.abs(np.asarray(r['mean']['spectrum'])-feedback_measured['exact']['spectrum'])/np.asarray(feedback_measured['exact']['spectrum']))
    feedback_table.append(f"| {seed} | {t['after']['fidelity']:.6f} | {t['after']['energy_error_per_site']:.6f} | {t['after']['ground_probability_tv']:.6f} | {t['minimum_training_lineage_ess']:.2f} | {r['errors']['histogram_tv']:.6f} | {r['errors']['q2_error']:.6f} | {relative:.2%} | {r['minimum_descendant_ess']:.2f} |")
display(Markdown(chr(10).join(feedback_table)))
fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharex=True,sharey=True)
feedback_q=np.asarray(feedback_measured['exact']['q_grid'])
for seed,ax in enumerate(axes):
    g=next(r for r in feedback_guides if r['seed']==seed)
    r=next(r for r in feedback_measured['rows'] if r['seed']==seed)
    ax.plot(feedback_q,feedback_measured['exact']['mass'],'ks-',ms=3,label='Exact ground state')
    ax.plot(feedback_q,g['mass'],'^--',ms=4,label='Final guide: exact enumeration')
    ax.errorbar(feedback_q,r['mean']['mass'],yerr=r['descriptive_repeat_se']['mass'],
                fmt='o-',ms=3,capsize=3,label='Independent GFMC: two-repeat mean')
    ax.set(title=f'L=2, N=12 | training seed {seed}',xlabel='Overlap q',ylabel='P(q)')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Final-guide validation | Gamma=0.60, h=0.15, disorder60000 | bars: descriptive repeat SE')
fig.tight_layout();plt.show()
''')
    md('''**Interpretation and uncertainty.** The final variational guide energy errors
remain 0.00363/0.00315 per site. After independent GFMC, overlap TV is
0.00697/0.00908 and maximum connected-spectrum relative error is 1.76%/1.43%;
all four measurements avoid ancestry alarms. Configuration-probability TV
and overlap TV are different statistics. Two-repeat SEs are descriptive, not
confidence bounds on population, projection-time or starting-state bias.

Frozen random/up/down Gibbs tests use 256 new chains/start, 500 burn sweeps,
then 100 samples separated by ten sweeps. Negative-phase sufficient-statistic
RMS errors against exact model expectations are 0.0039–0.0059; the largest
component discrepancy is 3.08 chain-wise SE. These checks do not prove that
adaptive negative chains were equilibrated throughout training or at L6.

Five mathematical/sampler/restart regressions passed, including exact
interrupted-fit continuation on CPU and CUDA separately. All 12 before/after
exact evaluations were independently recomputed on CPU; checkpoint continuity,
raw arrays and 56 training hashes were checked. The complete archive includes
all stints, models, optimizer/RNG checkpoints and the first-stint ancestry alarms.

### Finite L6 feedback pilot and existing campaigns

The L2 screen supports a **diagnostic** L6 comparison. The pilot launched at
2026-11-03T14:40:16Z with one sequential worker and no wall-clock deadline:
Gamma=.60, disorders60000/60001, seeds0/1, matching the 400-target-update VMC
checkpoint in each control. Each context receives six feedback stints under
the same pilot training settings, frozen random/up/down sampling checks, then
two matched final-guide measurements. Total: four contexts, 24 stints and
eight GFMC measurements. Model hashes differ; measurement settings and RNG
identities match the controls. No seed or alarmed measurement is selected away.
L6 ground-state convergence remains unresolved; its pilot outputs are pending.
The following saved status is a checkpoint, not a live monitor.''')
    code('''feedback_snapshot=feedback_load('campaign_snapshot.json')
display(Markdown('**Saved campaign status: '+feedback_snapshot['utc']+'**'))
feedback_status=['| Campaign | Alive at snapshot | Fits/milestones | Completed runs | Ancestry alarms | Failures |',
                 '|---|---|---:|---:|---:|---:|']
for row in feedback_snapshot['rows']:
    status=row['status']; fits=status.get('completed_milestones',status.get('completed_extensions',status.get('completed_fits','—')))
    feedback_status.append(f"| {row['campaign']} | {row['alive']} | {fits} | {row['saved_runs']} | {row['ancestry_alarms']} | {len(row['failures'])} |")
display(Markdown(chr(10).join(feedback_status)))
print('L6 feedback launch:',feedback_load('l6_launch/launch.json'))
''')
    notebook.cells[1:1] = cells
    for cell in notebook.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('# L6 convergence first'):
            cell.source = (root/'code/transformer_qmc/L6_CONVERGENCE.md').read_text()
    nb.write(notebook, path)
    if execute:
        NotebookClient(notebook, timeout=1200, kernel_name='python3',
            resources={'metadata':{'path':str(root)}}, allow_errors=False).execute()
        nb.write(notebook, path)
        html,_ = HTMLExporter().from_notebook_node(notebook)
        path.with_suffix('.html').write_text(html)
        print('Executed notebook cells:',len(notebook.cells),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    update(args.root,args.execute)


if __name__ == '__main__':
    main()
