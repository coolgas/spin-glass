"""Append a frozen, explicitly incomplete sampled-training snapshot to the report."""
from pathlib import Path
import json
import shutil

import nbformat
from nbclient import NotebookClient


def main():
    root=Path(__file__).resolve().parents[2]
    data=root/'data/kagome/ltfim_simple_guides_v1'
    live=data/'sampled/review.json'
    report=json.loads(live.read_text())
    # This explicit snapshot prevents notebook outputs and figures from silently
    # changing as the independent background training process completes tasks.
    snapshot=data/'sampled_snapshot.json'
    snapshot.write_text(json.dumps(dict(rows=report['rows'],
        campaign_status=json.loads((data/'sampled/status.json').read_text()),
        scope='Frozen partial snapshot; order is sequential, not a random sample of tasks.'),indent=2)+'\n')
    path=root/'notebook/simple_guiding_wavefunctions.ipynb'
    notebook=nbformat.read(path,as_version=4)
    marker='## 5. Sampled training on AutoDL: frozen snapshot'
    if any(marker in c.source for c in notebook.cells):
        first=next(i for i,c in enumerate(notebook.cells) if marker in c.source)
        notebook.cells=notebook.cells[:first]
    notebook.cells.extend([nbformat.v4.new_markdown_cell(r"""## 5. Sampled training on AutoDL: frozen snapshot

This follow-up trains on samples only. Eight parallel-tempering levels target
$\psi^{2\beta}$, with 256 chains per level, local spin flips, global inversion,
uniform refreshes at $\beta=0$, and exact replica exchanges. Cached fields make
each proposal inexpensive. Adaptive training samples are not convergence data.

Compare direct training at the target field with continuation from $\Gamma=1.5$
through 0.75 to the target. Each stage uses 400 Adam updates. **Continuation uses
more total updates:** this experiment cannot distinguish schedule effects from
additional compute. All seeds are retained, and no exact-state label enters
training. The original Hamiltonian is restored at the target stage.

After parameters are frozen, new random/up/down starts are measured separately.
Each uses 200 warmup and 200 measurement sweeps, recording every fourth sweep.
For L2 only, post-training enumeration measures ground fidelity and the total
variation distance between sampled histograms and the fitted guide's exact
probability. Finite histograms and autocorrelation contribute to that distance.
Temperature round trips alone do not establish valley equilibration.

The snapshot is partial and tasks run in a fixed order; it is not an unbiased
summary of the whole planned campaign. L4 feasibility tasks may still be pending.
"""),nbformat.v4.new_code_cell("""sampled=json.loads((folder/'sampled_snapshot.json').read_text())
print('Frozen sampled-training fits:',len(sampled['rows']))
print('Campaign state at snapshot:',sampled['campaign_status'])
for r in sampled['rows']:
    t=r['task'];e=r.get('exact_evaluation')
    if e is None:
        print(t,'larger-L feasibility only');continue
    tv=[d['tv_from_exact_guide_probability'] for d in r['diagnostics']]
    print(f\"Gamma={t['gamma']} s={t['sample_id']} {t['kind']:8s} seed={t['seed']} {t['mode']:6s}: \"
          f\"fidelity={e['fidelity']:.6f}, spectral error={100*e['max_spectral_error']:.2f}%, \"
          f\"guide histogram TV={min(tv):.4f}..{max(tv):.4f}\")

complete=[]
for r in sampled['rows']:
    t=r['task']
    if t['mode']!='direct' or 'exact_evaluation' not in r:continue
    matches=[a for a in sampled['rows'] if a['task']==dict(t,mode='anneal')]
    if matches:complete.append((r,matches[0]))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for i,(direct,anneal) in enumerate(complete):
    for r,marker in [(direct,'o'),(anneal,'x')]:
        e=r['exact_evaluation'];t=r['task']
        axes[0].plot(i,e['fidelity'],marker,color=colors[t['kind']])
        axes[1].plot(i,100*e['max_spectral_error'],marker,color=colors[t['kind']])
labels=[f\"{a['task']['sample_id']} {a['task']['kind']} s{a['task']['seed']}\" for a,b in complete]
for ax in axes:
    ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right');ax.grid(alpha=.25)
axes[0].set_yscale('log');axes[0].set_ylabel('Ground-state fidelity')
axes[1].set_yscale('log');axes[1].set_ylabel('Maximum unprojected spectral error (%)')
fig.suptitle('Sampled VMC: circles = direct; crosses = continuation (more updates)')
fig.tight_layout();fig.savefig(figure_dir/'sampled_snapshot.png',dpi=180,bbox_inches='tight')
fig.savefig(figure_dir/'sampled_snapshot.pdf',bbox_inches='tight');plt.show()
"""),nbformat.v4.new_markdown_cell("""The full-basis and sampled-training experiments answer different questions.
The former demonstrates that simple ansätze can represent useful low-field
guides. The latter tests whether sampling and optimization can find them. A
successful L2 guide is not yet a scalable protocol: matched-compute schedules,
more optimization seeds, L4 mixing, and GFMC sensitivity checks remain necessary.
""")])
    audit=data/'audit/review.json'
    if audit.exists():
        notebook.cells.extend([nbformat.v4.new_markdown_cell(r"""## 6. Completed GFMC sensitivity audit

The 64 additional runs test 4096 walkers, doubled equilibration/forward times,
uniform initial states, and triangle step 0.05. Each audit group has two repeats.
Exact CT references use the actual forward-walking schedule, including growing
left projection over cycles and fixed right projection. Their discrepancy from
the ground state is finite-time bias; measured discrepancies from those
references contain population/sampling effects.

At $\Gamma=0.5$, neither longer projection nor larger population establishes
convergence. Uniform starts can improve apparent spectra while raising ancestry
alarms. Such runs remain unresolved. The table retains all groups and invalid
length fractions; bootstrap intervals from two runs are descriptive only.
"""),nbformat.v4.new_code_cell("""audit=json.loads((folder/'audit/review.json').read_text())
print('Audit status:',json.loads((folder/'audit/status.json').read_text()))
print('kind Gamma disorder method audit error(%) alarms invalid-xi-fraction')
for r in audit['rows']:
    print(r['kind'],r['gamma'],r['sample_id'],r['method'],r['audit'],
          round(100*r['max_relative_error'],3),r['alarms'],r['invalid_length_fraction'])
print('Gamma=0.5 CT error decomposition:')
for r in audit['rows']:
    if r['gamma']==.5 and r['method']=='ct':
        print(r['kind'],r['audit'],
              'finite-time error (%)',np.round(100*np.array(r['finite_projection_relative_error']),2),
              'sampling relative to finite target (%)',np.round(100*np.array(r['relative_sampling_error']),2))
""")])
    NotebookClient(notebook,timeout=180,kernel_name='python3',
        resources={'metadata':{'path':str(root)}}).execute()
    nbformat.write(notebook,path)
    print('Executed simple-guiding notebook with',len(report['rows']),'frozen sampled fits')


if __name__=='__main__':main()
