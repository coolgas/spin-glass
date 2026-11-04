"""Build an executed, compact notebook from downloaded ALiBi refinement data."""

import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
import torch

from .experiment import load_checkpoint, write_json
from .unified import UnifiedTransformer
from .physics import make_af_sample, exact_reference
from .refine_l2 import diagnostics


def main():
    root=Path(__file__).resolve().parents[2]
    results=root/'data/kagome/transformer_af_alibi_l2_v1'
    selected=json.loads((results/'selected.json').read_text())
    torch.set_num_threads(2)
    sample=make_af_sample(2,0)
    ham=sample.hamiltonian(.3)
    reference=exact_reference(ham,.5,sample.points,2)
    state=load_checkpoint(results/selected['best_checkpoint'],'cpu')
    model=UnifiedTransformer(ham,.5,**state['metadata']['architecture']).double()
    model.load_state_dict(state['model'])
    model.set_hamiltonian(ham,.5)
    checked=diagnostics(model,ham,sample,reference)
    assert abs(checked['normalization']-1)<1e-10
    assert abs(checked['energy']-selected['diagnostics']['energy'])<1e-5
    write_json(results/'selected_double_precision_check.json',checked)
    notebook=nbformat.v4.new_notebook()
    notebook.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    notebook.cells=[nbformat.v4.new_markdown_cell(r'''# Improving the 12-spin ALiBi wavefunction

This is a measured AutoDL benchmark on the **same** periodic Kagome Hamiltonian:
L=2, N=12, positive random bonds K∈[0.5,1.5], disorder ID 0, Γ=0.5, h=0.3.

The final model uses ALiBi. We keep the shared model's architecture (width 32,
two causal blocks, four heads, one static context block) and specialize its
weights to this Hamiltonian. This experiment does not establish that the
specialized checkpoint retains accuracy on other sizes or disorder samples.

Four sampled-VMC runs compare shared-checkpoint initialization against a fresh
ALiBi initialization, each with seeds 101 and 102. Each run uses 4,000 updates,
batch 512, and an Adam learning rate decaying from 0.001 to 0.0001. The
original shared model had only 200 updates per Hamiltonian context.

**Training uses sampled energy gradients, without exact ground-state labels.**
Every 200 updates we enumerate the learned distribution over 4,096 states for
validation. Checkpoints are selected by lowest enumerated variational energy;
the exact ground state is used to assess energy and observable errors. This is
a benchmark on the training Hamiltonian, not a held-out generalization test.

Accuracy targets were set before this run: energy error <10⁻⁴ per spin,
relative raw-overlap error <1%, and relative connected-susceptibility error <5%.
'''),nbformat.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
ROOT=Path.cwd()
if not (ROOT/'code/transformer_qmc').exists():
    ROOT=ROOT.parent
RESULTS=ROOT/'data/kagome/transformer_af_alibi_l2_v1'
summary=json.loads((RESULTS/'summary.json').read_text())
selected=json.loads((RESULTS/'selected.json').read_text())
reference=json.loads((RESULTS/'exact.json').read_text())
baseline=json.loads((ROOT/'data/kagome/transformer_af_dual_v1/benchmark/small_system.json').read_text())
rows=[('Original shared ALiBi',baseline['unified']),('Earlier fixed model',baseline['fixed'])]
rows += [(r['task']['initialization']+' ALiBi, seed '+str(r['task']['seed']),r['diagnostics']) for r in summary]
table=['| Model | E/N | Energy error/spin | Raw overlap | Connected χ |', '|---|---:|---:|---:|---:|']
table.append(f"| Exact | {reference['energy']/12:.9f} | 0 | {reference['q2']:.7f} | {reference['chi_connected']:.7f} |")
for label,d in rows:
    table.append(f"| {label} | {d['energy']/12:.9f} | {(d['energy']-reference['energy'])/12:.3g} | {d['q2']:.7f} | {d['chi_connected']:.7f} |")
display(Markdown('\\n'.join(table)))
display(Markdown('All values above use full enumeration of the learned state, so they have no Monte Carlo sampling error.'))
'''),nbformat.v4.new_code_cell('''fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
for r in summary:
    directory=RESULTS/Path(r['best_checkpoint']).parent
    trace=json.loads((directory/'validation.json').read_text())
    label=f"{r['task']['initialization']}, seed {r['task']['seed']}"
    for ax,key in zip(axes,['energy_error_per_spin','q2_relative_error','chi_connected_relative_error']):
        ax.semilogy([t['step'] for t in trace],[max(abs(t[key]),1e-12) for t in trace],label=label)
for ax,target,title in zip(axes,[1e-4,.01,.05],['Energy error per spin','Relative raw-overlap error','Relative connected-χ error']):
    ax.axhline(target,color='black',linestyle='--',label='Target')
    ax.set(xlabel='Additional VMC updates',ylabel='Absolute error magnitude',title=title)
axes[0].legend(fontsize=8)
fig.savefig(RESULTS/'alibi_l2_accuracy.png',dpi=180)
fig.savefig(RESULTS/'alibi_l2_accuracy.pdf')
plt.show()
'''),nbformat.v4.new_code_cell('''d=selected['diagnostics']
display(Markdown(f"**Selected model:** {selected['name']}, step {selected['best_step']}.\\n\\n"
                 f"Energy error per spin: **{d['energy_error_per_spin']:.3g}**; "
                 f"raw-overlap error: **{100*d['q2_relative_error']:.3f}%**; "
                 f"connected-susceptibility error: **{100*d['chi_connected_relative_error']:.3f}%**. "
                 f"Squared overlap with the exact ground state: **{d['fidelity']:.8f}**.\\n\\n"
                 f"All three targets met: **{d['qualified']}**.\\n\\n"
                 f"Checkpoint: `{selected['best_checkpoint']}`"))
'''),nbformat.v4.new_markdown_cell('''## What this establishes

The table and traces distinguish architecture capability from the quality of
the initial mixed-context training. Compare both initializations and both seeds;
do not infer that pretraining always helps from one selected checkpoint.

The selected ALiBi state is specialized to this 12-spin Hamiltonian. Applying it
as a frozen projector guide still requires projection-time, walker-population,
and ancestry checks. Accurate larger-size or held-out-disorder performance is
a separate qualification task. The earlier fixed-position state is retained as
a benchmark, while the selected final state uses ALiBi.
''')]
    path=root/'notebook/alibi_12site_accuracy.ipynb'
    NotebookClient(notebook,timeout=120,kernel_name='python3').execute(cwd=str(root))
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    print(path)
    print('Selected',selected['name'],selected['diagnostics'])


if __name__=='__main__':
    main()
