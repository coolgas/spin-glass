"""Executed larger-size ALiBi refinement report; no exact-accuracy claims."""

import json
import hashlib
from pathlib import Path
import nbformat
from nbclient import NotebookClient
import torch

from .experiment import load_checkpoint, write_json
from .physics import make_af_sample
from .unified import UnifiedTransformer
from .vmc import local_energy


def main():
    root=Path(__file__).resolve().parents[2]
    results=root/'data/kagome/transformer_af_alibi_sizes_v1'
    checks=[]
    torch.set_num_threads(2)
    for run in json.loads((results/'summary.json').read_text()):
        final=results/run['final_checkpoint']
        latest=final.parent/'checkpoint.pt'
        # The original runner records the latest archive's hash. PyTorch
        # archives saved under different basenames need not have identical bytes.
        assert hashlib.sha256(latest.read_bytes()).hexdigest()==run['checkpoint_sha256']
        state=load_checkpoint(final,'cpu')
        last=load_checkpoint(latest,'cpu')
        assert state['step']==run['step']
        for key,value in state['model'].items():
            torch.testing.assert_close(value,last['model'][key],rtol=0,atol=0)
        ham=make_af_sample(run['L'],0).hamiltonian(.3)
        model=UnifiedTransformer(ham,.5,**state['metadata']['architecture'])
        model.load_state_dict(state['model'])
        spins=model.sample(8,torch.Generator().manual_seed(8800+run['seed']))
        assert torch.isfinite(local_energy(model,ham,.5,spins)).all()
        checks.append(dict(name=run['name'],final_checkpoint=run['final_checkpoint'],
                           final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                           latest_sha256=run['checkpoint_sha256'],weights_identical=True,reload_verified=True))
    write_json(results/'checkpoint_verification.json',checks)
    # Store executable analysis source separately from the frozen training tree.
    write_json(results/'report_provenance.json',dict(
        sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                 for name in ('report_sizes.py','audit_size_observables.py')},
        checks='latest archive hash, exact model weight equality, final reload and direct flip ratios'))
    notebook=nbformat.v4.new_notebook()
    notebook.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    notebook.cells=[nbformat.v4.new_markdown_cell(r'''# ALiBi refinement at 48, 108, and 192 spins

We extend the 12-spin refinement to periodic Kagome **L=4,6,8**, with
**48,108,192 spins**, respectively. Every comparison within a size uses the
same positive-random AF bonds (disorder ID 0), Γ=0.5, and h=0.3.

**Completed result:** both seeds lower the energy at every size. The refined
energy per spin is approximately −1.09727 (48 spins), −1.11604 (108 spins),
and −1.06935 (192 spins). These are improvements relative to the original
shared model, not errors relative to an exact reference.

The independent observable audit finds raw-overlap and connected-χ seed
differences within about 1.3 combined sampling standard errors at 48/108 spins.
At 192 spins, raw overlap is **0.48053 vs 0.47289** and connected χ is
**0.22159 vs 0.23322**, with differences about 13.4 and 4.3 combined sampling
standard errors. Magnetization differences are resolved at all sizes.
**The 192-spin observables are not yet stable across optimization seeds.**
Additional optimization/capacity and projection checks remain necessary;
energy agreement alone is insufficient.

Each Hamiltonian receives **two independent sampled-VMC refinements**, seeds
201 and 202, initialized from the original shared ALiBi model. We preserve the
width-32 architecture, two causal blocks, four heads, and the static context
encoder. Each run uses 2,400 updates, batch 128, and Adam with learning rate
0.0005→0.00005. The 12-site-specialized checkpoint is not used here.

**The predeclared final checkpoint is evaluated**, avoiding selection of a
checkpoint using a noisy energy minimum. Both seeds are shown. Final estimates
use 32 independent batches of 256 configurations (8,192 energy samples), plus
independent four-replica draws for the observables. Interim evaluations use
separate random streams. Exact diagonalization is unavailable at these sizes.

Read the results as follows:

- Lower energy at the **same Hamiltonian** is a variational improvement if
  the difference is resolved relative to its sampling error.
- Lower local-energy variance is useful evidence but does not certify a
  ground state; an excited eigenstate also has zero variance.
- Compare raw overlap and connected susceptibility across training seeds.
  Differences beyond sampling error indicate residual variational sensitivity.
- Neither seed agreement nor a training plateau establishes ground-state
  convergence. Projection and population checks remain necessary.

The earlier fixed-position models and original shared ALiBi model are baselines
with different training budgets. This comparison does not isolate ALiBi's effect.
'''),nbformat.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
ROOT=Path.cwd()
if not (ROOT/'code/transformer_qmc').exists():
    ROOT=ROOT.parent
RESULTS=ROOT/'data/kagome/transformer_af_alibi_sizes_v1'
BASELINE=ROOT/'data/kagome/transformer_af_dual_v1'
summary=json.loads((RESULTS/'summary.json').read_text())
status=json.loads((RESULTS/'status.json').read_text())
display(Markdown(f"**Campaign state:** {status['state']}; {len(summary)}/6 final evaluations downloaded."))
shared={r['L']:r['vmc'] for r in json.loads((BASELINE/'unified_alibi/evaluation.json').read_text()) if r['sample_id']==0}
fixed={L:[] for L in (4,6,8)}
for path in sorted((BASELINE/'fixed_size').glob('*/evaluation.json')):
    s=json.loads((path.parent/'settings.json').read_text())
    if s['L'] in fixed and s['gamma']==.5 and s['h']==.3:
        fixed[s['L']].append((s['train_seed'],json.loads(path.read_text())))
table=['| Spins | Model | Energy/spin ± SE | Variance/N | Raw q² ± SE | Connected χ ± SE |',
       '|---:|---|---:|---:|---:|---:|']
def row(n,label,r):
    def value(key,scale=1):
        return f"{r[key]['mean']/scale:.6f} ± {r[key]['se']/scale:.6f}"
    return f"| {n} | {label} | {value('energy',n)} | {r['local_energy_variance']/n:.6f} | {value('q2')} | {value('chi_connected')} |"
for L in (4,6,8):
    n=3*L*L
    table.append(row(n,'Original shared ALiBi',shared[L]))
    for seed,r in fixed[L]:
        table.append(row(n,f'Earlier fixed, seed {seed}',r))
    for r in summary:
        if r['L']==L:
            table.append(row(n,f"Refined ALiBi, seed {r['seed']}",r['sampled']))
display(Markdown('\\n'.join(table)))
display(Markdown('SE denotes Monte Carlo sampling error for the individual variational state, not its unknown error relative to the exact ground state. Variances are point estimates.'))
'''),nbformat.v4.new_code_cell('''table=['| Spins | Seed | ΔE/N versus original shared ± SE | Variance reduction factor |',
       '|---:|---:|---:|---:|']
improvements=[]
for r in summary:
    original=shared[r['L']]
    refined=r['sampled']
    n=r['N']
    delta=(refined['energy']['mean']-original['energy']['mean'])/n
    se=np.hypot(refined['energy']['se'],original['energy']['se'])/n
    factor=original['local_energy_variance']/refined['local_energy_variance']
    table.append(f"| {n} | {r['seed']} | {delta:.6f} ± {se:.6f} | {factor:.2f}× |")
    improvements.append(dict(L=r['L'],N=n,seed=r['seed'],energy_change_per_spin=delta,
                             combined_se=se,variance_reduction=factor))
display(Markdown('\\n'.join(table)))
display(Markdown('Negative ΔE means lower energy on the identical Hamiltonian. The ratio of estimated variances is descriptive; it is not a measured projector speedup.'))
(RESULTS/'improvements.json').write_text(json.dumps(improvements,indent=2)+'\\n')
'''),nbformat.v4.new_code_cell('''fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
for ax,L in zip(axes,(4,6,8)):
    n=3*L*L
    for r in summary:
        if r['L']!=L:
            continue
        directory=RESULTS/Path(r['final_checkpoint']).parent
        trace=json.loads((directory/'validation.json').read_text())
        ax.errorbar([t['step'] for t in trace],
                    [t['sampled']['energy']['mean']/n for t in trace],
                    yerr=[t['sampled']['energy']['se']/n for t in trace],
                    marker='o',capsize=3,label=f"ALiBi seed {r['seed']}")
        ax.errorbar([r['step']+80],[r['sampled']['energy']['mean']/n],
                    yerr=[r['sampled']['energy']['se']/n],fmt='x',color='black',capsize=3)
    for seed,r in fixed[L]:
        ax.axhline(r['energy']['mean']/n,linestyle='--',alpha=.5,label=f'Fixed seed {seed}')
    ax.set(title=f'L={L}: {n} spins',xlabel='Additional VMC updates',ylabel='Energy per spin')
    ax.legend(fontsize=8)
fig.suptitle('ALiBi refinement: independent validation, with final evaluation shown as black ×',fontsize=11)
fig.savefig(RESULTS/'alibi_sizes_energy.png',dpi=180)
fig.savefig(RESULTS/'alibi_sizes_energy.pdf')
plt.show()
'''),nbformat.v4.new_code_cell('''fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
for index,L in enumerate((4,6,8)):
    n=3*L*L
    records=[('Original shared',shared[L],index-.22,'black','s')]
    for offset,r in enumerate(sorted([r for r in summary if r['L']==L],key=lambda r:r['seed'])):
        records.append((f"Refined seed {r['seed']}",r['sampled'],index+.02+.18*offset,
                        ['tab:blue','tab:orange'][offset],'o'))
    for label,r,x,color,marker in records:
        axes[0].plot(x,r['local_energy_variance']/n,marker=marker,color=color,
                     linestyle='none',label=label if index==0 else None)
        for ax,key in zip(axes[1:],['q2','chi_connected']):
            ax.errorbar(x,r[key]['mean'],yerr=r[key]['se'],fmt=marker,color=color,capsize=3)
for ax,title in zip(axes,['Local-energy variance per spin','Variational raw overlap ⟨q²⟩','Variational connected χ']):
    ax.set(xticks=range(3),xticklabels=['48','108','192'],xlabel='Number of spins',title=title)
axes[0].legend(fontsize=8)
fig.suptitle('Seed sensitivity and variance: statistical errors do not include variational bias',fontsize=11)
fig.savefig(RESULTS/'alibi_sizes_observables.png',dpi=180)
fig.savefig(RESULTS/'alibi_sizes_observables.pdf')
plt.show()
'''),nbformat.v4.new_code_cell('''table=['| Spins | Quantity | Seed 201 − seed 202 | Combined sampling SE | Difference / SE |',
       '|---:|---|---:|---:|---:|']
comparisons=[]
for L in (4,6,8):
    runs=sorted([r for r in summary if r['L']==L],key=lambda r:r['seed'])
    if len(runs)!=2:
        continue
    n=3*L*L
    for key,label in [('energy','Energy/spin'),('q2','Raw q²'),('chi_connected','Connected χ')]:
        a,b=[r['sampled'][key] for r in runs]
        scale=n if key=='energy' else 1
        delta=(a['mean']-b['mean'])/scale
        se=np.hypot(a['se'],b['se'])/scale
        ratio=delta/se if se else None
        table.append(f"| {n} | {label} | {delta:.6g} | {se:.6g} | {ratio:.2f} |")
        comparisons.append(dict(L=L,quantity=key,delta=delta,se=se,difference_over_se=ratio))
display(Markdown('\\n'.join(table)))
display(Markdown('These are descriptive differences between two trained states. Ratios well beyond a few sampling standard errors reveal sensitivity; they are not a calibrated multiple-testing procedure or a ground-state error estimate.'))
(RESULTS/'seed_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\\n')
'''),nbformat.v4.new_code_cell('''audit_path=RESULTS/'observable_audit.json'
if audit_path.exists():
    audit=json.loads(audit_path.read_text())
    table=['| Spins | Seed | Raw q² ± SE | Connected χ ± SE | Magnetization ± SE |',
           '|---:|---:|---:|---:|---:|']
    for r in audit:
        def estimate(key):
            v=r['observables'][key]
            return f"{v['mean']:.6f} ± {v['se']:.6f}"
        table.append(f"| {3*r['L']**2} | {r['seed']} | {estimate('q2')} | {estimate('chi_connected')} | {estimate('magnetization')} |")
    display(Markdown('### Independent observable audit\\n\\n32,768 four-replica groups per checkpoint, from fresh sampling streams.\\n\\n'+'\\n'.join(table)))
    for L in (4,6,8):
        pair=sorted([r for r in audit if r['L']==L],key=lambda r:r['seed'])
        if len(pair)==2:
            comparisons=[]
            for key,label in [('q2','raw q²'),('chi_connected','connected χ'),('magnetization','magnetization')]:
                a,b=[r['observables'][key] for r in pair]
                z=(a['mean']-b['mean'])/np.hypot(a['se'],b['se'])
                comparisons.append(f"{label}: **{z:.2f}**")
            display(Markdown(f"L={L}, seed difference / combined sampling SE — "+'; '.join(comparisons)))
else:
    display(Markdown('Independent higher-statistics observable audit not yet downloaded.'))
'''),nbformat.v4.new_markdown_cell('''## Checkpoints and next interpretation

All completed final ALiBi checkpoints are retained with explicit L, spin count,
fields, disorder ID, initialization, and seed in the filename. `summary.json`
contains relative final paths and the original latest-archive SHA-256 hashes.
`checkpoint_verification.json` maps both archives to their separate hashes and
records exact equality of their model weights plus local reload/ratio checks.
`campaign.json` records source
hashes and training settings; each run stores its actual bonds and history.

These are Hamiltonian-specific ALiBi guides. There is no assumption that a
specialized checkpoint retains shared-model accuracy on other instances.
Direct transformer ratios are used at all these sizes, without state-space
enumeration. To qualify physical predictions, compare further training/capacity
changes and independent projected estimates with varying projection times,
populations, and branching cadence. A phase boundary cannot be inferred from
these three single-disorder field points.
''')]
    path=root/'notebook/alibi_larger_sizes.ipynb'
    NotebookClient(notebook,timeout=120,kernel_name='python3').execute(cwd=str(root))
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    print(path)


if __name__=='__main__':
    main()
