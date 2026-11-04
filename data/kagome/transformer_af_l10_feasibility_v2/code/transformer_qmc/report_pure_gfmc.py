"""Execute a readable constant-guide versus network-guide GFMC report."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    args = parser.parse_args()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    notebook.cells = [md(r'''# Ordinary and neural-guided GFMC: matched 12-spin benchmark

**Pure GFMC** here means ordinary propagation with a constant guide
$\psi_T(s)=1$. **Guided GFMC** uses the same frozen shared seed501 transformer
$\psi_T(s)=\sqrt{p_\theta(s)}$. Both use the same continuous-time projector,
branching cadence, walker counts, times, disorder samples, and seed list.
Initial samples follow the corresponding trial probability. Exact ground states
provide evaluation targets only. No optimizer is constructed.

Energy is a mixed estimator. Connected susceptibility uses four physical
replicas and forward walking in **both** methods. Thus “pure” in the title
describes the guide; pure diagonal estimation is common to both methods.
All forward cycles are retained, including those with poor ancestry.
Error bars are standard errors across four independent populations.

The network uses a complete 12-spin lookup of its frozen amplitudes.
Recorded cost includes propagation and initial sampling, and excludes lookup
construction, cache loading, and direct neural inference. Efficiency ratios
here assess the stochastic algorithms with lookup rates. Larger-system
efficiency needs direct-inference costs and its own convergence checks.
'''), code(f'''from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
ROOT=Path({str(args.results.resolve())!r})
summary=json.loads((ROOT/'summary.json').read_text())
manifest=summary['manifest']
rows=summary['rows']
assert not manifest['reserved_test_ids_inspected']
display(Markdown(f"Completed populations: **{{summary['completed_populations']}}/{{summary['expected_populations']}}**; complete=**{{summary['completed']}}**. Completed groups: **{{len(rows)}}**."))
display(manifest['design'])
if (ROOT/'status.json').exists(): display(json.loads((ROOT/'status.json').read_text()))
display(Markdown('Frozen guide SHA-256: `'+manifest['checkpoint_sha256']+'`'))
'''), md('''## Initial guide quality

Fidelity and local-energy variance are evaluated by enumeration. They describe
the initial trial state, before stochastic projection. Poor trial fidelity can
coexist with useful importance sampling; the projected observables below decide
whether the guide helps in practice.
'''), code('''table=['| Γ | h | K | Disorder | Guide | Initial fidelity | Initial Var(E local) |','|---|---|---|---|---|---:|---:|']
seen=set()
for r in rows:
    context=(r['gamma'],r['h'],r['K'],r['sample_id'],r['guide'])
    if context in seen: continue
    seen.add(context)
    table.append(f"| {r['gamma']:g} | {r['h']:g} | {r['K']:g} | {r['sample_id']} | {r['guide']} | {r['initial_fidelity']:.6f} | {r['initial_local_energy_variance']:.6f} |")
display(Markdown('\\n'.join(table)))
'''), md('''## Matched stochastic results

Energy errors are per spin; susceptibility is the connected spin-glass
susceptibility. Each row combines four independent populations. The smallest
ancestry ESS and largest lineage include every forward cycle. Agreement of
energy alone does not establish correlation convergence.
'''), code('''table=['| Γ | h | K | Disorder | Guide | W | Kτ forward | (E−E₀)/N ± SE | χ ± SE | Exact χ | Min ancestry ESS | Largest lineage | Seconds |','|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    e=r['observables']['energy']; c=r['observables']['chi_connected']
    table.append(f"| {r['gamma']:g} | {r['h']:g} | {r['K']:g} | {r['sample_id']} | {r['guide']} | {r['walkers']} | {r['forward_Ktau']:g} | {e['bias']/12:.6f} ± {e['se']/12:.6f} | {c['mean']:.5f} ± {c['se']:.5f} | {c['exact']:.5f} | {r['min_descendant_ess']:.2f} | {r['max_largest_lineage']:.3f} | {r['seconds']:.2f} |")
display(Markdown('\\n'.join(table)))
'''), md('''## Population and forward-time dependence

Biases are plotted against exact ground-state targets. Lines connect forward
times at fixed population and method. Error bars represent statistical
uncertainty; projection and population bias require stability as settings grow.
'''), code('''contexts=manifest['design']['contexts']
fig,axes=plt.subplots(3,len(contexts),figsize=(5*len(contexts),10),squeeze=False,constrained_layout=True)
for col,context in enumerate(contexts):
    for sid in manifest['design']['sample_ids']:
        for W in manifest['design']['walkers']:
            for kind,style in [('constant','--'),('network','-')]:
                group=sorted([r for r in rows if all(r[k]==v for k,v in context.items()) and r['sample_id']==sid and r['walkers']==W and r['guide']==kind],key=lambda r:r['forward_Ktau'])
                if not group: continue
                x=[r['forward_Ktau'] for r in group]
                label=f'{kind}, s={sid}, W={W}'
                for ax,metric,factor in [(axes[0,col],'energy',12),(axes[1,col],'chi_connected',1)]:
                    ax.errorbar(x,[r['observables'][metric]['bias']/factor for r in group],yerr=[r['observables'][metric]['se']/factor for r in group],marker='.',linestyle=style,label=label)
                axes[2,col].plot(x,[r['min_descendant_ess']/W for r in group],marker='.',linestyle=style,label=label)
    for row,ylabel in enumerate(['(E−E₀)/N','χ−χ exact','Minimum ancestry ESS / W']):
        axes[row,col].set(xlabel='Kτ forward',ylabel=ylabel)
        if row<2: axes[row,col].axhline(0,color='k',lw=.7)
    axes[0,col].set_title(', '.join(f'{k}={v:g}' for k,v in context.items()))
axes[-1,-1].legend(fontsize=6,bbox_to_anchor=(1.02,1))
fig.savefig(ROOT/'pure_vs_guided_convergence.png',dpi=160)
plt.show()
'''), md('''## Measured uncertainty and lookup cost

The ratio below is `(constant SE² × constant seconds) / (network SE² × network
seconds)` for matched settings. Values above one favor the network in variance
per measured cost. Four populations give a noisy variance estimate. Bias must
be assessed separately in the table above, and these lookup measurements
exclude direct transformer inference.
'''), code('''lookup={(r['gamma'],r['h'],r['K'],r['sample_id'],r['walkers'],r['forward_Ktau'],r['guide']):r for r in rows}
table=['| Γ | h | K | Disorder | W | Kτ forward | Constant/network seconds | Energy efficiency ratio | χ efficiency ratio |','|---|---|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    if r['guide']!='constant': continue
    k=(r['gamma'],r['h'],r['K'],r['sample_id'],r['walkers'],r['forward_Ktau'],'network')
    n=lookup.get(k)
    if n is None: continue
    ratios={}
    for metric in ['energy','chi_connected']:
        denom=n['observables'][metric]['se']**2*n['seconds']
        ratios[metric]=r['observables'][metric]['se']**2*r['seconds']/denom if denom else float('nan')
    table.append(f"| {r['gamma']:g} | {r['h']:g} | {r['K']:g} | {r['sample_id']} | {r['walkers']} | {r['forward_Ktau']:g} | {r['seconds']/n['seconds']:.3f} | {ratios['energy']:.3f} | {ratios['chi_connected']:.3f} |")
display(Markdown('\\n'.join(table)))
'''), md('''## Deterministic projection references

These curves project the constant and network trial amplitudes exactly in
the 12-spin Hilbert space. Fidelity and full connected-matrix errors measure
remaining projection error without walker noise or finite-population bias.
The pure projected state differs from a finite-forward-time stochastic
estimator; these curves serve as projection diagnostics.
'''), code('''p=ROOT/'projection_reference.json'
if p.exists():
    references=json.loads(p.read_text())
    assert references['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert references['input_sha256']==manifest['input_sha256']
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    table=['| Γ | h | K | Disorder | Guide | Fidelity at Kτ16 | Fidelity at Kτ64 | C RMS error at Kτ64 |','|---|---|---|---|---|---:|---:|---:|']
    for r in references['rows']:
        t=r['trajectory']; end=t[-1]; mid=next(v for v in t if v['Ktau']==16)
        table.append(f"| {r['gamma']:g} | {r['h']:g} | {r['K']:g} | {r['sample_id']} | {r['guide']} | {mid['fidelity']:.6f} | {end['fidelity']:.6f} | {end['connected_correlation_rms_error']:.6f} |")
        label=f"{r['guide']}, Γ={r['gamma']:g}, K={r['K']:g}, s={r['sample_id']}"
        style='--' if r['guide']=='constant' else '-'
        axes[0].semilogy([v['Ktau'] for v in t],[max(1-v['fidelity'],1e-15) for v in t],style,label=label)
        axes[1].semilogy([v['Ktau'] for v in t],[max(v['connected_correlation_rms_error'],1e-15) for v in t],style,label=label)
    display(Markdown('\\n'.join(table)))
    axes[0].set(xlabel='Kτ',ylabel='Infidelity'); axes[1].set(xlabel='Kτ',ylabel='Connected C RMS error')
    axes[1].legend(fontsize=5,bbox_to_anchor=(1.02,1))
    fig.savefig(ROOT/'pure_vs_guided_exact_projection.png',dpi=160)
    plt.show()
    finite=references.get('finite_forward_rows',[])
    finite_lookup={(r['gamma'],r['h'],r['K'],r['sample_id'],r['guide'],r['forward_Ktau']):r for r in finite}
    if finite:
        display(Markdown('### Exact finite-forward-time targets\\n\\nFor a tag at time $t$ followed for $f$, the infinite-population ancestor probability is proportional to $\\phi_t(s)\\phi_f(s)$, where $\\phi_t=e^{-tH}\\psi_T$. The table averages the same cycle times as the stochastic run. This separates forward-time bias from finite-population error.'))
        table=['| Γ | K | Disorder | Guide | W | Kτ forward | Stochastic χ ± SE | Infinite-population finite-time χ | Exact ground-state χ |','|---|---|---|---|---:|---:|---:|---:|---:|']
        for r in rows:
            k=(r['gamma'],r['h'],r['K'],r['sample_id'],r['guide'],r['forward_Ktau'])
            target=finite_lookup.get(k)
            if target is None: continue
            c=r['observables']['chi_connected']
            table.append(f"| {r['gamma']:g} | {r['K']:g} | {r['sample_id']} | {r['guide']} | {r['walkers']} | {r['forward_Ktau']:g} | {c['mean']:.5f} ± {c['se']:.5f} | {target['chi_connected']:.5f} | {c['exact']:.5f} |")
        display(Markdown('\\n'.join(table)))
else:
    display(Markdown('Deterministic projection references pending.'))
'''), md('''## Reading the benchmark

Compare methods within each fixed Hamiltonian and setting. Both should approach
the same exact result as projection, forward time, and population grow. A
useful guide reduces uncertainty or required population at comparable bias.
Ancestry collapse, disagreement between populations, and time dependence all
remain visible. This first comparison does not establish convergence at larger
sizes or across disorder, and does not determine a phase boundary.
''')]
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    NotebookClient(notebook, timeout=300, kernel_name='python3').execute()
    nbformat.write(notebook, args.notebook)
    print(args.notebook.resolve())


if __name__ == '__main__':
    main()
