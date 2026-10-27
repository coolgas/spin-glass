"""Build and execute a report using completed frozen-scan artifacts only."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--direct-preview', type=Path)
    parser.add_argument('--direct-pilot', type=Path)
    args = parser.parse_args()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    notebook.cells = [md(r'''# Frozen shared ALiBi: field/bond scans and guided GFMC

The main seed501 final won the predefined eight-disorder development comparison
(score 0.291550). Its **single parameter state is frozen across all sizes and
contexts**. This report reads completed records; the progress table distinguishes
partial campaigns from completed stages. Reserved disorders 40000–40007 are unused.

We use $H=\sum K_{ij}Z_iZ_j-\Gamma\sum X_i-h\sum Z_i$ with
$K_{ij}=K(1+0.5u_{ij})$ and matched relative disorder. The full grid is
$\Gamma\in\{0.25,0.5,0.75,1,1.5,2,3\}$,
$h\in\{0,0.15,0.3,0.6,1\}$, $K\in\{0.5,1,2\}$.
Only $K=1$ and the three training anchors have direct training support.
All other contexts measure transfer. Exact states are evaluation references.

GFMC uses the same positive $\psi=\sqrt p$. The 12-spin lookup caches the
**network** exactly. Energy is a mixed estimator; diagonal observables use
four-replica forward walking. Error bars are across independent populations,
and all forward cycles are retained. Projection, population, forward-time,
and branching-cadence dependence must be assessed together. Larger-size
variational measurements and short throughput pilots do not establish
larger-size ground-state accuracy or finite-size scaling.
'''), code(f'''from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
ROOT=Path({str(args.results.resolve())!r})
manifest=json.loads((ROOT/'manifest.json').read_text())
assert manifest['checkpoint_sha256']=='5f9fa33b04102ddff2e43179440514021090ed3aaf228ce67dce53c2139e2544'
assert not manifest['reserved_tests_inspected']
display(Markdown('Frozen checkpoint SHA-256: `'+manifest['checkpoint_sha256']+'`'))
data={{}}
table=['| Stage | Completed records | Expected | Complete |','|---|---:|---:|---|']
for stage in ('benchmark','exact','projection','sampled','gfmc'):
    p=ROOT/(stage+'_summary.json')
    d=json.loads(p.read_text()) if p.exists() else {{'rows':[],'completed':False,'expected_rows':'pending'}}
    data[stage]=d['rows']
    table.append(f"| {{stage}} | {{len(d['rows'])}} | {{d['expected_rows']}} | {{d['completed']}} |")
display(Markdown('\\n'.join(table)))
if (ROOT/'supervisor.json').exists(): display(json.loads((ROOT/'supervisor.json').read_text()))
'''), md('## Exact 12-spin transfer map\n\nMeans below use available development disorders; the counts expose partial coverage. Energy errors are per spin. Fidelity and connected-correlation errors carry independent information.'), code('''rows=data['exact']
table=['| K | h | Γ | Disorders | Mean fidelity | Minimum fidelity | Mean energy error/N | C RMS error | Absolute χ error |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
for K in manifest['K']:
    for h in manifest['h']:
        for g in manifest['gamma']:
            group=[r for r in rows if (r['K'],r['h'],r['gamma'])==(K,h,g)]
            if not group: continue
            table.append(f"| {K:g} | {h:g} | {g:g} | {len(group)} | {np.mean([r['fidelity'] for r in group]):.5f} | {min(r['fidelity'] for r in group):.5f} | {np.mean([r['energy_error_per_spin'] for r in group]):.6f} | {np.mean([r['connected_correlation_rms_error'] for r in group]):.5f} | {np.mean([abs(r['chi_connected']-r['exact_chi']) for r in group]):.5f} |")
display(Markdown('\\n'.join(table)))
if rows:
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for col,K in enumerate(manifest['K']):
        for row,metric,title in [(0,'fidelity','Fidelity'),(1,'connected_correlation_rms_error','Connected C RMS error')]:
            array=np.full((len(manifest['h']),len(manifest['gamma'])),np.nan)
            for j,h in enumerate(manifest['h']):
                for i,g in enumerate(manifest['gamma']):
                    group=[v[metric] for v in rows if (v['K'],v['h'],v['gamma'])==(K,h,g)]
                    if group: array[j,i]=np.mean(group)
            ax=axes[row,col]
            im=ax.imshow(array,origin='lower',aspect='auto',vmin=0,vmax=1 if row==0 else None)
            ax.set_xticks(range(len(manifest['gamma'])),manifest['gamma'])
            ax.set_yticks(range(len(manifest['h'])),manifest['h'])
            ax.set(xlabel='Γ',ylabel='h',title=f'{title}, K={K:g}')
            fig.colorbar(im,ax=ax)
    fig.savefig(ROOT/'exact_transfer_maps.png',dpi=160)
    plt.show()
'''), md(r'''## Common energy scaling

For matched disorder, $H(K,\Gamma,h)=K H(1,\Gamma/K,h/K)$, so exact
ground-state observables are invariant at fixed field ratios. Network fidelity
between two commonly scaled Hamiltonians tests learned scale response; its
parameters are unchanged. Only pairs present in the executed grid are compared.
'''), code('''lookup={(r['sample_id'],r['gamma'],r['h'],r['K']):r for r in rows}
scaling=[]
for r in rows:
    if r['K']==1: continue
    other=lookup.get((r['sample_id'],r['gamma']/r['K'],r['h']/r['K'],1.))
    if other is None: continue
    def cache(row):
        name=f"L2_s{row['sample_id']}_g{row['gamma']:g}_h{row['h']:g}_K{row['K']:g}".replace('.','p')
        with np.load(ROOT/'exact'/(name+'.npz')) as d: return np.exp(d['logs']),d['exact'].copy()
    a,ea=cache(r); b,eb=cache(other)
    scaling.append(dict(K=r['K'],h=r['h'],gamma=r['gamma'],sample_id=r['sample_id'],
                        network_scale_fidelity=float((a@b)**2),exact_scale_fidelity=float((ea@eb)**2),
                        energy_scale_residual=abs(r['exact_energy']/r['K']-other['exact_energy'])))
if scaling:
    display(Markdown(f"Common-scale checks: **{len(scaling)}** pairs; minimum exact scale fidelity **{min(r['exact_scale_fidelity'] for r in scaling):.12f}**; maximum energy scaling residual **{max(r['energy_scale_residual'] for r in scaling):.3g}**; mean network scale fidelity **{np.mean([r['network_scale_fidelity'] for r in scaling]):.6f}** (minimum **{min(r['network_scale_fidelity'] for r in scaling):.6f}**)."))
else: display(Markdown('No completed common-scale pairs yet.'))
'''), md('## Exact projection reference\n\nThese curves establish 12-spin time dependence without population or ancestry bias. The pure projected energy and GFMC mixed-energy reference are recorded separately.'), code('''projected=data['projection']
if projected:
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for r in projected:
        t=r['trajectory']; x=[v['Ktau'] for v in t]
        label=f"Γ={r['gamma']:g},h={r['h']:g},K={r['K']:g},s={r['sample_id']}"
        for ax,metric in zip(axes,['fidelity','connected_correlation_rms_error','energy_error_per_spin']):
            y=[1-v[metric] if metric=='fidelity' else v[metric] for v in t]
            ax.semilogy(x,np.maximum(y,1e-15),'.-',label=label)
    for ax,title in zip(axes,['Infidelity','Connected C RMS error','Pure energy error/N']): ax.set(xlabel='Kτ',ylabel=title)
    axes[-1].legend(fontsize=6,bbox_to_anchor=(1.03,1))
    fig.savefig(ROOT/'exact_projection_scan.png',dpi=160)
    plt.show()
else: display(Markdown('Exact projection stage pending.'))
'''), md('## Measured direct-guide cost\n\nEvery rate call evaluates all single-spin flip ratios. Timings include GPU synchronization; these determine feasible larger-size projection budgets.'), code('''cost=data['benchmark']
table=['| L | N | h | States | Sample seconds | All-flip seconds | Mean escape rate |','|---|---:|---:|---:|---:|---:|---:|']
for r in cost:
    table.append(f"| {r['L']} | {3*r['L']**2} | {r['h']:g} | {r['states']} | {r['sample_seconds']:.4f} | {r['flip_ratio_seconds']:.4f} | {r['mean_escape_rate']:.3f} |")
display(Markdown('\\n'.join(table)))
table=['| L | h | Walkers / replicas | Physical time | Wall seconds | Weight ESS |','|---|---:|---|---:|---:|---:|']
for p in sorted(ROOT.glob('direct_step*.json')):
    r=json.loads(p.read_text())
    table.append(f"| {r['L']} | {r['h']:g} | {r['walkers']} / {r['replicas']} | {r['tau']} | {r['seconds']:.3f} | {r['weight_ess']:.2f} |")
display(Markdown('\\n'.join(table)))
'''), md('## Sampled larger-size variational measurements\n\nErrors shown here are Monte Carlo standard errors across fresh direct-sampling batches. They do not include wavefunction bias. Only two disorder realizations enter each point; this is a scan, not a disorder-converged phase boundary.'), code('''sampled=data['sampled']
table=['| L | Points | Mean energy/N | Minimum χ | Maximum χ | Total seconds |','|---|---:|---:|---:|---:|---:|']
for L in (4,6,8):
    group=[r for r in sampled if r['L']==L]
    if group: table.append(f"| {L} | {len(group)} | {np.mean([r['energy']['mean']/(3*L*L) for r in group]):.6f} | {min(r['chi_connected']['mean'] for r in group):.5f} | {max(r['chi_connected']['mean'] for r in group):.5f} | {sum(r['seconds'] for r in group):.1f} |")
display(Markdown('\\n'.join(table)))
if sampled:
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,L in zip(axes,(4,6,8)):
        for h in manifest['h']:
            group=[r for r in sampled if r['L']==L and r['K']==1. and r['h']==h]
            for sid in manifest['sampled_ids']:
                values=sorted([r for r in group if r['sample_id']==sid],key=lambda v:v['gamma'])
                if values: ax.errorbar([v['gamma'] for v in values],[v['chi_connected']['mean'] for v in values],yerr=[v['chi_connected']['se'] for v in values],marker='.',label=f'h={h:g},s={sid}')
        ax.set(xlabel='Γ',ylabel='Variational connected χ',title=f'L={L}, K=1')
    axes[-1].legend(fontsize=6)
    fig.savefig(ROOT/'sampled_variational_scan.png',dpi=160)
    plt.show()
'''), md('## Selected-guide stochastic GFMC\n\nPopulation dependence is compared at fixed physical context. The largest population and forward time also receive an equilibration-time check and a finer branching-cadence check. Low ancestry ESS is reported without dropping cycles.'), code('''gfmc=data['gfmc']
table=['| Γ | h | K | Disorder | W | Kτ forward | Kτ branch | Kτ equil | E−E₀ | χ ± SE | Exact χ | Min ancestry ESS | Largest lineage |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in gfmc:
    e=r['observables']['energy']; c=r['observables']['chi_connected']
    table.append(f"| {r['gamma']:g} | {r['h']:g} | {r['K']:g} | {r['sample_id']} | {r['walkers']} | {r['forward_Ktau']:g} | {r['branch_Ktau']:g} | {r['equilibration_Ktau']:g} | {e['bias']:.5f} | {c['mean']:.5f} ± {c['se']:.5f} | {c['exact']:.5f} | {r['min_descendant_ess']:.2f} | {r['max_largest_lineage']:.3f} |")
display(Markdown('\\n'.join(table)))
if gfmc:
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for col,h in enumerate((0.,.3)):
        for sid in (10000,10001):
            for W in (512,2048,8192):
                values=sorted([r for r in gfmc if (r['gamma'],r['h'],r['K'],r['sample_id'],r['walkers'],r['branch_Ktau'],r['equilibration_Ktau'])==(.5,h,1.,sid,W,.1,16.)],key=lambda v:v['forward_Ktau'])
                if not values: continue
                for ax,metric in zip(axes[:,col],['chi_connected','energy']):
                    ax.errorbar([v['forward_Ktau'] for v in values],[v['observables'][metric]['bias'] for v in values],yerr=[v['observables'][metric]['se'] for v in values],marker='.',label=f's={sid},W={W}')
        for ax,metric in zip(axes[:,col],['χ−χ exact','E−E₀']):
            ax.axhline(0,color='k',lw=.8); ax.set(xlabel='Kτ forward',ylabel=metric,title=f'Γ=0.5,h={h:g},K=1')
    axes[0,-1].legend(fontsize=7)
    fig.savefig(ROOT/'selected_guide_gfmc_convergence.png',dpi=160)
    plt.show()
'''), md('## Direct-network 48-spin GFMC pilot\n\nThese runs use network inference during every continuous-time trajectory, with exact memoization of repeated states. Small walker populations and short projection times are pilot settings. They establish execution and measured cost; convergence remains to be demonstrated.'), code(f'''direct_paths=[Path(p) for p in {[str(p.resolve()) for p in (args.direct_preview, args.direct_pilot) if p is not None]!r}]
table=['| Run | Disorder | h | W | Equilibration | Forward | Branch | E/N ± SE | χ ± SE | Min ancestry ESS | Largest lineage | Seconds |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for directory in direct_paths:
    p=directory/'summary.json'
    if not p.exists():
        display(Markdown(directory.name+': no completed group of independent populations yet.'))
        continue
    d=json.loads(p.read_text())
    assert d['manifest']['checkpoint_sha256']==manifest['checkpoint_sha256']
    display(Markdown(directory.name+f": {{len(d['rows'])}} completed settings; campaign complete={{d['completed']}}."))
    for r in d['rows']:
        e=r['observables']['energy']; c=r['observables']['chi_connected']
        table.append(f"| {{directory.name}} | {{r['sample_id']}} | {{r['h']:g}} | {{r['walkers']}} | {{r['equilibration_time']:g}} | {{r['tau_forward']:g}} | {{r['tau_block']:g}} | {{e['mean']/48:.6f}} ± {{e['se']/48:.6f}} | {{c['mean']:.5f}} ± {{c['se']:.5f}} | {{r['min_descendant_ess']:.2f}} | {{r['max_largest_lineage']:.3f}} | {{r['seconds']:.1f}} |")
display(Markdown('\\n'.join(table)))
'''), md('''## Interpretation limits

The selected guide is the best measured shared candidate under the frozen
development criterion. Low-field fidelity is uneven. Transfer to new field
and bond scales must be read from the scan above. Exact 12-spin projection
can converge while finite-population susceptibility remains biased; energy
agreement alone is insufficient. Ground-state claims at larger sizes require
their own population, projection, forward-time, and cadence checks.
''')]
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    NotebookClient(notebook, timeout=600, kernel_name='python3').execute()
    nbformat.write(notebook, args.notebook)
    print(args.notebook.resolve())


if __name__ == '__main__':
    main()
