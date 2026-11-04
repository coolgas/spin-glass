"""Execute a comparison notebook from actual shared-optimization artifacts."""

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
    continuation = (args.results/'selection_policy.json').exists()
    introduction = r'''# Shared low-field optimization experiments

Every run has one parameter set for $L=2,4,6,8$; all four sizes contribute to
every update. The comparison starts from one frozen corrected ALiBi checkpoint.
Adam and stochastic reconfiguration (SR) see the same anchor schedule and, for
matched seeds, the same disorder sequence. Samples diverge as the guides change.
The continuous-field run uses 50% anchor replay and 50% log-uniform
$\Gamma\in[0.4,2.2]$, uniform $h\in[0,0.4]$. Exact states are evaluation-only.
The completion ablation adds the independent-spin quantum energy response of
future nearest neighbors to each candidate cell. Its gate starts at zero;
future/future interactions still have to be learned by the network.

With $O=\partial_\theta\log\psi$, SR uses the equal-context metric
$S=\frac14\sum_c \mathrm{Cov}_c(O,O)/N_c$ and the physical force
$g=\frac14\sum_c2\mathrm{Cov}_c(O,E_{\rm loc})/N_c$.
It solves $(S+\lambda I)d=g$ in sample space, then applies $\theta\leftarrow\theta-\eta d$.
The effective rate limits the largest sampled RMS change in $\log\psi$ to 0.05.
This is a sampled linear trust limit, not an exact bound on the new-state KL.

Validation IDs 10000–10001 guide development; IDs 40000–40007 remain reserved.
Energy, fidelity, connected susceptibility, and local-energy variance must be
read together: a near-excited eigenstate can have a small variance. Finite-size
spectra distinguish this failure from ordinary broad high-energy contamination.
'''
    if continuation:
        introduction = introduction.replace('The comparison starts from one frozen corrected ALiBi checkpoint.',
            'These continuations start from checkpoints chosen by the recorded development comparison.')
        introduction = introduction.replace('Adam and stochastic reconfiguration (SR) see the same anchor schedule and, for\nmatched seeds, the same disorder sequence. Samples diverge as the guides change.',
            'The selected family uses two independent training seeds; a conservative SR branch and a different candidate family provide further comparisons. Initializations and schedules are listed below.')
        introduction = introduction.replace('The continuous-field run uses 50% anchor replay and 50% log-uniform\n$\\Gamma\\in[0.4,2.2]$, uniform $h\\in[0,0.4]$. Exact states are evaluation-only.',
            'All four continuations use the declared discrete anchor schedule. Exact states are evaluation-only.')
        introduction = introduction.replace('Its gate starts at zero;', 'Its initial gate is inherited from the selected checkpoint;')
        introduction = introduction.replace('Validation IDs 10000–10001 guide development;', 'Development IDs 10000–10007 select these continuations; routine curves use 10000–10001;')
    notebook.cells = [md(introduction), code(f'''from pathlib import Path
import hashlib, json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
ROOT=Path({str(args.results.resolve())!r})
runs={{}}
for path in sorted(ROOT.glob('*/campaign.json')):
    if path.parent.name.startswith('benchmark'): continue
    manifest=json.loads(path.read_text())
    history=json.loads((path.parent/'history.json').read_text()) if (path.parent/'history.json').exists() else []
    status=json.loads((path.parent/'status.json').read_text()) if (path.parent/'status.json').exists() else {{}}
    validations=[dict(json.loads(p.read_text()), step=int(p.stem.split('_')[-1])) for p in sorted(path.parent.glob('validation_*.json'))]
    runs[path.parent.name]=dict(manifest=manifest,history=history,status=status,validations=validations)
assert runs, 'No executed experiments found'
identities={{r['manifest']['initialization_sha256'] for r in runs.values()}}
if not {continuation!r}:
    assert len(identities)==1, 'Controlled comparison requires identical initialization'
    display(Markdown('**Shared initialization SHA-256:** `'+next(iter(identities))+'`'))
else:
    decision=json.loads((ROOT/'selection.json').read_text())
    display(Markdown('**Development selection score:** '+decision['policy']['selection']))
    ranking=['| Candidate | Score | Low-field infidelity | C RMS error | Absolute χ error | High-field retained |','|---|---:|---:|---:|---:|---|']
    for candidate in decision['ranked']:
        ranking.append(f"| {{Path(candidate['checkpoint']).parent.name}} / {{Path(candidate['checkpoint']).name}} | {{candidate['score']:.6f}} | {{candidate['mean_low_field_infidelity']:.6f}} | {{candidate['mean_low_field_C_rms']:.6f}} | {{candidate['mean_low_field_absolute_chi_error']:.6f}} | {{candidate['high_field_retained']}} |")
    display(Markdown('\\n'.join(ranking)))
    final_selection=ROOT/'final_selection.json'
    if final_selection.exists():
        final_decision=json.loads(final_selection.read_text())
        assert not final_decision['reserved_tests_inspected']
        ranking=['| Candidate | Phase | Final development score | Minimum high-field fidelity |','|---|---|---:|---:|']
        for candidate in final_decision['ranked']:
            ranking.append(f"| {{Path(candidate['checkpoint']).parent.name}} / {{Path(candidate['checkpoint']).name}} | {{candidate['phase']}} | {{candidate['score']:.6f}} | {{candidate['lowest_high_field_fidelity']:.6f}} |")
        display(Markdown('**Completed continuation selection**\\n\\n'+'\\n'.join(ranking)))
        display(Markdown(final_decision['candidate_scope']+' '+final_decision['interpretation']))
    table=['| Run | Method | Continuous fields | Completion | Initialization SHA-256 |', '|---|---|---|---|---|']
    for name,r in runs.items():
        m=r['manifest']
        table.append(f"| {{name}} | {{m['method']}} | {{m['continuous']}} | {{m['architecture'].get('completion',False)}} | `{{m['initialization_sha256']}}` |")
    display(Markdown('\\n'.join(table)))
table=['| Run | Status | Updates | Low/high batch | Peak tensor GB | Training seconds |', '|---|---|---:|---:|---:|---:|']
for name,r in runs.items():
    hist=r['history']; m=r['manifest']
    table.append(f"| {{name}} | {{r['status'].get('state','starting')}} | {{hist[-1]['step'] if hist else 0}} / {{m['steps']}} | {{m['batch']}} / {{m['high_batch']}} | {{max([v['peak_gpu_bytes'] for v in hist],default=0)/1e9:.2f}} | {{hist[-1]['seconds'] if hist else 0:.1f}} |")
display(Markdown('\\n'.join(table)))
for name,r in runs.items():
    verification=ROOT/name/'transfer_verification.json'
    if verification.exists():
        identity=json.loads(verification.read_text())
        checkpoint=ROOT/name/identity['checkpoint']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==identity['sha256']
        display(Markdown('**'+name+' verified final checkpoint:** `'+identity['sha256']+'`'))
'''), code('''table=['| Run | Γ | h | Validation error/spin | Fidelity | Lowest fidelity | Mean absolute χ error | Variance/spin |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for name,r in runs.items():
    if not r['validations']: continue
    rows=r['validations'][-1]['rows']
    for gamma,h in ((.5,0.),(.5,.3),(2.,.3)):
        selected=[v for v in rows if v['L']==2 and v['sample_id'] in (10000,10001) and (v['gamma'],v['h'])==(gamma,h)]
        if not selected: continue
        table.append(f"| {name} | {gamma} | {h} | {np.mean([v['energy_error_per_spin'] for v in selected]):.6f} | {np.mean([v['fidelity'] for v in selected]):.5f} | {min(v['fidelity'] for v in selected):.5f} | {np.mean([abs(v['chi_connected']-v['exact_chi']) for v in selected]):.5f} | {np.mean([v['variance_per_spin'] for v in selected]):.6f} |")
display(Markdown('\\n'.join(table)))
'''), code('''table=['| Run | Latest validation | Γ | h | Mean fidelity | Change from step 0 | Mean error/spin | Mean absolute χ error |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,r in runs.items():
    if not r['validations']: continue
    latest=max(r['validations'],key=lambda v:v['step'])
    initial=next((v for v in r['validations'] if v['step']==0),None)
    if initial is None: continue
    for gamma,h in ((.5,0.),(.5,.3),(2.,.3)):
        current=[v for v in latest['rows'] if v['L']==2 and v['sample_id'] in (10000,10001) and (v['gamma'],v['h'])==(gamma,h)]
        start=[v for v in initial['rows'] if v['L']==2 and v['sample_id'] in (10000,10001) and (v['gamma'],v['h'])==(gamma,h)]
        if not current or not start: continue
        fidelity=np.mean([v['fidelity'] for v in current])
        start_fidelity=np.mean([v['fidelity'] for v in start])
        table.append(f"| {name} | {latest['step']} | {gamma} | {h} | {fidelity:.5f} | {fidelity-start_fidelity:+.5f} | {np.mean([v['energy_error_per_spin'] for v in current]):.6f} | {np.mean([abs(v['chi_connected']-v['exact_chi']) for v in current]):.5f} |")
display(Markdown('**Latest measured change from each run relative to its step-0 validation**\\n\\n'+'\\n'.join(table)))
'''), code('''common_steps=set.intersection(*(set(v['step'] for v in r['validations']) for r in runs.values()))
if common_steps:
    matched=max(common_steps)
    table=['| Run | Matched updates | Γ | h | Mean error/spin | Mean fidelity | Change in fidelity | Mean absolute χ error |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in runs.items():
        current=next(v for v in r['validations'] if v['step']==matched)
        initial=next(v for v in r['validations'] if v['step']==0)
        for gamma,h in ((.5,0.),(.5,.3),(2.,.3)):
            selected=[v for v in current['rows'] if v['L']==2 and v['sample_id'] in (10000,10001) and (v['gamma'],v['h'])==(gamma,h)]
            start=[v for v in initial['rows'] if v['L']==2 and v['sample_id'] in (10000,10001) and (v['gamma'],v['h'])==(gamma,h)]
            fidelity=np.mean([v['fidelity'] for v in selected])
            table.append(f"| {name} | {matched} | {gamma} | {h} | {np.mean([v['energy_error_per_spin'] for v in selected]):.6f} | {fidelity:.5f} | {fidelity-np.mean([v['fidelity'] for v in start]):+.5f} | {np.mean([abs(v['chi_connected']-v['exact_chi']) for v in selected]):.5f} |")
    display(Markdown('**Largest validation update shared by every run**\\n\\n'+'\\n'.join(table)))
for horizontal in ('updates','seconds'):
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for column,(gamma,h) in enumerate(((.5,0.),(.5,.3),(2.,.3))):
        for name,r in runs.items():
            seconds={v['step']:v['seconds'] for v in r['history']}
            seconds[0]=0.
            points=[]
            for validation in r['validations']:
                selected=[v for v in validation['rows'] if v['L']==2 and v['sample_id'] in (10000,10001) and (v['gamma'],v['h'])==(gamma,h)]
                step=validation['step']
                if not selected or (horizontal=='seconds' and step not in seconds): continue
                x=step if horizontal=='updates' else seconds[step]/60
                points.append((x,np.mean([v['energy_error_per_spin'] for v in selected]),np.mean([v['fidelity'] for v in selected]),np.mean([abs(v['chi_connected']-v['exact_chi']) for v in selected])))
            if points:
                p=np.asarray(points)
                for row in range(3): axes[row,column].plot(p[:,0],p[:,row+1],'.-',label=name)
        axes[0,column].set_title(f'Γ={gamma}, h={h}')
        for row,label in enumerate(('Energy error / spin','Fidelity','Mean absolute χ error')):
            axes[row,column].set_ylabel(label)
        axes[2,column].set_xlabel('Shared updates' if horizontal=='updates' else 'Elapsed minutes')
        axes[1,column].set_ylim(0,1.01)
    axes[0,0].legend(fontsize=8)
    fig.savefig(ROOT/f'validation_{horizontal}.png',dpi=180)
    fig.savefig(ROOT/f'validation_{horizontal}.pdf')
    plt.show()
display(Markdown('Elapsed time starts after initialization validation and includes preceding in-run validations and resource contention. Curves show measured validation points; no accuracy is inferred between them.'))
'''), code('''table=['| Run | L | Γ | h | Sampled E/spin ± SE | Connected χ ± SE |', '|---|---:|---:|---:|---:|---:|']
for name,r in runs.items():
    if not r['validations']: continue
    for v in r['validations'][-1]['rows']:
        if v['L']==2: continue
        q=v['sampled']; n=3*v['L']**2
        table.append(f"| {name} | {v['L']} | {v['gamma']} | {v['h']} | {q['energy']['mean']/n:.6f} ± {q['energy']['se']/n:.6f} | {q['chi_connected']['mean']:.5f} ± {q['chi_connected']['se']:.5f} |")
display(Markdown('\\n'.join(table)))
display(Markdown('Larger-size values are variational sample estimates; no exact reference or ground-state fidelity is inferred from them. Equal update counts and recorded wall time answer different optimization-efficiency questions.'))
'''), code('''spectra=[('initial',ROOT/'initial_low_field_spectrum.json')]
spectra += [(name,path) for name in runs for path in sorted((ROOT/name).glob('low_field_spectrum_*.json'))]
for name,path in spectra:
    if path.exists():
        diagnostic=json.loads(path.read_text())
        checkpoint=Path(diagnostic['checkpoint']) if name=='initial' else ROOT/name/diagnostic['checkpoint']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==diagnostic['sha256']
        table=['| ID | Γ | h | Accessible gap | Fidelity | First excited weight | First 16 weight | ΔE/gap |','|---:|---:|---:|---:|---:|---:|---:|---:|']
        for v in diagnostic['rows']:
            table.append(f"| {v['sample_id']} | {v['gamma']} | {v['h']} | {v['accessible_gap']:.6f} | {v['ground_fidelity']:.5f} | {v['spectral_weights'][1]:.5f} | {v['first_16_weight']:.5f} | {v['energy_excess_over_gap']:.3f} |")
        display(Markdown('**'+name+' — '+path.name+'**\\n\\n'+'\\n'.join(table)))
'''), md('''## Frozen-guide correlations and field response

The following uses full 12-spin enumeration on the two development disorders.
RMS and maximum correlation errors exclude the matrix diagonal. The field scan
uses h=0.15, outside the discrete anchors, and does not adapt the guide.
'''), code('''comparisons=[]
for filename in ('initial_correlation_comparison.json','correlation_comparison.json'):
    path=ROOT/filename
    if path.exists():
        diagnostic=json.loads(path.read_text())
        for run in diagnostic['runs']:
            checkpoint=Path(run['checkpoint'])
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==run['sha256']
            comparisons.append(dict(run,label='initial' if filename.startswith('initial') else checkpoint.parent.name))
if comparisons:
    table=['| Checkpoint | ID | Γ | h | Fidelity | RMS connected-C error | Max connected-C error | Absolute Δχ |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in comparisons:
        for v in run['rows']:
            table.append(f"| {run['label']} step{run['step']} | {v['sample_id']} | {v['gamma']} | {v['h']} | {v['fidelity']:.5f} | {v['connected_correlation_rms_error']:.5f} | {v['connected_correlation_max_error']:.5f} | {abs(v['chi_connected']-v['exact_chi']):.5f} |")
    display(Markdown('\\n'.join(table)))
    # The lowest-field biased context exposes competing patterns directly.
    selected=[(run,[v for v in run['rows'] if (v['sample_id'],v['gamma'],v['h'])==(10000,.5,.3)]) for run in comparisons]
    selected=[(run,rows[0]) for run,rows in selected if rows]
    if selected:
        fig,axes=plt.subplots(1,len(selected),figsize=(4*len(selected),4),squeeze=False,constrained_layout=True)
        for ax,(run,v) in zip(axes[0],selected):
            error=np.asarray(v['connected_correlation'])-np.asarray(v['exact_connected_correlation'])
            image=ax.imshow(error,cmap='RdBu_r',vmin=-.5,vmax=.5)
            ax.set_title(run['label']+f" step{run['step']}")
        fig.colorbar(image,ax=list(axes[0]),label='Connected C error')
        fig.savefig(ROOT/'connected_correlation_errors.png',dpi=180)
        fig.savefig(ROOT/'connected_correlation_errors.pdf')
        plt.show()
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for run in comparisons:
        gammas=sorted({v['gamma'] for v in run['rows'] if v['h']==.15})
        if not gammas: continue
        selected=[[v for v in run['rows'] if (v['gamma'],v['h'])==(gamma,.15)] for gamma in gammas]
        axes[0].plot(gammas,[np.mean([v['fidelity'] for v in group]) for group in selected],'.-',label=run['label'])
        axes[1].plot(gammas,[np.mean([v['connected_correlation_rms_error'] for v in group]) for group in selected],'.-',label=run['label'])
    axes[0].set(ylabel='Mean fidelity',ylim=(0,1.01))
    axes[1].set_ylabel('Mean off-diagonal connected-C RMS error')
    for ax in axes:
        ax.set_xlabel('Γ at h=0.15')
        if ax.lines: ax.legend(fontsize=8)
    fig.savefig(ROOT/'field_response.png',dpi=180)
    fig.savefig(ROOT/'field_response.pdf')
    plt.show()
'''), md('''## Exact imaginary-time projection reference

Each curve applies normalized exp[−τ(H−E₀)]ψ to all 4096 configurations. It
measures how long the frozen guide takes to approach the exact finite-size
ground state. It provides a target for stochastic projection, whose walker
population, branching interval, and forward-time errors still need convergence
studies. These curves establish no accuracy for systems larger than 12 spins.
'''), code('''projections=[]
paths=[ROOT/'initial_exact_projection.json']+sorted(ROOT.glob('*/exact_projection_final.json'))
for path in paths:
    if not path.exists(): continue
    diagnostic=json.loads(path.read_text())
    assert hashlib.sha256(Path(diagnostic['checkpoint']).read_bytes()).hexdigest()==diagnostic['sha256']
    projections.append(dict(diagnostic,label='initial' if path.name.startswith('initial') else path.parent.name))
if projections:
    table=['| Guide | ID | Γ | h | Initial F | Final τ | Final F | Final C RMS error | First grid τ with F≥0.99 | First grid τ with F≥0.999 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in projections:
        for v in run['rows']:
            trajectory=v['trajectory']; first,last=trajectory[0],trajectory[-1]
            reached=[next((str(q['tau']) for q in trajectory if q['fidelity']>=threshold),'not reached') for threshold in (.99,.999)]
            table.append(f"| {run['label']} | {v['sample_id']} | {v['gamma']} | {v['h']} | {first['fidelity']:.6f} | {last['tau']} | {last['fidelity']:.8f} | {last['connected_correlation_rms_error']:.6g} | {reached[0]} | {reached[1]} |")
    display(Markdown('\\n'.join(table)))
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for column,(gamma,h) in enumerate(((.5,0.),(.5,.3),(2.,.3))):
        for run in projections:
            selected=[v['trajectory'] for v in run['rows'] if (v['gamma'],v['h'])==(gamma,h)]
            times=[q['tau'] for q in selected[0]]
            means=np.mean([[[max(0.,1-q['fidelity']),q['connected_correlation_rms_error'],abs(q['chi_connected']-q['exact_chi'])] for q in trajectory] for trajectory in selected],axis=0)
            for row in range(3): axes[row,column].semilogy(times,np.maximum(means[:,row],1e-15),'.-',label=run['label'])
        axes[0,column].set_title(f'Γ={gamma}, h={h}')
        for row,label in enumerate(('Mean infidelity','Mean connected-C RMS error','Mean absolute χ error')): axes[row,column].set_ylabel(label)
        axes[2,column].set_xlabel('Imaginary time τ')
    axes[0,0].legend(fontsize=8)
    fig.savefig(ROOT/'exact_projection.png',dpi=180)
    fig.savefig(ROOT/'exact_projection.pdf')
    plt.show()
'''), md('''## Stochastic projector convergence study

The frozen network is cached exactly on all 12-spin configurations for this
CPU study. Independent four-replica walker populations estimate connected
susceptibility. Error bars are across independent populations; every forward
cycle is included regardless of ancestry quality. A small ancestry ESS flags
unreliable forward walking and is not a reason to discard a result.
'''), code('''for path in sorted(ROOT.glob('projector_*/summary.json')):
    study=json.loads(path.read_text())
    base_settings=len(study['manifest']['walkers'])*len(study['manifest']['forward'])
    extra_cadence=max(0, len(study['manifest']['blocks'])-1)
    expected=len(study['manifest']['ids'])*2*(base_settings+extra_cadence)*study['manifest']['seeds']
    available=len(list(path.parent.glob('s*_seed*.json')))
    display(Markdown(f"**{path.parent.name}** — completed: {study['completed']}; summarized settings: {len(study['rows'])}; independent population records: {available}/{expected}."))
    if not study['completed']:
        display(Markdown('Partial snapshot: the resumable study is still running; missing settings are not interpreted as zero or omitted data.'))
    table=['| ID | h | Walkers | Forward τ | Branch Δτ | χ ± population SE | Exact χ | E error/spin ± SE | Min ancestry ESS | Largest lineage |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in study['rows']:
        c=v['observables']['chi_connected']; e=v['observables']['energy']
        table.append(f"| {v['sample_id']} | {v['h']} | {v['walkers']} | {v['tau_forward']} | {v['tau_block']} | {c['mean']:.5f} ± {c['se']:.5f} | {c['exact']:.5f} | {e['bias']/12:.6f} ± {e['se']/12:.6f} | {v['min_descendant_ess']:.2f} | {v['max_largest_lineage']:.3f} |")
    display(Markdown('\\n'.join(table)))
    focus=[v for v in study['rows'] if v['walkers']==max(study['manifest']['walkers']) and v['tau_forward']==max(study['manifest']['forward'])]
    if focus:
        table=['| ID | h | Branch Δτ | χ estimate ± population SE | Exact χ | Difference / SE | Min descendant ESS | Largest lineage | Healthy populations |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for v in focus:
            q=v['observables']['chi_connected']; z=(q['mean']-q['exact'])/q['se']
            table.append(f"| {v['sample_id']} | {v['h']} | {v['tau_block']} | {q['mean']:.5f} ± {q['se']:.5f} | {q['exact']:.5f} | {z:+.2f} | {v['min_descendant_ess']:.1f} | {v['max_largest_lineage']:.3f} | {v['healthy_populations']}/{v['independent_populations']} |")
        display(Markdown('**Longest forward-time check at the largest population**\\n\\n'+ '\\n'.join(table)))
    if (10000,.3) in {(v['sample_id'],v['h']) for v in study['rows']}:
        population=[v for v in study['rows'] if (v['sample_id'],v['h'],v['tau_forward'],v['tau_block'])==(10000,.3,max(study['manifest']['forward'],key=float),study['manifest']['blocks'][0])]
        if population:
            table=['| Walkers | χ estimate ± population SE | Exact χ | Min descendant ESS | Largest lineage |','|---:|---:|---:|---:|---:|']
            for v in sorted(population,key=lambda x:x['walkers']):
                q=v['observables']['chi_connected']
                table.append(f"| {v['walkers']} | {q['mean']:.5f} ± {q['se']:.5f} | {q['exact']:.5f} | {v['min_descendant_ess']:.1f} | {v['max_largest_lineage']:.3f} |")
            display(Markdown('**Population dependence at ID10000, h=0.3, longest forward time**\\n\\n'+'\\n'.join(table)))
        display(Markdown('At these 12-spin settings, the projected energy estimates can be close to the exact energy while connected susceptibility remains sensitive to disorder, forward time, walker population and branching cadence. Ancestry diagnostics and independent-population errors are reported together; agreement of energy alone does not establish observable convergence.'))
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,(sample_id,h) in zip(axes.flat,((10000,0.),(10000,.3),(10001,0.),(10001,.3))):
        rows=[v for v in study['rows'] if (v['sample_id'],v['h'])==(sample_id,h)]
        for block in sorted({v['tau_block'] for v in rows}):
            for walkers in sorted({v['walkers'] for v in rows}):
                selected=sorted([v for v in rows if (v['walkers'],v['tau_block'])==(walkers,block)],key=lambda v:v['tau_forward'])
                if not selected: continue
                ax.errorbar([v['tau_forward'] for v in selected],[v['observables']['chi_connected']['mean'] for v in selected],yerr=[v['observables']['chi_connected']['se'] for v in selected],fmt='.-',label=f'W={walkers}, Δτ={block}')
        if rows: ax.axhline(rows[0]['observables']['chi_connected']['exact'],color='black',ls='--',label='Exact')
        ax.set(title=f'ID{sample_id}, h={h}',xlabel='Forward time',ylabel='Connected χ')
        if ax.lines: ax.legend(fontsize=7)
    fig.savefig(path.parent/'convergence.png',dpi=180)
    fig.savefig(path.parent/'convergence.pdf')
    plt.show()
'''), md('''These are controlled optimization experiments, not a qualification for finite-size scaling. A useful remedy must improve unseen-disorder observables and fidelity consistently, retain its high-field accuracy, and survive independent seeds. The reserved disorders should be inspected only after choosing the continuation from development results.''')]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'remedies_cell_{index}', 'exec')
    root = Path(__file__).resolve().parents[2]
    NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata':{'path':str(root)}}).execute()
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.notebook)
    print('Executed', args.notebook)


if __name__ == '__main__':
    main()
