"""Executed shared-symmetry campaign report, suitable for pilot and final runs."""

import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--notebook',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    md=nbformat.v4.new_markdown_cell
    code=nbformat.v4.new_code_cell
    n=nbformat.v4.new_notebook()
    n.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    n.cells=[md(r'''# Corrected shared ALiBi: measured progress and qualification

One parameter set serves 12, 48, 108, and 192 spins. The guide fixes a
bond-defined canonical cell origin (averaging ties), guaranteeing consistency
under joint translations of spins and bonds. At h=0 it mixes probabilities of
s and −s, with consistent sampling and amplitude ratios. The base ALiBi decoder
also receives bond-weighted fields from earlier cells and a local-energy logit
prior. These inputs are causal. This model is not claimed to enforce rotations
or reflections, and canonicalization is discontinuous in bond inputs.

Training uses sampled physical VMC, all four sizes before each optimizer step,
258 disorder IDs per size, and three field pairs. No exact ground-state labels
enter training. Full enumeration at 12 spins supplies deterministic diagnostics;
larger-size measurements have sampling uncertainty and do not establish exact
ground-state accuracy.

This report distinguishes fixed validation curves from noisy changing-context
training energies. IDs 0 and 1 are training samples; 2–9 are retrospective
held-out samples; 10000–10001 are validation. The manifest records the test IDs
(20000–20007 for the pilot, 30000–30007 for the first main run). Once used to guide another experiment, inspected
test samples become development data and require a new confirmatory test set.
'''),code(f'''from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
RESULTS=Path({str(args.results.resolve())!r})
status=json.loads((RESULTS/'status.json').read_text())
manifest=json.loads((RESULTS/'campaign.json').read_text())
display(Markdown('**Recorded status:** `'+str(status)+'`'))
display(Markdown(f"**Training schedule:** {{manifest['steps']}} updates, batch {{manifest['batch']}} per size; seed {{manifest['seed']}}."))
display(Markdown('**Reserved final-test IDs:** `'+str(manifest['final_test_ids'])+'`'))
validations=[json.loads(p.read_text()) for p in sorted(RESULTS.glob('validation_*.json'))]
evaluation=json.loads((RESULTS/'evaluation.json').read_text()) if (RESULTS/'evaluation.json').exists() else None
rows=evaluation['rows'] if evaluation else (validations[-1]['rows'] if validations else [])
if evaluation:
    import hashlib
    checkpoint=RESULTS/evaluation['checkpoint']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==evaluation['sha256']
    display(Markdown('**Verified final checkpoint SHA-256:** `'+evaluation['sha256']+'`'))
'''),code('''table=['| L2 disorder | Γ | h | Energy error/spin | Fidelity | Connected χ | Exact χ | RMS local m |','|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    if r['L']==2:
        table.append(f"| {r['sample_id']} | {r['gamma']} | {r['h']} | {r['energy_error_per_spin']:.6g} | {r['fidelity']:.6f} | {r['chi_connected']:.6f} | {r['exact_chi']:.6f} | {r['rms_site_magnetization']:.3g} |")
display(Markdown('\\n'.join(table)))
'''),code('''groups=[('Training examples', [0,1]),
        ('Retrospective held-out', list(range(2,10))),
        ('Validation', manifest['validation_ids']),
        ('Reserved test', manifest['final_test_ids'])]
table=['| Split | Γ | h | Count | Mean error/spin | Worst error/spin | Mean fidelity | Lowest fidelity | Mean absolute χ error |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,ids in groups:
    for gamma,h in manifest['fields']:
        selected=[r for r in rows if r['L']==2 and r['sample_id'] in ids and (r['gamma'],r['h'])==(gamma,h)]
        if not selected: continue
        errors=[r['energy_error_per_spin'] for r in selected]
        fidelities=[r['fidelity'] for r in selected]
        chi_errors=[abs(r['chi_connected']-r['exact_chi']) for r in selected]
        table.append(f'| {name} | {gamma} | {h} | {len(selected)} | {np.mean(errors):.6f} | {max(errors):.6f} | {np.mean(fidelities):.6f} | {min(fidelities):.6f} | {np.mean(chi_errors):.6f} |')
display(Markdown('\\n'.join(table)))
display(Markdown('The two listed training examples do not measure accuracy across all 258 training disorders. Split averages summarize these recorded samples; they are not population guarantees.'))
'''),code('''fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
for gamma,h in ((.5,0.),(.5,.3),(2.,.3)):
    points=[]
    for v in validations:
        selected=[r for r in v['rows'] if r['L']==2 and r['sample_id']>=10000 and (r['gamma'],r['h'])==(gamma,h)]
        if selected:
            step=int(v['checkpoint'].split('step')[1].split('.')[0])
            points.append((step,np.mean([r['energy_error_per_spin'] for r in selected]),np.mean([r['fidelity'] for r in selected]),np.mean([abs(r['chi_connected']-r['exact_chi']) for r in selected])))
    if points:
        x,e,f,c=np.array(points).T
        axes[0].plot(x,e,'o-',label=f'Γ={gamma}, h={h}')
        axes[1].plot(x,f,'o-',label=f'Γ={gamma}, h={h}')
        axes[2].plot(x,c,'o-',label=f'Γ={gamma}, h={h}')
axes[0].set(xlabel='Shared updates',ylabel='Validation mean energy error/spin')
axes[1].set(xlabel='Shared updates',ylabel='Validation mean fidelity',ylim=(0,1.01))
axes[2].set(xlabel='Shared updates',ylabel='Validation mean absolute χ error')
for ax in axes:
    if ax.lines: ax.legend()
fig.savefig(RESULTS/'validation.png',dpi=180)
fig.savefig(RESULTS/'validation.pdf')
plt.show()
'''),code('''table=['| Spins | Disorder | Γ | h | Energy/spin ± SE | Variance/spin | Connected χ ± SE |','|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    if r['L']>2:
        s=r['sampled'];N=3*r['L']**2
        table.append(f"| {N} | {r['sample_id']} | {r['gamma']} | {r['h']} | {s['energy']['mean']/N:.6f} ± {s['energy']['se']/N:.6f} | {s['local_energy_variance']/N:.6g} | {s['chi_connected']['mean']:.6f} ± {s['chi_connected']['se']:.6f} |")
display(Markdown('\\n'.join(table)))
'''),code('''table=['| Spins | Disorder | Γ | χ − Nq² at h=0 | Paired batch SE |',
       '|---:|---:|---:|---:|---:|']
for r in rows:
    if r['L']>2 and r['h']==0:
        s=r['sampled'];N=3*r['L']**2
        differences=np.array(s['chi_connected']['batches'])-N*np.array(s['q2']['batches'])
        se=differences.std(ddof=1)/np.sqrt(len(differences))
        table.append(f"| {N} | {r['sample_id']} | {r['gamma']} | {differences.mean():.6f} | {se:.6f} |")
display(Markdown('For an exactly spin-flip symmetric guide, χ = Nq². The independent-replica estimators fluctuate; the paired batch uncertainty below includes their covariance.'))
display(Markdown('\\n'.join(table)))
auditpath=RESULTS/'symmetry_audit.json'
if auditpath.exists() and evaluation:
    audit=json.loads(auditpath.read_text())
    assert audit['sha256']==evaluation['sha256'] and audit['passed']
    largest=max(r['max_translation_logp_error'] for r in audit['rows'])
    display(Markdown(f'**Final checkpoint symmetry audit passed:** largest translation log-probability difference {largest:.3g}; precision {audit["precision"]}. L2 is enumerated; larger-size checks use sampled configurations.'))
'''),md(r'''## Low-field physics: spectral diagnosis

For a normalized guide, write ψ = Σₙ cₙ|n⟩. Its energy excess is
E−E₀ = Σₙ>₀ |cₙ|²(Eₙ−E₀), whereas fidelity is |c₀|². Thus small gaps
make energy a weak discriminator of the state and its correlations. The bound
1−F ≤ (E−E₀)/Δ uses total energy excess, not error per spin. At h=0 this
guide is exactly even, so the relevant Δ is the next even-sector gap; the
lower odd-sector level cannot carry guide weight.

The following frozen-checkpoint diagnostics use exact 12-spin spectra. They
identify finite-size competition between low-energy states; they do not establish
a thermodynamic phase or an exponentially small tunnelling gap. Spectral weight
is a wavefunction overlap, not a probability of occupying a classical basin.
'''),code('''spectral_paths=sorted(RESULTS.glob('low_field_spectrum_*.json'))
for spectral_path in spectral_paths:
    diagnostic=json.loads(spectral_path.read_text())
    import hashlib
    frozen=RESULTS/diagnostic['checkpoint']
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==diagnostic['sha256']
    display(Markdown(f"**Spectral audit: update {diagnostic['step']}**, checkpoint `{diagnostic['checkpoint']}`."))
    table=['| Disorder | Γ | h | Accessible gap | Total energy excess / gap | Ground weight | First excited weight | First 16 weight |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in diagnostic['rows']:
        table.append(f"| {r['sample_id']} | {r['gamma']} | {r['h']} | {r['accessible_gap']:.6f} | {r['energy_excess_over_gap']:.3f} | {r['ground_fidelity']:.6f} | {r['spectral_weights'][1]:.6f} | {r['first_16_weight']:.6f} |")
    display(Markdown('\\n'.join(table)))
if not spectral_paths:
    display(Markdown('No spectral audit has been recorded for this campaign.'))
'''),code('''path=RESULTS/'correlation_comparison.json'
if path.exists():
    diagnostic=json.loads(path.read_text())
    table=['| ID | Γ | h | Fidelity | RMS connected-C error | Max connected-C error | Absolute Δχ |','|---:|---:|---:|---:|---:|---:|---:|']
    for run in diagnostic['runs']:
        assert hashlib.sha256(Path(run['checkpoint']).read_bytes()).hexdigest()==run['sha256']
        for v in run['rows']:
            table.append(f"| {v['sample_id']} | {v['gamma']} | {v['h']} | {v['fidelity']:.6f} | {v['connected_correlation_rms_error']:.6f} | {v['connected_correlation_max_error']:.6f} | {abs(v['chi_connected']-v['exact_chi']):.6f} |")
    display(Markdown('**Final frozen-guide correlations and field response (L2)**\\n\\n'+'\\n'.join(table)))
path=RESULTS/'exact_projection_final.json'
if path.exists():
    diagnostic=json.loads(path.read_text())
    assert hashlib.sha256(Path(diagnostic['checkpoint']).read_bytes()).hexdigest()==diagnostic['sha256']
    table=['| ID | Γ | h | Initial F | Final τ | Final F | Final C RMS error |','|---:|---:|---:|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for v in diagnostic['rows']:
        first,last=v['trajectory'][0],v['trajectory'][-1]
        table.append(f"| {v['sample_id']} | {v['gamma']} | {v['h']} | {first['fidelity']:.6f} | {last['tau']} | {last['fidelity']:.8f} | {last['connected_correlation_rms_error']:.6g} |")
        label=f"ID{v['sample_id']}, Γ={v['gamma']}, h={v['h']}"
        times=[q['tau'] for q in v['trajectory']]
        axes[0].semilogy(times,[max(1e-15,1-q['fidelity']) for q in v['trajectory']],'.-',label=label)
        axes[1].semilogy(times,[max(1e-15,q['connected_correlation_rms_error']) for q in v['trajectory']],'.-',label=label)
    display(Markdown('**Exact 12-spin projection reference**\\n\\n'+'\\n'.join(table)))
    display(Markdown('Normalized exp[−τ(H−E₀)]ψ on all 4096 configurations gives a finite-size imaginary-time target. Walker population, branching and forward-time convergence remain to be established independently.'))
    axes[0].set_ylabel('Infidelity'); axes[1].set_ylabel('Connected-C RMS error')
    for ax in axes: ax.set_xlabel('Imaginary time τ')
    axes[0].legend(fontsize=7)
    fig.savefig(RESULTS/'exact_projection.png',dpi=180)
    fig.savefig(RESULTS/'exact_projection.pdf')
    plt.show()
'''),md('''## Comparison and decision

Symmetry enforcement is a mathematical property, not an accuracy certificate.
At zero field, connected χ must equal Nq² because local magnetizations vanish;
the value can still be inaccurate if even-sector correlations are wrong.
Compare held-out energy errors and fidelities to the previous checkpoint,
retain all training outcomes, and inspect low-field versus high-field behavior.
All larger-size results use identical frozen weights; no specialist adaptation
occurs during evaluation.
'''),code('''ROOT=Path.cwd()
if not (ROOT/'code/transformer_qmc').exists(): ROOT=ROOT.parent
oldpath=ROOT/'data/kagome/transformer_af_shared_accuracy_v1/symmetry_diagnostics.json'
if oldpath.exists() and rows:
    old=json.loads(oldpath.read_text())
    pairs=[]
    for r in rows:
        if r['L']==2 and r['sample_id'] in range(2,10) and (r['gamma'],r['h'])==(.5,.3):
            prior=next(x for x in old['rows'] if (x['sample_id'],x['gamma'],x['h'])==(r['sample_id'],r['gamma'],r['h']))['states']['original']
            pairs.append((r['sample_id'],prior['energy_error_per_spin'],r['energy_error_per_spin'],prior['fidelity'],r['fidelity']))
    table=['| Held-out disorder | Previous error/spin | New error/spin | Previous fidelity | New fidelity |','|---:|---:|---:|---:|---:|']
    for s,a,b,c,d in pairs:
        table.append(f'| {s} | {a:.6f} | {b:.6f} | {c:.6f} | {d:.6f} |')
    display(Markdown('\\n'.join(table)))
    if pairs:
        values=np.array(pairs)
        display(Markdown(f"**Mean retrospective held-out error/spin:** {values[:,1].mean():.6f} → {values[:,2].mean():.6f}."))
''')]
    for index,cell in enumerate(n.cells):
        if cell.cell_type=='code':
            compile(cell.source,f'generated report cell {index}','exec')
    NotebookClient(n,timeout=120,kernel_name='python3').execute(cwd=str(root))
    nbformat.validate(n)
    nbformat.write(n,args.notebook)
    print(args.notebook)


if __name__=='__main__':
    main()
