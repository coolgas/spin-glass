"""Compare weakened-coupling RG with the unchanged full L2 target on AutoDL."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from kagome_qea.exact import operator_matrix
from .triangle_effective_benchmark import digest, lift, write_json, xor_mass
from .triangle_effective_interaction_scan import GAMMAS, exact_small, physical_hamiltonian
from .triangle_effective_rg import triangles


LAMBDAS = (0., .1, .3, .5, .7, .85, 1.)
DISPLAY_LAMBDAS = (.3, .5, .7, .85, 1.)
COLORS = ('#0072B2', '#009E73', '#E69F00', '#CC79A7', '#D55E00')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.results, args.output
    (out/'exact').mkdir(parents=True, exist_ok=True)
    # Existing scientific archives are immutable; new contexts live here.
    for gamma in GAMMAS:
        for lam in LAMBDAS:
            key = f'l2_g{gamma:g}_l{lam:g}'
            source = root/'exact'/(key+'.json')
            destination = out/'exact'/source.name
            if source.exists() and not destination.exists():
                row = json.loads(source.read_text())
                assert digest(source.with_suffix('.npz')) == row['raw_sha256']
                shutil.copy2(source.with_suffix('.npz'), destination.with_suffix('.npz'))
                shutil.copy2(source, destination)
            exact_small(out, gamma, lam)
            print(f'Saved {key}', flush=True)

    q = np.linspace(-1, 1, 13)
    masses, records, references, hashes = {}, [], [], {}
    for gamma in GAMMAS:
        _, original, target_ham, target_mask = physical_hamiltonian(2, 1.)
        target_matrix = operator_matrix(target_ham, gamma)
        target_key = f'l2_g{gamma:g}_l1'
        with np.load(out/'exact'/(target_key+'.npz')) as data:
            target_state = data['exact_state'].copy()
            target_mass = data['mass'].copy()
        target_energy = float(target_state@(target_matrix@target_state))
        target_residual = float(np.linalg.norm(target_matrix@target_state-target_energy*target_state))
        assert target_residual < 1e-8
        np.testing.assert_allclose(target_mass, xor_mass(target_state**2, 12), atol=1e-12)
        masses[gamma, 'target', 0] = target_mass
        references.append(dict(gamma=gamma, lambda_reference=1., energy=target_energy,
            residual=target_residual, q=float(target_mass@q), q2=float(target_mass@(q*q))))
        first = triangles(original, gamma, 8)
        for lam in LAMBDAS:
            key = f'l2_g{gamma:g}_l{lam:g}'
            path = out/'exact'/(key+'.json')
            row = json.loads(path.read_text())
            assert digest(path.with_suffix('.npz')) == row['raw_sha256']
            hashes[key] = dict(json=digest(path), npz=digest(path.with_suffix('.npz')))
            _, _, modified_ham, mask = physical_hamiltonian(2, lam)
            modified_matrix = operator_matrix(modified_ham, gamma)
            with np.load(path.with_suffix('.npz')) as data:
                np.testing.assert_array_equal(data['bonds'], original.bonds)
                np.testing.assert_array_equal(data['original_J'], original.J)
                np.testing.assert_array_equal(data['mask'], target_mask)
                np.testing.assert_allclose(data['J'], original.J*np.where(mask, lam, 1.), atol=0, rtol=0)
                modified_mass = data['mass'].copy()
                np.testing.assert_allclose(modified_mass,
                    xor_mass(data['exact_state']**2, 12), atol=1e-12)
                masses[gamma, lam, 'modified'] = modified_mass
                for m in (4, 6, 8):
                    psi = lift(dict(sites=first['sites'], bases=data[f'bases_m{m}']),
                               data[f'state_m{m}'], m)
                    np.testing.assert_allclose(psi@psi, 1, atol=1e-11)
                    mass = data[f'mass_m{m}'].copy()
                    np.testing.assert_allclose(mass, xor_mass(psi**2, 12), atol=1e-11)
                    mass_error = float(abs(mass-target_mass).sum()/2)
                    truncation = float(abs(mass-modified_mass).sum()/2)
                    model_bias = float(abs(modified_mass-target_mass).sum()/2)
                    assert abs(model_bias-truncation) <= mass_error+1e-10
                    assert mass_error <= model_bias+truncation+1e-10
                    if m == 8:
                        assert truncation < 1e-7
                        if lam == 1: assert mass_error < 1e-7
                    target_expectation = float(psi@(target_matrix@psi))
                    target_energy_excess = (target_expectation-target_energy)/12
                    assert target_energy_excess > -1e-10
                    # Energies measured under H(lambda=1) remain variational for the target.
                    record = dict(gamma=gamma, lambda_rg=lam, lambda_reference=1., m=m,
                        target_tv=mass_error, modified_model_truncation_tv=truncation,
                        model_change_tv=model_bias, q=float(mass@q), q2=float(mass@(q*q)),
                        target_q_error=float(abs((mass-target_mass)@q)),
                        target_q2_error=float(abs((mass-target_mass)@(q*q))),
                        target_energy_excess_per_spin=target_energy_excess,
                        modified_energy=float(psi@(modified_matrix@psi)))
                    records.append(record)
                    masses[gamma, lam, m] = mass
    lookup = {(r['gamma'], r['lambda_rg'], r['m']): r for r in records}
    shifts = []
    for lam in LAMBDAS:
        for a, b in zip(GAMMAS, GAMMAS[1:]):
            exact_delta = masses[b, 'target', 0]-masses[a, 'target', 0]
            rg_delta = masses[b, lam, 6]-masses[a, lam, 6]
            change_error = float(abs(rg_delta-exact_delta).sum()/2)
            bound = lookup[a, lam, 6]['target_tv']+lookup[b, lam, 6]['target_tv']
            assert change_error <= bound+1e-10
            shifts.append(dict(lambda_rg=lam, gamma_from=a, gamma_to=b,
                exact_target_field_tv=float(abs(exact_delta).sum()/2),
                rg_field_tv=float(abs(rg_delta).sum()/2), field_change_error=change_error,
                endpoint_error_sum=bound))

    figures = []

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'{name}.{ext}', dpi=150)
        plt.close(fig)
        figures.append(name+'.png')
        return name+'.png'

    def image_cell(text, name):
        return nbformat.v4.new_markdown_cell(text+f'\n\n![{name}](attachment:{name})',
            attachments={name: {'image/png': base64.b64encode((out/name).read_bytes()).decode()}})

    text = r'''# Can weakened-intertriangle RG reproduce the original full L2 model?

**The exact reference always has lambda=1.** At each Gamma, compare that fixed
12-spin target with RG calculated using weaker connecting bonds, including
lambda_RG=0.7. The question is how far those bonds can be weakened while still
representing the **original model** and its field response.

Draw each positive antiferromagnetic bond once from
$K^{(0)}_{ij}\sim U(0.5,1.5)$ (disorder60000). A denotes the internal edges of
disjoint orientation0 triangles; B denotes edges connecting different triangles.
There are 12 A and 12 B bonds, with two of each touching every spin.

$$H_{\rm target}=\sum_A K^{(0)}_{ij}Z_iZ_j+\sum_B K^{(0)}_{ij}Z_iZ_j
-h\sum_i Z_i-\Gamma\sum_i X_i,\qquad h=0.15.$$

For RG only, replace the connecting term by
$\lambda_{\rm RG}\sum_B K^{(0)}_{ij}Z_iZ_j$, then project into m states per
triangle. The same original bond arrays, fields and geometry are used in every
comparison. Stored couplings use J=-K. Fields are in units of mean internal K=1.

| Calculation | Internal K | Connecting K |
| :--- | :--- | :--- |
| Exact target, always lambda=1 | U(0.5, 1.5) | U(0.5, 1.5) |
| RG lambda=0.3 | same original draws | U(0.15, 0.45) |
| RG lambda=0.5 | same original draws | U(0.25, 0.75) |
| RG lambda=0.7 | same original draws | U(0.35, 1.05) |
| RG lambda=0.85 | same original draws | U(0.425, 1.275) |
| RG lambda=1 | same original draws | original U(0.5, 1.5) |

The main distribution figure uses m=6: four triangle sites, solved dimension
$6^4=1296$, versus $2^{12}=4096$ for the full target. The error figure also
shows m=4 (dimension 256) and m=8 (dimension 4096). L2 uses one triangle
projection; no superblock chi truncation is used.

All plotted distributions refer to **physical spins**, after lifting the RG
wavefunction back into the 12-spin basis:

$$q=\frac1{12}\sum_i z_i^{(a)}z_i^{(b)},\qquad
P(q)=\sum_{z,z'}|\psi(z)|^2|\psi(z')|^2
\mathbf1\!\left[q=\frac1{12}\sum_i z_i z'_i\right].$$

Two replicas are independent. The 13 discrete bins have spacing 1/6; the
vertical axis is probability mass per bin. All probabilities are summed
deterministically from numerical ground states, without replica sampling noise.
At h=0.15 local polarization contributes to the overlap; a shift alone does
not identify a glass transition. Results below describe one disorder sample.
'''
    cells = [nbformat.v4.new_markdown_cell(text)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True,
                             layout='constrained')
    for ax, gamma in zip(axes.flat, GAMMAS):
        ax.plot(q, masses[gamma, 'target', 0], '--', color='black', lw=1.8,
                label='Exact full target (lambda=1), dashed', zorder=10,
                dash_capstyle='butt')
        for lam, color in zip(DISPLAY_LAMBDAS, COLORS):
            ax.plot(q, masses[gamma, lam, 6], '-', marker='.', color=color,
                    ms=5, lw=1.8, label=f'RG m=6, lambda={lam:g}, solid')
        ax.set(title=f'Gamma={gamma:g}', xlabel='Physical overlap q',
               ylabel='Probability mass per bin', xlim=(-1.02, 1.02), ylim=(0, None))
        ax.grid(alpha=.2)
    fig.suptitle('Unchanged exact L2 target versus weakened-intertriangle RG | h=0.15')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=3)
    name = save(fig, 'fixed_target_pq')
    cells.append(image_cell('## Compare with the same full model at every Gamma\n\n'
        '**Black dashed: exact original full model, lambda=1. Colored solid: RG at '
        'the lambda values listed in the legend.** Each panel fixes Gamma and contains '
        'all five RG multipliers. All RG curves use m=6. Compare colors within '
        'each panel to see the cost of weakening those bonds, and compare panels '
        'to follow the field response. Where exact and RG lambda=1 agree, the '
        'colored solid curve remains visible through the gaps in the black dashed '
        'reference. Per-curve errors are in the table below. All panels share axes.', name))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True,
                             layout='constrained')
    for ax, gamma in zip(axes.flat, GAMMAS):
        for m, color, style, label in (
            (4, '#0072B2', '--', 'RG m=4 versus full target'),
            (6, '#D55E00', '-', 'RG m=6 versus full target'),
            (8, '#777777', ':', 'm=8: bond modification alone')):
            ax.plot(LAMBDAS, [lookup[gamma, lam, m]['target_tv'] for lam in LAMBDAS],
                    marker='o', ms=4, ls=style, color=color, label=label)
        ax.axhline(.05, c='black', lw=1, ls='--', alpha=.5, label='TV=0.05 reference')
        ax.axvline(.7, c='#E69F00', lw=1, alpha=.5)
        ax.set(title=f'Gamma={gamma:g}', xlabel='Intertriangle multiplier in RG',
               ylabel='TV error relative to exact lambda=1', xlim=(-.02, 1.02), ylim=(0, 1))
        ax.grid(alpha=.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    name = save(fig, 'fixed_target_errors')
    cells.append(image_cell('## How much weakening can the original-model comparison tolerate?\n\n'
        'TV=½ Σq |P_RG(q;lambda_RG)−P_exact(q;1)|. Zero means identical distributions. '
        'The horizontal line at 0.05 is a descriptive accuracy reference for P(q). '
        'The vertical line marks lambda=0.7. Lines join the computed multipliers '
        '0, 0.1, 0.3, 0.5, 0.7, 0.85 and 1.\n\n'
        'With m=8 every triangle state is retained: its error measures the effect '
        'of changing the bonds alone. It recovers the original target only at '
        'lambda=1. Increasing m can reduce truncation error but cannot undo the '
        'modified Hamiltonian. Total distribution error need not decrease '
        'monotonically with m because model bias and truncation can partly cancel. '
        'The gray curve is a diagnostic, not an additive component of TV.', name))

    text = ['## Numerical comparison with the unchanged target', '',
        'Every entry below uses the exact lambda=1 reference. The last column '
        'measures the RG state energy under **H_target**, minus the exact target '
        'ground energy, per physical spin. This avoids comparing energies of '
        'different Hamiltonians as though they were a variational error.', '',
        '| Gamma | lambda_RG | TV m6 | TV m8: bond change alone | q² error m6 | target energy excess/spin m6 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for gamma in GAMMAS:
        for lam in DISPLAY_LAMBDAS:
            a, b = lookup[gamma, lam, 6], lookup[gamma, lam, 8]
            text.append(f"| {gamma:g} | {lam:g} | {a['target_tv']:.5f} | {b['target_tv']:.5f} | {a['target_q2_error']:.5f} | {a['target_energy_excess_per_spin']:.6f} |")
    text += ['', '**At lambda_RG=0.7:**', '']
    for gamma in GAMMAS:
        a = lookup[gamma, .7, 6]
        text.append(f"- Gamma={gamma:g}: TV={a['target_tv']:.5f}; "
                    f"q² error={a['target_q2_error']:.5f}.")
    passing = [g for g in GAMMAS if lookup[g, .7, 6]['target_tv'] < .05]
    text += ['', 'On this sample, lambda_RG=0.7 has distribution TV below 0.05 at '
        + (', '.join(f'Gamma={g:g}' for g in passing) if passing else 'none of the sampled Gamma values')
        + '. This assesses the physical P(q) distribution only; energy and '
        'correlation errors remain separate criteria.', '',
        'Smaller lambda makes the modified triangle problem easier to truncate '
        'when the relevant excitation denominators remain finite, but the target '
        'comparison includes the cost of changing the physical bonds. The tradeoff '
        'is field dependent; accuracy for the weaker model alone is insufficient.', '',
        '## Does the weakened RG preserve the target field response?', '',
        'For lambda_RG=0.7, compare the change of the entire RG distribution '
        'with the change of the unchanged target distribution. The change error is '
        '½ Σq |[P_RG(b)−P_RG(a)]−[P_target(b)−P_target(a)]|.', '',
        '| Gamma interval | exact target motion TV | RG lambda=0.7 motion TV | error in distribution change |',
        '| :--- | ---: | ---: | ---: |']
    for s in shifts:
        if s['lambda_rg'] == .7:
            text.append(f"| {s['gamma_from']:g} → {s['gamma_to']:g} | {s['exact_target_field_tv']:.5f} | {s['rg_field_tv']:.5f} | {s['field_change_error']:.5f} |")
    text += ['', '**Scope:** one L2 disorder, four sampled Gamma values, seven RG '
        'multipliers and m=4,6,8. This establishes how the chosen weakened RG '
        'states compare with the original small-system target. It does not yet '
        'validate L4 superblock truncation, disorder averages or L6 scaling.', '',
        'All scientific execution and figures were produced on AutoDL. Raw '
        'bonds, original and modified couplings, exact and retained states, '
        'isometries, solver residuals and source hashes are preserved with this '
        'report. The previous same-lambda comparisons and L4 plots are archived '
        'separately and have been removed from this notebook.']
    cells.append(nbformat.v4.new_markdown_cell('\n'.join(text)))
    nb = nbformat.v4.new_notebook(cells=cells, metadata=dict(
        execution_location='AutoDL', exact_reference_lambda=1., rg_lambdas=LAMBDAS,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        kernelspec=dict(display_name='Python 3', language='python', name='python3')))
    nbformat.validate(nb)
    nbformat.write(nb, out/'triangle_effective_interaction_scan.ipynb')
    write_json(out/'comparison.json', dict(reference_lambda=1., references=references,
        rg_lambdas=LAMBDAS, measurements=records, field_shifts=shifts,
        raw_hashes=hashes, checks_passed=True))
    shutil.copy2(Path(__file__), out/'report_source.py')
    source = Path(__file__).parent
    write_json(out/'provenance.json', dict(
        source_sha256=digest(Path(__file__)),
        sources={name: digest(source/name) for name in (
            Path(__file__).name, 'triangle_effective_interaction_scan.py',
            'triangle_effective_benchmark.py', 'triangle_effective_rg.py',
            'physics.py', 'triangle_projector.py')},
        raw_hashes=hashes,
        files={p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'provenance.json'}))
    print(json.dumps(dict(cells=len(cells), figures=len(figures), raw_contexts=len(hashes),
        lambda07=[r for r in records if r['lambda_rg'] == .7 and r['m'] == 6])), flush=True)


if __name__ == '__main__':
    main()
