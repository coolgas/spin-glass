"""Audit saved L2 states and make the exact-versus-RG notebook on AutoDL."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from ising.hamiltonian import Hamiltonian
from kagome_qea.exact import operator_matrix
from .triangle_effective_benchmark import digest, lift, write_json, xor_mass
from .triangle_effective_interaction_scan import GAMMAS, LAMBDAS, H, physical_hamiltonian
from .triangle_effective_rg import triangles


COLORS = {4: '#0072B2', 6: '#D55E00', 8: '#009E73'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.results, args.output
    out.mkdir(parents=True, exist_ok=True)
    q = np.linspace(-1, 1, 13)
    records, contexts, masses, raw_hashes = [], [], {}, {}
    reference_bonds = None
    for lam in LAMBDAS:
        for gamma in GAMMAS:
            key = f'l2_g{gamma:g}_l{lam:g}'
            path = root/'exact'/(key+'.json')
            row = json.loads(path.read_text())
            raw = path.with_suffix('.npz')
            assert digest(raw) == row['raw_sha256']
            raw_hashes[key] = dict(npz=digest(raw), json=digest(path))
            sample, original, ham, mask = physical_hamiltonian(2, lam)
            matrix = operator_matrix(ham, gamma)
            first = triangles(original, gamma, 8, lam=lam)
            with np.load(raw) as data:
                np.testing.assert_array_equal(data['bonds'], sample.bonds)
                np.testing.assert_allclose(data['original_J'], sample.J, atol=0, rtol=0)
                np.testing.assert_allclose(data['J'], ham.J, atol=0, rtol=0)
                np.testing.assert_array_equal(data['mask'], mask)
                assert mask.sum() == 12 and (~mask).sum() == 12
                if reference_bonds is None:
                    reference_bonds = data['original_J'].copy()
                np.testing.assert_array_equal(data['original_J'], reference_bonds)
                state = data['exact_state']
                np.testing.assert_allclose(state@state, 1, atol=1e-12)
                energy = float(state@(matrix@state))
                residual = float(np.linalg.norm(matrix@state-energy*state))
                assert residual < 1e-8
                mass = data['mass'].copy()
                np.testing.assert_allclose(mass, xor_mass(state**2, 12), atol=1e-12)
                masses[gamma, lam, 0] = mass
                contexts.append(dict(gamma=gamma, lam=lam, energy=energy,
                    residual=residual, q=float(mass@q), q2=float(mass@(q*q))))
                for m in (4, 6, 8):
                    # Use the archived isometries, not newly selected eigenvectors.
                    archived = dict(sites=first['sites'], bases=data[f'bases_m{m}'])
                    psi = lift(archived, data[f'state_m{m}'], m)
                    np.testing.assert_allclose(psi@psi, 1, atol=1e-11)
                    pq = data[f'mass_m{m}'].copy()
                    np.testing.assert_allclose(pq, xor_mass(psi**2, 12), atol=1e-11)
                    assert pq.min() > -1e-12 and abs(pq.sum()-1) < 1e-11
                    estimate = float(psi@(matrix@psi))
                    error = (estimate-energy)/12
                    tv = float(abs(pq-mass).sum()/2)
                    saved = next(c for c in row['checks'] if c['m'] == m)
                    np.testing.assert_allclose(tv, saved['tv'], atol=1e-11)
                    np.testing.assert_allclose(error, saved['energy_error_per_site'], atol=1e-10)
                    assert error > -1e-10
                    fidelity = float(abs(psi@state)**2)
                    if m == 8:
                        assert tv < 1e-7 and abs(error) < 1e-8 and fidelity > 1-1e-7
                    record = dict(gamma=gamma, lam=lam, m=m, tv=tv,
                        energy_error_per_spin=error, fidelity=fidelity,
                        q=float(pq@q), q2=float(pq@(q*q)),
                        mean_q_error=float(abs((pq-mass)@q)),
                        q2_error=float(abs((pq-mass)@(q*q))))
                    records.append(record)
                    masses[gamma, lam, m] = pq
    lookup = {(r['gamma'], r['lam'], r['m']): r for r in records}
    shifts = []
    for lam in LAMBDAS:
        for a, b in zip(GAMMAS, GAMMAS[1:]):
            exact_delta = masses[b, lam, 0]-masses[a, lam, 0]
            motion = float(abs(exact_delta).sum()/2)
            for m in (4, 6, 8):
                rg_delta = masses[b, lam, m]-masses[a, lam, m]
                change_error = float(abs(rg_delta-exact_delta).sum()/2)
                endpoint_errors = lookup[a, lam, m]['tv']+lookup[b, lam, m]['tv']
                assert change_error <= endpoint_errors+1e-11
                shifts.append(dict(lam=lam, gamma_from=a, gamma_to=b, m=m,
                    exact_field_tv=motion, rg_field_tv=float(abs(rg_delta).sum()/2),
                    field_change_error=change_error, endpoint_error_sum=endpoint_errors,
                    endpoint_error_over_motion=endpoint_errors/motion if motion > 1e-12 else None,
                    exact_delta_q2=float(exact_delta@(q*q)),
                    rg_delta_q2=float(rg_delta@(q*q))))
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

    text = r'''# Exact L2 versus triangle RG: varying Gamma at fixed lambda

The primary comparison is now the **same 12-spin physical model**, solved in
the full spin space and after retaining m states per triangle. Each lambda
section below compares exact and RG P(q) at Gamma=0.25, 0.5, 1 and 2, with
longitudinal field h=0.15. All curves at a given parameter pair share the
**identical saved bonds**. This is one disorder realization (disorder60000).

We first sample positive antiferromagnetic bonds independently from
$K^{(0)}_{ij}\sim U(0.5,1.5)$. Choose disjoint orientation0 triangles, call
their internal bonds A and their connecting bonds B, and use

$$H(\lambda)=\sum_{(ij)\in A}K^{(0)}_{ij}Z_iZ_j
+\lambda\sum_{(ij)\in B}K^{(0)}_{ij}Z_iZ_j
-0.15\sum_i Z_i-\Gamma\sum_i X_i.$$

| lambda | K inside triangles | K connecting triangles | Physical meaning |
| ---: | :--- | :--- | :--- |
| 0 | U(0.5, 1.5) | zero | independent frustrated triangles |
| 0.1 | U(0.5, 1.5) | U(0.05, 0.15) | weakly connected triangles |
| 0.3 | U(0.5, 1.5) | U(0.15, 0.45) | intermediate connection |
| 1 | U(0.5, 1.5) | U(0.5, 1.5) | original kagome bond law |

No bonds are redrawn as Gamma or lambda changes. L2 has 24 bonds: 12 A and
12 B, with two of each touching every spin. Lambda is dimensionless; fields
are in units of the mean internal K, which is 1. Stored couplings use J=-K.
Changing lambda changes the Hamiltonian; increasing m improves its approximation.

| Calculation | Physical spins | Triangle sites | Total solved dimension |
| :--- | ---: | ---: | :--- |
| Full L2 | 12 | — | 2^12 = 4096 |
| RG m=4 | 12 | 4 | 4^4 = 256 |
| RG m=6 | 12 | 4 | 6^4 = 1296 |
| Full triangle space m=8 | 12 | 4 | 8^4 = 4096 |

This L2 test has one projection step. The later superblock retention chi is
not used. At m=8 no triangle state is discarded, so exact recovery checks
the projection and reconstruction.

The measured quantity is the **physical** overlap between two independent
replicas drawn from the ground-state Born distribution:

$$q=\frac1{12}\sum_i z_i^{(a)}z_i^{(b)},\qquad
P(q)=\sum_{z,z'}|\psi(z)|^2|\psi(z')|^2
\mathbf{1}\!\left[q=\frac1{12}\sum_i z_i z'_i\right].$$

There are 13 bins, with spacing 1/6. The vertical axis is probability mass
per bin. RG wavefunctions are lifted back to all 12 physical spins before
measuring q. Every probability is evaluated by exact summation over the saved
states, so these L2 curves have **no replica sampling noise**. Lines join
discrete bins. The full spin reference is a numerical ground-state eigensolve.
At h=0.15, mean q includes local polarization:
$\langle q\rangle=12^{-1}\sum_i\langle Z_i\rangle^2$.
Neither this mean nor a finite-size peak alone establishes glass order.
'''
    cells = [nbformat.v4.new_markdown_cell(text)]
    for lam in LAMBDAS:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True,
                                 layout='constrained')
        for ax, gamma in zip(axes.flat, GAMMAS):
            ax.plot(q, masses[gamma, lam, 0], 'o-', c='black', lw=2, ms=5,
                    label='Full 12-spin exact')
            for m, style, marker in ((4, '--', 's'), (6, '-.', '^'), (8, ':', '.')):
                ax.plot(q, masses[gamma, lam, m], color=COLORS[m], ls=style,
                        marker=marker, ms=4, lw=1.4, label=f'RG m={m}')
            ax.set(title=f'Gamma = {gamma:g}', xlim=(-1.02, 1.02), ylim=(0, None),
                   xlabel='Physical overlap q', ylabel='Probability mass per bin')
            ax.grid(alpha=.2)
            ax.text(.02, .97, '\n'.join(f"TV(m={m}) = {lookup[gamma,lam,m]['tv']:.4g}"
                    for m in (4, 6)), transform=ax.transAxes, va='top', fontsize=9)
        fig.suptitle(f'L2: exact versus RG | lambda={lam:g} | h=0.15 | identical bonds')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4)
        name = save(fig, f'l2_exact_rg_lambda{lam:g}')
        cells.append(image_cell(f'## Fixed lambda={lam:g}: does RG track the exact distribution?\n\n'
            'Read the four panels in order to increase Gamma. Black is the full-spin '
            'reference; blue, orange and green retain 4, 6 and all 8 states per triangle. '
            'Overlapping curves indicate agreement, including the required m=8 recovery. '
            'TV=½ Σq |P_RG(q)−P_exact(q)|; zero means identical distributions. '
            'All four panels share the same vertical scale within this figure.', name))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharex=True, sharey=True,
                             layout='constrained')
    gamma_colors = ('#0072B2', '#009E73', '#E69F00', '#CC79A7')
    for ax, lam in zip(axes, LAMBDAS):
        for gamma, color in zip(GAMMAS, gamma_colors):
            ax.plot(q, masses[gamma, lam, 0], 'o-', c=color, ms=3, label=f'Gamma={gamma:g}')
            ax.plot(q, masses[gamma, lam, 6], '--', c=color, lw=1.5)
        ax.set(title=f'lambda={lam:g}', xlabel='Physical overlap q', xlim=(-1, 1), ylim=(0, None))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Probability mass per bin')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=4)
    fig.suptitle('Field response | solid + circles: full-spin exact | dashed: RG m=6')
    name = save(fig, 'l2_gamma_overlays_exact_rg')
    cells.append(image_cell('## How does P(q) move with Gamma at each lambda?\n\n'
        'Here **color identifies Gamma**. For every color the solid curve with circles '
        'is exact and the dashed curve is RG m=6. Each panel fixes lambda. These overlays '
        'show whether RG reproduces the motion of the full physical distribution.', name))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey='row',
                             layout='constrained')
    for col, lam in enumerate(LAMBDAS):
        for row, power in enumerate((1, 2)):
            ax = axes[row, col]
            for m in (0, 4, 6):
                values = [masses[g, lam, m]@(q**power) for g in GAMMAS]
                ax.plot(GAMMAS, values, marker='o' if m == 0 else '^',
                    ls='-' if m == 0 else '--', color='black' if m == 0 else COLORS[m],
                    label='Full-spin exact' if m == 0 else f'RG m={m}', ms=4)
            ax.set(xscale='log', xticks=GAMMAS, xticklabels=[f'{g:g}' for g in GAMMAS],
                   ylim=(0, None), title=f'lambda={lam:g}' if row == 0 else None)
            ax.grid(alpha=.2)
            if row == 1: ax.set_xlabel('Transverse field Gamma')
    axes[0, 0].set_ylabel('Mean physical overlap <q>')
    axes[1, 0].set_ylabel('Second moment <q²>')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=3)
    name = save(fig, 'l2_moments_exact_rg')
    cells.append(image_cell('## Does RG reproduce the field dependence of the moments?\n\n'
        'Top: mean q, including the longitudinal-field polarization. Bottom: mean q². '
        'The shared axes make lambda panels comparable. These are the four computed '
        'Gamma values; connecting lines are guides, not extra calculations.', name))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout='constrained')
    for col, m in enumerate((4, 6)):
        for row, field in enumerate(('tv', 'energy_error_per_spin')):
            values = np.array([[lookup[g, lam, m][field] for lam in LAMBDAS] for g in GAMMAS])
            im = axes[row, col].imshow(values, cmap='magma', vmin=0,
                                      vmax=.23 if row == 0 else .04, aspect='auto')
            for i in range(4):
                for j in range(4):
                    axes[row, col].text(j, i, f'{values[i,j]:.2g}', ha='center', va='center',
                        color='black' if values[i,j] > (.14 if row == 0 else .025) else 'white')
            axes[row, col].set(title=f'm={m}: '+('distribution TV error' if row == 0 else 'energy error / spin'),
                xticks=range(4), xticklabels=[f'{lam:g}' for lam in LAMBDAS],
                yticks=range(4), yticklabels=[f'{g:g}' for g in GAMMAS], ylabel='Gamma', xlabel='lambda')
            fig.colorbar(im, ax=axes[row, col], shrink=.8)
    name = save(fig, 'l2_exact_errors')
    cells.append(image_cell('## Does smaller lambda help convergence?\n\n'
        'At fixed Gamma and m, weaker connecting bonds usually reduce the error here. '
        'At lambda=0 the ground state is a product of triangle ground states, already '
        'contained in each retained space, and RG is exact up to numerical precision. '
        'Increasing m from 4 to 6 markedly improves the connected cases. '
        'The two columns share color limits within each row; numbers show small errors '
        'that the common color scale can hide. Energy errors are in units of mean internal K=1.\n\n'
        'This gives a useful controlled starting point: connecting bonds mix local '
        'triangle states, so weakening them suppresses that mixing when the relevant '
        'excitation denominators stay finite. Small bare lambda alone is not a guarantee '
        'near small denominators, nor does it make an iterative solver necessarily faster. '
        'The numerical question here is convergence of the retained approximation.\n\n'
        '**Why it matters:** lambda allows a direct test from independent triangles to '
        'the original lattice. It is a promising route to a validated weak-coupling '
        'effective theory. To claim accuracy for the original kagome model we must '
        'still demonstrate convergence at lambda=1, and then across larger sizes. '
        'A small-lambda physical result describes the model with those weaker bonds.', name))

    table = ['## Measured accuracy and field motion', '',
        'All 16 parameter pairs are included below. Errors compare each RG result with its own full-spin reference.', '',
        '| lambda | Gamma | TV m4 | TV m6 | energy error/spin m6 | q² error m6 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for lam in LAMBDAS:
        for gamma in GAMMAS:
            a, b = lookup[gamma, lam, 4], lookup[gamma, lam, 6]
            table.append(f"| {lam:g} | {gamma:g} | {a['tv']:.6f} | {b['tv']:.6f} | {b['energy_error_per_spin']:.7f} | {b['q2_error']:.6f} |")
    table += ['', 'To compare approximation error with physical field response, use the exact TV distance '
        'between adjacent Gamma distributions. The endpoint error sum is TV(RG,exact) at '
        'both ends; by the triangle inequality it bounds the error in the distribution change. '
        'A small ratio to the exact motion means the field response is well resolved at these '
        'sampled points. This ratio is descriptive and is not a thermodynamic criterion.', '',
        '| lambda | Gamma interval | exact field TV | m6 endpoint error sum | error sum / motion |',
        '| ---: | :--- | ---: | ---: | ---: |']
    for s in shifts:
        if s['m'] == 6:
            ratio = s['endpoint_error_over_motion']
            table.append(f"| {s['lam']:g} | {s['gamma_from']:g} → {s['gamma_to']:g} | {s['exact_field_tv']:.6f} | {s['endpoint_error_sum']:.6f} | {ratio:.4f} |")
    worst = max(records, key=lambda r: r['tv'] if r['m'] == 6 else -1)
    recovery = max(r['tv'] for r in records if r['m'] == 8)
    table += ['', f"Across this grid, the largest m=6 TV error is **{worst['tv']:.6f}** "
        f"at lambda={worst['lam']:g}, Gamma={worst['gamma']:g}. The largest m=8 recovery "
        f"TV error is **{recovery:.2e}**. Distribution accuracy and energy accuracy are "
        'different tests: an accurate P(q) does not imply that every energy tolerance passes.', '',
        'The 16 archived raw hashes, all physical distributions reconstructed from the '
        'saved states and isometries, physical energies, norms, reference residuals, '
        'shared disorder arrays, and m=8 recovery were checked on AutoDL. Numerical '
        'checks and per-context measurements are saved in `l2_comparison.json`.', '',
        '**Scope for the next step:** this is strong evidence for the first triangle '
        'projection on this L2 sample. L4 adds a separate superblock truncation, whose '
        'chi convergence remains unresolved. Use the L2 response to choose follow-up '
        'parameters, then validate L4 retention and disorder dependence before interpreting '
        'an L6 calculation as a size trend. No L6 calculation is included here.']
    cells.append(nbformat.v4.new_markdown_cell('\n'.join(table)))
    old_path = root/'triangle_effective_interaction_scan.ipynb'
    old = nbformat.read(old_path, as_version=4)
    shutil.copy2(old_path, out/'previous_interaction_notebook.ipynb')
    cells.append(nbformat.v4.new_markdown_cell('## Appendix: previous L4 retention screen\n\n'
        'The primary benchmark above compares exact and RG **on the same L2 system**. '
        'The archived L4 figures below retain the earlier study for reference. L4 has '
        'a different size-dependent disorder draw and an additional superblock projection; '
        'its retention agreement is a sensitivity check. Its sample histograms have '
        'replica noise, unlike the deterministic L2 curves above.'))
    for cell in old.cells:
        if cell.get('attachments'):
            cell.metadata['l4_archive_appendix'] = True
            cells.append(cell)
    nb = nbformat.v4.new_notebook(cells=cells, metadata=dict(execution_location='AutoDL',
        primary_benchmark='Same-model full-spin L2 versus first-stage triangle RG',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        kernelspec=dict(display_name='Python 3', language='python', name='python3')))
    nbformat.validate(nb)
    notebook = out/'triangle_effective_interaction_scan.ipynb'
    nbformat.write(nb, notebook)
    write_json(out/'l2_comparison.json', dict(contexts=contexts, measurements=records,
        field_shifts=shifts, raw_hashes=raw_hashes, numerical_checks_passed=True,
        all_m8_recovery_passed=True, max_m8_tv=recovery, no_sampling_noise=True))
    shutil.copy2(Path(__file__), out/'report_source.py')
    source_paths = [Path(__file__), *[Path(__file__).with_name(n) for n in (
        'triangle_effective_benchmark.py', 'triangle_effective_interaction_scan.py',
        'triangle_effective_rg.py', 'physics.py', 'triangle_projector.py')]]
    write_json(out/'provenance.json', dict(notebook_sha256=digest(notebook),
        previous_notebook_sha256=digest(old_path), source_sha256=digest(Path(__file__)),
        imported_sources={str(p.name): digest(p) for p in source_paths},
        raw_hashes=raw_hashes,
        files={p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'provenance.json'}))
    print(json.dumps(dict(contexts=len(contexts), rg_models=len(records), max_m6_tv=worst['tv'],
        max_m8_tv=recovery, cells=len(cells), figures=len(figures))), flush=True)


if __name__ == '__main__':
    main()
