"""Build explanatory notebooks from preserved triangle and simple-guide results."""
from pathlib import Path

import nbformat as nb


ROOT = Path(__file__).resolve().parents[2]


def main():
    cells = [nb.v4.new_markdown_cell(r"""# Simpler guiding wavefunctions: Jastrow and RBM

**Question:** is a cheaper, geometry-independent positive guide more useful
than the current transformer-based pipeline on the difficult low-field cases?

We compare an all-pairs Jastrow wavefunction and real RBMs trained from scratch.
These are **selected $L=2$ diagnostics**, not finite-size scaling. Training uses
full-basis energy expectations to isolate ansatz/optimization behavior from
training-chain mixing. Exact eigenvectors are loaded **after fitting**, only for
evaluation. Larger-system sampled training remains unqualified.

This does not isolate ALiBi: the optimizer, per-disorder fitting, initialization,
and compute budget differ from the transformer pipeline. It does test whether
much simpler guides can avoid its observed metastable-sector failure.
"""), nb.v4.new_markdown_cell(r"""## 1. Positive wavefunctions and cheap flip ratios

The two ansätze are
$$\log\psi_J(s)=\sum_i a_i s_i+\sum_{i<j}u_{ij}s_is_j,$$
$$\log\psi_{\rm RBM}(s)=\sum_i a_i s_i+
\sum_{\mu=1}^{\alpha N}\log[2\cosh(b_\mu+\sum_iW_{i\mu}s_i)].$$

Both have analytic single-spin ratios. For Jastrow,
$$\log\frac{\psi(s^{(i)})}{\psi(s)}=-2s_i(a_i+\sum_{j\ne i}u_{ij}s_j).$$
For RBM, cache $\theta_\mu=b_\mu+\sum_iW_{i\mu}s_i$ and replace it by
$\theta_\mu-2W_{i\mu}s_i$ for a flip. Neither ansatz needs token positions.

The direct RBM sampling adapter uses **two hidden copies**, since the target is
$\psi^2$, not $\psi$. Its finite Gibbs chains are not automatically independent
or equilibrated. The L2 GFMC experiment below instead uses enumerated guide
probabilities for initialization and lookup; this is explicitly a small-system
accelerator.
"""), nb.v4.new_code_cell("""from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt

root = Path.cwd().resolve()
if not (root / 'code/transformer_qmc').exists():
    root = root.parent
sys.path.insert(0, str(root / 'code'))
from transformer_qmc.correlation_lengths import sha
from transformer_qmc.collective_campaign import label
from transformer_qmc.production_protocol import identity, geometry
from transformer_qmc.independent_replica_projector import spectrum_from_moments

folder = root / 'data/kagome/ltfim_simple_guides_v1'
screen = json.loads((folder / 'screen/review.json').read_text())['rows']
manifest = json.loads((folder / 'screen/manifest.json').read_text())
for row in screen:
    name = f"{label(row['gamma'],row['sample_id'])}_{row['kind']}_seed{row['seed']}"
    assert sha(folder / 'screen' / (name+'.npz')) == row['npz_sha256']
    assert row['source_sha256'] == manifest['source_sha256']
print('Validated fits:', len(screen))
print('Training settings:', manifest['settings'])
contexts = [(.25,50025),(.25,50059),(.5,50043),(.75,50004)]
kinds = ['jastrow','rbm_1','rbm_2']
selected = {(g,s,k): min([r for r in screen if
    (r['gamma'],r['sample_id'],r['kind']) == (g,s,k)], key=lambda r:r['energy'])
    for g,s in contexts for k in kinds}
print('Selection uses variational energy only; all seeds are retained.')
"""), nb.v4.new_markdown_cell(r"""## 2. Every initialization, including metastable failures

Each fit uses 800 Adam steps followed by at most 200 L-BFGS iterations, with
three independent initializations per ansatz/context. The variational objective
is $\langle\psi|H|\psi\rangle/\langle\psi|\psi\rangle$ for the original LTFIM.
No fidelity, correlation, or exact-state label enters the objective or selection.

The plots distinguish energy, ground-state fidelity, and spectral error. A
low-variance excited-sector fit remains a failure, even when its energy looks
reasonable. Selecting the minimum-energy seed helps but does not guarantee
finding the ground sector in a larger system.
"""), nb.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(13,4))
colors = dict(jastrow='C0',rbm_1='C1',rbm_2='C2')
for j,kind in enumerate(kinds):
    for c,(g,s) in enumerate(contexts):
        rows = [r for r in screen if (r['gamma'],r['sample_id'],r['kind'])==(g,s,kind)]
        x = c+(j-1)*.22+np.linspace(-.035,.035,len(rows))
        values = [[r['energy_error'] for r in rows],
                  [r['fidelity'] for r in rows],
                  [100*r['max_spectral_relative_error'] for r in rows]]
        for ax,y in zip(axes,values):
            ax.scatter(x,y,color=colors[kind],s=28,label=kind if c==0 else None)
for ax in axes:
    ax.set_xticks(range(4),[f'{g}, {s}' for g,s in contexts],rotation=25,ha='right')
    ax.set_xlabel('Gamma, disorder');ax.grid(alpha=.25)
axes[0].set_yscale('log');axes[0].set_ylabel('Total variational energy error')
axes[1].set_yscale('log');axes[1].set_ylabel('Ground-state fidelity')
axes[2].set_yscale('log');axes[2].set_ylabel('Maximum spectral error (%)')
axes[0].legend();fig.suptitle('All 36 fits: L2 full-basis training diagnostic')
fig.tight_layout()
figure_dir=root/'figures/simple_guides';figure_dir.mkdir(parents=True,exist_ok=True)
fig.savefig(figure_dir/'all_seeds.png',dpi=180,bbox_inches='tight')
fig.savefig(figure_dir/'all_seeds.pdf',bbox_inches='tight')
plt.show()
print('Lowest-energy seeds:')
for (g,s,k),r in selected.items():
    print(f"{g:.2f} {s} {k:8s} seed={r['seed']} E error={r['energy_error']:.6g} "
          f"fidelity={r['fidelity']:.9f} spectral error={100*r['max_spectral_relative_error']:.4f}%")
"""), nb.v4.new_markdown_cell(r"""## 3. Projection can still matter

The following is an **offline exact propagation diagnostic**, not a Monte Carlo
result. It evaluates the symmetric pure state proportional to
$[e^{-\tau H}\psi_T]^2$. This differs from the GFMC forward-walking schedule,
whose left projection grows over measurement cycles and whose right projection
is fixed. We do not treat these exact curves as that estimator's finite-time
target.
"""), nb.v4.new_code_cell("""for (g,s,k),r in selected.items():
    print(f'{g:.2f} {s} {k:8s}:', ', '.join(
        f\"tau={x['tau']:g}: {100*x['max_spectral_relative_error']:.4f}%\"
        for x in r['projection']))
"""), nb.v4.new_markdown_cell(r"""## 4. Matched GFMC with frozen selected guides

For each ansatz/context, freeze the minimum-energy fit and compare continuous
time and Hamiltonian-aware triangle GFMC at step 0.1. Each run uses four
independently resampled populations, 1024 walkers per population, equilibration
32, forward time 32, and 16 measurement cycles. There are four whole-run repeats.

Connected spectra use products of moments from disjoint populations. We verify
each raw archive hash and reconstruct the spectrum from saved moments. Whole
runs, not walkers, are the uncertainty units. The plots show all selected
contexts, including failed guides. Four repeats and one population/time setting
do not establish controlled ground-state convergence.
"""), nb.v4.new_code_cell("""projection = json.loads((folder/'projector/review.json').read_text())['rows']
project_manifest=json.loads((folder/'projector/manifest.json').read_text())
validated=0
for path in sorted((folder/'projector').glob('*/runs/*.json')):
    r=json.loads(path.read_text())
    assert sha(path.with_suffix('.npz'))==r['npz_sha256']
    assert r['provenance']['source_sha256']==project_manifest['source_sha256']
    with np.load(path.with_suffix('.npz')) as a:
        measured=spectrum_from_moments(a['magnetization_cycles'].mean(1),a['pair_cycles'].mean(1),geometry(2))
    np.testing.assert_allclose(measured,r['spectrum'],atol=1e-12)
    validated+=1
print('Validated GFMC runs:',validated)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for j,kind in enumerate(kinds):
    for method,marker,offset in [('ct','o',-.035),('triangle','x',.035)]:
        rows=[next(r for r in projection if (r['gamma'],r['sample_id'],r['kind'],r['method'])==(g,s,kind,method)) for g,s in contexts]
        x=np.arange(4)+(j-1)*.22+offset
        axes[0].plot(x,[100*r['max_relative_error'] for r in rows],marker,ls='none',color=colors[kind],label=f'{kind}, {method}')
        axes[1].plot(x,[r['minimum_descendant_ess'] for r in rows],marker,ls='none',color=colors[kind])
for ax in axes:
    ax.set_xticks(range(4),[f'{g}, {s}' for g,s in contexts],rotation=25,ha='right')
    ax.set_xlabel('Gamma, disorder');ax.grid(alpha=.25)
axes[0].set_yscale('log');axes[0].set_ylabel('Maximum spectral error (%)');axes[0].legend(fontsize=8)
axes[1].set_yscale('log');axes[1].set_ylabel('Minimum descendant ESS across four runs')
fig.suptitle('Energy-selected simple guides: matched L2 GFMC')
fig.tight_layout();fig.savefig(figure_dir/'gfmc_comparison.png',dpi=180,bbox_inches='tight')
fig.savefig(figure_dir/'gfmc_comparison.pdf',bbox_inches='tight');plt.show()
for r in projection:
    print(f\"{r['gamma']:.2f} {r['sample_id']} {r['kind']:8s} {r['method']:8s}: \"
          f\"spectral error={100*r['max_relative_error']:.3f}% alarms={r['alarms']}/{r['repeats']}\")
"""), nb.v4.new_markdown_cell(r"""## Interpretation and limits

The comparison tests **guide representation plus optimization**, and does not
prove that ALiBi caused the transformer failure. RBM and Jastrow also have
metastable optimization failures. The retained seed-level results make this
visible instead of reporting only successful fits.

The original finite-size target remains $\xi_R/L$ versus $\Gamma$ and versus
$L$ at $h=0.15$, with disorder spectra averaged before taking the length ratio.
Four selected L2 contexts cannot supply those curves. Before extrapolating,
simple guides need sampled training, mixing checks, direct-guide timing, and
population/projection/step-size convergence at larger L.

Sources: [guide definitions](../code/transformer_qmc/simple_guides.py),
[energy-only fitting](../code/transformer_qmc/simple_guide_benchmark.py),
[GFMC comparison](../code/transformer_qmc/simple_guide_projector.py), and
[method notes](../code/transformer_qmc/SIMPLE_GUIDES.md).
""")]
    notebook=nb.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nb.write(notebook,ROOT/'notebook/simple_guiding_wavefunctions.ipynb')

    path=ROOT/'notebook/collective_flip_guided_gfmc.ipynb'
    notebook=nb.read(path,as_version=4)
    marker='## 11. Executed Hamiltonian-aware triangle update'
    if not any(marker in c.source for c in notebook.cells):
        notebook.cells.extend([nb.v4.new_markdown_cell(r"""## 11. Executed Hamiltonian-aware triangle update

Split the Kagome triangles into two vertex-disjoint layers. Put each bond in its
unique triangle and divide each on-site longitudinal/transverse field equally
between its two triangles. Then $H=A+B$ exactly. Use
$$G_\delta=e^{-\delta A/2}e^{-\delta B}e^{-\delta A/2}.$$
Each factor is an exact positive eight-state local exponential. For block $K$,
sample $q(y|x)=K_{yx}\psi_T(y)/Z(x)$ and multiply the weight by
$Z(x)/\psi_T(x)$. Intermediate guide ratios telescope; the correction is a
**path correction**, not an endpoint approximation. This gives second-order
projection with no arbitrary extra spin-flip term.

Tests verify the original full Hamiltonian, operator symmetry, cubic one-step
error, weighted path moments, cache/direct equivalence, lattice covers through
L=10, and checkpoint recovery. The pilot below has only two short repeats per
setting and uses frozen enumerated L2 guides. Its exact references are computed
for the actual triangle product, including the finite-projection schedule.
"""),nb.v4.new_code_cell("""from transformer_qmc.triangle_campaign import assess as assess_triangle
triangle_folder=root/'data/kagome/ltfim_triangle_gfmc_v1/pilot'
triangle=assess_triangle(triangle_folder,snapshot/'benchmark/references.json')
print('Validated triangle-comparison pilot runs:',triangle['completed'])
for r in triangle['rows']:
    if r['guide']=='corrected' and r['delta']==.1:
        sampling=100*max(abs(np.asarray(r['relative_sampling_error'])))
        split=100*max(abs(np.asarray(r['split_ground_relative_bias'])))
        print(f\"{r['gamma']:.2f} {r['sample_id']} {r['method']:8s}: ground error \"
              f\"{100*r['max_relative_bias']:.3f}%, finite-target sampling error {sampling:.3f}%, \"
              f\"split-ground error {split:.5f}%\")
"""),nb.v4.new_markdown_cell("""### What changed scientifically

The triangle rule passed correctness checks but did not rescue the two difficult
corrected low-field guides. At step 0.1 their errors remain about 84% and 155%.
The sampler is within roughly 1% of its exact finite-time target there: the bad
target is already explained by the initial guide and projection schedule.

This motivated testing guide representation itself. See the executed
[Jastrow/RBM comparison](simple_guiding_wavefunctions.ipynb). Its small-system
results separate metastable optimization failures from successful simple fits;
they do not qualify larger-system sampled training.
""")])
        nb.write(notebook,path)


if __name__=='__main__':main()
