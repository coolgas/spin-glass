"""Build the self-contained triangle RG research-plan notebook; no simulations."""

import hashlib
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name("TRIANGLE_EFFECTIVE_THEORY_PLAN.md")
OUTPUT = ROOT / "notebook" / "triangle_effective_rg_research_plan.ipynb"


def schedule_svg():
    """An embedded diagram, independent of a notebook kernel or remote assets."""
    boxes = [
        (24, "Physical spins", "48 spin-1/2 sites", "Full dimension: 2^48"),
        (308, "Triangle blocks", "16 blocks / 3 spins each", "Trial: m states per block"),
        (592, "Superblocks", "4 blocks / 12 spins each", "Trial: chi states per block"),
        (876, "Final reduced solve", "4 interacting superblocks", "chi^4; 4^4 = 256 amplitudes"),
    ]
    elements = []
    for index, (x, title, description, dimension) in enumerate(boxes):
        elements.append(
            f'<rect x="{x}" y="62" width="260" height="120" rx="12" '
            'fill="#eef4fb" stroke="#385a7c" stroke-width="2"/>'
            f'<text x="{x + 130}" y="94" text-anchor="middle" '
            f'font-size="19" font-weight="bold">{title}</text>'
            f'<text x="{x + 130}" y="125" text-anchor="middle" '
            f'font-size="15">{description}</text>'
            f'<text x="{x + 130}" y="154" text-anchor="middle" '
            f'font-size="14">{dimension}</text>'
        )
        if index < len(boxes) - 1:
            elements.append(
                f'<path d="M {x + 262} 121 H {x + 278}" stroke="#385a7c" '
                'stroke-width="2" marker-end="url(#arrow)"/>'
            )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="258" '
        'viewBox="0 0 1160 258" role="img" aria-labelledby="title description">'
        '<title id="title">Required L4 renormalization benchmark</title>'
        '<desc id="description">48 physical spins become 16 triangle blocks, '
        'then 4 superblocks. Retaining four states per block at both levels '
        'leaves 256 amplitudes. Benchmark before proceeding to L6.</desc>'
        '<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" '
        'refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#385a7c"/>'
        '</marker></defs>'
        '<rect width="1160" height="258" fill="white"/>'
        '<g fill="#18314b" font-family="Arial, sans-serif">'
        '<text x="580" y="30" text-anchor="middle" font-size="20" '
        'font-weight="bold">L4 benchmark before L6</text>'
        + "".join(elements)
        + '<text x="580" y="214" text-anchor="middle" font-size="16">'
        'Transform the full Hamiltonian and physical measurement operators at each step.</text>'
        '<text x="580" y="241" text-anchor="middle" font-size="15">'
        'The tested fixed low-energy schedule failed the first L4 pilot; expansion is stopped.</text>'
        '</g></svg>'
    )


def build():
    plan_text = PLAN.read_text()
    digest = hashlib.sha256(plan_text.encode()).hexdigest()
    cells = [
        nbformat.v4.new_markdown_cell(
            r"""# Kagome real-space renormalization: research plan

**Goal:** reduce L6 enough to solve an effective Hamiltonian while preserving
physical-spin overlaps, eventually scaling $q_{\rm EA}$ and the physical
correlation length in a validated RG regime. Updated 4 November 2026: the five-schedule L4 pilot
fails the accuracy screen; the completed higher-field comparison captures the
qualitative trend with a bias toward lower overlap. **Current priority:
fixed original-model exact L2 versus RG with varied intertriangle strength, then L4
convergence and gradual L6 diagnostics before finite-size scaling.**
[Pilot](triangle_effective_l4_results.ipynb) ·
[Higher-field results](triangle_effective_higher_field_distributions.ipynb).

**What is $\chi$ (chi)?** It is the number of quantum states retained for each
superblock. First retain $m$ states per three-spin triangle; then combine four
triangles into a superblock with $m^4$ basis states. Diagonalize its internal
Hamiltonian and keep the lowest $\chi$ eigenstates:

$$W_B=(|v_{B,0}\rangle,\ldots,|v_{B,\chi-1}\rangle)
\in\mathbb C^{m^4\times\chi},\qquad W_B^\dagger W_B=I_\chi.$$

For $m=4$, this compresses 256 states to, for example, $\chi=4$ or 8.
Each retained state can be a superposition of many triangle configurations.
With $n_B$ superblocks, the final state has $\chi^{n_B}$ amplitudes:
$\chi^4$ for L4 and $\chi^9$ for L6. Increasing $\chi$ enlarges the retained
space at fixed $m$ and partition, but raises cost sharply; physical-overlap
accuracy must be checked. Keeping $\chi=m^4$ removes this second truncation,
while the earlier triangle truncation remains if $m<8$.

The full Hamiltonian is

$$H=\sum_{\langle ij\rangle}K_{ij}Z_iZ_j
-h\sum_i Z_i-\Gamma\sum_i X_i,
\qquad K_{ij}\sim U(0.5,1.5),\quad h=0.15,\quad N=3L^2.$$

**Order: exact small systems → L4 (48 spins) → L6 (108 spins).**
Keep each disorder fixed across solvers.
"""
        ),
        nbformat.v4.new_markdown_cell(
            "## 1. Renormalization schedule\n\n"
            "![L4 hierarchy: 48 physical spins, 16 triangles, 4 superblocks, "
            "then a reduced-space solve.](attachment:rg-schedule.svg)\n\n"
            "Keep h and Gamma fixed. Trial schedule: retain m=4 states per "
             "triangle, then chi=4 per 2-by-2 group of triangles. Solve the "
             "remaining interacting superblocks together.\n\n"
            "| Lattice | Full model: spin sites | First effective model: triangle sites | Second effective model: superblock sites |\n"
            "| --- | ---: | ---: | ---: |\n"
            "| L2 | 12 | 4 | 1 |\n"
            "| L4 | 48 | 16 | 4 |\n"
            "| L6 | 108 | 36 | 9 |\n"
            "| General even L | 3L² = N | L² = N/3 | L²/4 = N/12 |\n\n"
            "Each triangle site represents 3 physical spins and has m states; "
            "each superblock site represents 12 physical spins and has chi states. "
            "These effective sites have local dimensions m and chi, respectively. "
            "The total Hilbert-space dimensions are 2^N → m^(N/3) → chi^(N/12).\n\n"
             "| chi | L4: chi^4 amplitudes | L6: chi^9 amplitudes |\n"
            "| ---: | ---: | ---: |\n"
            "| 4 | 256 | 262,144 (2 MiB/vector) |\n"
            "| 8 | 4,096 | 134,217,728 (1 GiB/vector) |\n"
            "| 16 | 65,536 | Outside routine L6 budget |\n\n"
            "Use matrix-free iteration; multiple vectors and workspaces add memory.",
            attachments={"rg-schedule.svg": {"image/svg+xml": schedule_svg()}},
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 2. Block Hamiltonian and RG map

Choose disjoint triangles covering every site. Assign full on-site fields:

$$H_t=\sum_{\langle ij\rangle\subset t}K_{ij}Z_iZ_j
-h\sum_{i\in t}Z_i-\Gamma\sum_{i\in t}X_i,
\qquad H(\lambda)=\sum_t H_t+\lambda V.$$

$V$ contains intertriangle bonds; $\lambda=1$ is the original physical model.
At $\lambda=0$, retain the required triangle and superblock ground-state
subspaces to recover the decoupled endpoint; match any degeneracy convention.
Diagonalize each triangle and retain its lowest m states:

$$H_t|a_t\rangle=\epsilon_{ta}|a_t\rangle,\qquad
U_t=(|0_t\rangle,\ldots,|m-1_t\rangle),\qquad U=\bigotimes_t U_t.$$

$$H^{(1)}=U^\dagger H U
=\sum_t\operatorname{diag}(\epsilon_{ta})
+\lambda\sum_{\mathrm{inter}\ \langle ij\rangle}
K_{ij}\bar Z_{ti}\otimes\bar Z_{uj},\qquad \bar Z_{ti}=U_t^\dagger Z_iU_t.$$

At each later merge, diagonalize the internal superblock Hamiltonian and retain
chi states. Transform every interaction:

$$H^{(\ell+1)}=W_\ell^\dagger H^{(\ell)}W_\ell,
\qquad \mathcal U=U W_1\cdots W_s,\qquad
|\psi_{\rm approx}\rangle=\mathcal U|\phi_{\rm top}\rangle.$$

Keep general operator matrices. Projection may introduce positive off-diagonal
elements; a direct eigensolver can handle them.
"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 3. Retention and truncation error

Test m=2,4,6,8, then chi=2,4,8. Track cutoff gaps, boundary leakage, and
physical-observable errors. Preserve unresolved multiplets.

$$\Delta_t(m)=\epsilon_{t,m}-\epsilon_{t,m-1},\qquad
w_{\rm disc}=1-\langle\psi_{\rm exact}|\mathcal U\mathcal U^\dagger|\psi_{\rm exact}\rangle.$$

Internal and boundary bonds have comparable strength: local gaps alone cannot
justify truncation. Compare larger retained spaces and shifted partitions.
Weak coupling is useful relative to discarded-state excitation energies:

$$\eta=\frac{\|QH_{\rm boundary}P\|_2}{\Delta_{\rm connected}},
\qquad P=\text{retained projector},\quad Q=I-P.$$

Use connected retained/discarded denominators at each RG level. Nonpositive
or unresolved denominators invalidate this diagnostic; small eta is not an
error bound. Weak bare bonds alone do not control near-degenerate states.
If boundary entanglement limits accuracy, test density-matrix-selected states:

$$\rho_B=\operatorname{Tr}_{\rm env}|\Psi\rangle\langle\Psi|,\qquad
\rho_B|v_a\rangle=p_a|v_a\rangle,\qquad
w_\rho=1-\sum_{a<\chi}p_a,$$

retaining the largest $p_a$. Validate this refinement separately.
"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 4. Preserve physical-spin overlaps

$$q=\frac1N\sum_i z_i^{(a)}z_i^{(b)},\qquad
p(z)=|\langle z|\mathcal U|\phi_{\rm top}\rangle|^2,$$

$$P(q)=\sum_{z,z'}p(z)p(z')\,
\mathbf 1\!\left[q=N^{-1}\sum_i z_i z_i'\right],\qquad
\langle q^2\rangle=\frac1{N^2}\sum_{ij}\langle Z_iZ_j\rangle^2.$$

Transform physical products directly:

$$O_{\rm eff}=\mathcal U^\dagger O\mathcal U,\qquad
\mathcal U^\dagger Z_iZ_j\mathcal U
\ne(\mathcal U^\dagger Z_i\mathcal U)(\mathcal U^\dagger Z_j\mathcal U)
\quad\text{in general}.$$

Small systems: reconstruct p(z) and compute P(q) by XOR convolution.
L4/L6: validate sequential joint Born sampling through the hierarchy.
Independent block marginals lose interblock correlations.
"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 5. Finite validation schedule

| Stage | Comparison | Fixed cases |
| --- | --- | --- |
| A: algebra | Full-space recovery at m=8; operator and probability identities | Both triangle orientations |
| B: first truncation | Exact periodic L2, N=12, versus m=2,4,6,8 | 8 disorders; Gamma=.55,.60,.65,.70,1; lambda=0,.25,.5,1; both orientations |
| C: transfer | Exact open N=15 patch | 4 disorders; Gamma=.55,.70,1; lambda=1 |
| D: second truncation | Two coupled superblocks; chi=2,4,8 versus uncompressed first-level model | N=24; m=4; reference dimension 4^8=65,536; 4 disorders; Gamma=.55,.70,1 |
| E: required L4 | Periodic N=48 versus full-spin RBM and Jastrow GFMC separately | Gamma=.55,.56; disorders60000–60007; h=.15; lambda=1 |

Stage D tests additional RG error relative to the first-level model.
Its reference is not the full physical 24-spin solution.

At lambda=1 require, per case:

$$|\delta E|/N\le10^{-3},\qquad
|\delta\langle q^2\rangle|\le0.01,\qquad
\operatorname{TV}(P,P_{\rm ref})=\tfrac12\sum_q|P(q)-P_{\rm ref}(q)|\le0.05.$$

Also require maximum two-point error <=.05 and connected-spectrum error
<=max(10% of the larger magnitude, .001). Aim for half these limits to leave
room for later errors; resolve sampling uncertainty below the limits.
"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 6. Evidence and proposed continuation

**Completed current pilot:** original K~Uniform(0.5,1.5) on every bond, lambda=1,
Gamma=0.5 and h=0.15. The L4/L6 diagnostic completed on AutoDL:
eight disorders60000–60007 per size, (m,chi)=(4,2),(6,2),(6,3), 48 cases,
one CPU worker, no calculation failures or diagnostic alarms. Normalized physical
histograms receive equal disorder weights; bars show disorder SE only. Paired
retention changes preserve the shared disorders. At L6, increasing chi=2 to 3
at m=6 changes q² by +0.029764 ± 0.015917 and energy/spin by
-0.015096 ± 0.002590 (disorder SE). The dedicated local report is
`triangle_effective_l6_results.ipynb`. This
supersedes adaptive-lambda and higher-field choices for active work. Small
chi is preliminary; physical convergence and scaling are still unresolved.

**Completed L6 field scan:** Gamma=0.1,0.25,0.5,0.75,1.0 at the same original
bonds, h=0.15 and lambda=1. All eight disorders and all three retention
settings are matched across fields: 120 cases with 24 reused at Gamma=0.5,
one CPU worker. Report physical P(q), direct moments, paired field changes
and retention sensitivity. Every plotted bar is disorder SE. The dedicated
notebook is `triangle_effective_l6_gamma_scan.ipynb`. All 120 cases and their
raw archives, source snapshots, physical measurements, averages and paired
differences passed local verification, with no diagnostic alarms. Physical
convergence remains unresolved.

**Completed:** the five-schedule L4 pilot at Gamma=.55, disorder60000 fails the
energy screen (best discrepancy .05352/spin versus Jastrow). The higher-field
study completed 40 cases at Gamma=.60,.65,.70,.85,1.00, disorders60000–60007,
(m,chi)=(6,8), plus five (8,8) checks. It captures the field trend but shifts
P(q) toward lower overlap: TV of disorder means versus Jastrow is .139–.167.
The single-disorder m=8 checks change P(q) by TV=.018–.030 at fixed chi=8;
test chi convergence next. L4 used direct Lanczos eigensolves followed by
independent physical Born draws. GFMC comparisons retain guide disagreements
and ancestry alarms; they are not exact ground truth.

1. **Keep the original L2 model as the fixed reference.** Compare exact
   lambda=1 with RG at lambda=0,.1,.3,.5,.7,.85,1, m=4,6,8 and
   Gamma=.25,.5,1,2, at h=.15 using the same base bonds. At RG lambda=.7,
   m6 distribution TV errors versus the original model are .16068,.16632,
   .03727,.01209 in Gamma order. Keeping m8 isolates the effect of changing
   bonds; it does not recover the original model for lambda<1. Weaker bonds
   can reduce truncation while increasing target-model bias. Assess total
   target error and field response before selecting follow-ups.
2. **Use the completed all-bond scan as context.** Use $K=sK^{(0)}$, with
   $K^{(0)}\sim U(.5,1.5)$ and $s=1,.5,.2,.1$, using matched underlying draws.
   Scan Gamma=.05,.10,.25,.50 and h=0,.025,.075,.15: one 4-by-4 L4/N48 grid
   per bond range, exact L2 companions, and (m,chi)=(4,8),(6,8),(6,16).
   Every bond receives the same factor; lambda=1. Benchmark matched L4 models,
   leakage, physical correlations and partition stability before qualification.
   Divide energy tolerances by mean K so changing units cannot improve a screen.
3. **Search within the reliable region.** Refine Gamma and h at fixed bond law with
   equal disorder weighting. Inspect physical P(q), tails/secondary peaks and
   connected observables. At h=.15, polarization gives a raw overlap background:

   $$C_{ij}=\langle Z_iZ_j\rangle-\langle Z_i\rangle\langle Z_j\rangle,
   \qquad \chi_{\rm SG,conn}=\frac1N\sum_{ij}[C_{ij}^2]_{\rm dis}.$$

   Define connected Binder ratios and $\xi_L/L$ consistently from physical
   connected fields. Histogram shifts alone do not establish a transition;
   leave transition versus crossover open at h=.15.
4. **Scale only after convergence.** Use physical $L$, $N=3L^2$, identical
   blocking depth/rules and matched disorder laws. Increase m,chi as needed.
   Require truncation shifts below the size dependence and crossing uncertainty;
   check crossing drift, partition dependence, disorder resampling and fit-window
   stability across several sizes. Separate disorder, measurement, solver and
   truncation errors. Two sizes or two disorders cannot establish an exponent.

At lambda<1 the weakened intertriangle bonds define a **modified model**.
Hold that physical interpolation fixed when comparing partitions; changing
triangle orientation also changes its Hamiltonian. If supported, follow a
transition line $\Gamma_c(\lambda)$ toward lambda=1. Weakening bonds may remove
collective ordering, and required retention can grow near criticality.
Uniformly rescaling all energy scales offers no ground-state improvement:

$$H(aK,a\Gamma,ah)=aH(K,\Gamma,h),\qquad a>0,$$

so the ground-state wavefunction and physical P(q) remain unchanged.

**Completed complementary intertriangle screen:** a 4-by-4 L4/N48 P(q) grid with rows
Gamma=.25,.5,1,2 and columns lambda=0,.1,.3,1, at h=.15 and disorder60000.
Compare (m,chi)=(4,8),(6,8),(6,16), with an exact L2 companion. Overlay fields
at each lambda to identify where P(q) shifts while retained-space sensitivity
is small. Only lambda=.1, Gamma=1,2 pass the provisional screen, and their
mean-q change is -.00344 with sampling SE .00239. Larger low-field shifts
fail the chi energy screen. No broad shifting-and-stable interval is established.
This preserves a useful diagnostic alongside the primary all-bond scan.
Full-space exact L4 curves exist here only at lambda=0. Candidate intervals
need matched L4 benchmarks and disorder/partition checks before qualification.
For q_EA, declare the estimator and its longitudinal-field background before
fitting. A shifted P(q) alone does not establish collective glass order.

**Execution status:** the all-bond scan is complete on AutoDL (64 exact
L2 contexts and 192 L4 retained models, one disorder); none of its 64 contexts
passes the full provisional retention screen. Zero-field solvers use even
spin-flip parity to avoid arbitrary mixtures of near-degenerate states.
The intertriangle scan and its 16-context exact L2 comparison are complete.
[Exact L2 versus RG](triangle_effective_interaction_scan.ipynb) is the current
primary benchmark. Full Stage
B/C/D grids and the 320-case original L4 grid remain unlaunched. The bounded
48-case L4/L6 disorder pilot above is complete. Run
scientific computations on AutoDL; preserve disorder arrays, seeds, isometries,
raw measurements, failures and provenance.

[Full protocol, Section 9.1](../code/transformer_qmc/TRIANGLE_EFFECTIVE_THEORY_PLAN.md#91-proposed-research-path-weak-coupling--transition-search--finite-size-scaling).
Amorphous geometry remains an independent option.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "research_plan": {
                "status": "Fixed original-model exact L2 versus varied-coupling RG is primary; scans complete; L4 convergence before gradual L6 diagnostics and scaling",
                "presentation": "concise mathematical research plan",
                "protocol_sha256": digest,
                "scientific_compute_platform": "AutoDL",
            },
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell.id = f"triangle-rg-plan-{index:02d}"
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
