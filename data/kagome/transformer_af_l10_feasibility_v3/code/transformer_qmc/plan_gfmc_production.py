"""Build an executed, local-only notebook describing the staged GFMC rollout."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path,
                        default=Path('notebook/guided_gfmc_production_plan.ipynb'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text.strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(text.strip()))

    md(r'''
# Staged production plan: frozen-transformer GFMC

**Draft dated 27 October 2026. Primary goal: finite-size scaling and locating the phase boundary in $(\Gamma,h)$ under PBC. Qualify the projector and establish affordable, converged measurements across several sizes before concentrating the budget on disorder precision.**

The production program is designed to estimate $\Gamma_c(h)$, or $h_c(\Gamma)$ where identifiable, with uncertainty from sampling, projection, population size and finite-size corrections. Its supporting outputs are two-dimensional maps of $[\langle q^2\rangle]_D$, finite-size $[q_{\mathrm{EA}}]_D$ and explicitly defined susceptibilities. These maps guide targeted refinement and document the physics; locating a transition requires measurements across sizes. The longitudinal-field scan has seven values over $0\leq h/K\leq0.3$, matching the seven transverse-field values. The frozen guide's efficiency is assessed by the cost of obtaining converged scaling observables.

**Size allocation for review:** target $L=4,5,6,7,8,9,10$ (48, 75, 108, 147, 192, 243, 300 spins) on common transition-region cuts. Use $L=4,6,8,10$ as anchor sizes for the broader $7\times7$ observable maps, subject to feasibility. All clusters have PBC in both directions. Qualify odd sizes and check ordering-pattern compatibility before combining their scaling data. Converged seven-size cuts take priority over completing four full maps.

This notebook is a plan with locally executed evidence and workload calculations. All future run counts, thresholds and settings below are proposals. Numerical settings become production settings only after their qualification gates pass. Running this notebook reads local artifacts and performs small calculations.
''')

    code(f'''
from pathlib import Path
import json
from IPython.display import display, Markdown, Image

ROOT = Path({str(root)!r})
RESULTS = ROOT / 'data/kagome/transformer_af_disorder_gfmc_v1'
manifest = json.loads((RESULTS / 'manifest.json').read_text())
audit = json.loads((RESULTS / 'audit.json').read_text())
assert audit['passed'] and audit['complete']
assert audit['checkpoint_sha256'] == manifest['checkpoint_sha256']
display(Markdown(
    f"**Completed baseline:** L={{manifest['L']}}, N={{manifest['n_sites']}}, "
    f"D={{len(manifest['sample_ids'])}}, R={{len(manifest['seeds'])}}; "
    f"{{audit['populations']:,}} population runs, {{audit['caches']}} exact/guide caches, "
    f"{{audit['finite_forward']}} finite-forward reference contexts. Final audit passed.\\n\\n"
    f"**Frozen checkpoint:** `{{manifest['checkpoint_name']}}`  \\n"
    f"SHA-256: `{{manifest['checkpoint_sha256']}}`"
))
display(Image(filename=str(ROOT / 'figures/q2_vs_gamma.png'), width=1000))
''')

    md(r'''
## 1. What the completed study establishes

The 12-spin campaign covers $\Gamma/K=0.25,0.5,0.75,1,1.5,2,3$ and $h/K=0,0.3$, with 32 fresh disorders and four population repeats per point. It compares constant and frozen-transformer guides against exact ground states. The first eight disorders also have population, equilibration, forward-time and branching-cadence checks.

The guided $\langle q^2\rangle$ means closely follow the exact curve across much of the scan. The lowest-$\Gamma$ point retains a visible discrepancy. Larger populations gave the most consistent improvement; agreement in energy alone did not qualify the correlations. These findings motivate production work, but do not establish larger-size convergence.

At $h=0$, finite-size spin-flip symmetry forces our $q_{\mathrm{EA}}$ target to zero. The signed overlap estimator is sensitive to imperfect cancellation. Squared overlap is insensitive to independent global flips of the replicas; its improved agreement does not prove full sampling of the overlap distribution. Keep $q_{\mathrm{EA}}$ as a symmetry/sampling diagnostic.

The previous GFMC propagation used a complete lookup of the frozen network on the 12-spin Hilbert space. **That timing cannot be extrapolated to 48, 108, 192 or 300 spins.** Existing direct-inference pilots through $L=8$ establish that execution is possible; their short times and small populations do not qualify ground-state accuracy. The proposed $L=10$ extension still needs its own execution and cost benchmark. Reuse existing records for cost estimates only after checking their actual completion and configuration.
''')

    md(r'''
## 2. Fixed scientific specification

$$
H_J=\sum_{\langle ij\rangle}K_{ij}Z_iZ_j-\Gamma\sum_iX_i-h\sum_iZ_i,
\qquad K_{ij}\overset{\mathrm{iid}}{\sim}\mathcal U[0.5,1.5],\quad K=1.
$$

Use the kagome lattice with periodic boundaries and $N=3L^2$. Keep the bond width $w=0.5$ and use the full Cartesian product of these proposed production grids:

| Scan axis | Values in units $K=1$ | Number of values |
|---|---|---:|
| $\Gamma/K$ | $0.25,0.5,0.75,1,1.5,2,3$ | 7 |
| $h/K$ | $0,0.05,0.10,0.15,0.20,0.25,0.30$ | 7 |

This gives 49 field contexts per disorder, size, method, setting and independent repeat. The proposed $h$ spacing is $0.05$, preserving the original endpoints and adding five interior fields. Hold each bond realization fixed across both $\Gamma$ and $h$, methods, repeats and convergence settings. Identical sample IDs at different $L$ do not imply identical or nested bonds: the current disorder seed includes $L$.

Freeze the selected seed501 checkpoint recorded above. The main production claim concerns projection with this fixed guide. If retraining becomes necessary, make it a separate version, qualify it on development disorders, then freeze it before final evaluation. The earlier guide was trained at $L=2,4,6,8$; these sizes test performance on new disorders. The additions $L=5,7,9,10$ test unseen-size generalization and require separate guide and projector qualification.

| Role | Observable | Interpretation |
|---|---|---|
| Primary | $Q_2(J)=\langle q^2\rangle_J=N^{-2}\sum_{ij}\langle Z_iZ_j\rangle_J^2$ | Overlap magnitude and pair correlations |
| Primary companion at $h>0$ | $\chi_{\mathrm{conn},J}=N^{-1}\sum_{ij}\mathcal C_{ij}^2$ | Correlations beyond field-induced polarization |
| Numerical check | $E_J/N$ | Mixed energy estimator; assess alongside correlation convergence |
| Sampling check | $\langle q\rangle_J=N^{-1}\sum_i\langle Z_i\rangle_J^2$ | Zero at finite $N$, $h=0$, $\Gamma>0$ |
| Required scaling development | $\langle q^4\rangle$, connected overlap structure factors | Qualify dimensionless crossing diagnostics before final production |

Here $q=N^{-1}\sum_i Z_i^{(a)}Z_i^{(b)}$ and $\mathcal C_{ij}=\langle Z_iZ_j\rangle_J-\langle Z_i\rangle_J\langle Z_j\rangle_J$. All susceptibilities in this plan are equal-time quantities. The baseline $1/N$ is the diagonal contribution to $Q_2$ and its value for independent unbiased spins; it is not a phase boundary. At $h>0$, $Q_2$ also contains polarization contributions.

Report the equal-time overlap susceptibility $\chi_{\mathrm{SG}}^{\mathrm{eq}}=NQ_2$ alongside $\chi_{\mathrm{conn}}^{\mathrm{eq}}=N^{-1}\sum_{ij}\mathcal C_{ij}^2$, with their definitions attached to every map. Neither is automatically an imaginary-time-integrated response susceptibility. The finite-size $q_{\mathrm{EA}}=N^{-1}\sum_i\langle Z_i\rangle^2$ vanishes at $h=0$ for the unique finite-size symmetric ground state and contains field-induced polarization at $h>0$. Its map is a supporting observable; a thermodynamic glass-order claim requires a separately specified order of limits and symmetry-breaking protocol. Signed finite-sample estimates may fluctuate below zero and must retain their uncertainty.

Average per-disorder observables after forming replica moments. Use the existing four-replica forward-walking estimator for connected quantities, rather than squaring a noisy sample covariance. Retain every forward cycle and every disorder, including poor-ancestry cases.
''')

    md(r'''
## 3. Rollout and decision points

| Stage | Scope | Main question | Deliverable and gate |
|---|---|---|---|
| A — direct-path qualification | $L=2$, existing exact development benchmark | Does direct network inference reproduce the validated projector? | Exact-error, time/population and cache-invalidation checks pass |
| B — feasibility | $L=4$ ($N=48$), 4 development disorders, 3 $\Gamma$, 3 representative $h$ | Can direct GFMC support meaningful populations and projection times at an acceptable cost? | Measured throughput, memory, ancestry and a priced Stage C |
| C — convergence | $L=4$, 8 development disorders | Which parameters qualify $Q_2$ and connected correlations? | One-at-a-time escalation plus combined stress test meets prespecified tolerances |
| D — first production | $L=4$, 64 fresh disorders initially, full $7\times7$ field grid | Is the disorder average precise and stable? | Audited curves and maps with statistical and systematic uncertainty; extend to 128/256 if justified |
| E — size expansion | Anchor maps at $L=4,6,8,10$; add $L=5,7,9$ on transition cuts | Does the method remain feasible and converged as $N$ grows? | Qualify unseen sizes and cluster compatibility; repeat B/C for each admitted size/context |
| F — primary inference | Target seven qualified sizes $L=4,5,6,7,8,9,10$, matched field cuts | Where is the transition, and how stable is its extrapolation? | Crossing drift, scaling fits and a phase-boundary estimate with statistical and systematic uncertainty, or a stated unresolved region |

Advance on evidence, rather than a fixed calendar. A stage can yield a smaller validated domain: for example, publish converged moderate/high-$\Gamma$ results while low-$\Gamma$ points remain visibly unresolved. Predeclare this convergence-based rule and retain those unresolved points in the coverage report.

Reserve compute for qualifying all seven target sizes and for transition-region refinement before expanding the $L=4$ disorder count beyond 64. Begin size-transfer and cost checks once the direct path is qualified; odd-size qualification need not wait for completion of the anchor maps. The target largest system is $10\times10$ kagome unit cells, $N=3\times10^2=300$, with PBC in both directions. Use the PBC $L=2$ exact study as a numerical anchor; test explicitly whether it belongs in any asymptotic fit. Four anchor sizes can locate a candidate transition, but offer limited support for simultaneous exponent and correction fits. Seven sizes allow stronger size-window and compatibility tests without guaranteeing asymptotic scaling. If $L=10$ is only affordable on selected cuts, prioritize those common cuts across sizes and report the available size range explicitly.
''')

    md(r'''
## 4. Stage A: qualify the scalable implementation

Use eight of the existing 12-spin benchmark disorders as development cases and all 49 production field contexts. Reuse their existing exact references at $h=0,0.3$. Inventory compatible earlier references at the five interior fields; calculate the missing exact references and frozen-guide lookup tables before qualifying those contexts. At most $8\times7\times5=280$ additional exact/guide contexts are needed for this development subset. Store this extension in a new campaign directory with its own provenance.

1. Verify direct log amplitudes, all single-spin flip ratios, local energies and initialization probabilities against the frozen lookup for the same Hamiltonian. Exercise state-cache eviction and changes in $J$, $\Gamma$ and $h$. A cache must be scoped to the complete guide/context identity.
2. Compare constant-guide and direct-guide propagation on selected short cases. Use matched random streams where numerically practical; otherwise compare independent ensembles with uncertainty. Preserve four physical replicas per walker group and the same descendant weighting as the reference code.
3. Validate any throughput optimization at $L=2$ before applying it at larger $L$. Batching and exact memoization may change cost; reduced precision, clipped rates or modified resampling can change numerical behavior and need explicit checks.
4. On the hardest development points, extend the current ladder to $W=8192,32768$, $\tau_{\mathrm{eq}}=32,64,128$, and $\tau_{\mathrm{FW}}=8,16,32$ as needed. Change one parameter at a time, starting from the existing results; then test the chosen settings jointly. Start with four independent runs, adding repeats when the convergence comparison is too noisy to resolve.

**Gate A:** direct-versus-lookup numerical checks pass, the completed original audit remains reproducible, and correlation errors against exact are quantified under the proposed ladder. Persistent low-$\Gamma$ error triggers further diagnosis; it does not invalidate already qualified points. If new code changes the algorithm, create a new source snapshot and repeat the relevant checks.
''')

    md(r'''
## 5. Stage B: a bounded 48-spin feasibility pilot

Use four development disorders, $\Gamma=0.5,1,2$, representative fields $h=0,0.15,0.3$, both guides, and four independent population runs. This initial configuration contains $4\times3\times3\times2\times4=288$ runs before escalations. Begin with $W=64$ four-replica walker groups, $\tau_{\mathrm{eq}}=2$, energy-measurement time $1$, $\tau_{\mathrm{FW}}=1$, four cycles and branch interval $0.1$. This is an execution/cost probe. Escalate representative contexts to $W=256$ and $1024$, plus longer projection, before extrapolating production cost. Benchmark $h=0$ separately because the symmetric guide requires extra network work; include $h=0.05$ in the escalation checks to probe the small nonzero-field regime.

Record elapsed time for initialization, direct network queries, propagation and reporting; GPU memory; unique/requested state queries; effective states/second; event counts; and ancestry versus forward time. Include both cold and warm cache behavior. Count $W$ as the number of four-replica groups, not the number of individual configurations.

Price the full algorithm, including guide inference and setup. Compare guided and unguided methods in two ways: identical populations/times to expose estimator behavior, and equal end-to-end compute to assess practical benefit. At $L=2$ use exact RMSE; at $L=4$ use precision and convergence evidence, explicitly acknowledging the absence of exact ground-state truth.

**Gate B:** the measured cost projects to a feasible Stage C within a recorded GPU-hour and memory budget, with room for repeats and failures. A production runner needs its own configuration, atomic outputs and budget cap; the existing hard-coded short-pilot runner is only a starting point. If $W\sim10^3$ and longer forward times are unaffordable, improve inference or projection before expanding disorders. A large count of short, collapsed runs cannot replace this gate.
''')

    md(r'''
## 6. Stage C: qualify populations, projection and ancestry at 48 spins

Use eight development disorders: four fixed in advance and four added by a documented difficult-case rule based on runtime/ancestry, not agreement with a desired curve. Start at $\Gamma=0.5,1,2$ and $h=0,0.15,0.3$. Include $\Gamma=0.25$ and $h=0.05$ as stress cases before admitting them to the qualified scan. Extend the convergence comparisons to the remaining $\Gamma=0.75,1.5,3$ and $h=0.10,0.20,0.25$ before applying a setting policy to their combinations. All 49 contexts require recorded qualification evidence; apparent smoothness in $h$ does not establish convergence.

| Parameter | Candidate ladder | What it diagnoses |
|---|---|---|
| Walker groups $W$ | $512,2048,8192$; extend only if feasible | Population bias and loss of independent lineages |
| Equilibration $\tau_{\mathrm{eq}}$ | $16,32,64$ | Initial-state dependence and incomplete projection |
| Forward time $\tau_{\mathrm{FW}}$ | $4,8,16$ | Residual trial-state dependence of replica observables |
| Branch interval $\Delta\tau$ | $0.1,0.05$; add $0.025$ when needed | Finite-population sensitivity to resampling cadence |
| Forward cycles / repeats | $16,32$ cycles; $R=4$, then $8$ if needed | Statistical precision; cycles may be correlated |

These are candidate brackets, not certified values transferred from 12 spins. Start from the strongest feasible configuration suggested by B. Escalate each parameter separately, then combine the chosen larger population and longer times to detect interactions. Add checkpoints along a run where practical, preserving the same target definition.

Inspect mixed/growth energies, $Q_2$, $\chi_{\mathrm{conn}}/N$, overlap signs and initial-state dependence. Use guide-initialized runs and independent alternative initial states, with their initialization protocol recorded. At $h=0$, monitor the overlap distribution and spin-flip symmetry even if $Q_2$ looks stable. A symmetry-restoring move is a possible later algorithm change, requiring separate qualification.

Report descendant ESS, unique ancestors and largest-lineage fraction for **each** forward cycle and their distributions over disorders. As an initial alarm, any cycle with descendant ESS below 8 or a lineage fraction above 0.5 triggers review and a stronger run; these are failure alarms, not accuracy certificates. Never drop an unfavorable cycle to improve agreement. If longer forward time improves bias but collapses ancestry, increase population or change the separately validated algorithm.
''')

    md(r'''
## 7. Quantitative qualification and precision targets

Set the tolerances before evaluating the final disorders. The following are provisional absolute tolerances, in units $K=1$:

| Observable | Desired final disorder-average SE | Maximum unresolved numerical shift |
|---|---:|---:|
| $Q_2$ | $0.005$ | $0.002$ |
| $\chi_{\mathrm{conn}}/N$ | $0.005$ | $0.002$ |
| $E/N$ | $0.002$ | $0.001$ |

For each escalation form per-disorder differences between the stronger and candidate settings, averaging independent repeats within each disorder. Estimate the uncertainty of the mean difference using paired disorder resampling and independent-run variation. Pairing refers to the same Hamiltonian; random seeds do not make two different algorithms' trajectories identical.

An operational equivalence gate for observable $O$ is

$$
|\overline{\Delta O}|+1.96\,\operatorname{SE}(\overline{\Delta O})\leq\epsilon_O.
$$

Use this as a prespecified approximate 95% criterion, checking bootstrap intervals and repeat stability when the distribution is skewed. Require it for the final two relevant population/time levels and the combined stress test, at each qualified field context. If only two levels have been measured, add a third before claiming a plateau. If the interval is too wide, gather more repeats; an insignificant difference is not an equivalence result. Tolerance changes require a new documented target, never a silent relaxation after seeing results.

Check per-disorder absolute shifts and tails as well as their signed average, so opposite biases cannot cancel unnoticed. Broad confidence intervals from eight pilot disorders can require a larger qualification subset. At $L=2$, also compare directly to exact truth. At larger sizes this procedure establishes empirical stability over the tested range, not a rigorous upper bound on ground-state bias.

On the final sample, preselect at least eight audit disorders and repeat the stronger settings; expand that subset when numerical tails appear. Stronger runs on flagged disorders supplement this fixed subset. Report a separate systematic-sensitivity envelope, aiming for numerical uncertainty below half the achieved statistical SE. Tightening the statistical SE later can therefore require stronger numerical qualification.
''')

    md(r'''
## 8. Stage D: a fresh disorder average and defensible error bars

Once C passes, freeze the setting policy, source tree and checkpoint. Allocate a new development pool and final pool in a manifest after checking existing IDs. A suggested candidate allocation is IDs `60000:60032` for development and `61000:61256` for up to 256 final disorders; these ranges require a repository-wide collision/training-split check before reservation. Existing IDs `40000:40008` were reserved for another purpose and remain reserved.

Begin with the first 64 final disorders at $L=4$, four independent runs per point, all 49 combinations of the seven $\Gamma$ and seven $h$ values, and the qualified settings. Prioritize completing all matched contexts for a disorder. Add disorders in prespecified blocks of 64, up to 256, if the target precision is unmet. This is a sequential precision policy; report each checkpoint and the stopping rule. Treat ordinary pointwise SE/intervals as descriptive under that stopping policy unless a sequentially valid procedure is implemented.

For the main large-system campaign, run the guided method throughout. Retain both guides across the full $7\times7$ grid on a fixed 16-disorder subset of the final pool for matched and equal-cost comparisons. Enlarge the baseline subset if the comparison is inconclusive. The code cell below also prices a full two-method design. Fix the subset before reading its results. Changes to either field grid based on a pilot must be frozen before collecting the final pool; a later denser scan becomes a separately declared extension.

Apply the full survey design to the four anchor sizes as resources allow. At $L=5,7,9$, start with the common selected transition cuts, using 64 fresh disorders and four repeats per admitted point plus the same 16-disorder unguided comparison policy. Specify the unique $(\Gamma,h)$ contexts in a manifest so intersecting cuts are counted once. Choose cuts from development evidence, then freeze them before final confirmation; compare sizes at the same field values.

Generate an independent propagation RNG for each $(L,J,\Gamma,h,\mathrm{method},\mathrm{setting},r)$ using a stable, stored seed mapping. Keep disorder generation separate. The old campaign reused seed labels across contexts; its disorder bars condition on those streams. Fresh independent streams simplify the next campaign's uncertainty accounting.

For independent disorder and run streams,

$$
\widehat O_J=\frac1R\sum_{r=1}^{R}\widehat O_{J,r},\qquad
\overline O=\frac1D\sum_{J=1}^{D}\widehat O_J,\qquad
\widehat{\operatorname{SE}}(\overline O)=\frac{\operatorname{sd}_J(\widehat O_J)}{\sqrt D}.
$$

The last expression includes finite-run Monte Carlo scatter. Show a separate estimate of that contribution,

$$
\widehat{\operatorname{Var}}_{\mathrm{MC}}(\overline O)
=\frac{1}{D^2}\sum_{J=1}^{D}\frac{s_J^2}{R},
$$

where $s_J^2$ is the sample variance across the independent run means. Do not add this again to the disorder-level variance above. If estimating the intrinsic disorder variance, subtract the average within-disorder Monte Carlo variance with explicit finite-sample uncertainty. If MC dominates, buy more independent repeats or longer validated measurements; if disorder scatter dominates, buy more disorders.

Use paired disorder bootstrap intervals for method differences and derived ratios, retaining the entire $(\Gamma,h)$ grid for a disorder together. Label intervals as pointwise; claims across the full map require simultaneous uncertainty assessment. Use independent run-level variance to assess MC contamination, avoiding naive nested resampling that counts it twice. Inspect distributions and leave-one-disorder-out sensitivity. Retain rare disorders at equal weight; rerun difficult cases to reduce uncertainty without selecting by observable value.
''')

    md(r'''
## 9. Workload and budget worksheet

The following counts cover the full $7\times7$ grid and are a planning envelope for one production setting per point. They exclude development ladders and the stronger final-sample audit. Compared with the original two-field design, there are $7/2=3.5$ times as many production runs at each disorder count. This is a run-count multiplier; the GPU-hour multiplier depends on measured field-specific timings. A population run contains four physical replicas per walker group. Any field context awaiting qualification remains a proposed run, not a certified result.

Use the feasibility pilot to measure end-to-end seconds per run separately by size, field, $\Gamma$ and method. For a serial GPU schedule,

$$
T_{\mathrm{GPU\ hours}}\approx\frac{1}{3600}\sum_{c\in\mathrm{contexts}}n_c\,t_c.
$$

Include guide loading/setup, convergence ladders, the final audit and retries. Initially reserve 30% contingency on top of measured estimates and display the underlying timing quantiles. If jobs share a GPU, measure aggregate throughput under that exact concurrency; summing overlapped wall times is not device-hours. The relative pilot cost is more useful than a calendar promise at this stage.
''')

    code('''
# Explicit proposed production grids; the manifest above describes the completed baseline.
gammas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
fields = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
repeats = 4
baseline_disorders = 16
display(Markdown(
    f'**Proposed production grid:** {len(gammas)} Gamma values × '
    f'{len(fields)} h values = {len(gammas)*len(fields)} contexts per disorder.  \\n'
    f'Gamma/K: `{gammas}`  \\n'
    f'h/K: `{fields}`'
))
lines = [
    '| Final disorders per size | Guided runs | Extra unguided runs (16 disorders) | Recommended total | Full two-method total |',
    '|---:|---:|---:|---:|---:|'
]
for disorders in (64, 128, 256):
    guided = disorders * len(gammas) * len(fields) * repeats
    baseline = baseline_disorders * len(gammas) * len(fields) * repeats
    lines.append(f'| {disorders} | {guided:,} | {baseline:,} | {guided+baseline:,} | {2*guided:,} |')
display(Markdown('\\n'.join(lines)))
anchor_sizes = [4, 6, 8, 10]
additional_cut_sizes = [5, 7, 9]
scaling_sizes = sorted(anchor_sizes + additional_cut_sizes)
initial_per_size = (64 + baseline_disorders) * len(gammas) * len(fields) * repeats
anchor_total = len(anchor_sizes) * initial_per_size
display(Markdown(
    f'**Anchor-map envelope:** {len(anchor_sizes)} sizes '
    f'(L = {anchor_sizes}) × {initial_per_size:,} runs/size '
    f'= {anchor_total:,} runs. '
    'The full L=10 survey is conditional on feasibility and convergence; '
    'qualification, audit and transition-region refinement are additional.'
))
per_context = (64 + baseline_disorders) * repeats
cut_rows = [
    '| Unique field contexts C on common cuts | Seven-size cut campaign alone | Extra L=5,7,9 runs | Anchor maps + extra sizes, if cuts lie on survey grid |',
    '|---:|---:|---:|---:|'
]
for contexts in (7, 14, 21):
    added = len(additional_cut_sizes) * per_context * contexts
    cut_rows.append(
        f'| {contexts} | {len(scaling_sizes)*per_context*contexts:,} | '
        f'{added:,} | {anchor_total+added:,} |'
    )
display(Markdown('\\n'.join(cut_rows)))
display(Markdown(
    '**Illustrative counts:** C is the number of unique field pairs, not the number of cuts. '
    'For shared cuts containing C_off new points outside the survey grid, add '
    f'{len(anchor_sizes)*per_context:,} × C_off anchor-size runs to the last column. '
    'Count every size/context once; qualification, stronger audits and extra precision runs are additional.'
))

size_rows = ['| L | N | Uncorrelated baseline 1/N | Status in this plan |',
             '|---:|---:|---:|---|']
for L in [2] + scaling_sizes:
    sites = 3*L*L
    status = 'Completed exact benchmark' if L == 2 else (
        'First production target, conditional on gates' if L == 4 else (
            'Target largest size; unseen-size qualification required' if L == 10
            else ('Transition cuts; unseen-size and compatibility checks' if L in additional_cut_sizes
                  else 'Anchor maps and transition cuts; conditional expansion')))
    size_rows.append(f'| {L} | {sites} | {1/sites:.6f} | {status} |')
display(Markdown('\\n'.join(size_rows)))

# Fill from the Stage B/C timing records before scheduling production.
# Keys should identify L, h, Gamma and method; values are end-to-end seconds/run.
measured_seconds_per_run = {}
display(Markdown('**GPU-hour estimate pending:** larger-system production timings have not been inserted.'))
''')

    md(r'''
## 10. Stage E/F: size expansion and physical conclusions

Develop the anchor sizes $L=4,6,8,10$ alongside the additional cut sizes $L=5,7,9$, repeating the cost and convergence gates at each size. Keep the same disorder distribution and PBC, with common field points on each scaling cut. New sizes may require more walkers, longer times or a different validated projector; do not carry settings over solely because the energy looks reasonable. If the frozen guide becomes too costly or inaccurate, finish the validated size/domain and develop a separately versioned improvement.

### Seven-size design and cluster compatibility

Four sizes provide useful exploratory crossings but only limited tests of finite-size corrections. Neighboring-size crossings share measurements and are correlated; six adjacent-size crossings from seven sizes are also not six independent estimates. The primary target is a stable critical-field estimate. Report critical exponents only if size-window and fit-form tests support them; do not simultaneously fit a flexible scaling function and unconstrained correction exponents merely because seven sizes are available.

Before pooling sizes, verify site/bond counts, periodic connectivity, allowed reciprocal momenta and consistent physical lattice geometry. Check whether candidate conventional ordering wavevectors fit each periodic cluster. Compare odd/even sequences and other relevant commensurability classes, using disorder-averaged structure factors and scaling residuals. Random bonds do not automatically remove finite-cluster compatibility effects. Set the admission/analysis rules using development data and physical geometry, then freeze them before final confirmation. If sequences differ, analyze compatible families or a justified correction model; preserve and report every size rather than dropping inconvenient results after seeing the boundary.

The additions $L=5,7,9$ require frozen-guide construction, ratio/cache and observable checks as well as the usual convergence ladders. Successful even-size inference does not qualify odd sizes. Use alternative initial states and independent runs to diagnose slow-mode or cluster effects near the crossing region.

Start with common cuts suggested by the anchor survey, add the three intermediate sizes at those field values, and refine the crossing brackets across all admitted sizes. Test increasing minimum size, leave-one-size-out fits, compatible-family fits, and the influence of $L=10$. Report how the inferred critical field and any exponent move under each test. If only one compatibility family has enough useful sizes, report that effective size count. More points within $4\leq L\leq10$ improve these tests but do not extend the maximum length: persistent drift among the largest sizes may require systems beyond $L=10$ or an unresolved extrapolation.

### Gate for the $10\times10$ PBC target

1. **Check unseen-size construction and estimator arithmetic.** Verify 300 sites, periodic nearest-neighbor connectivity and the intended disorder reconstruction. Load the frozen checkpoint without changing parameter shapes; check finite log probabilities, samples, all-spin flip ratios, context changes and cache eviction. Compare selected flip ratios with direct log-amplitude differences. Check integer accumulation and overlap/moment normalization at 300 spins; an overflow or a sequence-length assumption can invalidate a large-size run even when the model loads.
2. **Measure H800 feasibility early.** Start with four development disorders at $\Gamma=0.5,1,2$ and $h=0,0.15,0.3$, using bounded short jobs and the Stage B protocol. Benchmark inference chunk sizes and worker concurrency anew at $L=10$. Measure initialization, all-flip query cost, complete event propagation, GPU/RAM peaks and ancestry. Escalate feasible populations and projection times before estimating production cost. Memory fit alone is insufficient; local proposals require many network evaluations and long-time convergence may dominate cost.
3. **Qualify ground-state observables.** Repeat the population, equilibration, forward-time and cadence ladders on selected cuts, including the candidate transition region and small nonzero fields. Verify fourth moments and connected structure factors, alternative starts and independent repeats. The new size lies outside the guide's training sizes, so quantify guide quality and efficiency on development cases; neither successful inference nor smooth energy establishes correlation convergence.
4. **Prioritize useful scaling data.** First fund matched transition brackets at $L=4,5,6,7,8,9,10$, using a fixed initial 16-disorder development subset at $L=10$ for qualification. Final confirmation uses the separately reserved fresh pool, initially 64 disorders per admitted cut with four repeats, expanding as precision requires. If the 49-context survey is unaffordable, publish a clearly partial $L=10$ map and invest in converged crossing cuts. Never reduce projection quality merely to fill the map.
5. **Admit or defer the size explicitly.** Include $L=10$ in boundary fits only after its numerical sensitivity is below the crossing-error budget. Compare fits with and without the smaller sizes and with and without $L=10$. If qualification fails, record the failure and use the qualified size range while optimizing or separately retraining the guide. Seven target sizes still require restrained fit complexity and checks of scaling assumptions.

The worksheet prices four anchor maps and selected cuts at the three added sizes. For $C$ unique field contexts at one size, the initial 64-disorder guided campaign plus the 16-disorder unguided comparison costs $(64+16)\times4\times C=320C$ runs. Seven-size cuts alone cost $2240C$ runs; adding $L=5,7,9$ to matching existing anchor data costs $960C$ runs. New refined field points also require anchor-size runs, at $1280$ per new common field pair. These counts exclude qualification, stronger audits and extra precision runs. Measure seconds/run at each size before converting counts into GPU-hours.

Plot $[Q_2]_D$, finite-size $[q_{\mathrm{EA}}]_D$, $[\chi_{\mathrm{SG}}^{\mathrm{eq}}]_D$ and $[\chi_{\mathrm{conn}}^{\mathrm{eq}}]_D$ against $\Gamma$ at each of the seven $h$ values, and against $h$ at each of the seven $\Gamma$ values, for each qualified size. Add maps over $(\Gamma,h)$ with unresolved contexts visibly marked, and show each size's $1/N$ reference for $Q_2$ where useful. Add uncertainty on changes with size. The $7\times7$ grid is a survey; any proposed transition region needs a prospectively declared denser grid and convergence checks there.

At $h=0$, a candidate dimensionless crossing diagnostic to qualify before production is

$$
g_L=\frac12\left(3-\frac{[\langle q^4\rangle]_D}{[\langle q^2\rangle]_D^2}\right).
$$

This is a ratio of disorder-averaged moments, not an average of per-disorder ratios. Record the convention and obtain its uncertainty by paired disorder resampling. Qualify fourth moments independently: they may require more samples than $Q_2$. Connected structure-factor ratios provide another candidate, after momentum and lattice conventions are checked. At nonzero field use a separately defined connected diagnostic; the uncentered Binder ratio includes polarization.

Finite-size crossings or a large $Q_2$ alone do not establish a spin-glass phase. Examine drift, finite-size corrections and possible conventional magnetic order using appropriate spin structure factors before making a glass interpretation. A thermodynamic $q_{\mathrm{EA}}$ claim also needs a stated symmetry-breaking/order-of-limits protocol, beyond the finite-size signed-overlap estimator used here.

### Phase-boundary inference protocol

1. **Define the crossing observables first.** Qualify the Binder convention above at $h=0$ and a connected overlap structure-factor ratio at finite field. One candidate is $R_L=1-S_{\mathrm{conn}}(\mathbf k_{\min})/S_{\mathrm{conn}}(0)$, where $S_{\mathrm{conn}}(\mathbf k)=N^{-1}\sum_{ij}\mathcal C_{ij}^2\exp[i\mathbf k\cdot(\mathbf r_i-\mathbf r_j)]$. Specify kagome basis positions, reciprocal primitive vectors and equivalent minimum-momentum directions; validate translation, reality, normalization and the four-replica estimator against exact results. Do not import a square-lattice correlation-length formula without deriving its geometry factors. Check the theoretical scaling of the equal-time quantity at this quantum transition before assuming a dimensionless ratio has size-independent crossings; a time-integrated diagnostic may require additional estimator development.
2. **Survey common cuts across sizes.** Use fixed-$h$ cuts in $\Gamma$ and fixed-$\Gamma$ cuts in $h$ on the shared PBC grid. Inspect candidate crossing intervals, conventional magnetic order and distribution tails. A smooth map or a finite-size susceptibility peak alone does not locate a critical point. Allow for no transition in the scanned domain, first-order behavior, or a crossover.
3. **Refine crossing brackets prospectively.** Add common field values for every participating size where the coarse scan suggests crossings. Choose spacing so interpolation uncertainty is smaller than the target critical-field uncertainty, and check it with an additional refinement. Freeze the refinement rule and use fresh confirmation disorders to assess choices made using development data. Agree on the desired uncertainty in $\Gamma_c(h)$ or $h_c(\Gamma)$ after pilot slopes and costs are available; the observable tolerance $0.002$ alone is not a critical-field error budget.
4. **Fit crossing drift and size dependence.** If a continuous transition and the chosen scaling form are supported, examine $R_L(t)=F(tL^{1/\nu})+L^{-\omega}G(tL^{1/\nu})$, with $t=\Gamma-\Gamma_c(h)$ or $h-h_c(\Gamma)$. Use the linear extent $L$, not $N=3L^2$, as the scaling length. Avoid unconstrained multi-parameter fits to only a few sizes. Vary the smallest included size, field window, correction terms and interpolation method, and compare multiple independently motivated diagnostics. Report instability when these choices move the inferred boundary substantially.
5. **Propagate covariance through the full inference.** Resample each disorder's entire field grid and moment set together, recomputing ratios, interpolation, crossings and fits in each replicate. Current disorder generation gives independent samples across sizes, so bootstrap each size independently; pairing numerical sample IDs across $L$ would impose false covariance. Account for independent-run MC uncertainty without counting it twice. Report pointwise boundary intervals and any simultaneous band distinctly, plus numerical-convergence and fit-choice sensitivity.
6. **Requalify near the candidate transition.** Increase projection/equilibration times and populations as size grows; slow modes and reduced gaps can require more stringent settings precisely where the boundary is inferred. Confirm that numerical shifts in the inferred crossing are below its declared error budget. Ancestry, alternative initial states, fourth moments and connected estimators all need checks at the refined points. Ground-state projection times must be converged separately at every size; fixed times do not establish the zero-temperature limit.

The release target is a table of candidate critical fields versus $h$ (or versus $\Gamma$), the sizes and observables used, crossing brackets, uncertainty components and fit stability. Include unresolved regions explicitly. The two-dimensional observable maps accompany this table and the underlying finite-size curves.
''')

    md(r'''
## AutoDL execution strategy for the scaling program

**Read-only resource check, 27 October 2026:** the instance has an NVIDIA H800 PCIe with 81,559 MiB GPU memory, a 20-core CPU quota and a 100 GiB RAM cap. At the check it had zero GPU memory allocated and 0% GPU utilization; the data volume had about 28 GiB free. This is a point-in-time observation, not ongoing monitoring or a claim that production has started.

Use AutoDL as the main compute platform. Keep a persistent, resumable queue of scientifically useful, qualified work so completed tasks release resources directly to the next task. Prioritize direct-path checks and estimator development, then larger-size feasibility/convergence, matched size cuts, transition-region refinement, and additional disorder precision. Prepare CPU exact references, audits and local reports concurrently with GPU inference when measured CPU/RAM headroom permits.

- **Measure end-to-end throughput.** Profile GPU inference, CPU event propagation, transfers and I/O. Sweep inference batch/chunk sizes and benchmark one, two and four concurrent independent workers within measured memory limits. Choose concurrency by completed valid samples per device-hour and estimator precision per device-hour. Record utilization, peak GPU/RAM use, event rate and ancestry; high GPU utilization alone is not a scientific efficiency metric.
- **Feed the H800 efficiently.** Batch queries across walkers/replicas and compatible contexts, preserving Hamiltonian-specific conditioning and cache identity. Compare a shared inference service with independent workers if profiling shows CPU event simulation leaves the GPU idle. Validate batching, cache eviction, precision changes and RNG isolation against the 12-spin reference before accepting an optimization. Use CPU thread caps based on the actual 20-core quota, allowing room for existing processes and the supervisor.
- **Keep the pipeline supplied.** Maintain manifests for ready, running, completed, failed and awaiting-qualification tasks. Dispatch only tasks whose prerequisites pass; use independent stored seeds and atomic completion records. Increase concurrency only after checking other jobs and aggregate throughput. Record a heartbeat, last successful output, GPU samples and failure counts; surface stalled work and stop repeated failing configurations. Refill from the next qualified size/field/disorder task instead of rerunning completed contexts.
- **Preserve recoverability.** Stage each campaign and immutable source snapshot under `/root/autodl-tmp`, reuse the existing Python/CUDA environment, and run the supervisor with `nohup` or `tmux`. Checkpoint at bounded intervals and mirror results/logs locally. Estimate output growth before dispatch and retain disk headroom; the observed 28 GiB free is a planning constraint. Preserve resumable artifacts and record errors before retrying with a changed configuration.
- **Budget by the scientific milestone.** Record measured GPU-hours for qualification, multi-size crossing cuts, refinement and broad maps separately. Reallocate idle resources to ready work that improves a convergence gate or boundary uncertainty. A process budget cap pauses dispatch and saves state; the AutoDL instance stays powered on. Never attach an automatic instance shutdown to completion or idle detection.

The first execution deliverable is a tested dispatcher and measured H800 throughput for the qualified direct path, followed by a priced multi-size scaling schedule. Full production dispatch follows the numerical qualification gates in this notebook.
''')

    md(r'''
## 11. Reproducible execution and release criteria

Each new campaign receives its own directory, immutable manifest and source snapshot. Record lattice/bonds or their verified seed reconstruction, checkpoint SHA-256, dependencies/device/precision, independent seed mapping, all projector settings, cache policy, per-run timings, convergence decisions and failed attempts. Resume only against the same identity; changed settings get new record keys. Write outputs atomically and preserve diagnostics and partial progress.

Keep development, qualification and final inference artifacts distinguishable. Store each independent run and its cycle-level estimates, ancestry and energy traces. Extend the current audit to cover the new sizes and direct path: it currently assumes the completed 12-spin campaign. Verify expected/actual coverage, duplicate keys, nonfinite data, provenance and disorder reconstruction before releasing plots. An executed notebook must distinguish planned, running, complete and qualified results.

Launch bounded compute batches with a recorded budget, checkpoint/mirror schedule and stop/resume behavior. A process cap stops dispatch and preserves the instance for resumption. Production timing and fair-use choices depend on the selected compute platform and are finalized after B.

**A production result is ready when:**

- The direct-path checks and provenance audit pass, and every published context has a documented qualification status.
- Population, time and cadence sensitivity satisfy the declared tolerances; poor-ancestry cases are resolved or remain explicitly unresolved.
- Final-disorder statistical precision meets the declared target, or a budget-limited precision is reported honestly; systematic sensitivity is shown separately.
- Matched-method comparisons distinguish fixed-settings behavior from equal-cost efficiency and include all inference/setup costs.
- Figures, tabulated data and the executed report can be regenerated from the frozen artifacts; missing or difficult disorders are visible in coverage.
- Phase-boundary claims include multi-size crossing evidence, projection checks at the refined fields, covariance-aware uncertainty, fit-window/size-window sensitivity and checks for conventional order. Where these fail, report the finite-size data and unresolved boundary.

## 12. Immediate next implementation batch

1. Inventory the existing local direct-pilot records and reserve new disorder/seed pools after a collision and training-split audit.
2. Define and exact-check the scaling estimators, including fourth moments, connected overlap structure factors and conventional-order diagnostics; document the finite-field and quantum scaling assumptions.
3. Add a configurable direct-inference campaign runner and persistent AutoDL dispatcher with four-replica estimator parity, stable seed mapping, timing/ancestry records, atomic resume and a process budget cap. Complete Stage A checks for that runner and inference optimizations.
4. Run the bounded Stage B 48-spin pilot and benchmark H800 batching/concurrency. Price all seven sizes early, including unseen-size checks at $L=5,7,9,10$ and odd/even/ordering-pattern compatibility, then populate a budget that prioritizes seven-size crossing cuts and refinement over full anchor maps.
5. Freeze the Stage C qualification matrix and numerical tolerances across the $7\times7$ grid; use pilot slopes to set a critical-field precision target and choose production settings, disorder counts and the common-size refinement schedule.

The first milestone is **a validated direct projector with qualified scaling observables and a priced seven-size schedule**, anchored by the 48-spin pilot and targeting up to 300 spins. The proposed 64-disorder survey at each anchor size contains 12,544 guided runs and 3,136 additional unguided runs, or 15,680 in total. Four full anchor maps would require 62,720 runs; selected cuts at $L=5,7,9$ add $960C$ runs for $C$ unique common field contexts, with refined points at anchor sizes costed separately. Qualification and audits are additional. Prioritize converged, compatible seven-size cuts and stable phase-boundary inference as the AutoDL queue is filled; full maps remain conditional on the remaining budget.

### Local evidence and implementation references

- [Completed disorder study](disorder_averaged_guided_gfmc.ipynb)
- [Baseline specification and artifact layout](../code/transformer_qmc/DISORDER_GFMC.md)
- [Frozen-guide and earlier direct-pilot documentation](../code/transformer_qmc/FROZEN_SCAN_GFMC.md)
- [Direct-inference pilot](../code/transformer_qmc/direct_gfmc_pilot.py)
- [Continuous-time projector and replica tags](../code/transformer_qmc/projector.py)
- [Disorder generation and exact observable definitions](../code/transformer_qmc/physics.py)
- [Baseline manifest](../data/kagome/transformer_af_disorder_gfmc_v1/manifest.json)
- [Baseline audit](../data/kagome/transformer_af_disorder_gfmc_v1/audit.json)
''')

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    NotebookClient(notebook, timeout=60, kernel_name='python3').execute()
    nbformat.validate(notebook)
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.notebook)
    print(args.notebook.resolve())


if __name__ == '__main__':
    main()
