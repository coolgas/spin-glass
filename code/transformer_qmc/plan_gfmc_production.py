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

**Protocol and execution record dated 27 October 2026. Primary goal: finite-size scaling and locating the phase boundary in $(\Gamma,h)$ under PBC. Qualify the projector and establish affordable, converged measurements across several sizes before concentrating the budget on disorder precision.**

The production program is designed to estimate $\Gamma_c(h)$, or $h_c(\Gamma)$ where identifiable, with uncertainty from sampling, projection, population size and finite-size corrections. Its supporting outputs are two-dimensional maps of $[\langle q^2\rangle]_D$, finite-size $[q_{\mathrm{EA}}]_D$ and explicitly defined susceptibilities. These maps guide targeted refinement and document the physics; locating a transition requires measurements across sizes. The longitudinal-field scan has seven values over $0\leq h/K\leq0.3$, matching the seven transverse-field values. The frozen guide's efficiency is assessed by the cost of obtaining converged scaling observables.

**Production-line size allocation:** include $L=4,5,6,7,8,9,10$ (48, 75, 108, 147, 192, 243, 300 spins) on common transition-region cuts. The anchor sizes $L=4,6,8,10$ all have the same full $7\times7$ production scope, computations and acceptance standards. All clusters have PBC in both directions. Qualify odd sizes and check ordering-pattern compatibility before combining their scaling data. Converged seven-size cuts take scheduling priority within the planned full anchor surveys.

**Rollout decision, 27 October 2026:** include $L=10$ as a full 300-spin production case under the same protocol as $L=4,6,8$. It receives all 49 field contexts, the same observable calculations, initial disorder/repeat allocation, guide comparisons, convergence tests, precision targets and audits. Every size advances through the same context-specific qualification gates before final-data collection and boundary fitting. The successful short $L=10$ benchmarks establish execution, not completion of those gates.

This notebook combines the agreed survey scope, locally executed evidence, workload calculations and a timestamped execution snapshot. Qualification has launched; final-disorder collection remains gated. The 64-disorder/four-repeat initial allocation is retained at every anchor size. Numerical settings become production settings only after their qualification gates pass. Running this notebook reads local artifacts and performs small calculations.
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

The previous GFMC propagation used a complete lookup of the frozen network on the 12-spin Hilbert space. **That timing cannot be extrapolated to 48, 108, 192 or 300 spins.** Existing direct-inference pilots through $L=8$ establish that execution is possible; their short times and small populations do not qualify ground-state accuracy. New bounded $L=10$ probes are documented in the measured feasibility assessment below. They establish execution and short-run cost only; affordable, converged production still requires qualification. Reuse existing records for cost estimates only after checking their actual completion and configuration.
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
| E — size expansion | Full anchor surveys at $L=4,6,8,10$ under one protocol; add $L=5,7,9$ on matched transition cuts | Does the method remain feasible and converged as $N$ grows? | Repeat B/C and the common acceptance criteria for every size/context; qualify unseen sizes and cluster compatibility |
| F — primary inference | Target seven qualified sizes $L=4,5,6,7,8,9,10$, matched field cuts | Where is the transition, and how stable is its extrapolation? | Crossing drift, scaling fits and a phase-boundary estimate with statistical and systematic uncertainty, or a stated unresolved region |

Advance on evidence, rather than a fixed calendar. A stage can yield a smaller validated domain: for example, publish converged moderate/high-$\Gamma$ results while low-$\Gamma$ points remain visibly unresolved. Predeclare this convergence-based rule and retain those unresolved points in the coverage report.

**Current $L=10$ readiness:** the v1/v2/v3 benchmarks establish completed execution and short-run throughput measurements. Float32 remains a candidate inference mode; ensemble-level precision equivalence, longer-time/population convergence, qualified crossing observables and a measured production budget are pending. Record qualification and final-data status separately for every size/context under the common policy. Ready contexts can advance while others are being qualified.

Reserve compute for qualifying all seven target sizes and for transition-region refinement before expanding the $L=4$ disorder count beyond 64. Begin size-transfer and cost checks once the direct path is qualified; odd-size qualification need not wait for completion of the anchor maps. The target largest system is $10\times10$ kagome unit cells, $N=3\times10^2=300$, with PBC in both directions. Use the PBC $L=2$ exact study as a numerical anchor; test explicitly whether it belongs in any asymptotic fit. Four anchor sizes can locate a candidate transition, but offer limited support for simultaneous exponent and correction fits. Seven sizes allow stronger size-window and compatibility tests without guaranteeing asymptotic scaling. Budget limits can pause any size/context; pending anchor contexts remain part of the full survey scope.

### Shared production protocol

| Requirement | Common rule, including $L=10$ |
|---|---|
| Field coverage | All 49 $(\Gamma,h)$ combinations at each anchor size; matched prospective crossing refinements across participating sizes |
| Final sampling | Initially 64 fresh disorders and four independent population repeats per context; the same precision-driven extension policy |
| Guide comparison | Frozen guide throughout, plus the same preselected 16-disorder unguided subset and fixed-settings/equal-cost comparisons |
| Computations | Energy, $\langle q^2\rangle$, finite-size $q_{\mathrm{EA}}$, explicitly defined equal-time susceptibilities, qualified fourth moments, connected structure factors and conventional-order diagnostics |
| Qualification | The same development-selection policy, population/equilibration/forward-time/cadence ladders, alternative starts, ancestry checks and combined stress test |
| Acceptance and uncertainty | The common Section 7 tolerances, qualified crossing-error targets, disorder/run uncertainty analysis and fixed final-audit subset of at least eight disorders |
| Precision and provenance | The same validated inference-precision policy, double-precision projector arithmetic, frozen checkpoint/source identity, atomic outputs and coverage audit |

Equal standards do not imply identical converged walker counts or projection times. Select settings independently at each size/context using the same ladders and acceptance criteria, escalating when needed. No size receives looser tolerances or reduced estimator coverage to fit a runtime target. All anchor sizes retain the full grid in their completion accounting, even when common transition cuts are scheduled first.
''')

    md(r'''
## 4. Stage A: qualify the scalable implementation

Use eight of the existing 12-spin benchmark disorders as development cases and all 49 production field contexts. Reuse their existing exact references at $h=0,0.3$. Inventory compatible earlier references at the five interior fields; calculate the missing exact references and frozen-guide lookup tables before qualifying those contexts. At most $8\times7\times5=280$ additional exact/guide contexts are needed for this development subset. Store this extension in a new campaign directory with its own provenance.

1. Verify direct log amplitudes, all single-spin flip ratios, local energies and initialization probabilities against the frozen lookup for the same Hamiltonian. Exercise state-cache eviction and changes in $J$, $\Gamma$ and $h$. A cache must be scoped to the complete guide/context identity.
2. Compare constant-guide and direct-guide propagation on selected short cases. Use matched random streams where numerically practical; otherwise compare independent ensembles with uncertainty. Preserve four physical replicas per walker group and the same descendant weighting as the reference code.
3. Validate any throughput optimization at $L=2$ before applying it to larger-size production. For the measured float32 candidate, retain float64 as the comparison implementation, exact-check the small-system estimators, then compare independent float32/float64 ensembles on representative larger-size development contexts, including $L=10$. Assess $Q_2$, connected correlations and the chosen crossing observables alongside energy using prespecified precision-equivalence tolerances. Preserve double-precision rate/weight arithmetic. Batching and exact memoization may change cost; reduced precision, clipped rates or modified resampling can change numerical behavior and need explicit checks.
4. On the hardest development points, extend the current ladder to $W=8192,32768$, $\tau_{\mathrm{eq}}=32,64,128$, and $\tau_{\mathrm{FW}}=8,16,32$ as needed. Change one parameter at a time, starting from the existing results; then test the chosen settings jointly. Start with four independent runs, adding repeats when the convergence comparison is too noisy to resolve.

**Gate A:** direct-versus-lookup numerical checks pass, the completed original audit remains reproducible, and correlation errors against exact are quantified under the proposed ladder. Persistent low-$\Gamma$ error triggers further diagnosis; it does not invalidate already qualified points. If new code changes the algorithm, create a new source snapshot and repeat the relevant checks.
''')

    md(r'''
## 5. Stage B: a bounded 48-spin feasibility pilot

Use four development disorders, $\Gamma=0.5,1,2$, representative fields $h=0,0.15,0.3$, both guides, and four independent population runs. This initial configuration contains $4\times3\times3\times2\times4=288$ runs before escalations. Begin with $W=64$ four-replica walker groups, $\tau_{\mathrm{eq}}=2$, energy-measurement time $1$, $\tau_{\mathrm{FW}}=1$, four cycles and branch interval $0.1$. This is an execution/cost probe. Escalate representative contexts to $W=256$ and $1024$, plus longer projection, before extrapolating production cost. Benchmark $h=0$ separately because the symmetric guide requires extra network work; include $h=0.05$ in the escalation checks to probe the small nonzero-field regime.

Record elapsed time for initialization, direct network queries, propagation and reporting; GPU memory; unique/requested state queries; effective states/second; event counts; and ancestry versus forward time. Include both cold and warm cache behavior. Count $W$ as the number of four-replica groups, not the number of individual configurations.

Price the full algorithm, including guide inference and setup. Compare guided and unguided methods in two ways: identical populations/times to expose estimator behavior, and equal end-to-end compute to assess practical benefit. At $L=2$ use exact RMSE; at $L=4$ use precision and convergence evidence, explicitly acknowledging the absence of exact ground-state truth.

**Gate B:** the measured cost projects to a feasible Stage C within a recorded GPU-hour and memory budget, with room for repeats and failures. A production runner needs its own configuration, atomic outputs and budget cap; the existing hard-coded short-pilot runner is only a starting point. If $W\sim10^3$ and longer forward times are unaffordable, improve inference or projection before expanding disorders. A large count of short, collapsed runs cannot replace this gate.

**Implemented continuation (27 October 2026):** after all 288 base runs and their supervisor complete, a shared-lock controller dispatches 144 representative escalation runs on fixed development disorder 60000, at $\Gamma=0.5,2$ and $h=0,0.05$, with both guides, float64 and four independent repeats. This selection targets low-field ancestry and high-field cost without selecting by observable agreement. It is a bounded feasibility subset; Stage C still requires eight development disorders.

| Escalation | W | Equilibration | Forward time | Coverage |
|---|---:|---:|---:|---|
| Small-field baseline | 64 | 2 | 1 | h=0.05; reuse base pilot at h=0 |
| Population level 1 | 256 | 2 | 1 | All four selected field pairs |
| Population level 2 | 1024 | 2 | 1 | All four selected field pairs |
| Equilibration check | 1024 | 4 | 1 | All four selected field pairs |
| Forward-time check | 1024 | 2 | 2 | All four selected field pairs |

Measurement time 1, four cycles and branch interval 0.1 are fixed throughout. Population comparisons and each time comparison change exactly one parameter. Independent seeds are checked across both queues. The controller preserves the original 17:46:47 UTC deadline, checkpoints at expiry and never automatically admits Stage C or final data.

`assess_stage_b.py` exports context coverage, independent-run/disorder uncertainty, all cycle-level ancestry and a provisional Stage C cost worksheet. W64-based linear cost extrapolation is an explicit scenario until checked against W256/W1024 and longer-time measurements. Equal-cost guide comparisons, event counts and separate cold/warm-cache and timing-component profiles remain required Gate B work; the current records do not provide all these diagnostics.
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

Apply the full survey design identically to $L=4,6,8,10$, with 49 contexts, initially 64 fresh disorders, four repeats and the same 16-disorder unguided comparison policy at each size. At $L=5,7,9$, retain the planned common selected transition cuts under the same per-context computations and numerical standards. Specify the unique $(\Gamma,h)$ contexts in a manifest so intersecting cuts are counted once. Choose cuts from development evidence, then freeze them before final confirmation; compare sizes at the same field values.

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
    'All four anchor sizes share the full survey design and qualification standards; '
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
        'Full 49-context survey and matched cuts; common production protocol' if L in anchor_sizes
        else 'Transition cuts; common numerical standards and compatibility checks')
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

### Applying the common protocol at $10\times10$ PBC

$L=10$ has the same full production scope and acceptance standards as the other anchor sizes. The bounded execution benchmarks are complete and should be reused as evidence; the work below applies the shared protocol at 300 spins. Preserve their source snapshots and records. These are size-specific checks within the common qualification process.

1. **Check unseen-size construction and estimator arithmetic.** Verify 300 sites, periodic nearest-neighbor connectivity and the intended disorder reconstruction. Load the frozen checkpoint without changing parameter shapes; check finite log probabilities, samples, all-spin flip ratios, context changes and cache eviction. Compare selected flip ratios with direct log-amplitude differences. Check integer accumulation and overlap/moment normalization at 300 spins; an overflow or a sequence-length assumption can invalidate a large-size run even when the model loads.
2. **Measure H800 feasibility early.** Start with four development disorders at $\Gamma=0.5,1,2$ and $h=0,0.15,0.3$, using bounded short jobs and the Stage B protocol. Benchmark inference chunk sizes and worker concurrency anew at $L=10$. Measure initialization, all-flip query cost, complete event propagation, GPU/RAM peaks and ancestry. Escalate feasible populations and projection times before estimating production cost. Memory fit alone is insufficient; local proposals require many network evaluations and long-time convergence may dominate cost.
3. **Qualify the same ground-state observables.** Apply the Stage C development-selection policy and the common population, equilibration, forward-time and cadence ladders to all 49 contexts, plus prospective crossing refinements. Verify fourth moments and connected structure factors, alternative starts and independent repeats. The new size lies outside the guide's training sizes, so quantify guide quality and efficiency on development cases; neither successful inference nor smooth energy establishes correlation convergence. Apply the same Section 7 tolerances and crossing-error criteria used for the other sizes.
4. **Complete the same final-data design.** Use the separately reserved fresh pool, initially 64 disorders and four independent repeats at every one of the 49 field contexts, with the same 16-disorder unguided comparison policy and stronger audit on at least eight preselected disorders. Extend by the shared precision rule. Schedule matched transition brackets first across $L=4,5,6,7,8,9,10$ while retaining the full anchor grids as required coverage. A budget-limited pause leaves outstanding contexts pending; it does not redefine $L=10$ as a cut-only or completed case.
5. **Promote qualified contexts explicitly.** Include $L=10$ in boundary fits only after its numerical sensitivity is below the crossing-error budget. Compare fits with and without the smaller sizes and with and without $L=10$. If a context fails qualification, retain it in the pipeline as unresolved and use the qualified size/domain while optimizing or separately retraining the guide. Seven target sizes still require restrained fit complexity and checks of scaling assumptions.

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

### Ancestry-onset hypothesis, exact P(q), and targeted refinement

**User-directed investigation:** retain the small-$\Gamma$ regime as a physics target. Test whether the onset of ancestry loss marks a difficult physical region by comparing it with exact small-system overlap distributions, then refine common field cuts around candidate physical crossings and attempt finite-size extrapolation. The suggestion that ancestry onset coincides with a phase transition is a hypothesis; sampler collapse also depends on the guide and numerical parameters.

For each disorder define the equal-time two-copy distribution
$P_J(q)=\sum_{s,t}|\psi_J(s)|^2|\psi_J(t)|^2\,\mathbf1[q=N^{-1}\sum_i s_i t_i]$.
Average distributions over equally weighted disorders. At $h=0$, keep the full signed distribution and verify spin-flip symmetry; at nonzero $h$, report polarization and connected diagnostics alongside $P(q)$. Small exact clusters show crossovers rather than thermodynamic singularities.

1. **Completed exact-distribution implementation:** `assess_overlap_onset.py` reuses 392 existing $L=2$, 12-spin exact ground states across eight disorders and 49 fields. It computes the full $P_J(q)$ using XOR convolution, checks its first/second/fourth moments against existing references and compares it with the frozen guide's unprojected $P_J(q)$. It exports CSV, PNG/PDF and Markdown with input hashes. This requires no new diagonalization.
2. **Onset scan after current escalation review:** at $h=0,0.05$, propose common $\Gamma=2,1.75,1.5,1.25,1,0.75,0.5$, four fixed development disorders, four independent repeats and $W=256,1024$. Hold equilibration2/measurement1/forward1/cycles4/branch0.1 fixed. Compare alarm fraction, ESS/W and lineage distributions. A diagnostic half-alarm crossing is reported only if measured points bracket it; inspect nonmonotonicity and dependence on $W$, projection time and guide. These are proposed development tasks, not automatically dispatched or qualified final data.
3. **Local interpolation and confirmation:** use the onset bracket to prioritize a physical-observable survey. At each fixed $h$, refine actual Binder/connected-ratio crossing brackets by common midpoint fields across all admitted sizes. Interpolate only within measured brackets; preserve multiple crossings and report missing ones. Refine again until field interpolation error is subdominant. Use fresh confirmation disorders after selecting brackets with development data.
4. **Finite-size extrapolation:** first inspect qualified even-size curves $L=4,6,8,10$, then compatible odd-size cuts $L=5,7,9$. Use $L$ as length. Exclude $L=2$ from the primary thermodynamic fit because of its short extent and potential large corrections, but show it as an exact benchmark. Fit candidate crossing drift $\Gamma_\times(L_1,L_2)=\Gamma_c+A\ell^{-\lambda}$ with $\ell=\sqrt{L_1L_2}$ as an explicitly phenomenological convention. Prefer fixed-ratio compatible pairs where coverage permits. Vary fixed positive $\lambda$ (initial sensitivity values 1,2,3), minimum size and pair family; do not infer an unconstrained correction exponent from sparse crossings. Do not interpret this drift convention as a derived universal law.
5. **Covariance and scaling:** resample each disorder's entire field curve and moment set together, bootstrap different sizes independently, recompute all shared-size crossings jointly and propagate their covariance through drift fits. Test $R_L=F((\Gamma-\Gamma_c)L^{1/\nu})+\cdots$ only if a continuous transition and the equal-time ratio's scaling are supported. The current narrow size range may not constrain $\nu$ or $\Gamma_c$ reliably; test alternative behavior and conventional order. Compare drift models and report numerical, statistical, interpolation and size-window sensitivity separately.
6. **Executable support and present limit:** `finite_size_crossings.py` implements bracketed crossings and conditional drift fits from joint bootstrap samples; independent synthetic tests check interpolation and recovery. Present data contain exact $N=12$ references and unqualified $N=48$ development runs. They do not yet support a measured physical crossing, a thermodynamic extrapolation or an exponent estimate. Record these as pending rather than fitting ancestry alarms as an order parameter.

The active base pilot and its bounded population/time escalations continue. Additional field and size tasks follow measured feasibility and numerical qualification. Full anchor-grid scope and common acceptance criteria remain unchanged.
''')

    md(r'''
## Measured L=10 feasibility evidence

The assessment below is generated from saved H800 measurements, with immutable source snapshots and checkpoint identity. Its cost scenarios are deliberately conditional. The full survey run counts in Section 9 remain scope calculations, not a funded or throughput-qualified schedule. Direct-inference optimization and validation must pass before dispatching the proposed large population/time ladders.

**Measured decision:** 300-spin PBC construction, checkpoint loading and short four-replica GFMC run successfully on the current H800. Inference is the dominant cost. At $\Gamma=0.5$ on one disorder, 32 walker groups evolved for total imaginary time $0.2$ in about 85 s ($h=0.3$) or 164 s ($h=0$) using float64. A separate float32-inference probe took about 4.2 s and 8.2 s, respectively, with small sampled rate differences but no long-run accuracy qualification. Preserve float64 as the comparison path; prioritize precision validation and measured throughput improvements before larger production allocations. Even a 20-fold speedup does not by itself price the full campaign: the old 12-spin population/time settings remain expensive under a simple extrapolation and have not been qualified at $L=10$.
''')

    code('''
assessment = ROOT / 'data/kagome/transformer_af_l10_feasibility_v1/ASSESSMENT.md'
if assessment.exists():
    display(Markdown(assessment.read_text()))
else:
    display(Markdown('Measured assessment not present locally; regenerate it from the saved feasibility records.'))
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

## 12. Execution handoff and remaining qualification

1. **Implemented:** local/remote disorder-ID collision checks and frozen-checkpoint training-split audit. Development IDs `60000:60032` and final IDs `61000:61256` use half-open ranges; final-disorder tasks remain gated.
2. **Implemented numerical estimators:** fourth moments, connected overlap spectra and conventional sublattice spin structure factors, with deterministic replica/Fourier identity tests. Establishing their physical scaling behavior and crossing-error targets remains part of qualification.
3. **Launched:** the resumable campaign runner and AutoDL dispatcher, with stable seed mapping, source/checkpoint verification, atomic outputs, branch-boundary RNG/state checkpoints, ancestry records and a 12-hour initial process window. Exact references and direct-guide checks cover eight development disorders across all 49 fields. Numerical precision and long-time convergence remain open gates.
4. **Corrected execution:** after the specified $L=10$ task completed, the preliminary short-probe queue was paused. The active queue now follows Section 5 exactly: $L=4$, $W=64$, equilibration 2, energy measurement 1, forward time 1, four cycles, branch interval 0.1, four development disorders, nine field pairs, two guides and four repeats (288 runs). Both methods use float64. Review cost and ancestry, then escalate representative cases to $W=256,1024$ and longer times, including $h=0.05$, before Stage C. Preserve the same eventual computations and standards at $L=10$ and the full anchor scope; the tiny probes are not Stage B completion.
5. **Implemented follow-up:** `research_followup.py` waits for the base pilot, then runs the Section 5 representative escalation matrix above using the same supervisor lock and deadline. The local watcher follows the handoff and refreshes the scientific review. Stage C admission remains pending measured feasible cost, completed profiling and review of ancestry.
6. Freeze the Stage C qualification matrix and numerical tolerances across the $7\times7$ grid; use pilot slopes to set a critical-field precision target and choose production settings, disorder counts and the common-size refinement schedule.

The first milestone is **a validated direct projector with qualified scaling observables and a priced seven-size schedule**, anchored by the 48-spin pilot and targeting up to 300 spins. The proposed 64-disorder survey at each anchor size, including $L=10$, contains 12,544 guided runs and 3,136 additional unguided runs, or 15,680 in total. Four full anchor maps require 62,720 runs; selected cuts at $L=5,7,9$ add $960C$ runs for $C$ unique common field contexts, with refined points at anchor sizes costed separately. Qualification and audits are additional. Prioritize converged, compatible seven-size cuts and stable phase-boundary inference as the AutoDL queue is filled; budget limits affect scheduling, and unfinished anchor-grid work remains visible in completion accounting.

### Local evidence and implementation references

- [Completed disorder study](disorder_averaged_guided_gfmc.ipynb)
- [Baseline specification and artifact layout](../code/transformer_qmc/DISORDER_GFMC.md)
- [Frozen-guide and earlier direct-pilot documentation](../code/transformer_qmc/FROZEN_SCAN_GFMC.md)
- [Direct-inference pilot](../code/transformer_qmc/direct_gfmc_pilot.py)
- [Continuous-time projector and replica tags](../code/transformer_qmc/projector.py)
- [Disorder generation and exact observable definitions](../code/transformer_qmc/physics.py)
- [Baseline manifest](../data/kagome/transformer_af_disorder_gfmc_v1/manifest.json)
- [Baseline audit](../data/kagome/transformer_af_disorder_gfmc_v1/audit.json)
- [Live campaign mirror and execution audit](../data/kagome/transformer_af_production_v1/STATUS.md)
- [Shared production protocol](../code/transformer_qmc/production_protocol.py)
- [Resumable production runner](../code/transformer_qmc/production_runner.py)
''')

    md(r'''
## Launched campaign: audited snapshot

AutoDL campaign: `/root/autodl-tmp/transformer-af-production-20261027-v1`. The original 27,328-task preliminary queue is paused at 11,267 completed tasks. Its large-size settings were shorter than Section 5 and do not satisfy the notebook's Stage B pilot or Stage C convergence requirements. It is retained as execution/precision evidence, not reported as progress through those gates.

**User-directed handoff, 27 October 2026:** the $L=10$, $\Gamma=1$, $h=0$ float64 probe (`b5758ffa8a5e6ee833cf4dc5`) completed before the handoff. `notebook_stage_b.py` now runs the exact Section 5 288-run pilot in `notebook_stage_b/`, using the frozen verified projector, float64, and the unchanged 17:46:47 UTC process deadline. `active_stage.json` selects this queue for monitoring. The original source, manifests, results and checkpoints are retained. The four anchor sizes still retain the same full 49-context final production scope.

Matched eight-task $L=10$ concurrency trials took 58.67, 59.87 and 60.43 seconds with one, two and four GPU workers, respectively. One GPU worker was selected; CPU reference, ensemble and unguided work runs concurrently. A completion-driven scheduler replaces an initial polling delay. Its queue-selection collision was corrected, the incident was archived, and valid completed results and checkpoints were preserved.

All 3,136 small-system precision ensemble runs and 784 precision-specific finite-forward references are complete. At this short setting, 40 of 147 approximate paired-disorder 95% bounds satisfy the declared tolerances; 107 remain unresolved. These pointwise approximate bounds do not establish long-time precision equivalence, and unresolved bounds do not establish a float32 bias. Float64 remains the comparison implementation.

**Authorized continuation, 28 October 2026:** the base pilot completed 288/288 and the escalation paused at 114/144 when the original window expired. `resume_stage_b.py` verified the frozen dependencies, task identities and available checkpoints, then resumed the 30 unfinished guided time checks in a separate 24-hour window, from 28 October 01:23:56 UTC to 29 October 01:23:56 UTC. The initial remaining-work estimate was 17.3 GPU-hours. `active_continuation.json` points to the separate launch record; the original launch and scientific manifests remain preserved. Terminal handling evaluates physical-observable convergence, saves Markdown/CSV/JSON/plots and records an outcome before the local watcher reports completion. Stage C admission remains pending.

The following is a timestamped local mirror, not a live connection. `watch_production.py` refreshes the mirror and structural audit. Resource telemetry, operational incident records, immutable scientific source snapshots and per-task progress remain available in the campaign directory. The initial process budget pauses at saved state and never powers off the instance.
''')

    code('''
production_status = ROOT / 'data/kagome/transformer_af_production_v1/STATUS.md'
if production_status.exists():
    display(Markdown(production_status.read_text()))
else:
    display(Markdown('Execution snapshot not mirrored yet; see the remote campaign status.'))
''')

    md('''
## Stage B research assessment

This timestamped assessment is regenerated from the mirrored base pilot and escalation records. Incomplete coverage is explicit. Disorder averages use complete sets of four independent repeats; one-disorder escalation uncertainty measures Monte Carlo variation only. Cycle observations remain diagnostics rather than independent repeats. Gate status depends on scientific evidence, not the number of completed tasks.
''')
    code('''
stage_b_review = ROOT / 'data/kagome/transformer_af_production_v1/STAGE_B_REVIEW.md'
if stage_b_review.exists():
    display(Markdown(stage_b_review.read_text()))
else:
    display(Markdown('Stage B scientific assessment has not yet been generated.'))
''')

    code('''
physical_review = ROOT / 'data/kagome/transformer_af_production_v1/PHYSICAL_CONVERGENCE.md'
if physical_review.exists():
    display(Markdown(physical_review.read_text()))
    physical_image = physical_review.parent / 'physical_convergence/guided_estimates.png'
    if physical_image.exists():
        display(Image(filename=str(physical_image)))
''')

    code('''
overlap_note = ROOT / 'data/kagome/transformer_af_production_v1/overlap_onset/RESEARCH_NOTE.md'
if overlap_note.exists():
    display(Markdown(overlap_note.read_text()))
    overlap_image = overlap_note.parent / 'exact_guide_pq.png'
    if overlap_image.exists():
        display(Image(filename=str(overlap_image)))
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
