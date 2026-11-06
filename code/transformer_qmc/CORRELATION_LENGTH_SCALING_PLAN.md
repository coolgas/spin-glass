# Kagome RG: correlation lengths and the accessible size frontier

> **Scientific priority updated 2026-11-05:** focus on finite-size q_EA and
> connected equal-time spin-glass susceptibility, with magnetization as
> polarization context. The current definitions, qualification tolerances and
> execution priorities are in `CORRELATION_LENGTH_SCALING_OUTLINE.md` and the
> generated `triangle_overlap_retention.ipynb`. Correlation-length objectives
> below document the earlier plan; they no longer define the primary target.

Planning notebook, 2026-11-04. This proposes a new study following the completed
L6 field scan. Writing this plan launches no simulations. Scientific evaluation,
timing and figure generation belong on AutoDL; the notebook is available locally.

**Updated immediate scope:** prioritize L4,L6,L8,L10 (48,108,192,300 spins)
as the four-size scaling campaign, using RG + DMRG to reach L8/L10. Larger-size
ladders below are future extensions. Measure R and Q_conn=S_R(0)/N together;
raw <q²> remains a comparison observable because h=0.15 produces a polarization
background. Four stable size curves support an exploratory joint scaling test,
with common critical field and correlation-length exponent, disorder bootstrap,
field-window checks and sensitivity to dropping L4. They do not alone establish
a transition or precise exponents. The concise notebook outline is the current
execution priority.

**Execution update:** the authorized common field grid is Gamma=0.1,0.2,...,1.5
(15 fields, step 0.1), with a common step-0.025 refinement interval if justified.
This supersedes the sparse five-field launch proposal below. Begin with archive
analysis and exact/DMRG validation, then the L8/L10 frontier. Campaigns run to
completion with deliberate resource limits; the proposed wall-clock caps below
are superseded by the standing AutoDL runtime policy. No machine shutdown.

## 1. Objective and fixed physical problem

Measure the equal-time connected-overlap correlation length versus transverse
field and system size. Establish the largest **numerically affordable** size and
the largest **observable-stable** size separately. These may differ substantially.

Use periodic kagome samples with N=3L² spins,

$$H=\sum_{\langle ij\rangle}K_{ij}Z_iZ_j-0.15\sum_iZ_i-\Gamma\sum_iX_i,
\qquad K_{ij}\overset{\mathrm{iid}}\sim U(0.5,1.5).$$

Keep lambda=1, all physical bonds, h=0.15, and the existing disorder generator.
Do not obtain larger sizes by weakening interblock bonds. Retention and block
partition choices are numerical controls, not changes to the target model.
The nonzero longitudinal field makes raw overlap sensitive to polarization;
connected-overlap lengths are the primary observable. A crossing or a transition
is a hypothesis, not an expected outcome that the analysis must produce.

**Expanded solver direction:** use the RG Hamiltonian as the input to a many-body
solver. Prioritize DMRG/MPS as the general-sign route; assess guided GFMC after
a concrete stoquasticity/sign-gauge audit. Compare shallow RG plus a stronger
solver with deeper RG plus a small exact top solve. The ambition is L12–L24,
with L32 a conditional stretch experiment, while preserving the distinction
between a finished calculation and trustworthy finite-size scaling.

## 2. Observable, geometry and normalization

For each disorder d, evaluate physical-spin expectations in the approximate
ground state:

$$m_i=\langle Z_i\rangle,\quad C_{ij}=\langle Z_iZ_j\rangle-m_im_j,
\quad S_{R,d}(\mathbf k)=\frac1N\sum_{ij}C_{ij}^2
\cos[\mathbf k\cdot(\mathbf r_i-\mathbf r_j)].$$

Include diagonal terms: C_ii=1-m_i². Use physical coordinates including the
three-site kagome basis, never just superblock centers. This is an equal-time
connected-overlap length, not an imaginary-time-integrated susceptibility or a
complete classification of magnetic/glassy order.

Nearest-neighbor distance is one. Primitive vectors are a1=(2,0), a2=(1,sqrt(3));
the physical side length is 2L. Use reciprocal vectors satisfying
a_i·b_j=2 pi delta_ij. The three shortest directions are b1/L, b2/L,
(b1+b2)/L, each of norm k_min=2 pi/(sqrt(3)L).

Average spectra over equally weighted disorders **before** taking their ratio:

$$\overline S_0=[S_{R,d}(0)]_D,\qquad
\overline S_{\min}=\frac13\sum_{a=1}^3[S_{R,d}(\mathbf k_a)]_D,$$
$$\xi_R=\frac1{k_{\min}}\sqrt{\frac{\overline S_0}{\overline S_{\min}}-1},
\qquad R=\xi_R/L.$$

This matches `correlation_lengths.length` and the established convention in
`FINITE_SIZE_GOAL.md`: xi_R is in nearest-neighbor spacings and R is xi_R/L,
which is twice xi_R/(2L). Label that convention on every plot. Use the continuum
second-moment prefactor consistently; any lattice-dispersion alternative must
be derived for this geometry and reported as a separate convention check.
The finite-size quantity is an estimator; an Ornstein–Zernike interpretation
is not automatically justified at all fields and sizes.

Save all three directional spectra and lengths before shell averaging. Block
partition anisotropy can be concealed by the shell average. S0<=0, Smin<=0 or a
negative ratio produces an unresolved length; never clip it to zero or drop the
disorder to make a plot look smooth. Also retain Q=1-Smin/S0 as a diagnostic
when the square-root estimator is ill-conditioned; Q has the same input spectra
and is not independent evidence.

## 3. First results can reuse the saved L4/L6 states

The completed L6 scan contains 120 cases: five fields (0.1,0.25,0.5,0.75,1.0),
eight disorders 60000–60007 and three schedules (m,chi)=(4,2),(6,2),(6,3).
The original-bond fixed-field pilot also contains 24 L4 cases at Gamma=0.5.
Their NPZ files already contain `magnetization`, `zz`, `points`, bonds and seeds.
Compute connected spectra directly from these arrays on AutoDL, preserving
input hashes. No new eigensolve or physical replica sampling is required for
this first pass. Preserve the source archives unchanged.

Deliver an initial diagnostic notebook with L6 R(Gamma) for all three schedules,
directional spectra, paired retention differences, and the L4/L6 comparison at
Gamma=0.5. These are pilot estimates from eight disorders. The existing zero
solver alarms and physical-energy consistency do not establish xi convergence;
energy and overlap have already shown retention sensitivity.

Check the observable independently on L2 (12 spins) at Gamma=0.1,0.5,1,2,
using disorders 60000 and 60001. Compare direct exact ground-state contractions,
the existing `exact_spectra`, and the new contraction. Check a product-state
limit, diagonal terms, physical-coordinate phases and all shortest momenta.
For L2 first-level RG use m=4,6,8; m=8 must recover the full physical result up
to numerical tolerance. L2 is an observable/implementation anchor, not evidence
that L6 or larger truncations are accurate.

## 4. Why the present implementation stops growing quickly

After triangles and 2x2 groups of triangles, the existing hierarchy has
B=(L/2)² superblocks and a dense state vector of dimension D=chi^B. Matrix-free
Hamiltonian application avoids a D² matrix but still needs vectors of length D.
The eigensolver uses multiple Krylov vectors. `physical_correlations` additionally
stores about N inserted vectors, an 8ND-byte float64 payload before copies.

The next cell tabulates exact dimension arithmetic. A 32-vector column is a
planning envelope, not a measured peak-RSS bound: eigensolver workspaces, term
contractions, retained operators, BLAS temporaries and measurements add memory.
The current solver does not explicitly fix ncv; freeze it in the future manifest.

{RESOURCE_TABLE}

Local superblock diagonalization is another cost: it diagonalizes an m^4 by m^4
dense matrix per block. One float64 matrix requires about 12.8 MiB for m=6 and
128 MiB for m=8, with additional eigensolver workspace. Increasing chi changes
the final solve dramatically even when local construction is affordable.

**Near-term expectation:** L8 (192 spins), chi=2 is the first practical step.
L8, chi=3 is a large-state feasibility benchmark, not a default production
setting. L10 (300 spins), chi=2 also needs measurement redesign and measured
timing. L10, chi=3 and L12 at the present depth are excluded from the initial
launch grid by their state dimensions. Available hardware has not been measured
for this plan; there is no promised wall time or maximum qualified L.

## 5. Remove the measurement bottleneck before enlarging the scan

Correlation lengths require four Fourier components of C_ij², not thousands
of overlap-histogram replica pairs. Use deterministic physical moments when
affordable; skip `joint_samples` in the new correlation-only production driver.
The existing generic `run_case` always draws replicas and is not the final
large-size driver without this change.

Implement tiled physical correlations: process limited blocks of inserted
vectors, contract cross-block pairs, and accumulate the four Fourier sums.
Keep m_i and the small N by N correlation matrix if affordable, but avoid N
simultaneous length-D vectors. Within a block, use W†(Z_i Z_j)W; multiplying
(W†Z_iW)(W†Z_jW) loses discarded-space contributions. Validate the tiled
routine against the current method on L4/L6 before increasing size. Record the
time/memory tradeoff, including repeated operator application.

If deterministic measurement becomes too expensive, a separately validated
four-independent-replica estimator is available: define
d_i=(z_i^a-z_i^b)(z_i^c-z_i^d)/2. Then E[d_i d_j]=C_ij², and the mean of
|sum_i d_i exp(i k·r_i)|²/N estimates S_R(k). This avoids squaring a noisy
estimate of C_ij. Validate against deterministic spectra; retain independent
batches and make measurement noise subdominant to disorder variation. Sampling
can still cost O(D) per conditional step and does not remove the top-state limit.

## 6. Staged size and field campaign

| Stage | Scope | Purpose and exit decision |
| --- | --- | --- |
| A: existing-data analysis | 120 saved L6 cases + 24 L4 fixed-field cases | First R curves; quantify retention and directional sensitivity without new solves |
| B: estimator validation | L2, 4 fields, 2 disorders; L4/L6 tiled checks | Exact observable agreement and projection-product tests |
| C: cost frontier | L8, then conditional L10; 2 disorders, Gamma=0.1,0.5,1.0 | Benchmark low/intermediate/high fields; measure complete cost and RSS |
| C2: effective-model solvers | DMRG/MPS on L4/L6, then L8/L10/L12; GFMC sign audit on the same small cases | Validate energies and physical spectra against exact reduced-space solves |
| D: matched pilot | L4,L6 and passing L8/L10; 8 disorders, 5 existing fields | R(Gamma), R(L), retention and partition comparisons |
| E: recursive prototype | L8,L12, then conditional L16,L24 | Validate extra depth and seek larger affordable states |
| E2: solver frontier | Best validated route at L16/L24, conditional L32 | Increase width and retention; keep torus scaling separate from cylinder pilots |
| F: scaling statistics | Only sizes/settings passing observable sensitivity; 32 then 64 disorders | Resolve size trends and conditional crossing analysis |

For stage C, begin with (m,chi)=(6,2) at L8. Try (6,3) only after memory and
matvec forecasts pass. The two schedules over two disorders and three fields
give at most 12 L8 cases. L10 starts with six (6,2) cases only if its full
solve and tiled measurement fit the budget. Do not label a large chi=2-only
run retention-converged. Obtain its second retention comparison with a validated
recursive route or keep it explicitly exploratory.

Stage D adds the missing L4 fields and reuses all existing matching cases.
Keep the same disorders across Gamma and numerical settings at each L. Size
ensembles are independent: identical sample IDs at different L do not mean
identical physical disorder. Eight disorders are for development/cost estimates.

For representative Gamma=0.1,0.5,1.0 and disorders 60000,60001, test m=4,6,8
at a fixed feasible chi, then chi at fixed m. At L6 include chi=4 and conditional
chi=6 cost/sensitivity probes before declaring the chi=2/3 comparison adequate.
Use both triangle orientations and block origins (0,0),(1,0),(0,1),(1,1) on
this audit subset. Fix the production partition before the full scan; report
partition shifts as numerical sensitivity, not extra independent disorders.

The first field grid remains 0.1,0.25,0.5,0.75,1.0. Add 1.5 and 2.0 as
high-field anchors if the pilot has not reached a short-range regime. Refine
candidate crossing intervals to spacing 0.05, then 0.025 only if uncertainties
and numerical stability warrant it. Every retained size gets the same refined
interval; preserve a broad grid to avoid selecting only apparent crossings.

## 7. Larger systems require recursive RG with physical operators

Extend blocking by grouping 2x2 retained blocks again. Let r count 2x2 blocking
levels after triangles. Then B_top=(L/2^r)² and D_top=chi_r^B_top. Retentions
are a schedule (m,chi_1,...,chi_r), not a single interchangeable chi. Carry every
projected boundary interaction and physical observable through all levels.

| Candidate | Spins | Blocking levels r | Top blocks | Example top dimension | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| L8 | 192 | 2 | 4 | chi_2=4: 256 | Compare extra depth with the current L8 hierarchy |
| L12 | 432 | 2 | 9 | chi_2=4: 262,144 | First larger recursive target |
| L16 | 768 | 3 | 4 | chi_3=4: 256 | Adds another truncation level; audit against r=2 where feasible |
| L24 | 1,728 | 3 | 9 | chi_3=4: 262,144 | Conditional stretch target after L12/L16 validation |

Small top dimensions do not imply accurate long-distance correlations. Fixed
small retention at each added level can erase interblock correlations and create
spurious saturation, crossings or scale dependence. Never infer attainable
physical accuracy from the table alone. Track truncation spectra and vary every
retention level independently where feasible. Block-energy cutoff gaps are
diagnostics, not error bounds for xi.

Do not form an explicit 2^(physical block size) decoder beyond the current
12-spin blocks. Represent isometries as a tree. Ascend single-site and pair
operators, combining physical products at their lowest common ancestor; use
tree contractions for physical observables and any later joint sampling.
Validate coarse Hamiltonian actions, Hermiticity, orthonormality and physical
expectations against explicit small cases and the existing two-level code.

Compare L8 at r=1 and r=2 under retention increases, and L16 at r=2 and r=3
where r=2 is affordable. There are natural fixed-ratio families (4,8,16) and
(6,12,24), but they change depth; report that fact on every scaling plot.
Use a common-depth subset such as (4,8,12) at r=2, with independent L4 r=1
validation, to expose depth effects. Mixed-depth data enter a shared scaling
fit only after these bridges support observable stability.

If recursive low-energy RG fails the xi stability checks, stop extrapolating
its size sequence. The next method-development decision is an improved retained
basis or a separately benchmarked many-body solver (for example an MPS on the
coarse lattice). A narrow-strip MPS has its own geometry and entanglement limits;
it is not an automatic substitute for two-dimensional square-torus scaling.

## 7A. DMRG on the effective Hamiltonian: preferred ambitious route

Replace the full top vector by an MPS over retained blocks. Initially use the
existing shallow hierarchy, retaining chi=2,3,4,6,8 local states as budgets permit;
denote the MPS bond dimension by **D_MPS** to avoid confusing it with local chi.
This preserves more local states without paying chi^B for a full state vector.
DMRG tolerates the effective Hamiltonian's mixed-sign matrix elements, although
optimization and entanglement remain difficult.

Build an MPO from the actual on-site and pair terms in `Operator.terms` using a
snake ordering of the two-dimensional block lattice. Retain all periodic wrap
bonds, coefficients and local matrices. An open-boundary MPS can represent a
periodic physical Hamiltonian by including its wrap interactions in the MPO;
it does not justify deleting those bonds. Freeze any MPO compression tolerance
and verify its action against the exact reduced Hamiltonian on random small
vectors. Different physical bonds can contribute to the same block pair; sum
them correctly rather than replacing them by one guessed coupling.

Use an established arbitrary-local-dimension DMRG implementation (for example
TeNPy or quimb, after checking installed support, precision and API) rather than
writing the optimizer from scratch. Pin its version in the run environment.
Start with two-site sweeps and a D_MPS ladder 32,64,128,256; extend to 512 and
1024 only after measured costs support it. These are trial budgets, not expected
requirements. Record MPO bond dimensions, discarded weights, entanglement
entropies, sweep energies, residual/energy variance where affordable, convergence
thresholds and both low-entanglement and randomized starts. Optimize and measure
at every ladder endpoint; a small last-sweep energy change is insufficient.

**Benchmark before scaling:** disorders 60000,60001 and Gamma=0.1,0.5,1.0 at
L4 and L6, initially (m,chi)=(6,2),(6,3): 24 model contexts, reusing exact reduced
states where available. Match energy per spin, S0, all three S(k_min), physical
m_i and representative C_ij. Target |delta R|<0.002 and <1% spectral differences
against the exact reduced state when resolved, plus energy/spin discrepancy
<1e-6. Near-zero spectra require absolute tolerances specified from the exact
values before judging a run. These solver-validation targets are tighter than
the production retention budget and do not measure RG truncation error.

Then use six contexts per frontier size (two disorders, three fields) at
L8,L10,L12, followed conditionally by L16,L24 and L32 (3072 physical spins).
At the shallow hierarchy these have 16,25,36,64,144,256 MPS sites, respectively.
The periodic coarse-lattice width, not just site count, drives difficulty.
Maintain a retention comparison at each candidate size, then D_MPS convergence
at each retention. First vary these separately; a larger D_MPS cannot repair
lost local states. Resource envelopes depend roughly on B, local dimension,
MPO dimension and powers of D_MPS, so replace generic asymptotic estimates with
measured sweep and observable costs before expanding.

Evaluate physical m_i and C_ij via MPS operator insertions and cached transfer
environments; preserve exact projected same-block products. Accumulate the
physical Fourier spectra, not coarse-pseudospin correlations. For recursive
blocks, combine this with tree operator ascent. Benchmark measurement cost as
well as optimization: N² physical pairs can dominate a nominally fast solve.

Compare shallow-RG DMRG to recursive-RG DMRG/exact on overlapping L8/L12 cases.
This directly tests whether extra blocking removes correlations that a stronger
many-body solver would preserve. If low-energy block bases remain unstable,
investigate environment-informed/density-matrix retained bases as a new method
variant with fresh provenance and validation, rather than simply increasing L.

Long cylinders may permit much longer systems, but they are a separate width
scaling study. For a cylinder use physical finite-width observables or a carefully
defined transfer-matrix length and its own boundary convention. Do not combine
cylinder lengths or an MPS transfer length with the torus second-moment xi_R/L
curve. An MPS transfer length in particular may describe a different operator
channel from the squared connected physical-spin correlations used here.

## 7B. Guided GFMC on the effective model: conditional complementary route

The physical transverse-field Ising model is stoquastic in the Z basis. A local
energy-basis RG projection need not be. For each small benchmark assemble H_eff
and quantify positive off-diagonal entries relative to a documented tolerance.
Test diagonal sign gauges by graph consistency: for every nonzero off-diagonal
edge, demand s_x s_y=-sign(H_xy). A contradictory cycle excludes a diagonal
sign fix for that assembled basis. For larger cases, a common local gauge that
makes all terms stoquastic is a sufficient certificate; failure of that test
does not by itself prove that every possible gauge or basis fails.

A positive guiding function changes importance weights but cannot remove an
off-diagonal sign obstruction. Do not feed an uncertified mixed-sign H_eff into
the existing positive-weight projector, delete positive entries, or substitute
absolute values. A signed-walker or fixed-node variant is a separate algorithm:
measure sign decay/variance or fixed-node bias against exact small cases and
give it its own qualification. It is not the default large-size production path.

If a useful retention/basis passes the sign gate, implement block configurations
x_b in {0,...,chi-1}, the full allowed local/pair transitions, transformed
observables and an importance-sampled projector. In the sign-free gauge,

$$\widetilde H_{yx}=\Psi_T(y)H_{yx}/\Psi_T(x),\qquad
E_L(x)=\sum_y H_{xy}\Psi_T(y)/\Psi_T(x).$$

Use a strictly positive, efficiently evaluable categorical guide: begin with a
product/pair-Jastrow guide, then compare a categorical autoregressive model.
Preserve finite support, guide amplitude ratios, independent seeds and raw
training checkpoints. Validate transition rates, diagonal/branching terms and
the zero-variance exact-guide limit against reduced-space ED. The existing
binary-spin projector cannot simply be reused without this adaptation.

Physical projected Z and ZZ operators are generally off-diagonal in the retained
basis. Sampling the retained labels and treating them as physical spins is wrong.
The mixed local estimator (O Psi_T)/Psi_T generally gives a mixed expectation,
not the required pure ground-state value. Standard diagonal-observable forward
walking is insufficient for arbitrary projected operators. Develop and validate
a pure insertion, two-sided projection, or explicit physical-decoding estimator
before using GFMC spectra for xi. This observable gate is as important as the
Hamiltonian sign gate.

Start from the same exact L4/L6 benchmark contexts as DMRG. Use a bounded ladder
of 512,1024,2048 walkers, projection/equilibration times T,2T,4T chosen by pilot
gap/correlation diagnostics, two starts and at least two independent repeats.
Separate population, finite-projection, guide and pure-estimator sensitivities.
Track effective ancestry, weight concentration and any sign statistics; none
alone proves convergence. Compare total cost to a resolved physical R against
DMRG and exact reduced solves, including guide training and measurement.

**Solver selection:** prefer DMRG if the sign or pure-observable construction
blocks positive-weight GFMC. Retain GFMC as a complementary validated method
where it works; agreement across genuinely different solvers is useful only
after each one's truncation/projection errors are assessed. The largest-size
campaign uses the route with the best measured accuracy-cost tradeoff, not a
commitment to running every solver at every size.

## 8. Resource gates and bounded execution

Before a launch, inspect AutoDL memory, CPU allocation and existing workloads.
The present NumPy/SciPy eigensolver is CPU based; an available GPU does not
automatically accelerate it. Start with one worker and one BLAS thread; preserve
other campaigns. Benchmark construction, one matvec, eigensolve, physical
measurement and serialization separately. Record RSS and Krylov ncv, iterations
or matvec count, two solver starts and residuals. No resource measurements are
claimed by this plan.

Use at most half of currently available RAM for the first probe's forecast,
then refine that forecast from measured peak RSS before expanding concurrency.
Run a cheap dimension/memory check before allocating a trial state. Prespecify
a 30-minute first-case wall-time cap and a 2-hour total frontier pilot; if a
case exceeds the cap, save the failure and revise the budget rather than
silently extending a full grid. These are proposed budget limits, not runtime
estimates. Do not bypass a failed memory forecast to test an obviously huge D.

Count successful, failed and qualified cases separately. The largest finished
size and largest xi-stable size are separate entries in the final report.

## 9. Numerical sensitivity and disorder uncertainty

Initial targets, to be revised against pilot costs before production:

* Paired retention/partition changes in R: magnitude below 0.01, with a 95%
  disorder interval sufficiently narrow to support that statement. Prefer an
  interval contained within [-0.01,0.01] for an observable-stability designation.
* Inspect S0 and each nonzero-momentum spectrum separately; aim for changes
  below 5% where spectra are well resolved. A stable ratio can hide common bias.
* Two solver starts must agree in energy and relevant physical observables;
  use residuals and investigate near-degenerate states. State fidelity alone can
  be misleading near degeneracy. Passing these checks qualifies the reduced solve.
* Start production with 32 disorders per size, growing to 64 in prespecified
  batches. Target a disorder-only 95% interval half-width <=0.02 for R, and
  tighten when adjacent-size differences require it. Counts are targets, not
  proof of sufficient disorder coverage.

Use 5,000 whole-disorder bootstrap resamples with a saved seed and index array.
Reuse the same resampled disorder indices across fields, momentum components,
retentions and partitions at a given L. Draw size ensembles independently, while
reusing each size's bootstrap draw across all pairwise comparisons. Recompute
the ratio of mean spectra in each draw. Do not average individual-disorder xi.

Production plotted bars are **disorder uncertainty only** (95% bootstrap
intervals, labeled as such); single-disorder curves have no error bars. Plot
numerical-sensitivity shifts as separate curves/tables. If using sampled
measurements, quantify within-disorder uncertainty separately and increase
sampling until its contribution to R is below about one quarter of the disorder
uncertainty; the outer bootstrap can still inherit finite measurement noise.
Do not relabel a combined sampling/disorder interval as disorder-only.

Report invalid bootstrap fractions. With invalid resamples, do not silently
compute intervals from only the surviving draws: flag the xi interval unresolved,
show spectra and Q, and increase precision or report the limitation. Record
small roundoff-induced negatives as numerical diagnostics rather than inventing
a zero length. No disorder is removed because it is difficult or inconvenient.

## 10. Finite-size interpretation and conditional fits

First show R versus Gamma (one curve per L), R versus L (one curve per Gamma),
xi_R versus L, and S0/Smin separately. Report bootstrap intervals for adjacent-size
differences directly. Size differences within uncertainty do not establish
criticality, and decreasing R over accessible sizes does not prove asymptotic
saturation.

Only after numerical stability and disorder precision support a crossing:
bracket sign changes of R(L1,Gamma)-R(L2,Gamma), interpolate locally, and bootstrap
the entire operation. Record missing and multiple crossings in resamples;
never force one crossing. Compare coarse/fine field grids and small-size cuts.
Use `finite_size_crossings.crossing_brackets` as a starting utility, preserving
its limitations and handling invalid curves explicitly.

A collapse model such as

$$R(L,\Gamma)=F((\Gamma-\Gamma_c)L^{1/\nu})+
L^{-\omega}G((\Gamma-\Gamma_c)L^{1/\nu})$$

is a later conditional analysis. A four-parameter fit to a few sizes can produce
an attractive but uninformative collapse. Require at least four usable sizes,
a resolved field window, covariance-aware uncertainty, fit stability under
L_min/window/correction changes and held-out prediction checks. This is a minimum
screen, not a sufficiency theorem. Four sizes in a fixed-ratio family may require
extending beyond this initial frontier. With only L4/L6/L8, report finite-size
trends and crossing brackets, not a precise critical exponent. Equal-time spatial
lengths alone do not determine the dynamic exponent z.

## 11. Implementation and deliverables

Proposed implementation order:

1. `triangle_correlation_observables.py`: physical connected spectra, geometry
   validation and ratio-of-means bootstrap; independent exact/product-state tests.
2. `triangle_correlation_archive.py`: analyze saved L4/L6 arrays with hash checks;
   write a new archive without modifying the source results.
3. Tiled contractions in `triangle_effective_rg.py`, checked against the existing
   deterministic evaluator; no replica sampling in the correlation-only driver.
4. `triangle_correlation_frontier.py`: dimension checks, bounded L8/L10 probes,
   timing/RSS summaries, frozen manifests and resumable cases.
5. `triangle_recursive_rg.py`: explicit block tree and physical operator ascent;
   depth bridges before L12/L16/L24 production.
6. `triangle_effective_mps.py`: Hamiltonian-to-MPO conversion, DMRG convergence
   ladders and physical operator contractions, validated on exact reduced states.
7. `triangle_effective_sign_audit.py`, then conditional
   `triangle_effective_gfmc.py`: sign certificates, categorical transitions,
   importance sampling and validated pure physical measurements.
8. `triangle_correlation_scaling.py`: approved size/field grid, uncertainty,
   sensitivity and conditional crossing reports.

These are proposed modules, not claims that the implementation exists.
Archive under `data/kagome/triangle_rg_correlation_scaling_v1/`, with source
commit/snapshots, environment/BLAS versions, full run manifest, disorder arrays,
hashes, state/isometry tree, all retention spectra, both solver starts, physical
one-/two-point measurements or independent measurement batches, directional
spectra, timings/RSS, failures and bootstrap indices. Keep scalar spectra and
the ingredients needed to recompute them; figure files alone are insufficient.

The first results notebook should contain the reused-data diagnostic. The final
scaling notebook should contain embedded figures, explicit completed/qualified
coverage, a resource frontier table, retention/depth sensitivity, and unresolved
points. Keep the original-bond model and disorder-only plotting convention visible.

**Recommended first action after this plan:** compute the connected spectra from
the existing 144 L4/L6 case archives and validate the observable at L2. That
determines whether the promising next expense is larger retention at L6, a
bounded L8 timing probe, or improving the projection before a scaling campaign.

## 12. Source anchors and plan status

This plan was checked against `triangle_effective_rg.py` (hierarchy, eigensolver
and deterministic measurements), `triangle_effective_l6_adaptive.py` (saved
physical arrays), `production_protocol.py` (geometry and exact/four-replica
spectra), `correlation_lengths.py` (length convention), `finite_size_crossings.py`,
and `FINITE_SIZE_GOAL.md`. It extends the new RG direction; it does not restart
the older cancelled worldline campaigns. No new scientific results, resource
benchmarks, critical field, exponent or physical maximum size are asserted here.
