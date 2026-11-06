## Active execution update — D64 cap (2026-11-05)

The user requested resubmission with D64. The previous chi8 controller and
three D256 workers stopped at 05:13:21 UTC; all 85 old case files were
hash-verified unchanged. The replacement controller PID 556387 is verified
running at 05:17:22 UTC in
`/root/autodl-tmp/triangle-L6-m6-c8-D64-dense-20261105-v1`.

The same L6/m6/chi8 grid (Gamma=0.100:0.025:1.500; disorders 60000–60007;
456 cases) now uses D32 warmup and a D64 cap with two starts. Four workers
use four threads each. Existing compatible checkpoints are reused. All
original numerical thresholds remain diagnostic qualification flags;
unqualified finite-D results are retained and do not block other cases.
No stage above D64 is allowed. No physical-retention or critical-point claim
is implied. Archive resource guards and the no-deadline finite queue remain.

Small local evidence and restart policy:
`data/kagome/triangle_L6_m6_chi8_D64_dense_v1/README.md`.
Raw scientific artifacts and analysis remain remote. Notebook deliveries
belong directly in `notebook/`; uncertainty remains disorder-only.

This update supersedes older chi8 pilot-gating and large-D runtime plans below.

# Spin-glass observables: three-size RG + DMRG scan

**Current campaign: matched L6, m=6, chi=8 fallback; Gamma=0.100:0.025:1.500; disorders 60000–60007.**
**Execution update, 2026-11-05: chi3 baseline complete (456/456); chi8 feasibility pilots running; full chi8 dispatch remains gated.**
The earlier three-size m=8, chi=3/16 plan below is historical and stopped.
Do not resume it or its diagnostic without fresh user direction. L10 remains excluded.
Measure and analyze $q_{\mathrm{EA}}$, $\chi_{\mathrm{SG}}$ and
$\langle q^2\rangle$ together throughout the scan. Choose scaling targets
from their qualified size dependence and physical interpretation.
Periodic kagome, N=3L², K~Uniform(0.5,1.5), h=0.15, lambda=1.
Scientific solves, extraction, figures and fits run on AutoDL. Fetch concise
notebooks with embedded figures on demand; no persistent local fetcher.

## Matched chi8 fallback: active feasibility pilots

The user requested the same L6 m6 scan at chi16, authorizing chi8 if chi16
was not doable. Both chi16 and chi8 passed L4 exact validation, including
both-start and direct overlap checks. The 1/2/4-thread timing profiles used
the same initialization at L6 Gamma=1, disorder 60003. One-thread two-sweep
timing was 7.65 seconds for chi16 and 1.78 seconds for chi8. These short
D16 profiles measure cost only, not scientific convergence.

At 04:31:58 UTC, chi8 was selected because the chi16 archive format did not
fit the available disk budget. The measured chi16 model and first D32 stage
occupied 12,023,360 bytes; projecting that incomplete footprint across 456
cases already gives 5.106 GiB, versus 4.433 GiB usable after reserving 3 GiB.
The chi16 pilot PID 553074 was stopped after exact command verification;
saved artifacts were hash-verified unchanged. This is a current-storage
blocker, not a claim that chi16 is intrinsically impossible.

Chi8 controller PID 553232 adopts existing pilots and maintains four
four-thread workers within the 20-core allocation. Each worker reserves
13.5 GiB for the full D32/64/128/256/512 ladder. Four matched pilots cover
(Gamma,disorder)=(0.1,60000),(1.0,60003),(1.5,60000),(1.0,60000).
The peak-region/high-field pilots require larger D than 64 at the current
snapshot. Intermediate truncation alarms remain recorded; qualification
requires the unchanged final-D and two-start gates.

Full 456-case dispatch requires all four pilots to qualify and a measured
whole-scan storage projection to fit. The projection uses 1.25 times the
largest completed archive plus active checkpoint reservations and 3 GiB
free. The 0.77 GiB checkpoint admission allowance is a conservative reserve,
not expected saved data per case. Resource deferrals preserve evidence;
there is no wall-clock deadline or automatic retry of failed cases.

Remote root: `/root/autodl-tmp/triangle-L6-m6-c8-dense-20261105-v1/`.
Small local evidence: `data/kagome/triangle_L6_m6_chi8_dense_v1/`.
Raw models, states, moments, checkpoints and failures remain remote. Any
result notebook is delivered directly into `notebook/`. The earlier large
campaigns remain stopped, and the unrelated overlap campaign is preserved.

## Completed affordable L6 chi3 baseline

Completed at 2026-11-05 04:09:31 UTC: 456/456 qualified cases, no failures
or exclusions, and complete eight-disorder groups at all 57 fields. The
controller ran for 562.85 seconds, reusing 40 cases and computing 416 new
ones. New compressed raw files total 787,799,290 bytes (about 0.734 GiB),
retained remotely. Final disk headroom was 7.49 GiB; maximum worker RSS
was 0.680 GiB. Maximum eigenpair residual was 9.11e-10; all both-start
observable differences were below 3.68e-10.

The disorder-mean connected susceptibility has a broad maximum near
Gamma=1.0–1.025 (sampled maximum 1.54356 at 1.0). Individual sample maxima
span Gamma=0.775–1.275. Mean qEA falls from 0.93847 at Gamma=.1 to
0.002572 at 1.5; raw q2 falls from 0.881947 to 0.011682. These are
fixed-retention finite-size results, not critical-field estimates.

The user selected m=6, chi=3 and the 57-field 0.10–1.50 grid at spacing
0.025, with eight matched disorders. Total: 456 cases, including 40 reused
archived points and 416 new cases. Physics, disorder seeds, solver starts
and numerical thresholds are unchanged from the sparse L6 field scan.
The final retained Hamiltonian has dimension 3^9=19,683 and is solved by
two-start Lanczos. Both-start physical observable agreement is checked.
Original numerical/sampling alarms are retained and excluded from plotted means.
No physical-retention convergence or finite-size scaling is claimed.

At 2026-11-05 03:55:10 UTC, the remaining four expensive L8 correlation
workers and D1024 diagnostic PID 515820 received SIGTERM after exact argv
verification. Their exit was verified. All saved artifacts were preserved;
the unrelated overlap controller was preserved. Both large-scan controllers
and the broad high-chi extension remain stopped. Evidence resides under
`/root/autodl-tmp/triangle-L6-m6-c3-dense-20261105-v1/stop-expensive-20261105T035510Z/`.

The replacement controller PID 550628 launched at 2026-11-05 04:00:08 UTC,
with twelve one-thread workers and no artificial deadline. Measured preflight
case time was 10.67 seconds with peak RSS 0.67 GiB; the archived reference
reproduction differed by less than 3e-11 in every saved observable.
Existing compressed cases average about 1.8 MiB; about 0.8 GiB is expected
for the new cases based on measured archives, subject to actual growth.
Admission maintains memory and disk headroom and preserves failures.

Remote campaign: `/root/autodl-tmp/triangle-L6-m6-c3-dense-20261105-v1/`.
Local report: `notebook/triangle_L6_m6_chi3_dense.ipynb`. Small evidence:
`data/kagome/triangle_L6_m6_chi3_dense_v1/`. Raw results stay remote;
notebooks must be fetched directly into `notebook/`, never into `data/`.
Figures use individual curves without error bars and complete eight-disorder
means with 5,000 whole-disorder bootstrap resamples, shared across fields
and observables. Missing or excluded points break lines.

## Historical three-size execution choices

The user authorized execution of the following size and retention comparison:

* Initial coverage: 57 fields from .1 to 1.5 at spacing .025, eight disorders
  per size, matched across fields and chi. Start with pilots before committing
  resources to the dense grid. Eight disorders may be insufficient for exponents.
* Accuracy: keep every original solver gate and add direct raw-q2 checks:
  .5% final-D relative drift and 1% start/exact agreement. These numerical
  tolerances are qualification goals, not established error bounds.
* Scheduling: preserve active solves and hold the waiting broad chi=8,16,32
  extension. At the user's subsequent direction to focus on available plots,
  the requested and original controllers were also held under the shared
  dispatch lock. Existing workers continue; keep new dispatch held pending
  further direction after the resource upgrade.
* Feasibility: use per-size profiles and pilots to determine whether chi=16
  is affordable and solver-qualified. If a stage exceeds resources or fails
  accuracy checks, review the limitation before dense expansion.
* Scaling: examine q_EA, connected and raw chi_SG, and raw q2 together. Fit
  each retention chi separately using appropriate observable-specific forms;
  the paper's raw-q2 collapse is one candidate analysis. Assess retention
  shifts, bootstrap uncertainty, window sensitivity and smooth polarization
  backgrounds. A transition is a hypothesis to test.

The isolated `source-v5` snapshot passed ten regression checks and a scientific
preflight covering both-start checkpoint recovery, retained-space exact and
observable qualification, unchanged reference energy, and resource deferral.
This notebook documents execution; opening it does not trigger dispatch.

## Observables and paper correspondence

At every physical point and each solver start, obtain physical moments
$z_i=\langle Z_i\rangle$ and $G_{ij}=\langle Z_iZ_j\rangle$ from the same
state and decode them through both RG levels. Save the following observables
with matched disorder IDs:

| Observable | Definition for one disorder | Role |
| --- | --- | --- |
| Finite-size Edwards–Anderson moment | $q_{\mathrm{EA}}^{(N)}=N^{-1}\sum_i z_i^2$ | Local frozen-moment and polarization diagnostic |
| Connected equal-time spin-glass susceptibility | $\chi_{\mathrm{SG}}^{\mathrm{conn}}=N^{-1}\sum_{ij}(G_{ij}-z_i z_j)^2$ | Size dependence of connected correlations |
| Raw equal-time spin-glass susceptibility | $\chi_{\mathrm{SG}}^{\mathrm{raw}}=N^{-1}\sum_{ij}G_{ij}^2$ | Raw overlap fluctuations, including polarization |
| Raw replica second moment | $\langle q^2\rangle=N^{-2}\sum_{ij}G_{ij}^2$ | Correspondence with the paper's replica observable |

Here chi_SG denotes a physical susceptibility; retention chi=3 or 16 is
a separate numerical parameter. Report both susceptibility conventions
explicitly. Include i=j and retain diagonal/off-diagonal contributions.
All four observables use the same solved state; measurement and extraction
costs should be included in the profiles, with no separate ground-state
campaign required for each observable. Disorder averages are taken after
computing each sample's observable.

For independent identical replicas, the paper's overlap operator
q=N^-1 sum_i Z_(a,i) Z_(b,i) gives

$$\langle q^2\rangle=\frac1{N^2}\sum_{ij}\langle Z_iZ_j\rangle^2.$$

This is saved as `q2_raw`. The raw equal-time susceptibility is exactly
$\chi_{\mathrm{SG}}^{\mathrm{raw}}=N\langle q^2\rangle$, so their scaling
curves encode the same information with different size normalization.
One candidate analysis follows https://arxiv.org/abs/2505.05117 using

$$[\langle q^2\rangle]\ell^{b_q}
=f((\Gamma-\Gamma_c)\ell^{1/\nu}),\qquad \ell=\sqrt N=\sqrt3L.$$

Fit our own Gamma_c, b_q and nu, separately at each chi. The paper studies
zero longitudinal field, amorphous arrays and long-range 1/r^6 interactions;
our finite-field random-bond kagome model differs. Its critical parameters
and fitted window are reference results and are not constraints on our fits.
Three sizes support exploratory scaling with limited tests of corrections.
A visually good collapse alone does not establish a transition.

Also retain magnetization and connected correlation length, with
$C_{ij}=G_{ij}-z_i z_j$. At h=0.15 the raw overlap and finite-size q_EA
include smooth polarization, so a nonzero q_EA alone does not establish
spin-glass order. Analyze q_EA and connected chi_SG as scientific outputs
alongside raw q2. Taking h to zero is not a prerequisite for investigating
a transition at fixed h. Equal-time susceptibility and zero-frequency
response are different quantum observables.

## Dense scan and pilots

| Setting | Sizes | Fields | Disorders | Cases before reuse |
| --- | --- | --- | --- | ---: |
| m=8, chi=3 | 4,6,8 | 0.100:0.025:1.500 (57) | 60000–60007 | 1,368 |
| m=8, chi=16 | 4,6,8 | same matched grid | same IDs | 1,368 |

Full authorized grid: 2,736 cases, each with independent solver starts 0 and 1. Reuse only
compatible, hash-verified results; a reused failure remains excluded.
Disorders match across fields and settings at fixed size; ensembles are
independent across sizes even when numerical IDs coincide.

The validation requires exact retained-space L4 benchmarks at
Gamma=.5, disorder 60000 for each chi. Profile each L/chi at Gamma=.5.
For all three sizes, qualify both disorders
60000–60001 at Gamma=.1,.5,1 at both retentions before expanding that size. Independent
validated groups may progress while other cases or the D1024 diagnostic run.
No dense expansion follows a failed validation or unresolved pilot.

The manifest generator is `triangle_correlation_grid.py`; output is
`data/kagome/triangle_overlap_dense_scan_v1/manifest.json`. A prepared
manifest is distinct from a launched controller. The isolated controller is
`triangle_overlap_dense.py`; its remote `results/status.json` records actual dispatch.
The regenerated manifest contains only the authorized three sizes and two
retentions; the previous manifest is preserved as `manifest_before_execution.json`.
Remote root: `/root/autodl-tmp/triangle-overlap-dense-scan-20261105-v1`.
The older four-size `triangle_correlation_dmrg --phase scan` is historical.

## Accuracy and retention

m=8 retains the entire triangle Hilbert space. A four-triangle block has
4096 states, so chi=16 is partial retention. This is the requested comparison
setting, not an assertion of physical convergence. Preserve the prior
retention ladder, failures, and ongoing workers. The waiting extension is held
while this comparison has priority, with compatible completed results reused
after hash and observable qualification checks.

Keep original two-start D32,64,128,256,512 energy, spectra, correlation-ratio,
sweep, canonical and truncation gates unchanged: final-D energy drift
<10^-7 per spin, spectra drift <.5%, correlation-ratio change <.002,
sweep energy drift <10^-8 per spin, canonical error <10^-8, no truncation
failure, plus start agreement and exact retained-space validation. Here D is
the MPS bond dimension and is distinct from RG retention chi.
The production criteria additionally require, separately for each
start, final-step q_EA absolute drift <=.001,
connected susceptibility relative drift <=.5%, and **raw q2 relative drift
<=.5%**. Both starts and exact retained-space benchmarks must agree within
.001 in q_EA and 1% in each susceptibility and raw q2. Relative drifts use
the larger absolute endpoint as denominator. The original spectra gates
remain prerequisites. Added checks may extend the ladder within D512;
no failed original solve is rescued by them.

Compare each matched disorder, plus paired mean shifts, between chi=3 and 16.
Prospective retention tolerances are q_EA absolute change <=.01, connected
susceptibility relative change <=2%, and raw q2 relative change <=2%.
Two settings assess sensitivity; agreement alone does not demonstrate a
converged chi ladder. Numerical solver convergence and physical retention
convergence must be reported separately. Keep block-partition sensitivity
as a further systematic check.

## Execution sequence and review points

1. Execute the authorized scientific scope, initial disorder coverage,
   accuracy goals and scheduling policy with explicit validation gates.
2. Audit compatible existing results, then run exact retained-space validation
   and performance profiles for the requested settings. Record feasibility,
   checkpoint growth, runtime and any unresolved numerical checks.
3. Run the matched retention pilots. Plot individual disorders and paired
    chi=3 to 16 shifts in q_EA, connected/raw chi_SG and raw q2. Review whether the
   dense comparison is affordable and scientifically useful.
4. Produce the dense curves for groups that pass the agreed checks. Show
   coverage, failures and excluded cases alongside disorder-only uncertainty.
5. Attempt exploratory scaling only when common qualified curves support it.
   Report fit stability and retention sensitivity, including inconclusive fits.

The initial pilot suite contains 36 physical cases (three sizes, three fields,
two disorders, two chi settings), each with two starts. The two L4 validation
cases at Gamma=.5, disorder 60000 are included in those 36. Larger-size
profiles are performance measurements and do not replace these cases.

## Current evidence and feasibility limits

At 2026-11-05 03:21 UTC, requested controller PID 543928 and original
controller PID 543375 were stopped under the shared dispatch lock after
verifying their command lines. All eight active solves were preserved.
Remote hold evidence is under
`triangle-overlap-dense-scan-20261105-v1/plot-hold-20261105T032100Z/`.
Held status files are historical; they do not refresh as orphaned workers finish.

The available-results report snapshot after 03:31 UTC extracts 217 distinct
cases: 214 qualify and three failed original solves remain excluded.
It includes 23 requested m=8, chi=3/16 cases, 33 qualified L8 retention
cases, and 144 archived exact retained-space cases; these views overlap.
Requested pilot coverage out of six per size/retention is L4 6/6 at both
chi values, L6 6/6 at chi=3 and 1/6 at chi=16, L8 4/6 at chi=3 and
0/6 at chi=16. All seven available chi=3 to 16 pairs (six L4 and one L6)
exceed at least one prospective retention tolerance. No scaling fit was attempted.

The isolated report source and full extraction are remote under
`/root/autodl-tmp/triangle-overlap-available-20261105-v1/`.
The concise notebook with five embedded figures is local at
`notebook/triangle_overlap_available.ipynb`. Small evidence and figure previews
are under `data/kagome/triangle_overlap_available_v1/`; notebooks must be fetched
directly into `notebook/`, never staged under `data/`. The eight-disorder L6
scan uses 5,000 whole-disorder bootstrap resamples with seed 20261105;
individual curves and fixed two-disorder means have no error bars.
At the final report audit, five of the eight preserved solves remained
active, three had completed, and diagnostic PID 515820 and unrelated
controller PID 335206 remained active. Raw artifacts remain remote.

The handover at 2026-11-05 02:52 UTC activated requested controller PID 543928
from tested `source-v5`. Original controller PID 543375 uses tested `source-v4`.
The original campaign had 32/54 L8 cases passing and 35 completed. All six
active workers were adopted with their existing PIDs and no duplicate live
case owners. Waiting broad-extension controller PID 540660 was stopped and
recorded as held. The D1024 diagnostic and unrelated overlap campaign continue.

The first requested-controller audit reused five qualified compatible cases.
The new L4 chi=16 validation also passed the solver, two-start, exact and
observable gates. The subsequent status snapshot had 17 qualified cases.
Both L4 profiles completed and selected one thread for throughput; subsequent
profiles and pilots are dispatched subject to shared resource admission.
An initial live verification used 16 worker threads and 54 GiB of memory reservations.
Admission includes unused RAM plus reclaimable inactive file pages; active
cache, shared memory and anonymous memory remain charged. This removes a
measured cache accounting bottleneck while preserving the 16 GiB headroom.

The L6 chi=16 D16 timing profile initially exceeded TeNPy's default truncation
limit. Its failure, log and inputs are archived under remote `handover-v5`.
The explicit profile rerun completed, recording truncation and canonical error
as timing diagnostics; scientific production thresholds remain unchanged.
All six profiles are complete. L6 chi=16 selected two threads and estimated
21.36 GiB at D256 versus 78.91 GiB at D512, so D512 exceeds the 72 GiB
worker budget and will defer if required. Active scientific workers survived
this controller update and were adopted.

At L4, Gamma=.5, disorder 60000, chi=3 versus chi=16 gives q_EA
0.61131 versus 0.71581, connected chi_SG 0.29143 versus 0.15571, and raw q2
0.38914 versus 0.52408. Retention shifts exceed the prospective tolerances
despite excellent retained-space solver convergence. This single-disorder
comparison establishes sensitivity at this point, not physical convergence
or an ensemble conclusion. Consult `initial_results.json` and `verification.json`
in the local campaign evidence directory for the timestamped snapshots.

About 8.49 GiB remained on the data disk. This is a material limit for archived
high-chi checkpoints and the full scan. Profile storage before
committing to dense production; any archival/storage solution must preserve
inputs, checkpoints, isometries and provenance. There is no reliable full-scan
ETA before representative chi=16 solves. Existing retention evidence does
not yet qualify chi=16 as physically converged.

The later plotting audit measured 8.39 GiB free. The earlier 5 TB figure was
a conservative provisioning recommendation, not measured saved-data demand.
The storage estimator includes a fixed 0.4 GiB per-case safety allowance
(about 1.17 TB over 2,736 cases), which must be distinguished from expected
archive size. No new dense dispatch is authorized by this historical plan
while the current plotting/resource hold is in effect.

## Resources and resumability

Preserve existing sources and active workers in their isolated snapshots.
Enforced budgets are 16 worker CPU threads across these correlation campaigns within the
20-core allocation, up to 72 GiB reserved worker RAM, 16 GiB memory headroom,
and 3 GiB disk headroom plus future checkpoint reservations. Include active
workers' future growth. Preserve the unrelated overlap campaign and diagnostic.
Use measured thread timing with identical profile states to select
throughput; this launch limits profiles to 1/2 threads to fit alongside active
workers. Short D16 profiles establish performance only. Validated float64
CUDA accelerates superblock diagonalization; TeNPy DMRG remains on CPU.

Keep one owner per case, atomic records, source and artifact hashes, input
disorder arrays, entropy/seeds, isometries, states, spectra, and failures.
Adopt workers after controller restart. Resource deferrals are unresolved
and are not retried automatically. No artificial campaign time limit.
Available disk space may block later stages; retain checkpoints and report
that limit explicitly. The instance remains on.

## Scaling analysis protocol

First plot complete common-grid q_EA, connected/raw chi_SG and raw q2 curves,
individual disorders without error bars, and means with disorder-only 95%
intervals. Require complete
matched disorder groups before paired retention comparisons. Inspect raw
and connected observables before choosing candidate critical-field windows.

Select candidate scaling analyses after inspecting the qualified curves.
For connected chi_SG, consider an independent susceptibility scaling form
$[\chi_{\mathrm{SG}}^{\mathrm{conn}}]=\ell^{y_{\mathrm{SG}}}f_{\mathrm{SG}}(x)$,
where $x=(\Gamma-\Gamma_c)\ell^{1/\nu}$ and the exponent is fitted rather
than assumed from a classical or zero-field model. For q_EA, assess whether
a smooth background can be distinguished from a singular size-dependent
contribution before proposing an order-parameter collapse; do not impose
the raw-q2 exponent on q_EA. Seek compatible critical fields across
observables when identifiable, while reporting disagreements. Raw chi_SG
and raw q2 obey the exact normalization above and are not independent
evidence: in two spatial dimensions their scaling powers differ by two.

For a raw-q2 collapse, fit the paper's ansatz using an interpolated self-consistent master
curve, with points predicted from other sizes. Scan fit windows supported
by all three sizes; record the objective, overlap of scaled domains,
parameter bounds and boundary hits. Estimate Gamma_c, b_q and nu for each
chi rather than importing paper values. Include whole-disorder bootstrap:
resample complete field curves at each size independently, preserving
field/retention/observable covariance within that size and saving resampling indices.
Report bootstrap distributions, failed fits and parameter degeneracies.

Vary the field window and omit the smallest size as a sensitivity check;
two-size fits are weaker diagnostics, not independent exponent estimates.
Compare chi fits on identical windows and bootstrap samples. Check a smooth
background alternative when polarization dominates. If parameters are
unidentifiable or shifts exceed disorder uncertainty, report that result
instead of assigning precise critical exponents. Eight initial disorders
are a pilot; expand coverage when measured variance and cost justify it.

Detailed earlier correlation-length notes are historical:
`code/transformer_qmc/CORRELATION_LENGTH_SCALING_PLAN.md`.
