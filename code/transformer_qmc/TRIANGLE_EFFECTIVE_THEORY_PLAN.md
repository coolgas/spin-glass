# Triangle effective theory and real-space renormalization: a finite study

Research protocol, 2026-11-04. **Status: five-schedule L4 diagnostic pilot
complete; the tested fixed low-energy schedule fails the accuracy screen.**
All five schedules exceed the chosen energy-discrepancy limit of .001 per
physical spin (.048 total energy for N=48). The best observed discrepancy is
.05352 per spin against the Jastrow GFMC comparison, for (m,chi)=(8,8).
GFMC is comparison data, not exact L4 ground truth. This pilot covers only
disorder60000, Gamma=.55, upward triangles and origin (0,0).

Algebra checks, initial exact L2 screens and independent second-projection
checks ran; the full Stage B/C/D validation grids and 320-case Stage E grid
did not run. Expansion to the full L4 grid and L6 is stopped pending a separate
accuracy study of larger or environment-adapted bases.
See the [measured results notebook](../../notebook/triangle_effective_l4_results.ipynb)
and [archived assessment](../../data/kagome/triangle_rg_l4_v1/results/ASSESSMENT.md).

A subsequently completed higher-field comparison covers Gamma=.60,.65,.70,.85,1.00
and disorders60000–60007 at (m,chi)=(6,8), plus five disorder60000 checks at
(8,8): 45/45 cases. It reproduces the qualitative field trend but shifts
physical P(q) toward lower overlap; TV distances of disorder means from Jastrow
GFMC are .1393,.1574,.1665,.1569,.1402. Removing triangle truncation changes
the five single-disorder distributions by TV=.018–.030 at fixed chi=8; this
motivates testing superblock retention but does not identify the entire error.
The 4096-dimensional effective models were solved with Lanczos-based `eigsh`,
followed by independent conditional Born measurement draws. See the
[higher-field results](../../notebook/triangle_effective_higher_field_distributions.ipynb)
and [raw data/provenance](../../data/kagome/triangle_rg_higher_fields_v1/README.md).
Physical accuracy, a transition and finite-size scaling remain unestablished.
Section 9.1 records the proposed weak-coupling continuation; it launches no run.

## 1. Question and intended outcome

Can a low-energy basis on disjoint kagome triangles reduce the cost of reliable
physical-spin overlap measurements in the current disordered transverse-field
Ising model?

The first objective is to establish whether a useful truncation exists. The
second is to determine whether its effective Hamiltonian and measurement rules
admit an efficient many-body solver. These are separate requirements. A reduction
in formal Hilbert-space dimension alone is insufficient.

The proposed extension is a multiscale real-space renormalization schedule:
physical spins -> triangles -> connected superblocks -> a final reduced solve.
The immediate deliverable is a reproducible small-system feasibility report,
followed by a required periodic L4 benchmark and a conditional, finite L6 test.
The study can conclude that triangle
truncation fails. That would identify a missing separation of energy scales and
prevent a larger campaign built on an uncontrolled approximation.

Changing to amorphous geometry is an independent research option, described in
Section 12. It is not a prerequisite for this study.

## 2. Fixed physical model and motivation

Use the existing periodic kagome Hamiltonian, without refitting disorder:

\[
H=\sum_{\langle ij\rangle}K_{ij}Z_iZ_j
  -h\sum_i Z_i-\Gamma\sum_i X_i,
\qquad h=0.15,\quad K_{ij}\sim U(0.5,1.5).
\]

The repository stores `J=-K`. There are `N=3L^2` physical spins; L6 means
`N=108`, not linear size `sqrt(N)`. Bond identities and order come from
`kagome_qea.lattice.build_kagome`, and the disorder generator is
`default_rng(SeedSequence([20261027,L,sample_id,1]))`, as implemented in
[`physics.py`](physics.py).

The motivation is the observed sensitivity of L6 overlap distributions to guide
training. For example, in the audited two-disorder comparison at Gamma=.60
and 4000 updates, training-seed histogram TV was approximately .185, whereas
projection-repeat TV after averaging the seeds was approximately .020. Close
energies and stable repeats from the same guide therefore do not establish
physical convergence. These values motivate the study; they are not exact
reference answers for the effective theory.

The disorder strength supplies no obvious small parameter. Bonds inside and
between our proposed blocks have the same distribution. A favorable isolated
triangle spectrum must be tested against its environment.

## 3. Exact block construction

### 3.1 Disjoint triangles and a diagnostic interpolation

Choose one orientation of triangles as blocks. Each orientation partitions all
sites into `B=L^2` disjoint triples. Every nearest-neighbor bond is either
internal to one selected block or connects two selected blocks.

For selected triangle `t`, define

\[
H_t=\sum_{\langle ij\rangle\subset t}K_{ij}Z_iZ_j
    -h\sum_{i\in t}Z_i-\Gamma\sum_{i\in t}X_i,
\]

\[
H(\lambda)=H_0+\lambda V,\qquad H_0=\sum_t H_t,
\qquad V=\sum_{\text{interblock }\langle ij\rangle}K_{ij}Z_iZ_j.
\]

Assign the full on-site fields to each disjoint block. Then `lambda=1` is
exactly the current model, while `lambda=0` is a solvable product of triangles.
Intermediate lambda values diagnose how truncation breaks down as interblock
couplings grow. Success at small lambda is not success for the physical model.

Repeat the construction with the opposite triangle orientation. At lambda=1
the full Hamiltonians are identical. At intermediate lambda the two
constructions define different diagnostic interpolations, each with its own
full-space reference. At lambda=1, orientation dependence after truncation is
an error diagnostic; do not average it away or select the better orientation
afterward. At lambda<1, compare each interpolation against its own matched
reference: changing the orientation also changes the physical Hamiltonian.

The current [`triangle_projector.py`](triangle_projector.py) already validates
the two disjoint layers and bond cover. Its local Hamiltonian divides on-site
fields between both layers and retains all eight states for a split propagator.
Reuse its geometry checks, but implement the block Hamiltonian above separately.

### 3.2 Local basis and projected interactions

Diagonalize each `8 x 8` block:

\[
H_t|a_t\rangle=\epsilon_{ta}|a_t\rangle,
\quad \epsilon_{t0}\le\cdots\le\epsilon_{t7}.
\]

Let `U_t` contain the lowest `m_t` orthonormal eigenvectors and define the
isometry and projector

\[
U=\bigotimes_t U_t,\qquad P=UU^\dagger.
\]

Start with uniform `m=2,4,6,8`. Retain eigenvectors in nested order, use a
documented phase convention, and store the complete eigensystem. Flag a cutoff
that divides a numerically unresolved degenerate multiplet. Such a cutoff is
not an acceptable fixed-m result; retain the whole multiplet and report the
actual block dimensions separately. At h=.15 no Z2 symmetry reduction is assumed.

The leading approximation is the exact variational projection

\[
H_{\rm proj}=U^\dagger H(\lambda)U
=\sum_t\operatorname{diag}(\epsilon_{t0},\ldots,\epsilon_{t,m_t-1})
+\lambda\sum_{\text{interblock }\langle ij\rangle}
 K_{ij}\,\bar Z_{ti}\otimes\bar Z_{uj},
\qquad \bar Z_{ti}=U_t^\dagger Z_i U_t.
\]

Group all physical bonds between the same block pair before testing the signs
of its interaction matrix. This matters on small periodic lattices where the
same pair of blocks can share several bonds.

At m=8 this is an exact change of basis for every lambda. For a fixed nested
basis, its ground energy is an upper bound to the full ground energy and must
decrease or remain constant as m increases. The other observables need not
converge monotonically.

### 3.3 Size reduction and its limits

For L6 there are 36 blocks:

| Local dimension m | Total dimension | log2 of dimension |
| --- | --- | --- |
| 8 | `8^36 = 2^108` | 108 |
| 6 | `6^36` | about 93.06 |
| 4 | `4^36 = 2^72` | 72 |
| 2 | `2^36` | 36 |

Even `2^36` is too large for a routine dense or full-vector L6 calculation on
the rented instance. A many-body representation remains necessary. The effective
block graph is two dimensional and contains loops; blocking does not remove
frustration or guarantee easier sampling. Report memory, interaction density,
measurement cost, and time to a specified accuracy alongside dimension.

For an isolated uniform classical AF triangle at h=Gamma=0, the six nonaligned
states form the lowest manifold. Random bonds split those six states, and
transverse fields mix configurations. This observation motivates testing m=6;
it does not justify m=2 or m=4, or impose a six-state constraint on every triangle
of the full lattice.

### 3.4 A concrete real-space renormalization schedule

The hierarchy below is a trial schedule to validate, not an assumed universal
RG flow. It is a sequence of Hamiltonian projections at fixed physical h and
Gamma. Changing lambda is a separate diagnostic interpolation; annealing Gamma
does not substitute for coarse graining.

For L6, start with this geometrical hierarchy:

| Level | Objects | Input space per new object | Trial retained dimension |
| --- | --- | --- | --- |
| 0 | 108 physical spins | 2 per spin | 2 |
| 1 | 36 disjoint three-spin triangles | 8 | m1=2,4,6,8 for benchmarks |
| 2 | 9 connected groups of four triangles | m1^4 | chi=2,4,8 as resources allow |
| Final solve | 9 interacting superblocks | chi^9 total | no further local truncation |

For the upward-triangle partition, group the cells
`(2a+r,2b+s)`, with `a,b=0,1,2` and `r,s=0,1`, into a 2-by-2 cell
superblock. Each contains 12 physical spins. Preserve periodic interactions
between superblocks. Derive a corresponding partition for the opposite
triangle layer from its actual block adjacency, then verify disjoint coverage
and connectivity before using it. Save explicit site maps for every hierarchy.

Benchmark all four upward-grid origins `(ox,oy)` with ox,oy in `{0,1}` modulo
L6 in local diagnostic work. The first full reduced solve uses origin `(0,0)`;
origin `(1,1)` is a prespecified partition sensitivity. The other origins are
reserved finite follow-ups if those two disagree. Opposite orientation is an
additional partition sensitivity, not a symmetry of a fixed disorder.

At each step ell, assemble each block Hamiltonian from every current term
whose support lies entirely inside that block. Terms crossing its boundary
remain explicit interactions. Diagonalize the internal Hamiltonian and select
an isometry W_ell; then transform the entire current Hamiltonian and required
physical measurement operators:

\[
H^{(\ell+1)}=W_\ell^\dagger H^{(\ell)}W_\ell,
\qquad \mathcal U=W_0W_1\cdots W_{\ell}.
\]

Here W0 is the product of triangle isometries. The final physical state is
`psi_approx=mathcal U phi_top`. Save the isometries and original Hamiltonian
alongside the reduced operators. Merely fitting new effective Ising couplings
would omit generally present off-diagonal and anisotropic operators.

In the leading projection of a pairwise Hamiltonian, interactions remain
between pairs of current blocks, with general operator matrices on their
endpoints. Track and sum every such term. Downfolding corrections can create
interactions involving more blocks; any later removal of these terms is an
additional approximation with its own measured error.

**Why this schedule could change the computational situation.** With m1=4,
each level-2 diagonalization has dimension 256. Retaining chi=4 leaves a final
space `4^9=262144`: one real double-precision vector occupies about 2 MiB.
This makes a matrix-free iterative ground-state solve a concrete candidate,
even if positive off-diagonal elements appear. It does not require the positive
weights used by the present GFMC code. Never construct a dense final matrix.
The actual memory includes multiple Krylov vectors and operator workspaces.

With chi=8 the final dimension is `8^9=134217728`, about 1 GiB per real
vector, before workspaces; this requires a measured memory/time feasibility
gate. chi=16 gives `16^9=68719476736`, about 512 GiB per real vector and is
outside the routine full-vector proposal. A successful chi=4 calculation is
therefore useful only if its much stronger truncation also passes accuracy
tests. The total map is a tree of local isometries capped by an unrestricted
nine-superblock state, which supplies a representation for measurement without
allocating a physical `2^108` vector.

**Retention and stopping rules.** The first implementation uses fixed m1 and
chi so that errors can be attributed to a particular level. At each merge,
record cutoff gaps, retained bandwidth, boundary coupling to discarded states,
and the change in physical observables on exactly solved clusters. A small
internal gap or large boundary leakage triggers retaining more states or
stopping that branch; it does not authorize a forced decimation. Maintain
complete unresolved multiplets. A higher retained dimension may exceed the
resource budget, in which case the schedule is reported as infeasible at that
accuracy rather than compressed further by assumption.

For error attribution, compare three nested calculations on a four-triangle
cluster: the full 12-spin solution, triangle truncation without the second
truncation, and triangle plus superblock truncation. Compare corresponding
physical observables and retained weights. On a cluster with only one resulting
superblock, retaining its exact internal ground state makes the second step
trivially exact for that isolated cluster. Therefore test two coupled
superblocks as well: the exact level-1 reference has dimension m1^8, which is
65536 for m1=4. This tests second-level boundary truncation relative to the
level-1 model, while making no full 24-spin exactness claim. Retaining each
superblock's complete input space must recover the level-1 reference.

The total physical-observable tolerances in Section 8 apply to the composed
map. Reserve half of each tolerance for each truncation level as a diagnostic
allocation; this is not a rigorous additive error bound. If only the composed
map passes on one partition, preserve the level-wise failures and expand
validation before claiming robustness.

**An optional refinement after the fixed schedule.** Internal block ground
states may miss states needed to entangle neighboring superblocks. If this is
the limiting error, solve small block-plus-environment problems, form the
block reduced density matrix, and retain its dominant eigenvectors. Define
the discarded density-matrix weight and validate it against physical P(q).
This introduces a density-matrix-based renormalization step. A later sweep can
update isometries with the rest of the hierarchy held fixed, accepting only
variational energy decreases, but overlap accuracy still needs separate checks.
Do not combine this refinement with the first fixed-basis benchmark or assume
its energy optimization guarantees an accurate glass distribution.

## 4. Diagnostics for whether truncation is physically justified

Record these before examining solver performance:

1. **Local cutoff gaps:** `Delta_t(m)=epsilon[t,m]-epsilon[t,m-1]`, together
   with the full spectrum, retained bandwidth, and incident bond strengths.
   Define this only for m<8. Do not turn a local gap into a global error bound.
2. **Couplings to discarded states:** for every adjacent block pair, measure
   the spectral norm of the matrix coupling its retained product subspace to
   states where at least one block is discarded. Compare it with the smallest
   discarded-minus-retained product-energy denominator for connected states.
   If a denominator vanishes, becomes negative, or is unresolved, label the
   perturbative diagnostic invalid. A ratio below one is not a convergence proof.
3. **Exact-state discarded weight on small clusters:** for exact ground state
   `psi`, compute local discarded weights and `1-<psi|P|psi>`. Also measure
   errors in its physical-spin observables. Global retained weight can shrink
   with system size even when local errors are modest; report both.
4. **Residual of the lifted approximate state:** for normalized reduced ground
   state `phi`, let `psi_m=U phi`, `E_m=<psi_m|H|psi_m>`, and
   `r=||(H-E_m)psi_m||`. For an exactly solved projected state this residual
   measures coupling out of the retained space, up to numerical error. A small
   energy variance without a resolved many-body gap does not certify P(q).
5. **Sensitivity to m and triangle orientation:** compare physical observables
   at the same full Hamiltonian. Orientation agreement and successive-m
   agreement are necessary screening evidence, not independent exact references.

A disorder with strong leakage remains in every report. The plan does not
discard hard instances, select a guide by its overlap result, or replace a
disorder to improve the summary.

## 5. Reconstruct physical-spin observables

### 5.1 One-copy observables and moments

The reduced state describes amplitudes in a local energy basis. Block labels
are not measured Ising spins. For any physical operator O use

\[
O_{\rm proj}=U^\dagger O U,
\qquad \langle O\rangle_m=\langle\phi|O_{\rm proj}|\phi\rangle.
\]

For two sites in different blocks, the projection of `Zi Zj` factors into the
two local projected operators. For sites in the same block, construct
`U_t^dagger Zi Zj U_t` directly. In general it differs from
`(U_t^dagger Zi U_t)(U_t^dagger Zj U_t)`. In particular, `Zi^2=I` must remain
exact in a physical-spin moment calculation.

For two independent copies of the same ground state and disorder,

\[
q=\frac1N\sum_i z_i^{(a)}z_i^{(b)},\quad
\langle q\rangle=\frac1N\sum_i\langle Z_i\rangle^2,\quad
\langle q^2\rangle=\frac1{N^2}\sum_{ij}\langle Z_iZ_j\rangle^2.
\]

Construct higher physical-spin products before projecting them as well. Record
raw P(q), q2, q4, magnetizations, two-point correlations, and connected spectra
using the conventions in [`physics.py`](physics.py) and
[`../kagome_qea/exact.py`](../kagome_qea/exact.py). At h=.15 the mean overlap
is generally nonzero. Raw overlap order and connected glass diagnostics must
both remain visible.

### 5.2 The full overlap distribution

For a reduced wavefunction phi, the physical measurement probabilities are

\[
p_m(z)=|\langle z|U|\phi\rangle|^2,
\qquad
P_m(q)=\sum_{z,z'}p_m(z)p_m(z')\,
\mathbf1\left[q=\frac1N\sum_i z_i z_i'\right].
\]

Equivalently, each local measurement has effects
`E_t(z_t)=U_t^dagger |z_t><z_t| U_t`, which sum to identity in the retained
space. These effects determine a joint measurement on the entangled reduced
state. Independently decoding each block from its marginal, or decoding sampled
block labels without interference, does not generally reproduce that joint
distribution.

For the exact small-system benchmark, lift phi into the original spin basis.
Compute the XOR convolution of p with itself using a Walsh-Hadamard transform,
then group XOR masks by Hamming weight: a mask of weight d gives `q=1-2d/N`.
This yields the complete P(q) in `O(N*2^N)` work without enumerating all
`4^N` replica pairs. Validate normalization and moment identities. Store the
original `N+1` bins; do not smooth or rebin before computing TV distance.

Before an L6 run, implement either validated joint physical-spin sampling or a
contraction of the corresponding two-replica measurement operators. Benchmark
its cost and error against exact small-system P(q). An approach that measures
only q2 is a partial observable method and cannot pass the P(q) milestone.

For the multiscale schedule, use the composed isometry mathcal U in place of U.
Prototype sequential Born sampling on the isometry tree and top state: each
physical outcome conditions the remaining joint state before the next outcome
is drawn. Validate exact outcome probabilities on small hierarchies and the
resulting P(q) against XOR reconstruction. Obtain replicas from independent
draws of the same joint distribution. Benchmark the contraction cost for the
actual top-state representation; independent sampling of superblock marginals
would discard entanglement and is not an acceptable shortcut.

## 6. Solver compatibility is an early gate

The original model is stoquastic in the physical Z basis. Local energy-basis
projection can generate positive off-diagonal matrix elements. The existing
positive-guide GFMC implementation cannot simply accept those matrices.

For each candidate m:

- Inspect the assembled small-system Hamiltonian, recording the number,
  magnitude, and distribution of positive off-diagonal elements above a stated
  numerical tolerance.
- Test whether diagonal sign choices can make all off-diagonal elements
  nonpositive. On a finite configuration graph this is a consistency problem
  for the relative signs along edges; a contradictory cycle rules out such a
  sign change in that basis. This test does not rule out general basis rotations.
- For L6, test a common set of local basis sign choices across all interaction
  terms. Making every term nonpositive with the same choices is a sufficient
  certificate. Failure of that local test is inconclusive unless a sign
  obstruction is established for the assembled operator.
- Keep m=8 as the exact basis-change control. Positive elements there are a
  basis issue despite the unchanged physical model.

Use exact diagonalization for the first benchmarks, regardless of sign. If a
useful m passes the sign and measurement gates, develop a block-state projector
with the exact projected interactions. If it does not, assess a tensor-network
or variational solver in a separate finite proposal. Do not silently replace
positive elements, discard generated interactions, or interpret a variational
energy as a converged overlap distribution.

The multiscale route in Section 3.4 also permits a final matrix-free exact
reduced-space solve. For that route, the sign check is informative rather than
a blocking requirement: memory, eigensolver convergence, and measurement cost
are the relevant solver gates. Positive elements still block reuse of the
current positive-weight GFMC without a further valid construction.

For any solver, separate truncation error, many-body optimization/projection
error, physical-measurement error, and sampling uncertainty. The small-system
exact projected solution isolates the first of these.

## 7. Prespecified finite benchmark

### Stage A: algebra, spectra, and isolated failure mechanisms

Implement only the block representation and exact small-system measurement
tools first. Required checks are:

- The disjoint site cover and complete internal/interblock bond partition for
  both orientations, including L2 and L6 geometry.
- Equality of the independently constructed physical Hamiltonian and the m=8
  representation, including on-site fields and all interblock bonds.
- Exactness of the product ground state at lambda=0 when every local ground
  state is retained; variational ordering across nested m at lambda>0.
- Physical operator identities, normalized decoded probabilities, P(q), and
  agreement between histogram moments and direct correlator contractions.
- A uniform triangle with analytic classical energies, plus disordered coupled
  triangles with nonzero Gamma, to expose sign and leakage mechanisms.

Use tolerances scaled to matrix norms and eigensolver residuals; target relative
algebraic discrepancies <=1e-10 in double precision. Persist near-degeneracies
and unresolved solver outcomes rather than forcing a pass.

Also prepare the L6 local-spectrum inventory for sample IDs 60000 and 60001 at
Gamma=.55,.60,.65,.70, both orientations. This requires 576 local 8x8
diagonalizations. It does not require solving the L6 many-body problem and does
not qualify any L6 truncation. All scientific execution takes place on AutoDL.

### Stage B: exact periodic L2 comparison

Fix the following grid before running:

| Axis | Values |
| --- | --- |
| Physical lattice | periodic L2, N=12, four disjoint blocks |
| sample_id | 60000 through 60007, equally weighted |
| Gamma | .55, .60, .65, .70, 1.00 |
| h | .15 |
| lambda | 0, .25, .50, 1.00 |
| block orientation | both |
| nominal retained m | 2, 4, 6, 8 |

This is 160 parameter/disorder tuples, 320 full-Hamiltonian references when
counted by orientation, and at most 1280 projected ground-state solves.
At lambda=1 the two orientations share the same full reference, reducing the
number of unique full references to 280. Exact duplicate operators can be
deduplicated by matrix identity, with all logical task records retained.
Nominal-m cases with an unresolved cutoff are flagged as specified in Section 3.

L2 and L6 sample IDs do not denote identical bond realizations: L enters the
disorder RNG. Store every bond array and its hash rather than claiming a
cross-size disorder match.

For every case save the full and projected ground energies, lowest resolved
gaps, residuals, local discarded weights, global retained weight, sign
diagnostics, physical P(q), q2, q4, site magnetizations, zz correlations, and
connected spectra. Retain the exact reduced eigenvectors and isometries so
every physical observable can be regenerated.

Stage B estimates behavior across a small fixed set of disorders. It is not
a thermodynamic or phase-transition calculation. Exact L2 success is necessary
evidence for proceeding, not an error bound at N=108.

### Stage C: one larger exact connected patch, conditional

If a candidate m passes Stage B, validate it on one preregistered open patch
made of five complete upward triangles (`N=15`). Use cell coordinates
`(0,0),(1,0),(2,0),(0,1),(1,1)` in the existing kagome coordinate convention,
retaining all nearest-neighbor bonds between the selected sites. This preserves
complete blocks while increasing the exact reference dimension to 32768.

Use sample IDs 60000 through 60003, Gamma=.55,.70,1.00, and lambda=1:
12 physical references. Draw patch bonds with
`SeedSequence([20261104,15,sample_id,0])`, in upper-triangle site-pair order,
and save the arrays. This is a separate diagnostic ensemble, not a crop of the
periodic L6 disorder. Compare the surviving candidate, the next larger nominal
m, and m=8: at most 36 projected solves, with duplicates removed.

This patch uses its complete upward-triangle partition. It is not an
opposite-orientation boundary comparison. Report its boundary and geometry
limitations explicitly. Add a tested N=15 observable path without weakening
the existing `N<=12` overlap-moment guard; XOR-based P(q) itself does not require
a dense replica-pair array or the existing four-point moment implementation.

### Stage D: validate the second renormalization step, conditional

After the single-level tests, use two connected 2-by-2 upward-cell superblocks,
with cell coordinates `(x,y)` for x=0,1,2,3 and y=0,1, open boundaries. The
physical patch has N=24 and eight triangle blocks. Fix sample IDs 60000 through
60003, Gamma=.55,.70,1.00, h=.15, lambda=1. Generate its physical bonds in
upper-triangle site-pair order using
`SeedSequence([20261104,24,sample_id,0])` and store their hashes.

Use m1=4 initially, provided it survived the earlier validation. The uncompressed
level-1 space has dimension 65536, permitting a matrix-free reference without
claiming to solve the full physical 24-spin problem. Compare chi=2,4,8 and
chi=256 (no second-level truncation) for each superblock: 12 level-1 references
and at most 48 two-level solves, with duplicate reference solves removed.
The chi=256 case is an algebraic recovery control; its representation need
not be constructed as a dense matrix.

For this stage, evaluate physical correlations and joint physical-spin samples
through the triangle isometries for both the level-1 reference and the two-level
approximation. Match the original N=24 histogram bins. Use independent samples,
report confidence intervals, and require that uncertainty can resolve the
Section 8 tolerances. Full physical probability enumeration is not needed.
This stage bounds observed additional RG error on its fixed level-1 models;
the first-level physical truncation error remains separately assessed by
Stages B and C. A passing second step cannot repair a failed first step by
assertion.

### Stage E: required periodic L4 benchmark before L6

Use the same physical model at L4, N=48, h=.15, lambda=1. The hierarchy is
48 spins -> 16 disjoint triangles -> 4 connected four-triangle superblocks.
For upward triangles group cells `(2a+ox+r,2b+oy+s)` modulo 4, with
a,b,r,s in `{0,1}`. Verify the opposite-layer partition from actual adjacency,
as for L6. Preserve all periodic boundary interactions, including distinct
physical bonds joining the same pair of superblocks.

The final space has dimension chi^4: 256, 4096, and 65536 for chi=4,8,16.
This permits a wider retention comparison than L6. It is an exact solve only
within the retained space, with physical truncation error still to assess.

Freeze Gamma=.55,.56 and sample IDs 60000 through 60007: 16 physical contexts
aligned with the existing low-field L4 comparisons. Import the archived L4
bond arrays and verify hashes, site order, fields, and periodic boundaries.
The model RNG is `SeedSequence([20261027,4,sample_id,1])`; a matching L6 sample
ID does not identify the same disorder. Archive baseline records and their
source hashes in the comparison manifest before running the benchmark.

Start with Gamma=.55, sample60000, upward triangles, origin `(0,0)`, (m,chi)=(4,4).
After measuring solve and joint-sampling cost, compare the fixed schedules
`(4,4),(4,8),(4,16),(6,8),(8,8)` for both orientations and origins `(0,0),(1,1)`.
The chi ladder at m=4 probes second-level sensitivity; the m ladder at chi=8
probes first-level sensitivity, including full triangle retention at m=8.
Only dimensions supported by earlier checks enter this grid. Larger-m results
do not establish nesting of the composed subspaces: superblock bases change.

The cap is 16 physical contexts x 4 partitions x 5 schedules = 320 cases,
with two independent eigensolver starts each (640 solves including the pilot).
Freeze finite measurement-batch sizes from pilot variance, sufficient to
resolve Section 8 limits. Record residuals, near-degeneracies, peak memory,
construction/solve/measurement time, and independent sampling seeds. If the
required retention or sampling precision is infeasible, preserve that outcome.

Compare physical E/N, q2, P(q), two-point functions, and connected spectra
against RBM-guided and Jastrow-guided full-spin GFMC separately. Match the
archived histogram bins and observable definitions. Keep training seeds,
projection repeats, and ancestry diagnostics visible. Missing baseline
observables remain unassessed; they cannot silently pass a criterion.

Require retention and partition differences within Section 8 limits, with
resolved statistical uncertainty, and agreement with both full-spin methods
where those methods agree. Preserve disorders60005/60006 as prespecified
disagreement diagnostics: earlier audits found substantial RBM–Jastrow P(q)
differences in these cases despite good within-guide repeats. Record the dated
baseline snapshot; these observations are not a new live campaign audit.
Do not choose a preferred guide after seeing the RG result or average away
per-disorder discrepancies. Ancestry alarms alone do not establish bias.

L4 GFMC supplies comparison data, not exact N=48 ground truth. A pass requires
the exact small-system tests, L4 retention/partition stability, and agreement
on resolved reference cases. Disagreement cases remain explicitly unresolved
and preclude a blanket physical-accuracy claim. Proceed to a diagnostic L6
pilot only with that limitation recorded; instability of RG itself blocks
expansion. L4 agreement also cannot establish accuracy at an untested field
such as Gamma=.70; the L6 field transfer needs its own sensitivity checks.

## 8. Acceptance criteria and error budget

For the physical Hamiltonian, lambda=1, use these engineering screening limits:

| Quantity | Maximum discrepancy from full-space exact reference |
| --- | --- |
| Energy per physical spin | .001 |
| q2 | .01 absolute |
| P(q) total variation | .05 |
| Any physical two-point function | .05 absolute |
| Each connected-spectrum component | `max(.10*max(abs(S_full),abs(S_reduced)), .001)` |

TV is `0.5*sum(abs(P_full-P_reduced))`. Apply each criterion per disorder,
field, and valid orientation; report maxima, medians, and the entire table.
An ensemble mean cannot conceal a failed case. Report q4, Binder quantities,
and gaps as additional diagnostics, without assigning them a retrospectively
chosen pass threshold. Ratios with a small denominator need their components
and uncertainty shown.

To leave room for solver error at L6, the preferred truncation target is half
each listed limit. A candidate that meets the outer limits but uses most of
the error budget is marginal and requires a revised precision allocation
before L6. Exact benchmark results have numerical residuals, not Monte Carlo
error bars; those residuals must be negligible relative to the screening limits.

The decision rules apply to both single-level and composed multiscale maps:

- **Promising reduction:** m=2 or m=4 meets every outer limit at lambda=1 in
  Stage B and C, preferably the half-limit targets, and has a feasible solver
  and physical-measurement path. The RG route additionally passes Stage D
  and the Stage E L4 assessment;
  a positive-weight projector additionally needs compatible signs. Proceed to
  a bounded L6 test.
- **Marginal reduction:** only m=6 passes, or passing m=4 has excessive
  interaction/measurement cost. Benchmark cost before proposing L6; the formal
  dimension change alone does not establish value.
- **Unsupported truncation:** no m<8 passes, or unresolved resonances and
  observable errors persist. Stop this uniform triangle construction and report
  the failure mechanism. No automatic larger campaign follows.

These limits screen feasibility. They are not accuracy requirements for a
critical-exponent fit or a guarantee that finite-size errors are controlled.

## 9. Conditional improvements, one at a time

Do not combine several repairs before learning why the leading projection fails.

**Virtual discarded-state corrections.** If discrepancies are small and
discarded-state denominators are safely separated on all relevant connected
clusters, test a second-order Hermitian downfolding on Stage B first. For
retained product states a,b and discarded intermediate states r of H0,

\[
(H_{\rm eff}^{(2)})_{ab}
=E_a\delta_{ab}+\lambda V_{ab}
+\frac{\lambda^2}{2}\sum_{r\in Q}V_{ar}V_{rb}
\left[\frac1{E_a-E_r}+\frac1{E_b-E_r}\right].
\]

This expression is a diagnostic finite-cluster construction. A local cutoff
gap does not separate the entire many-body retained spectrum from its discarded
spectrum; resonances invalidate a naive global use. For larger systems, derive
and validate the connected-cluster expansion and denominators explicitly.
Virtual processes can generate terms involving more than two blocks.

Transform observables with the same downfolding map and expansion order.
Second-order energies paired with bare projected observables do not define a
consistent corrected overlap theory. Corrected physical probabilities and
normalization need independent verification. Downfolded energies need not retain
the variational upper-bound property. If resonances occur, enlarge the retained
space or stop; do not regularize denominators silently.

**Nonuniform retained dimensions.** If failures concentrate on a few triangles,
consider a deterministic local leakage/gap rule for choosing m_t. Define and
freeze the rule using a designated development subset before testing new
disorders. Record `sum(log2(m_t))` and actual cost. This is a new finite study,
not permission to tune m_t until each known overlap passes.

**Environment-adapted blocks.** If isolated triangle states are consistently
poor, compare a basis from reduced density matrices of exactly solved coupled
blocks. Such a basis targets environmental entanglement but introduces a
training/transfer problem. It requires separate validation on held-out
disorders and does not supply a controlled expansion automatically.

### 9.1 Proposed research path: weak coupling → transition search → finite-size scaling

The higher-field results motivate continuing the method study. The working
hypothesis is that truncation becomes more reliable when couplings from retained
to discarded states are small relative to their excitation-energy denominators.
For a connected cluster at a given RG level, a diagnostic is

\[
\eta=\frac{\|Q H_{\rm boundary}P\|_2}{\Delta_{\rm connected}},
\qquad P=\text{retained-space projector},\quad Q=I-P.
\]

Here Delta_connected is the smallest positive discarded-minus-retained
unperturbed energy difference among states coupled by the boundary interaction,
as specified in Section 4. At the triangle level H_boundary=lambda V_boundary;
later levels use their actual projected boundary terms. Retain the individual
numerators and denominators, their distribution over blocks/disorders, and
invalid cases. Weak bare interaction alone is insufficient near degeneracies;
small eta is a diagnostic, not a global error bound or convergence proof.

The complementary diagnostic uses the interpolation in Section 3.1, keeping the archived K arrays and full
on-site fields fixed while multiplying only intertriangle bonds by lambda.
At lambda=0 the ground state is a product of triangle ground states. The
composed hierarchy reproduces it if both levels retain the needed product
ground-state subspace. Degenerate ground spaces require an explicit state or
ground-manifold convention, matched in reference and reduced calculations.
The triangle endpoint alone does not certify a later superblock cutoff.

For lambda<1 this is a modified kagome model with unequal strengths on the two
triangle orientations. A transition found there belongs to that model until
its continuation toward lambda=1 is demonstrated. Weakening every K at fixed
Gamma,h is another physical trajectory toward field-dominated behavior; it is
distinct from weakening only intertriangle bonds. Uniformly rescaling all
energies cannot improve the ground-state approximation at fixed retention:

\[
H(aK,a\Gamma,ah)=aH(K,\Gamma,h),\qquad a>0.
\]

Eigenvectors and physical P(q) are unchanged under this common rescaling.
Record relative interaction/field scales and actual Hamiltonians for every case.

1. **Establish a reliable region.** The primary screen lowers both endpoints
   of every bond distribution, at lambda=1, and scans Gamma,h as specified below.
   Use matched full-space L2 references for each physical Hamiltonian. Retain
   the completed intertriangle-only interpolation as a complementary diagnostic.
   Add intermediate points where accuracy changes. Validate the second
   truncation on Stage D before qualification on L4. On L4, use matched
   full-spin benchmarks for the same K,Gamma,h; existing original-law references
   cannot validate new bond laws or lambda<1 Hamiltonians.
   Compare m=4,6,8 and chi=4,8,16 where affordable, emphasizing chi sensitivity
   after the higher-field checks. Measure energy, physical P(q), q2, connected
   correlations, leakage, and partition dependence. For modified-model energy
   screens, declare the energy unit and tolerance in the manifest; common
   energy rescaling must not create an artificial pass. Use the existing
   dimensionless observable screens as an initial feasibility target, then
   tighten them for the intended phase-search resolution. Preserve failures
   and held-out disorder checks; do not infer a continuous reliable region
   from isolated passing points.

2. **Search for collective behavior in that region.** Refine Gamma,h at each
   promising fixed bond distribution using equal disorder weighting and reconstructed
   physical-spin observables. Examine physical P(q), its moments, tails and
   possible secondary peaks, together with spatial connected correlations
   C_ij=<ZiZj>-<Zi><Zj>. A candidate glass diagnostic is
   chi_SG,conn=N^(-1) sum_ij [C_ij^2]_dis; it differs from raw N<q2> at h=.15.
   Define any connected-overlap Binder ratio and correlation length from the
   same physical connected field, storing their underlying moments and
   structure factors. Fix the kagome sublattice/Fourier convention before
   extracting xi_L/L. Treat longitudinal polarization and the nonzero raw
   overlap background explicitly. Histogram shifts alone do not identify a
   transition. At h=.15 leave transition versus crossover open, and record
   the possibility that weak coupling suppresses collective ordering entirely.

3. **Attempt finite-size scaling only after retention convergence.** Use
   physical L (N=3L^2), matched boundaries and disorder laws, and the same
   triangle/superblock blocking depth and construction rules across sizes.
   The number of effective sites is not the physical linear size. Allow m,chi
   to grow as required by accuracy; freeze the convergence rule rather than
   assuming a fixed small chi is sufficient. Add sizes beyond L4/L6 only after
   a finite resource and accuracy assessment. Two small sizes or two disorders
   cannot establish an exponent. Test crossings of connected dimensionless
   quantities, their drift with size, and stability under retained dimensions,
   superblock origins, size-window changes and disorder resampling. At lambda=1
   also test triangle-orientation stability; at lambda<1 hold the physical
   interpolation fixed. Truncation shifts must be smaller than both the
   observed size dependence and the stated crossing/fit uncertainty. Carry
   disorder, measurement, solver and truncation errors separately. Establish
   the ground-state limit for every size before fitting; select a justified
   scaling form and correction terms only after a stable candidate crossing.

If a transition line Gamma_c(lambda) is supported, follow it toward lambda=1
while repeating these checks. Near criticality the required retained space may
grow despite weak bare boundary couplings. Failure of convergence or absence
of a stable crossing is a reportable outcome, not evidence for a critical
point. The deliverables are a reliability map, a transition/crossover assessment
within it, and a scaling study only where its error budget is resolved.
The eventual scaling targets are q_EA and a correlation length constructed
from physical connected glass correlations, with the field-induced background
in q_EA treated explicitly. The prerequisite is a working interaction interval
where physical P(q) exhibits a resolved field-driven change and the retained
description is stable. A changing P(q) can reflect local polarization alone;
it is a necessary exploratory signal here, not proof of collective glass order.

**Primary all-bond endpoint screen (authorized 2026-11-04).** Lower both ends
of the distribution for every bond: K=s K_original, with s=1,.5,.2,.1 and
K_original~U(.5,1.5). Thus the four distributions are U(.5,1.5), U(.25,.75),
U(.10,.30), U(.05,.15). Use identical base disorder arrays within each size;
L2 and L4 use distinct size-dependent seeds. Keep lambda=1; no triangle
orientation is privileged by the physical bond law. The initial grid uses
Gamma=.05,.10,.25,.50 (rows) and h=0,.025,.075,.15 (columns), one 4-by-4 L4
figure for each bond distribution, with an exact L2 companion. Compare
(m,chi)=(4,8),(6,8),(6,16), disorder60000, orientation0 and origin(0,0):
64 exact L2 contexts and 192 L4 models, 8192 independent replica pairs each.
Use two eigensolver starts; at h=0 explicitly restrict both exact and retained
solvers to even global spin-flip parity, since exponentially small tunnelling
splittings can otherwise produce arbitrary asymmetric mixtures. Preserve
the unrestricted pilot separately. Save cutoff, solver-start and sampling
alarms, and exclude alarmed panels from provisional candidates. Normalize
energy discrepancies by mean K before applying the .001/spin screen. This
keeps a smaller energy unit from producing a trivial screen improvement.
Adjacent passing contexts must also show resolved P(q) motion before choosing
a window. Raw results and source provenance belong under
`data/kagome/triangle_rg_bond_ranges_v1/`. The finite scan is complete:
192 L4 models and 64 exact L2 contexts, with no calculation failures and
zero contexts passing the full provisional retention screen. It provides
context for locating interesting Gamma,h values; it does not certify L4
physical accuracy or a transition. The immediate priority is the same-model
exact L2 comparison below.

**Complementary intertriangle screen (completed 2026-11-04).** Follow the 4-by-4 panel
format in the section "Selected 4 × 4 comparison: exact, guide and guided GFMC"
of `exact_guide_overlap_distributions.ipynb`, which itself uses L2/N12. The new
figure uses L4/N48, with rows Gamma=.25,.5,1,2 and columns lambda=0,.1,.3,1,
h=.15, disorder60000, triangle orientation0 and superblock origin(0,0).
Compare (m,chi)=(4,8),(6,8),(6,16): 48 L4 retained models, two solver starts
and 8192 independent physical replica pairs each. A 16-context exact L2
companion checks m=4,6,8 at the same parameter values; exact full-spin L4
P(q) is available only at the decoupled endpoint. Overlay fields at each
lambda to expose changes hidden by the panel separation. Preserve full
couplings, states, isometries, samples and uncertainty/provenance records.

This is a diagnostic screen of the new weak-coupling route. A provisional
candidate panel needs TV<.05 across both m and chi comparisons, chi energy
change<.001/spin and q2 change<.01, matching exact-L2 m=6 screens, and resolved
cutoffs. Declare candidates at nonzero lambda only. Inspect the field-shift
size alongside sampling error and retention sensitivity, and identify intervals
using neighboring passing panels rather than one point. L2 checks and L4
retention stability do not replace Stage D or matched full-spin L4 benchmarks;
no panel is certified accurate from this scan alone. Refine candidate lambda
and Gamma ranges and test disorders/partitions before scaling. Outputs belong
under `data/kagome/triangle_rg_interaction_v1/` with a dedicated results notebook.
The completed screen has only two nonzero-lambda candidates: lambda=.1 at
Gamma=1,2. Their mean-q change is -.00344 (sampling SE .00239), while the
larger lower-field shifts fail chi energy sensitivity. No broad interval with
both visible shifts and retention stability is established. Preserve this
meaningful diagnostic alongside the all-bond scan. The original full-grid
and L6 campaigns remain stopped.

**Previous same-Hamiltonian diagnostic (2026-11-04).** Compare the archived
full 12-spin ground state directly with first-stage triangle projections at
m=4,6,8 for Gamma=.25,.5,1,2 and lambda=0,.1,.3,1, h=.15. Each reference and
approximation uses identical physical bonds. The primary notebook now shows
one exact-versus-RG four-panel figure per fixed lambda, Gamma overlays,
physical moments and exact errors; previous L4 plots remain an appendix.
All L2 probabilities are deterministic XOR convolutions of reconstructed
physical Born distributions, without replica sampling uncertainty. This
tests the first truncation only; no chi truncation occurs in these L2 solves.

On this disorder, maximum m6 TV is .00426688; full m8 recovery has maximum
TV 9.60e-14. Smaller lambda generally improves errors at fixed m, with an
exact product-triangle endpoint at lambda=0. It is a useful validation path,
not a guarantee near small excitation denominators and not evidence for the
unchanged lambda=1 model. Compare adjacent exact field motion to the sum
of endpoint TV errors, which bounds error in the distribution change. Use
these comparisons to choose L4 retention/disorder follow-ups, then progress
gradually toward L6 diagnostics; first-stage L2 success does not resolve
the second-stage truncation. Results and audited source provenance are in
`data/kagome/triangle_rg_interaction_v1/l2_comparison_20261104/`.

**Corrected primary comparison: keep the original target fixed.** The exact
reference always uses lambda=1. Compare it against RG at lambda_RG=0,.1,.3,
.5,.7,.85,1 for Gamma=.25,.5,1,2, h=.15, m=4,6,8 and identical base disorder
arrays. The main notebook now contains only physical distributions versus
this fixed target, target-error curves and measured tables. The previous
same-lambda and L4 figures are archived separately. At lambda_RG=.7 the m6
target TV errors are .160675,.166320,.037272,.012092 in Gamma order. The
first two deviations persist at m8: weakening the bonds changes the model
even when the triangle space is complete. Compare total target error,
field-response error and energy expectations evaluated under the target
Hamiltonian. Do not substitute agreement with the weakened exact model for
agreement with the original model. Raw new contexts, original references,
states, isometries and provenance are preserved in
`data/kagome/triangle_rg_interaction_v1/l2_fixed_target_20261104/`.

## 10. Conditional L6 experiment

**Active scope correction (2026-11-04):** keep K~Uniform(0.5,1.5) on every
bond, lambda=1, Gamma=0.5 and h=0.15. The gradual L6 diagnostic now averages
eight disorders (60000–60007) at each of L4 and L6, with
(m,chi)=(4,2),(6,2),(6,3): 48 cases, one CPU worker. Use equal disorder
weights, histogram SE across disorders and paired retention differences.
This supersedes adaptive-lambda and higher-field choices for active work.
The previous adaptive campaign was stopped and its completed data archived.
Six original-bond single-disorder cases completed successfully and are reused
with their original source snapshot and recorded seeds. Current campaign:
`/root/autodl-tmp/triangle-rg-l6-uniform-g05-disorder8-20261104-v1/`.
Tiny chi is a diagnostic; L4/L6 physical accuracy and convergence remain open.
**Completed:** all 48 cases finished with no calculation failures or diagnostic
alarms. Raw archives, source snapshots, histograms, direct moments and disorder
averages were verified locally. At L6, chi=2→3 at m=6 changes q² by
+0.029764 ± 0.015917 and energy/spin by -0.015096 ± 0.002590 (paired disorder
SE). This retention sensitivity prevents a convergence claim. The dedicated
local notebook is `notebook/triangle_effective_l6_results.ipynb`; every plotted
bar is disorder SE.

**Completed L6 Gamma scan (2026-11-04):** Gamma=0.1,0.25,0.5,0.75,1.0
at fixed h=0.15, lambda=1 and original K~Uniform(0.5,1.5) on every bond.
Match disorders60000–60007 and all three retention settings at every field:
120 cases, including the 24 reused Gamma=0.5 cases. One CPU worker; preserve
source snapshots and raw diagnostics. Production plots show disorder SE only.
Report paired field changes and retention sensitivity at each field.
Remote: `/root/autodl-tmp/triangle-rg-l6-gamma-scan-20261104-v1/`.
Local delivery: `data/kagome/triangle_rg_l6_gamma_scan_v1/`; report:
`notebook/triangle_effective_l6_gamma_scan.ipynb`. All 120 cases completed
without calculation failures or diagnostic alarms. All raw hashes, source
snapshots, physical measurements, disorder averages, paired changes and
embedded figures were verified locally. Maximum physical-energy consistency
error was 5.12e-13. Retention sensitivity remains; convergence is unresolved.
The older grids below remain conditional proposals, not an active launch queue.

Enter this stage only after documenting the small-system and solver decisions.
For the multiscale route, Stages D and E are required, including the recorded
L4 reference limitations. Start with one pilot context:
Gamma=.55, sample60000, upward triangles, origin `(0,0)`, m1=4 and chi=4,
provided those dimensions pass earlier gates. Solve its final nine-superblock
Hamiltonian with matrix-free iteration and demonstrate validated physical-spin
sampling before expanding. Store residuals, low-energy gaps when resolved,
isometries, and physical measurements. A converged reduced-space eigenproblem
does not itself establish accurate physical L6 overlaps.

For the subsequent finite multiscale comparison, use Gamma=.55,.70,
samples60000,60001, both triangle orientations, and origins `(0,0)` and `(1,1)`:
16 contexts. The base schedule is `(m1,chi)=(4,4)`. Compare `(6,4)` to probe
first-level sensitivity if m1=6 passes prior checks; compare `(4,8)` to probe
second-level sensitivity only after the memory/time gate passes. This gives
16 base schedule/context cases and at most 32 additional sensitivity cases,
less any preserved pilot duplicate. Freeze the surviving dimensions and resource
allocation in the launch manifest before expanding. If a candidate dimension
fails its earlier gate, record that branch as unsupported rather than launching
it for the sake of completing the grid.

Use a second independent eigensolver start for each accepted schedule/context,
with overlap and energy residual checks that distinguish numerical uncertainty
from truncation bias: at most 96 eigensolver runs across the 48 cases, including
preserved pilot runs. Keep independent physical-measurement batches and seed
identities; choose their finite size from measured pilot variance so confidence
intervals resolve the screening limits. Exact reduced-space solves have no
guide-training seeds. Low-energy near-degeneracies require their own numerical
assessment and cannot be resolved by selecting the preferred overlap result.

The alternative single-level many-body-solver route has the following separate
budget; do not launch both routes automatically.
Start with Gamma=.55 and .70, samples 60000 and 60001, and both orientations:
eight contexts. Use the smallest passing m and the next larger passing
truncation; their actual dimensions must be reported. If the solver is
stochastic, retain two independent training/initialization seeds and two
independent measurement repeats per seed. This gives at most 64 base run
records for two truncations. Any population, projection-time, or representation
sensitivities require a separate finite grid specified before their results.

Before launching this grid:

1. Benchmark one context to select and record numerical budgets, peak memory,
   solver representation, measurement method, and timing. A failed pilot is
   preserved and assessed before expanding the grid.
2. Match the full physical Hamiltonian to the archived L6 bond hashes. Use the
   same normalization and physical-spin observables as the baseline campaign.
3. Set an uncertainty budget that can resolve the screening tolerances. Use
   independent runs and valid autocorrelation/block analysis; walker or
   descendant counts do not define the number of independent observations.
4. Compare total cost, including block construction and physical-spin decoding,
   with the full-spin baseline at comparable measured precision. Keep training,
   projection, and measurement timings visible.

Require agreement across retained dimensions, triangle orientations, independent
seeds, and solver sensitivities. Continue to report ancestry/weight diagnostics
where relevant. An alarm is evidence to investigate; it is not proof of a wrong
distribution. Absence of alarms is not proof of physical convergence.

The archived full-spin L6 results are comparison data, not ground truth. If the
reduced model is stable while the full-spin guides disagree, label it a promising
approximation with unresolved L6 truncation bias. Claims of physical convergence
require an independent accuracy argument. Two disorders cannot establish a
disorder-averaged phase diagram.

Practical success means reproducible physical overlaps plus a measurable cost
benefit. Use a factor of two reduction in total time to the same resolved
precision as an initial cost target, subject to measured shared-machine
throughput. If the full-spin baseline never attains that precision, report
cost and stability separately; do not assign an invented speedup.

## 11. Implementation, provenance, and execution

Proposed components under `code/transformer_qmc/`:

- `triangle_effective.py`: block partitions, full-field local Hamiltonians,
  isometries, projected physical operators, and interaction tensors, including
  a term-support representation that survives successive block merges.
- `triangle_effective_rg.py`: explicit hierarchy maps, superblock spectra,
  retained-state schedules, composed isometries, matrix-free top-state solves,
  and sequential joint physical-spin sampling.
- `triangle_effective_exact.py`: independent small-system actions,
  eigensolutions, lifted probabilities, XOR overlap histograms, and leakage.
- `triangle_effective_campaign.py`: finite task manifest, resumable execution,
  hashes, and explicit failures without automatic infinite retries.
- `triangle_effective_report.py`: case tables, acceptance decisions, error
  versus m/lambda/orientation, and cost summaries.
- `test_triangle_effective.py`: independent algebraic and physical-observable
  checks from Stage A, including cases that expose incorrect operator products
  and incorrect independent-block decoding.

Reuse geometry and established conventions, while keeping the independent full
physical Hamiltonian as the reference. Do not weaken existing exact-size guards
or modify ongoing campaigns to accommodate this study.

Proposed output root: `data/kagome/triangle_effective_v1/`, with manifest,
source snapshot/commit, environment, disorder arrays, site/block maps,
eigensystems, projected tensors, solver states, raw observables, failures,
diagnostics, and a final assessment. Provide an executed results notebook only
after the corresponding data exist. Hash the archived data and reconstruction
inputs; preserve an initial delivery if the report is later revised.

Run scientific computation and notebook execution on AutoDL. Inspect available
resources before choosing a finite concurrency allocation, preserving the
authorized running campaigns and machine quota. Measure per-task throughput
before offering an ETA. Never shut down the instance. L8/L10 and cancelled
worldline work remain outside this study.

Implementation order is Stage A, Stage B, assessment, conditional Stage C,
conditional Stage D for the RG route, required Stage E L4 benchmark, then a
recorded solver/L6 decision.
No scientific run is launched by writing this plan.

## 12. Independent geometry option

The alternative direction follows
[Brodoloni et al., arXiv:2505.05117](https://arxiv.org/abs/2505.05117), using
the [published coordinates](https://zenodo.org/records/15276170). Their model
uses distance-dependent `J0/r^6` interactions and h=0, with correlated positional
disorder. Our present model uses iid nearest-neighbor bond disorder and h=.15.
An amorphous arrangement changes the physical problem and retains `2^N` states.

If this option is chosen independently, begin with a fixed small subset of
published N=40 patches and matched regular kagome open patches. Prespecify the
interaction normalization, boundary handling, tail treatment, and longitudinal
field. Pass identical coupling matrices to every solver. Separate the geometry
effect from a field or interaction-range change, and measure computational
performance rather than assuming amorphous arrays converge more easily.

The current priority proposed here is to determine whether the existing model
admits a useful triangle reduction. Either outcome can inform a later geometry
study, but the two directions need not be implemented together.
