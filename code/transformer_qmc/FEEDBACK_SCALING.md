# L6 / L8 / L10 iterative feedback — 2026-11-03

## Priority and live finite scope

The user explicitly prioritized **L6, L8 and L10**. The proposed five-arm L2
continuation campaign was **never launched**. Small systems receive only short
implementation checks; new research compute targets the three requested sizes.
The already-running finite L6 pilot, L4 production and fixed-guide L6 controls
continue with their original isolated sources.

At **2026-11-03T15:12:24.982677Z**, three detached size workers launched under
`/root/autodl-tmp/self-learning-frontier-20261103-v1/`:

| Size | Sites | PID at launch | First operation |
|---|---:|---:|---|
| L6 | 108 | 393149 | Feedback from archived VMC400 control |
| L8 | 192 | 393150 | Repaired, small-learning-rate VMC warm start |
| L10 | 300 | 393151 | Repaired, small-learning-rate VMC warm start |

All were alive and computing at 15:12:46Z; this is a dated observation.
There is one sequential worker per size, no deadline, no retry loop, and no
polling monitor. Checkpoints permit interruption/resumption; the instance stays on.

Each size uses Gamma=.60, longitudinal h=.15, periodic kagome, the established
coupling/disorder convention, disorders60000/60001, and both seeds0/1. All
contexts remain in the record. Same numeric disorder ID at different sizes
does not mean the couplings are identical or nested.

Two paired arms per context:

- **Baseline:** one1024-walker population; persistent two-copy Gibbs negatives.
- **Robust:** four independently normalized1024-walker populations;16-temperature
  two-copy replica-exchange negatives; empirical guide-step backtracking.

Both arms start from the exact same checkpoint for that size/disorder/seed.
L6 reuses the archived400-target-update VMC checkpoint and both fixed-guide
measurements. L8/L10 construct new repaired VMC guides:400 updates at each
Gamma1.5,.75,.60, Adam lr.001,256 chains, eight temperatures, two sweeps/update.
Legacy failed/cancelled campaigns are not restarted.

Each feedback context has12 stints. A stint has equilibration8, collection12,
delta.1,64 saved walker configurations/block in total,1000 likelihood updates,
batch256,10 negative sweeps/update, and lr.001. For the robust arm, equal16
configurations/island/block retain equal island mass. No ancestry-based
reweighting, sample pruning, relabeling or favorable-stint selection is used.

Two-stint dispatch alternates contexts/arms within each size. Training of all
contexts is prioritized, with frozen guides retained at six and twelve stints
for subsequent independent measurement. New population initialization and
equilibration occur at every guide change. Adam, negative chains, training
walkers, per-island RNGs and all diagnostics have resumable checkpoints.

Total finite scope: **24 training contexts,288 stints,288000 likelihood updates**;
96 milestone-guide measurements plus16 new L8/L10 fixed-guide measurements;
eight existing L6 controls are reused and independently reconstructed. This
is112 new measurements plus eight retained controls, not an unlimited queue.

## What changed and why

At14:58:45Z the original L6 pilot's first four completed stints had minimum
training lineage ESS13.267,2.484,9.977,10.903; largest lineage fractions
.1621,.6240,.2656,.2227. The second stint alarmed despite minimum one-step
weight ESS841.95/1024. Each target had about7580–7597 unique configurations
out of7680. Configuration uniqueness and high one-step ESS therefore did not
prevent severe genealogical concentration. Finite negative-chain bias and
guide-update instability remain separate possible contributors.

### Tempered negative phase

The model remains log psi(s)=a.s+sum_j log[2cosh((sW+b)_j)]. Its cold density
is proportional to psi(s)^2. The new ladder is the marginal of a two-hidden-copy
joint distribution with all joint parameters scaled by beta:

`log q_beta(s) = 2 beta a.s + 2 sum_j log[2cosh(beta (sW+b)_j)]`.

This is **not** beta times log psi(s)^2 at intermediate beta. At beta=0 it is
uniform and at beta=1 it is exactly the required squared amplitude. Blocked
Gibbs conditionals and replica exchanges use this marginal consistently.
The16 beta values are uniformly spaced from0 to1; exchange fractions and
temperature round trips are recorded, including bottlenecks. This remains a
finite-chain likelihood approximation, not an exact negative expectation.

Prior method motivation: Desjardins et al., *Tempered Markov Chain Monte Carlo
for training of Restricted Boltzmann Machines*, AISTATS2010, PMLR9:145–152.
Our two-copy marginal implementation and its finite-state tests are specific
to this repository's amplitude convention. The self-learning update follows
Pilati, Inack and Pieri (2019), arXiv1907.00907; the EA context is Brodoloni and
Pilati, arXiv2407.05978. Neither paper establishes this kagome finite-field case.

### Empirical step bound

After each Adam proposal, evaluate the centered change in2 log psi on the
current positive/negative minibatch. Backtrack the parameter displacement by
powers of two until RMS<=.025 and max absolute change<=.15. Up to eight halvings
are allowed; otherwise exactly restore the old parameters. Preserve Adam's
observed-gradient moments, and record every scale, RMS, maximum and rejection.

These are empirical minibatch constraints, **not** a global KL bound, a bound
on unvisited configurations, or a guarantee that a whole stint moves little.
No exact ground-state information controls the optimizer.

### Independent population targets

Four islands maintain separate resampling, shifts, ancestors and random streams.
They pool equal sample counts for fitting while retaining all per-island lineage
alarms. This increases compute and population diversity; it does not eliminate
finite-population bias. The two launched arms combine interventions and are
not a causal ablation of each ingredient or a compute-matched comparison.
The earlier five-arm ablation utility remains unlaunched.

## Large-size acceleration and validation

The continuous-time propagator and log-domain resampling are unchanged. Batched
RBM rates now run on CUDA for the new training/measurement paths. CPU remains
the default for existing callers. Device choice is recorded independently of
scientific measurement task identity. CPU/CUDA roundoff may diverge RNG paths;
equal task seeds do not imply bitwise paired trajectories across devices.

Short same-host benchmark of1024-walker rate/local-energy kernels:

| Size | CPU seconds | CUDA seconds | Kernel speedup |
|---|---:|---:|---:|
| L6 | .48242 | .01848 |26.1x|
| L8 | .94276 | .03212 |29.4x|
| L10 |3.77973| .06671 |56.7x|

These are three-call averages under current shared load, not end-to-end or
campaign speedups. They motivate executing larger-size guide evaluations on
the GPU; they are not convergence results.

**19 tests passed on AutoDL, zero skipped/errors/failures**, before dispatch:
exact tiny-state Gibbs marginal/invariance, exchange/inversion detailed balance,
likelihood gradient, feedback fixed point, empirical step bounds and rejection,
finite frozen sampler frequencies, complete CPU/CUDA interrupted-fit continuation,
interrupted multi-island projection, GPU two-stint dispatch continuation,
CPU/CUDA rates and energies at all three sizes, actual raw GFMC reconstruction,
nonfinite/missing-screen rejection, and finite matched-scope checks.

No long L2 campaign was needed for these checks. They verify implementation;
the new robust algorithm has **not** completed a physical convergence study.

## Physical outputs and interpretation

Frozen random/up/down diagnostics run separately. Final physical measurements
use four independent branching populations,512 walkers/population, equil16,
forward16,eight cycles,two repeats at each milestone and for the starting guide.
The task/RNG identities match within each size/disorder/seed, including the
existing L6 controls. Model hashes, source hashes and device provenance differ.
No physical tags or averages cross adaptive guide updates.

Every measurement is retained even if frozen sampling alarms. Histograms, q,
q2, connected spectra, energy and genealogy are reconstructed from raw tags
and descendant counts. L6/L8/L10 comparisons require joint evidence from
guide-training saturation, population/time/start dependence, negative-phase
mixing and independent physical observables. Screening levels remain energy/site
.001, q2 .01, overlap-histogram TV .05, spectrum10% with absolute floor.001;
uncertainty must resolve those tolerances. Ancestry alarms remain ESS<8 or
largest lineage>.5. Completion or better energy alone does not qualify a size.

The robust arm is a proposed improvement. Its success, the larger-size overlap
distributions and any thermodynamic/critical inference are still pending.

## First later checkpoint, 15:20:07Z

L6 disorder60000/seed0 completed the first two robust-arm stints. Worst-island
training ESS was1.161 then14.858. The first stint had one island reach92.8%
lineage concentration; the other three minima were11.995,18.463,10.357.
All records and raw chains are retained. The first population target preceded
the new fit, so that alarm is not evidence that tempering caused collapse.
The second-stint improvement is also not physical convergence evidence.

The first tempered fit recorded76699 temperature round trips, minimum exchange
fraction.3585,1000 bounded steps and zero rejected steps. These transport and
step-control diagnostics do not certify equilibrated negative expectations.
L8 and L10 had reached warm-start Gamma1.5 updates75 and50, respectively;
their first feedback stints and all new pure-observable measurements were pending.
