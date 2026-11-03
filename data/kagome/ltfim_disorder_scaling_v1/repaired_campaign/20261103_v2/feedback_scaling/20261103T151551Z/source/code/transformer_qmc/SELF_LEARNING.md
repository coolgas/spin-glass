# Iterative PQMC–RBM feedback — 2026-11-03

The current L4 production, L6 training-budget anchors, and .85/1.00 references
use energy-only VMC followed by fixed-guide GFMC. They do not implement the
self-learning protocol in Brodoloni and Pilati, *Zero-temperature Monte Carlo
simulations of two-dimensional quantum spin glasses guided by neural network
states*, https://arxiv.org/abs/2407.05978, Sec. II.2. Increasing VMC updates does
not test that protocol. These campaigns remain useful as fixed-guide controls.

The feedback rule follows Pilati, Inack and Pieri, Phys. Rev. E 100, 043301
(2019), https://arxiv.org/abs/1907.00907, Secs. II.3–II.4. For guide psi_s,
equilibrated importance-sampled walkers follow f_s proportional to psi_s psi_0
in the infinite-population/projection limit. Fit a normalized density p_theta
to these walkers by minimizing KL(f_s || p_theta), then use sqrt(p_theta) as
the next guide. Ideally psi_(s+1) is proportional to sqrt(psi_s psi_0); the
log-amplitude error, modulo a constant, halves at each iteration. The mixed
walker distribution is a training target, not a pure-observable estimate.

Our RBM.forward represents log(psi_theta), so p_theta is proportional to
exp(2 RBM.forward). The negative phase samples psi_theta squared using the
existing two-hidden-copy Gibbs transition. The likelihood gradient is
2(E_model grad log psi - E_walkers grad log psi). Finite persistent Gibbs
chains approximate the negative expectation. This is the same ideal guide
update, with our existing squared-amplitude family rather than the paper's
single-RBM probability family; it is not an exact reproduction of their
architecture or training schedule. At h=.15 the visible and hidden biases
remain trainable; the EA paper's zero-field inversion symmetry cannot be imposed.

`self_learning_rbm.py` implements a finite resumable pilot. Each stint starts
a new population from the current guide, excludes initial equilibration, stores
mixed walker configurations, then fits the next guide. Restarting the population
avoids carrying a distribution with the previous guide into a new importance
transform. All projection steps and likelihood parameters/RNGs are checkpointed
every 25 steps, with immutable input/output models and raw walker/fit diagnostics
per completed stint. No physical measurements or forward tags span guide updates.
Training ancestry is recorded and does not trigger selection of favorable samples.

Initial validation: exact likelihood-gradient comparison on an enumerated tiny
system, exact invariance of the two-copy Gibbs kernel, and the idealized feedback
fixed point. Then run finite L2 kagome pilots at h=.15, Gamma=.60, disorder60000,
seeds0/1, six stints each, 1024 walkers, equilibration8 plus collection12,
delta=.1, 64 randomly selected walkers/block, 1000 Adam updates/stint, lr=.001,
256 persistent negative chains and ten Gibbs sweeps/update. Exact calculations
are evaluation only. Archive fidelity, variational energy, ground-probability TV,
and KL to each empirical target before/after fitting. These choices are a pilot,
not convergence settings or an assertion of monotonic exact improvement.

L6 deployment follows assessment of these finite pilots and restart validation.
Retain both disorders60000/60001 and seeds0/1 in any subsequent L6 comparison,
compare the same starting VMC checkpoints, and independently measure final guides
with the matched four-population overlap protocol. Check population, projection
time, training-negative-phase mixing, P(q), q2, connected spectra and ancestry
jointly. Finishing adaptive stints or lowering energy does not certify convergence.
The 2D EA results concern square-lattice Gaussian couplings at zero longitudinal
field; they motivate this method but do not qualify our kagome finite-field case.

## Completed L2 validation and finite L6 dispatch

Both six-stint L2 pilots finished on 2026-11-03. The five implementation
regressions passed on AutoDL, including bitwise continuation of model, Adam,
negative chains, projection population, both RNGs and summaries after an
interrupted fit on CPU and CUDA. CPU and CUDA trajectories need not match
each other. The tiny restart tests validate continuation, not physics accuracy.
`report_self_learning.py` independently recalculated all 12 before/after exact
evaluations on CPU, checked continuity of every guide, validated raw arrays,
and verified 56 training artifact hashes.

| Final guide | Seed 0 | Seed 1 |
|---|---:|---:|
| Fidelity with exact ground state | 0.989079 | 0.990386 |
| Variational energy error/site | 0.003632 | 0.003151 |
| Configuration probability TV | 0.060198 | 0.054781 |
| Exact-enumerated guide overlap TV | 0.020306 | 0.011489 |
| Final-stint training lineage ESS, minimum | 240.61 | 254.02 |

Guide energy errors still exceed the 0.001/site screen. Configuration-density
TV and overlap-histogram TV are distinct quantities. Fidelity and probability
errors need not improve monotonically at finite training/sampling budgets.

`self_learning_measurements.py` ran new frozen Gibbs chains from random/up/down
starts (256 chains, 500 burn sweeps, 100 draws separated by ten sweeps). Against
the exactly enumerated model expectation, negative-phase sufficient-statistic
RMS discrepancies were 0.0039–0.0059; the largest component discrepancy was
3.08 chain-wise SE. These are useful finite checks, not proof that the adaptive
negative phase equilibrated at every update, and do not transfer to L6.

All four independent final-guide GFMC runs completed with no ancestry alarm.
They use four independent populations, 512 walkers/population, equilibration16,
forward16, eight cycles, two repeats per seed, and fresh measurement RNGs.

| Two-repeat mean versus exact | Seed 0 | Seed 1 |
|---|---:|---:|
| Mixed energy error/site | -0.0000106 | 0.0000964 |
| Pure overlap q2 error | -0.001035 | -0.004879 |
| Pure P(q) TV | 0.006969 | 0.009082 |
| Maximum connected-spectrum relative error | 1.76% | 1.43% |
| Minimum descendant ESS across repeats/cycles/populations | 85.67 | 76.70 |

Each individual run is also within the finite screening tolerances. Two-repeat
SEs are descriptive; they do not resolve all uncertainty or bound population,
forward-time and initial-state bias. The implementation and observable checks
support a diagnostic L6 feedback comparison, not ground-state qualification.

`self_learning_l6_pilot.py` was launched at 2026-11-03T14:40:16Z, PID391669,
under `/root/autodl-tmp/self-learning-l6-20261103-v1/`. Its explicit finite scope:

- Gamma=.60, h=.15, periodic kagome L6/N108; disorders60000/60001, seeds0/1.
- Start from the corresponding frozen 400-target-update VMC checkpoints in
  the lower-field anchor. Copy and hash each checkpoint, training diagnostics,
  and both completed fixed-guide control measurements before dispatch.
- Six feedback stints/context, each with 1024 walkers, equil8/collect12,
  delta.1, 64 walkers/block, 1000 likelihood Adam updates, lr.001, 256 persistent
  negative chains and ten Gibbs sweeps/update. Fresh Adam for the changed
  objective; preserve it across feedback stints. No exact training input.
- New random/up/down frozen tempering checks after each final guide, then the
  matched four-population protocol and two repeats: eight final measurements.
  Measurement task identities/RNG seeds match the fixed-guide controls, while
  model hashes and output directories differ. No training tags are reused.
- One sequential worker; 4 contexts/24 stints/24000 likelihood updates. No
  wall-clock deadline or automatic extension. Preserve every seed, failure,
  raw array, checkpoint and ancestry alarm. Sampling alarms are recorded and
  do not select away final diagnostic measurements.

The two finite-scope/matched-task regression checks passed before dispatch.
The scheduler was verified alive with its first projection checkpoint. Its
L6 output remains pending. Existing L4 production and fixed-guide L6 campaigns
continue; broad L6 production and L8/L10 remain conditional on reliability.

Archive: `data/kagome/ltfim_disorder_scaling_v1/repaired_campaign/20261103_v2/`
`self_learning/`, with SHA-256-protected artifacts, completed L2 raw data,
models, optimizer/RNG checkpoints, source snapshots, all regression logs,
independent measurements, and the dated L6 launch/manifest. Separate complete
source snapshots preserve the exact L2 validation and L6 dispatch implementations.
