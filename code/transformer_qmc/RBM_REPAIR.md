# L6 RBM and branching repair (2026-11-03)

## Reproduced failures

The retained disorder is 60000, selected before the diagnostic. All interacting
controls use periodic kagome, N=3L², h=0.15, iid K in U(0.5,1.5), and J=-K.
Bond RNG is `default_rng(SeedSequence([20261027,L,60000,1]))`.

1. **Optimization:** the old N=108 Adam learning rate .01 is unstable in matched
   runs at Gamma=1.5 and 4. Global gradient norm clipping does not bound Adam's
   coordinate updates. With identical initial state, chains, ladder, and budget,
   .001 improves frozen sampled energy and transport. The default now uses .001
   for RBMs with N>48; overrides enter checkpoint signatures. This is an empirical
   remedy, not a proof that all weak-field fits converge.
2. **Arithmetic:** the original failed L6/Gamma=.6 uniform start sets the initial
   shift near -5.2 million. Its terminal log weights are near -519800. Ordinary
   weights all underflow to zero; the resampler's generic error obscures this.
   Log accumulation completes the same block but leaves ESS=3.65/512. Numerical
   stability and physical convergence are separate.

## Changes

- `qmc_numerics.systematic_log`: subtract a finite peak before exponentiation;
  distinguish empty, NaN/+inf, and all--inf masses.
- `projector.propagate`: accumulate log weights and log time integrals. Each
  interval contributes `log(w) + log(dt) + log(expm1(x)/x)`, with stable formulas
  for both signs of x. Signed local energy is combined as a time-weighted mean.
  No guide-ratio clipping, timestep approximation, or estimator change is added.
- Frozen random/up/down transport and energy checks precede guide selection.
  A context with no passing seed retains a visible failure; no disorder is
  substituted or selected for its overlap shape.
- Future repaired campaigns order weak fields across sizes before higher fields.
  Running legacy campaign source snapshots are preserved.
- Projector checkpoints identify the log-weight implementation; production
  provenance also includes the shared numerics hash.

## Validation

All computation and notebook execution take place on AutoDL.

`python -m unittest -v transformer_qmc.test_log_projector
transformer_qmc.test_sampled_simple_guides transformer_qmc.test_simple_guides
transformer_qmc.test_table_projector transformer_qmc.test_pure_gfmc
transformer_qmc.test_disorder_scaling transformer_qmc.test_core
transformer_qmc.test_correlation_remedy transformer_qmc.test_training_priority
transformer_qmc.test_parallel_disorder_scaling`

The 30-test suite includes exact short-time kernels, independent transverse spins,
N=108 exact product guides with extreme shifts, signed weighted-time integrals,
unchanged ordinary-range trajectories, sampler marginals at every temperature,
analytic RBM ratios, covariance gradients, checkpoint replay, and parallel reuse.

The measured control bundle is
`data/kagome/ltfim_disorder_scaling_v1/rbm_repair/20261103_v1/`.
`sha256.json` authenticates raw tags/counts, energies, model checkpoints,
source revisions, diagnostics, logs, and retained execution failures.
`reproduce_rbm_underflow` reruns the preserved failed first block.
`rbm_optimizer_diagnostic` runs the four matched step-size controls.
`strong_field_validation` performs training and independent-population projection;
`finalize_strong_field_validation` verifies and packages the completed results.

Gamma=4 projection compares two repeats for main, uniform-start, doubled-walker,
and doubled-time controls. L2 has an exact interacting reference; L6 has matched
RBM/Jastrow consistency checks only. Repeat SE from two repeats is exploratory,
not a precision certificate, and no disorder-averaged claim follows from one
disorder. No original near-unit L6 overlap is relabeled as a corrected result.

The separate 40-minute `rbm_weak_repair_pilot` prioritizes L6 Gamma=.55 then .60,
disorder 60000 and seeds 0/1. Its models and failures are retained independently;
it does not replace the planned eight-disorder weak-field ensembles.
