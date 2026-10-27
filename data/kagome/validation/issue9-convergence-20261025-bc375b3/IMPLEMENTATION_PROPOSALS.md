## Proposed implementation roadmap after bc375b3: improve the estimator, not just the cutoff

The maintainer has asked for concrete change proposals in the review feedback. Below is a staged implementation proposal, **not a claim that these unimplemented changes solve the problem**. Keep the verified low-field improvement in `bc375b3`. No need to replace the whole projector as the first step.

### 1. Make the stopping policy explicit and make failures auditable

**Targets:** `kagome_qea/pqmc.py:_project_once`, `_projection_record`, `_combine_populations`; `trial_overlap.py` save/cache/queue interfaces.

- Add explicit configuration for `min_effective_ancestors` (current default8) and `min_unique_ancestors` (current default2). Pass the chosen values through `project`, store them, and include them in cache identity. Preserve the current production defaults until a comparison justifies a change.
- Return per-population/per-cycle stop reason, achieved time, unique count and descendant Kish size at the accepted state; also store the attempted step's failing diagnostics when a rollback occurs. Add the largest descendant fraction and a bounded diagnostic time series or predeclared checkpoints. A single global minimum is not enough to explain where diversity disappears.
- Distinguish **weight ESS before resampling** from **descendant/ancestral ESS after resampling**. Equal weights immediately after resampling do not mean ancestral diversity has recovered.
- Add an explicit production-validation entry point that refuses qualification when any required cycle truncates. Keep valid incomplete archives for exploratory work, but do not present their average as a common requested-time estimator. Do not discard incomplete cycles and average only survivors, or retry seeds until enough complete: that would condition the selected sample on completion.
- Reserve `projection_complete` for attained-time completion and a separately named/report-level qualification decision for demonstrated convergence. A flag based only on completed time and population count is not an accuracy certificate.

**Acceptance:** synthetic stop/rollback tests; saved/reloaded settings and diagnostics consistent with the actual cycle arrays; the same nominal8/16 truncation is visibly rejected as a convergence test; constant defaults reproduce the current benchmark. This is the smallest useful engineering patch before changing sampling.

### 2. Prototype importance sampling with a positive guiding function

**Primary algorithmic proposal:** improve the distribution of walkers so branching has less work to do, rather than simply lowering the lineage safeguard. This is a candidate to benchmark, not a promised cure for L6.

There is already a reusable interface in `code/gfmc/guide.py` (`Guide.log_amplitude`, `Guide.log_ratio`, `ConstantGuide`) and guided-rate machinery in `code/gfmc/projector.py`. Adapt/shared-test that logic for the two-replica `kagome_qea/pqmc.py` path instead of inventing a second convention.

For a positive single-replica guide `g(s)`, use a product guide for the noninteracting pair, `G(sA,sB)=g(sA)g(sB)`. With the repository's off-diagonal Hamiltonian entry `-Gamma`, the proposed discrete importance-sampling implementation follows directly from the similarity transform `H_tilde = D_G H_pair D_G^(-1)`:

- Flip channel `(replica r, site i)` has rate `Gamma * exp(log_g(s_r_flipped)-log_g(s_r))`.
- The waiting-time escape rate is the **sum of all channel rates**, and the selected channel is sampled proportional to its rate. It is no longer a uniform site choice except on the constant-guide path.
- The pair local energy is `E_diag(sA)+E_diag(sB)-sum(channel_rates)`. Use this local energy consistently in weight propagation and population control, then divide pair-energy estimates by2. Recompute/update rates after each flip.
- Preserve the descendant-weighted pure-overlap estimator and verify it explicitly. A guided histogram alone samples the mixed distribution and is not a replacement for the pure susceptibility estimator. Do not add an arbitrary extra guide factor to q tags without deriving and testing the normalization.
- `_branch_dt` currently uses the diagonal-energy spread. In a guided path, use the actual **local-energy** spread for the analogous branching heuristic; retain a bounded maximum interval and benchmark any interval change. Avoid interpreting a finer resampling interval alone as a cure; it did not establish longer propagation in the implementation comment.
- Use stable log-ratio/rate handling and explicit invalid-value failures. Do not silently cap ratios or rates in a way that changes the intended transformed generator. Include guide type/parameters and all relevant implementation dependencies in the run fingerprint.

**First experimental guide:** a small positive Jastrow/Boltzmann-form ansatz, e.g. `log g_beta(s) = -beta * E_diag(s)/2`, so `log_ratio=-beta*DeltaE_diag/2`. Here beta is a **variational inverse-energy parameter**, not a temperature or the projection time. Beta0 gives the constant guide. Keep units/scaling explicit when sigma changes. Start with a small predeclared parameter family; optimize energy/variance in a separate pilot, freeze parameters, and evaluate on held-out seeds/disorder. Do not choose beta or the Kish floor solely to match exact chi on the validation cohort. A richer guide is only warranted if this simple one demonstrably falls short.

**Acceptance before relying on this path:**

1. Constant-guide/zero-parameter tests reproduce existing rates, energies, and estimator behavior; preserve the flat fast path for direct comparisons.
2. Tiny-state-space test: construct the transformed generator and compare short-time evolution with matrix exponentiation of the exact similarity-transformed Hamiltonian. Check the local-energy and channel-rate identities.
3. An **exact positive ground-state guide on tiny systems** is an oracle: local energy should be constant within solver residual/tolerance, and appropriately equilibrated pure-observable estimates should agree with exact values. This is a unit/diagnostic test, not a deployable L6 guide.
4. Compare flat versus candidate guide with identical declared budgets and held-out seeds: attained time, full lineage diagnostics, local-energy variance, independent-population uncertainty, and error against exact susceptibility. Better completion alone is insufficient.

The formulas above are a proposed adaptation to this discrete Hamiltonian; they are not copied as a continuum DMC implementation. Background: Casulleras & Boronat, Phys. Rev. B52,3654 (1995), DOI `10.1103/PhysRevB.52.3654`, explains guided mixed versus descendant-weighted pure estimates. Samaras & Hamer, Aust. J. Phys.52,637–657 (1999), DOI `10.1071/PH98092`, tests forward-walking correlations in the transverse Ising model and reports trial-wavefunction sensitivity. These sources motivate testing a guide; they do not validate the proposed Jastrow ansatz for this Kagome problem.

### 3. Separate equilibration diagnostics from the changing forward-walk window

**Targets:** `pqmc.py` block-recording/measurement setup and a separate reproducible validation driver.

- Store an equilibration trace and a fixed pre-forward measurement-window energy trace, with times. Present them separately from energies appended during forward walking. Otherwise changing the forward horizon also changes the energy-averaging window.
- For L6, predeclare an equilibration grid such as40/80/160/320 blocks at fixed walker count and fixed measurement window; use a bounded pilot and continue only if the shorter grids remain unstable. Compare independent random starts and independently generated low-energy starts, preserving correct dynamics and symmetry. Starting low in energy is not itself proof of equilibration.
- Assess stationary behavior across windows and start policies, then inspect susceptibility too. Energy agreement alone cannot certify a pure overlap observable.
- Keep uncertainty across genuinely independent populations. Eight forward cycles sharing the same evolving population are not eight independent replicas.

**Why priority is justified:** in the fresh-seed bc375b3 L6 check, doubling equilibration changes energy by **-1.858366 +/- .265795** in the paired comparison. This is not fixed by the new stopping criterion. Treat it as a separate dependency, not as something another cutoff adjustment can certify.

### 4. Use a predeclared validation matrix and accuracy criterion

Build a small command-line validation driver (the saved AutoDL `study.py` harnesses are a starting point), producing machine-readable raw population records plus a compact report. Do not bury production acceptance in an ad hoc `abs(error)<2*SE` test.

- Minimum diagnostic matrix: L2 samples0/1/2, low/intermediate/high fields, **achieved** time8 versus16, walker counts5000/10000/20000 where justified, and the separate equilibration checks. Reuse matched child indices for paired changes and reserve independent seed cohorts for confirmation. Add more disorder draws before any disorder-averaged production claim.
- Include the observed sample1/time8 and sample0/10000-walker deviations as predeclared holdouts. They are about3SE and2.38SE respectively in one cohort; this is a reason to investigate, **not** proof of a bug after many comparisons.
- Specify desired absolute/relative susceptibility accuracy **before** testing. The project has not yet set that numerical tolerance; do not invent one from the observed scatter. Report confidence intervals for exact-reference error and paired parameter shifts, and require sufficient precision to assess that tolerance. Huge error bars must not turn an inaccurate result into a pass.
- Account for paired covariance, many comparisons, repeated stopping/look decisions, and shared ancestry. For small population counts use an appropriate finite-sample interval rather than treating every SE multiple as an exact Gaussian p-value.
- Retain the deterministic finite-projection references as a diagnostic. For flat-guide sample0/Gamma=.2, the prior ideal time8 result remains about0.025% below ground state: agreeing within current MC noise does not make this bias identically zero. The appropriate time depends on the requested accuracy and the sample, and cannot be extrapolated blindly to L6.
- Pass L6 only after longer actually achieved projection and population/equilibration changes are stable to the predeclared accuracy. Do not use agreement between two identically stopped requests as evidence.

### Suggested patch order

1. **Explicit stopping configuration + complete diagnostics + fixed-window traces**, with unchanged defaults and regressions.
2. **Optional guided PQMC path + tiny exact-transform/oracle tests**.
3. **Held-out convergence campaign**, using a frozen guide and predeclared stopping policy.
4. Only after demonstrated success: choose production settings and deploy the validated revision. If the guide still cannot maintain usable ancestry at the needed horizon, evaluate a different pure-estimator/projector design as a separate measured effort rather than repeatedly lowering the floor.

This is proposed work for the implementing agent. **No new solver edits or AutoDL jobs were launched for this feedback.** Keep #9 open and production HOLD pending measured acceptance.
