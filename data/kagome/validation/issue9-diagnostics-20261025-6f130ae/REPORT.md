## Independent AutoDL review of 6f130ae: instrumentation verified, rollback defect needs priority

Reviewed commit `6f130ae809215bdcfc579239b0f6d2fdad540ac9` and the new implementation comment. This is useful progress on roadmap stage 1, but I cannot sign off that stage yet: an independent state-level test found a **pre-existing rejected-step rollback bug**, and the new archive fields have validation gaps. Keep #9 open and production HOLD.

### What is verified

- The full regression suite passes **37/37 locally and 37/37 on AutoDL**.
- A frozen, isolated AutoDL snapshot ran **30 population replays across 8 settings**, using 8 CPU workers, in **88.12 seconds**. All 30 exactly match the previous `bc375b3` records for susceptibility, q2/q4, mixed/growth energy, per-cycle q2, attained times, completion/truncation, and minimum lineage diagnostics. These are regression replays, not new independent accuracy evidence.
- The entire seed-0, 16-population L2/sample0/Gamma=.2/time8 cohort reproduces **11.566237500 +/- .016116452407228046**. Two-child replays additionally cover fresh-seed low field at time8/time16, intermediate field at time8/time16, L6 at time8/time16, and L6 with doubled equilibration.
- Actual-run checkpoint endpoints agree with accepted cycle diagnostics; stop codes agree with the failing floor; separate equilibration and measurement trace shapes/times agree with their windows. A real two-population archive roundtrip passes with explicitly patched small test dimensions.
- The production qualification helper remains conservatively false. This adds no ground-state convergence evidence; the previously identified attained-time and equilibration blockers remain.

### P1: rejected forward steps do not reliably restore the state used by the next cycle

**Targets:** `code/kagome_qea/pqmc.py`, the rejection branch around lines 402-404, accepted assignment around line 410, and end-of-cycle assignment around line 447 at this revision.

The sequence is:

1. After an accepted step, `saved = walkers` keeps a reference to the live walker objects.
2. The next `population_step` calls `_advance` on those objects **in place**, modifying spins, diagonal energies, and weights before resampling.
3. If the attempted resample fails a floor, `walkers = backup` and `shift = shift_backup` restore those local variables. However, `saved` still points to the already-advanced parents.
4. The cycle finishes with `walkers = saved`, discarding the restored backup. The next cycle tags and propagates states from the rejected attempt, with nonunit weights, while that attempted dt was excluded from the recorded attained time.

This is **not introduced by 6f130ae**. The same trace reproduces in `bc375b3`; exact regression equivalence therefore cannot validate the rollback semantics. It occurs when a rejection follows an accepted step. The first-step rejection case has a separately cloned initial `saved` population and needs its own regression too.

**Direct AutoDL evidence:** L6/sample0/Gamma=.05, 5,000 walkers, seed20261025 child0, 40 equilibration + 40 measurement blocks, requested time8, 8 cycles:

- Cycle 0 records accepted time **1.655503789920187** and rejects dt **0.0735155674859284**.
- At cycle end, `saved` differs from the pre-attempt backup in **2,711 spin configurations/diagonal energies and all 5,000 weights**.
- Saved weights span **0.038625753545440754 to 2.258516052753337**; the backup weights are all exactly 1.
- The next cycle begins with exactly those saved nonunit weights, not the backup. All seven observed inter-cycle transitions show the same leak; every cycle's rejected attempt changes its saved state.
- An independent 30-walker four-site chain trace also reproduces this on both revisions.

The just-finished cycle's q tags/ancestor labels are not changed by `_advance`, so its reported tag moment need not expose the error. The incorrect starting states and weights contaminate subsequent cycles. I have **not measured the corrected estimator's numerical shift**, and this finding alone does not quantify the bias or invalidate completed, nontruncated low-field trajectories.

**Suggested minimal implementation:** in the failure branch, make the state retained for both measurement and continuation the pre-attempt snapshot, e.g. assign `saved = backup` as well as `walkers = backup`, restore `shift`, and preserve rejected diagnostics separately. Prefer one authoritative accepted-state object so the end-of-cycle assignment cannot undo rollback. Preserve the intended RNG-consumption policy explicitly; restoring the physical state does not require silently changing it.

**Required tests:** force a rejection after at least one accepted step and compare every retained/next-cycle spin, diagonal, weight, tag, ancestor and shift against a deep pre-attempt snapshot; also force first-step rejection. Check that rejected energy/time contributions are excluded and endpoint checkpoints describe that same retained state. A control with no rejection must remain unchanged. Then rerun the truncated intermediate-field and L6 cases: old truncated numerical outputs are useful bug reproductions, not correctness targets.

### P2: new diagnostic archive validation accepts impossible/inconsistent records

Using both a real small-run archive and the repository's **unmodified production-shaped cache fixture** (32 populations x 8 cycles, 5,000 walkers, L6), each of these separate mutations still returns `result_ready(...) == True`:

| Mutation | Observed result |
|---|---|
| Rejected pre-resample weight ESS = -1 | accepted |
| Accepted weight ESS = 50,000 for 5,000 walkers | accepted |
| Largest descendant fraction = 1e-6 for 5,000 walkers | accepted |
| Stored unique floor = 2.9 | accepted after integer coercion |
| Rejected dt = 100 with requested time8 and accepted time1.5 | accepted |
| Finite checkpoint diagnostics = negative unique/Kish/weight ESS/fraction | accepted |
| Checkpoint times reverse from1.5 to0 | accepted |
| Last checkpoint unique count differs from cycle_unique/minimum | accepted |

These are malformed synthetic archives, **not evidence that the real replay archives contain these corruptions**. The original identity/revision/fingerprint checks remain useful; this is incomplete validation of the newly added diagnostic schema. The real-run endpoint checks in this review pass.

**Suggested patch:** share a diagnostic validator between recording and loading. Require integer counts/codes before coercion, `1 <= weight_ESS <= N`, `1 <= descendant_Kish <= unique <= N`, valid descendant fractions (at least `1/N`), and applicable accepted-floor conditions. Validate rejected states' feasible bounds as well as which floor failed; require `0 < rejected_dt <= requested-achieved` within tolerance. Require ordered checkpoint times, coherent all-NaN padding, valid diagnostics at every finite checkpoint, and a final tuple matching the corresponding cycle's achieved time/unique/Kish/weight ESS/fraction. Preserve NaN sentinels for genuinely inapplicable rejection fields. Add negative cases for all eight examples.

### P2/P3: parameter and numerical hardening

- `_resolve_floors(30, 8.0, 2.9)` silently becomes `(2, 8.0)`. Floors above the population (effective31 or unique31 for30 walkers) are also accepted despite being unattainable. Validate finite, exactly integral unique counts and enforce meaningful bounds before allocating/initializing walkers; raise consistent `ValueError`s. Stored `floor_unique=2.9` should likewise not become an apparently valid production floor2.
- `_weight_ess` is not scale invariant numerically: weights `[1,1]` return2, but `[1e200,1e200]` and `[1e-200,1e-200]` returnNaN despite every input being finite and positive. Scale by the maximum positive weight before summing/squaring, with explicit empty/invalid handling. This is a reproducible extreme-value defect in the new diagnostic, **not an observed overflow in the benchmark runs**.
- `energy_forward_trace` currently stores one arithmetic mean per cycle, not each accepted forward-step energy; it has no forward-step times. The comment's description of forward-step energies is therefore imprecise. Either name/document it as `energy_forward_cycle_mean`, or provide an actual bounded time-indexed trace if needed. The separate fixed-window traces are a real improvement.

### Updated implementation order

1. Fix and directly test rejected-step physical-state rollback.
2. Tighten new archive diagnostics, floor input validation and scale-stable weight ESS.
3. Rerun the affected truncated cohorts and separate fixed-window equilibration checks. State explicitly that corrected truncated outputs may differ from the old baselines.
4. Continue with optional positive-guided PQMC and tiny-system exact-transform/oracle tests, followed by frozen held-out convergence qualification.

The previous low-field completion improvement remains verified. The new diagnostics do **not** certify the intermediate-field/L6 estimator, and the state-restoration defect gives an additional concrete reason to keep production on hold.

### Reproducibility and status

- Local evidence: `data/kagome/validation/issue9-diagnostics-20261025-6f130ae/`.
- Remote evidence: `/root/autodl-tmp/issue9-diagnostics-20261025-6f130ae/`.
- Includes frozen source and hashes, `review.py`, `trace_kagome.py`, 30 full replay archives, `probes.json`, `production_archive_probes.json`, `kagome_rollback.json`, regression logs and `final.json`. **88 downloaded artifacts were hash-verified.**
- One initial review-harness attempt used a noncanonical archive directory and failed the path-identity check; the harness path was corrected, and the successful rerun is reported above. The original failure log is retained. No solver fix was made to get the review to pass.
- At **2026-10-25 23:52:50 UTC+8**, no review or production-scan processes remained; GPU utilization0%, memory0MiB. The instance remains on.
- Solver Python working-tree diff is empty; `git diff --check` passes. No solver source was edited, production deployment overwritten, or production scan restarted. Existing bytecode changes were left alone.

**Recommendation: accept the demonstrated instrumentation progress, but fix rollback before treating roadmap stage1 as complete or proceeding to production.**
