## Independent AutoDL verification of bc375b3: genuine low-field improvement; L6 and intermediate-field hold remains

Reviewed **bc375b3d79cd51a5d56bca04f656e0c17f6e803d**. The source snapshot was frozen from this commit and all staged Python files match the GitHub blobs. Production remains HOLD, but the new low-field improvement is real and should be credited.

### Verification performed

- Full local regression suite: **36 passed, 0 failed**; `git diff --check` passes.
- AutoDL follow-up: **272 population runs across 19 settings**, zero failures, 16 CPU workers, **479.1 seconds**, from **22:54:50 to23:02:49 UTC+8 on 2026-10-25**.
- 24 runs reproduce the author's original seed0 examples (16 at time8 and8 at time16). The other **248 runs use fresh seed20261025**, chosen before observing their outputs. These are independent child streams within each setting; matching child indices are reused for paired comparisons across settings.
- PBC, sigma1, h0, two replicas, tau_block=.2, 40 measurement blocks, 8 cycles. Base settings are 5000 walkers and40 equilibration blocks; variants are listed below.
- All susceptibility means, population SEs, normalization factors, and truncation counts were independently recomputed from downloaded raw population records. The regression/validation harness did not modify solver files.
- Exact values were checked using both the repository ground-state calculation and a separate global-spin-inversion-even-sector eigensolve/correlation contraction. All six exact references agree within1e-8, with independent residuals below1.4e-12.

### Main interpretation

**The old fractional-ancestor stopping obstruction is removed in the tested low-field L=2 cases.** Both requested times8 and16 are actually attained at Gamma=.2 on three samples, including fresh seeds, and no cycle hits the Kish floor in those completed cases. However, the new floor is not a universal convergence certificate. Intermediate field and L=6 still hit it, and fresh-seed scatter warrants more careful low-field qualification than simply declaring all walker/equilibration checks passed.

### Reproduction of the implementation comment

| Seed | Case | Pops | chi +/- SE | Exact | Actual times | Complete pops |
|---:|---|---:|---:|---:|---|---:|
| 0 | L2/sample0/Gamma=.2/request8 | 16 | 11.566237500 +/- .016116452 | 11.581661331 | 8–8 | 16/16 |
| 0 | L2/sample0/Gamma=.2/request16 | 8 | 11.574293750 +/- .019375369 | 11.581661331 | 16–16 | 8/8 |

These exactly reproduce the claimed means and errors. Minimum Kish size at request8 is24.3133, above the new floor8. This is a genuine attained-time extension, not the earlier false plateau. Reproducing the same seeds is not additional independent statistical evidence.

### Fresh-seed holdout results

SeedSequence(20261025).spawn(32)[child]. Errors are standard errors across population means. The difference/SE values are descriptive across many settings, not multiplicity-adjusted significance tests.

| L | Sample | Gamma | Walkers | Equil. blocks | Request | Pops | chi +/- SE | Exact | Difference/SE | Actual time range | Truncated cycles |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 2 | 0 | 0.2 | 5000 | 40 | 4 | 16 | 11.527002083 +/- 0.013258513 | 11.581661331 | -4.12 | 4.000000–4.000000 | 0/128 |
| 2 | 0 | 0.2 | 5000 | 40 | 8 | 32 | 11.591655469 +/- 0.012632183 | 11.581661331 | 0.79 | 8.000000–8.000000 | 0/256 |
| 2 | 0 | 0.2 | 5000 | 40 | 16 | 16 | 11.554813542 +/- 0.029676183 | 11.581661331 | -0.90 | 16.000000–16.000000 | 0/128 |
| 2 | 0 | 0.2 | 10000 | 40 | 8 | 16 | 11.556594271 +/- 0.010521349 | 11.581661331 | -2.38 | 8.000000–8.000000 | 0/128 |
| 2 | 0 | 0.2 | 5000 | 80 | 8 | 16 | 11.560173437 +/- 0.019844781 | 11.581661331 | -1.08 | 8.000000–8.000000 | 0/128 |
| 2 | 1 | 0.2 | 5000 | 40 | 8 | 16 | 9.786590104 +/- 0.028944892 | 9.873401403 | -3.00 | 8.000000–8.000000 | 0/128 |
| 2 | 1 | 0.2 | 5000 | 40 | 16 | 16 | 9.854280208 +/- 0.029838713 | 9.873401403 | -0.64 | 16.000000–16.000000 | 0/128 |
| 2 | 2 | 0.2 | 5000 | 40 | 8 | 16 | 11.604802083 +/- 0.019489583 | 11.603902558 | 0.05 | 8.000000–8.000000 | 0/128 |
| 2 | 2 | 0.2 | 5000 | 40 | 16 | 16 | 11.579240104 +/- 0.023363628 | 11.603902558 | -1.06 | 16.000000–16.000000 | 0/128 |
| 2 | 0 | 1 | 5000 | 40 | 8 | 16 | 4.654298958 +/- 0.088287928 | 4.666208800 | -0.13 | 1.446253–5.978631 | 128/128 |
| 2 | 0 | 1 | 5000 | 40 | 16 | 16 | 4.654298958 +/- 0.088287928 | 4.666208800 | -0.13 | 1.446253–5.978631 | 128/128 |
| 2 | 0 | 0.05 | 5000 | 40 | 8 | 8 | 11.974851042 +/- 0.001450194 | 11.973897341 | 0.66 | 8.000000–8.000000 | 0/64 |
| 2 | 0 | 3 | 5000 | 40 | 4 | 16 | 1.127770833 +/- 0.022056133 | 1.129636304 | -0.08 | 4.000000–4.000000 | 0/128 |
| 6 | 0 | 0.05 | 5000 | 40 | 8 | 8 | 8.735929282 +/- 5.002439312 | unavailable | — | 1.172734–4.099824 | 64/64 |
| 6 | 0 | 0.05 | 5000 | 40 | 16 | 8 | 8.735929282 +/- 5.002439312 | unavailable | — | 1.172734–4.099824 | 64/64 |
| 6 | 0 | 0.05 | 10000 | 40 | 8 | 8 | 8.072460590 +/- 3.608214525 | unavailable | — | 1.229430–6.449134 | 64/64 |
| 6 | 0 | 0.05 | 5000 | 80 | 8 | 8 | 9.084811806 +/- 5.144735727 | unavailable | — | 1.323175–4.443922 | 64/64 |

### What changes in the assessment

1. **L2/sample0/Gamma=.2 now has supportive independent evidence.** Fresh-seed request4 is11.527002 +/- .013259 (4.12SE low), while request8 with32 populations is11.591655 +/- .012632 (0.79SE above exact); all cycles complete. The matched first16 shift from4 to8 is+.063033 +/- .025469. Request16 also completes and agrees within0.91SE, although its uncertainty is larger.
2. **Low-field validation is promising, not uniformly settled.** Fresh sample1 at time8 is9.786590 +/- .028945 versus exact9.873401 (about3 reported SE low); time16 is9.854280 +/- .029839 (0.64SE low). The paired8-to16 change is+.067690 +/- .039128 (1.73SE), so this does not by itself establish a systematic time bias. Sample2 agrees at time8 and16. These variations should be checked with additional independent cohorts before strong coverage claims.
3. **Walker/equilibration stability is not contradicted decisively, but needs precision.** For fresh sample0, 10000 walkers/time8 gives11.556594 +/- .010521 (2.38SE low versus exact); the paired change from5000 walkers is-.033441 +/- .021864 (1.53SE). Doubling equilibration gives11.560173 +/- .019845; paired change-.029862 +/- .030118. Neither comparison proves instability, but agreement in the original small seed cohort should not be treated as certification.
4. **Intermediate field remains truncated despite good point agreement.** At Gamma1, all128 cycles truncate at1.446253–5.978631 for request8. Request16 produces exactly the same per-cycle q2 arrays and times for all16 matched populations. chi4.654299 +/- .088288 happens to agree with exact4.666209, but the attained-time requirement still fails.
5. **L6 remains a production blocker.** At5000 walkers, requests8 and16 have exactly identical cycle arrays/times in all8 populations; all64 cycles truncate at1.172734–4.099824. With10000 walkers, all64 still truncate at1.229430–6.449134. With80 equilibration blocks, all64 truncate at1.323175–4.443922.
6. **L6 equilibration is still not stable.** Doubling equilibration changes mean energy from-130.244846 to-132.103212; paired shift **-1.858366 +/- .265795**. Susceptibility has large uncertainty and is not diagnostic of the full production/disorder-average bias. No exact L6 reference was computed.
7. **Low/high-field controls behave sensibly in this fresh cohort.** Gamma=.05/time8 agrees within0.66SE and Gamma3/time4 within0.09SE; all cycles complete. This high-field fresh-seed result supports the interpretation that the earlier eight-population +2.24SE result was not alone evidence of a new algorithmic defect.

### Paired comparisons

All differences below are second minus first using shared child indices. Comparisons involving the32-population baseline use only its matched first16. A larger baseline is not incorrectly treated as the same matched sample.

| First | Second | Matched pops | chi shift +/- SE | Identical q2 cycles |
|---|---|---:|---:|---|
| reproduce_s0_g02_t8 | reproduce_s0_g02_t16 | 8 | -0.016115625 +/- 0.030921633 | no |
| fresh_s0_g02_t4 | fresh_s0_g02_t8 | 16 | +0.063033333 +/- 0.025468510 | no |
| fresh_s0_g02_t8 | fresh_s0_g02_t16 | 16 | -0.035221875 +/- 0.038512004 | no |
| fresh_s0_g02_t8 | fresh_s0_g02_w10000_t8 | 16 | -0.033441146 +/- 0.021864362 | no |
| fresh_s0_g02_t8 | fresh_s0_g02_eq80_t8 | 16 | -0.029861979 +/- 0.030117915 | no |
| fresh_s1_g02_t8 | fresh_s1_g02_t16 | 16 | +0.067690104 +/- 0.039127727 | no |
| fresh_s2_g02_t8 | fresh_s2_g02_t16 | 16 | -0.025561979 +/- 0.035451317 | no |
| fresh_s0_g1_t8 | fresh_s0_g1_t16 | 16 | +0.000000000 +/- 0.000000000 | yes |
| fresh_L6_g005_t8 | fresh_L6_g005_t16 | 8 | +0.000000000 +/- 0.000000000 | yes |
| fresh_L6_g005_t8 | fresh_L6_g005_w10000_t8 | 8 | -0.663468692 +/- 6.866439324 | no |
| fresh_L6_g005_t8 | fresh_L6_g005_eq80_t8 | 8 | +0.348882523 +/- 1.158681840 | no |

### Remaining acceptance and recommendation

- Credit the attained-time low-field improvement; the previous statement that L2/Gamma=.2 cannot reach8 with this implementation is superseded by this commit and these checks.
- Keep production HOLD for L6 and intermediate fields. Their new exact plateaus at requested8/16 still arise from the stopping rule. Completed projection and statistical agreement are separate from a general convergence guarantee.
- The threshold8 was selected after inspecting earlier finite-sample results. Matching the old truncated estimator is not, by itself, a reason that a threshold is unbiased. Fresh-seed tests help, but cases where the floor binds still need attained-time / population-size / estimator-stability evidence. This report does not establish that8 is wrong, nor that simply lowering/removing it is safe.
- Prioritize genuine L6 lineage/variance control and equilibration, and resolve the sample1/walker-count holdout deviations with adequately powered independent checks before universal low-field accuracy claims. Do not tune a stopping threshold solely to match a noisy exact-reference comparison.
- No production restart is justified by this partial improvement. The implementation comment correctly retains HOLD.

### Reproducibility and operational status

- Remote workspace: `/root/autodl-tmp/issue9-convergence-20261025-bc375b3/`.
- Local hash-verified copy: `data/kagome/validation/issue9-convergence-20261025-bc375b3/`.
- Preserved `study.py`, commit source snapshot, SHA-256 manifests, exact references,272 raw population records,19 summaries, `final.json`, and audited paired comparisons.
- At **23:03:19 UTC+8 on2026-10-25**, benchmark and production scan process lists were empty; GPU0%, memory0MiB. The instance remains on.
- No solver source files were modified, no production code deployment was overwritten, and no production scan was restarted.

**Issue #9 remains open. The low-field improvement is verified; the production hold remains.**
