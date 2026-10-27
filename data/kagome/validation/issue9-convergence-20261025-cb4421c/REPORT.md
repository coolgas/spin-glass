## AutoDL convergence study completed: finite-time bias and lineage cutoff remain blockers

Reviewed revision: **cb4421cfc87749d8276ae1a21f78572ebcfa1eac**.

The maintainer explicitly authorized this AutoDL campaign. It ran on **2026-10-25, 22:04:47–22:10:04 UTC+8**, using 16 of the available 20 CPU cores. **256 population runs across 19 settings completed in 316.6 seconds, with zero failures.** This is a CPU implementation; no GPU rewrite or solver modification was made.

All source files were staged from the exact reviewed commit into an isolated directory and SHA-256 verified. The deployed production directory was not overwritten. Random streams are `SeedSequence(0).spawn(32)[child]`: independent children within each setting, matched children across settings. Reusing the earlier review seeds deliberately tests reproducibility; repeated baseline results are not additional independent evidence.

### Main conclusion

**HOLD production.** The previously fixed diagnostics/cache checks work, but changing nominal time, walker count, or equilibration within the tested ranges does not establish ground-state susceptibility convergence. The next engineering target is the ability of the forward-walking estimator to reach a sufficiently long time while retaining useful ancestral diversity, plus L=6 equilibration/population-size control. This does not establish that the full continuous-time projector must be replaced.

### Controlled stochastic study

All cases use PBC, sigma=1, h=0, two replicas, tau_block=.2, 40 measurement blocks, and 8 forward cycles per population. Equilibration blocks and walker counts are varied as listed. Errors are **standard errors across independent population means**, not cycle errors or disorder errors. Completion is achieved-time completion, not proof of ground-state convergence. Statistical difference/SE values are descriptive; the many settings are not a multiplicity-corrected significance test.

| L | Sample | Gamma | Walkers | Equil. blocks | Requested time | Pops | chi_SG +/- SE | Actual time range | Truncated cycles | Exact chi | Difference/SE |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2 | 0 | 0.2 | 5000 | 40 | 4 | 32 | 11.518803125 +/- 0.008605568 | 4.000000–4.000000 | 0/256 | 11.581661331 | -7.30 |
| 2 | 0 | 0.2 | 5000 | 40 | 2 | 16 | 11.351603646 +/- 0.012885605 | 2.000000–2.000000 | 0/128 | 11.581661331 | -17.85 |
| 2 | 0 | 0.2 | 5000 | 40 | 8 | 16 | 11.553411979 +/- 0.012966277 | 4.126219–5.345606 | 128/128 | 11.581661331 | -2.18 |
| 2 | 0 | 0.2 | 5000 | 40 | 16 | 16 | 11.553411979 +/- 0.012966277 | 4.126219–5.345606 | 128/128 | 11.581661331 | -2.18 |
| 2 | 0 | 0.2 | 10000 | 40 | 4 | 16 | 11.526621094 +/- 0.006782884 | 4.000000–4.000000 | 0/128 | 11.581661331 | -8.11 |
| 2 | 0 | 0.2 | 10000 | 40 | 8 | 16 | 11.540451042 +/- 0.012772605 | 4.235632–5.131138 | 128/128 | 11.581661331 | -3.23 |
| 2 | 0 | 0.2 | 20000 | 40 | 8 | 16 | 11.551050000 +/- 0.004403902 | 4.456462–5.017585 | 128/128 | 11.581661331 | -6.95 |
| 2 | 0 | 0.2 | 5000 | 80 | 4 | 16 | 11.536679167 +/- 0.011387160 | 4.000000–4.000000 | 0/128 | 11.581661331 | -3.95 |
| 2 | 0 | 0.2 | 5000 | 160 | 4 | 16 | 11.551651042 +/- 0.014842391 | 4.000000–4.000000 | 0/128 | 11.581661331 | -2.02 |
| 2 | 1 | 0.2 | 5000 | 40 | 4 | 16 | 9.843410417 +/- 0.018913101 | 4.000000–4.000000 | 0/128 | 9.873401403 | -1.59 |
| 2 | 1 | 0.2 | 5000 | 40 | 8 | 16 | 9.846638021 +/- 0.018207598 | 4.663192–6.252080 | 128/128 | 9.873401403 | -1.47 |
| 2 | 0 | 0.05 | 5000 | 40 | 4 | 8 | 11.970719792 +/- 0.001996750 | 4.000000–4.000000 | 0/64 | 11.973897341 | -1.59 |
| 2 | 0 | 1 | 5000 | 40 | 4 | 8 | 4.530384375 +/- 0.130762801 | 1.354916–1.748720 | 64/64 | 4.666208800 | -1.04 |
| 2 | 0 | 1 | 5000 | 40 | 8 | 8 | 4.530384375 +/- 0.130762801 | 1.354916–1.748720 | 64/64 | 4.666208800 | -1.04 |
| 2 | 0 | 3 | 5000 | 40 | 4 | 8 | 1.181894792 +/- 0.023319764 | 4.000000–4.000000 | 0/64 | 1.129636304 | 2.24 |
| 6 | 0 | 0.05 | 5000 | 40 | 4 | 8 | 4.183618056 +/- 2.561442767 | 1.137953–1.570618 | 64/64 | not available | — |
| 6 | 0 | 0.05 | 5000 | 40 | 8 | 8 | 4.183618056 +/- 2.561442767 | 1.137953–1.570618 | 64/64 | not available | — |
| 6 | 0 | 0.05 | 10000 | 40 | 4 | 8 | 8.783179340 +/- 3.419757177 | 1.236163–1.571386 | 64/64 | not available | — |
| 6 | 0 | 0.05 | 5000 | 80 | 4 | 8 | 4.142490046 +/- 2.203994945 | 1.083280–1.565761 | 64/64 | not available | — |

### What the parameter tests show

- **L=2, Gamma=.2, 5000 walkers:** the 32-population time-4 result reproduces exactly: chi=11.518803125 +/- .008605568 versus exact 11.581661331, a -7.304-SE difference. All time-4 cycles complete. Time 2 is substantially worse, demonstrating a real forward-time dependence.
- **Requests 8 and 16 are an exact false plateau at L=2:** every one of the 16 paired populations has identical cycle q2 arrays and identical actual times (4.126219–5.345606). All 128 cycles truncate in either request.
- **More walkers do not unlock time 8:** at 10,000 walkers the actual times are 4.235632–5.131138; at 20,000, 4.456462–5.017585. All 128 cycles in each case truncate. At 20,000 walkers chi=11.551050 +/- .004404 remains about 6.95 reported SE below exact. The implementation uses `ancestor_floor=max(2,n_walkers//40)`, so increasing population size also increases the stopping threshold.
- **More L=2 equilibration is not a demonstrated fix:** compared with the matched first 16 baseline populations, doubling equilibration changes chi by +.006537 +/- .017881; quadrupling changes it by +.021509 +/- .017555. Neither shift is decisive. Both still use forward time4 and therefore retain its finite-forward-time bias.
- **Second disorder sample:** at L=2, Gamma=.2, every request8 cycle also truncates (actual 4.663192–6.252080). Agreement within roughly 1.5 reported SE does not prove projection convergence.
- **Intermediate field Gamma=1:** requests4 and8 produce exactly identical cycle arrays in all eight matched populations; actual times are 1.354916–1.748720.
- **Low/high-field checks:** at Gamma=.05 and3, all time4 cycles complete. Their eight-population estimates differ from exact by -1.59 and +2.24 reported SE respectively. The high-field result uses only the first eight of the earlier 16 production seeds; its higher mean is not an independent contradiction of the earlier 16-population result. Neither eight-population check certifies the parameter range.

### Deterministic separation of forward-time bias

Sparse-matrix propagation removes Monte Carlo noise and finite-population effects. For the flat guide and two independent replicas, propagate a uniform left vector for the forward time and a uniform right vector for the premeasurement time; normalized elementwise left*right weights determine the spin correlations and chi. The right premeasurement time at the first cycle is `(n_equilibrate+n_measure)*tau_block`, then increases by the forward time on successive untruncated cycles.

The table is the eight-cycle **infinite-population, untruncated** calculation for L=2/sample0/Gamma=.2 with 40 equilibration +40 measurement blocks. Exact ground-state chi=11.58166133075845:

| Forward time | Deterministic chi | Relative error vs ground state |
|---:|---:|---:|
| 2 | 11.346374710809 | -2.031545% |
| 4 | 11.533318870048 | -0.417405% |
| 8 | 11.578742913984 | -0.025199% |
| 16 | 11.581646927659 | -0.000124% |

At time4, increasing equilibration from 40 to160 blocks changes this deterministic mean only from 11.533318870 to11.533327512. Thus even ideal population size and much longer equilibration cannot turn a genuinely time4 forward estimator into the ground-state susceptibility. This isolates a finite-forward-time bias; it does not assign every stochastic discrepancy to that one mechanism.

Exact-reference checks: the repository residual/degeneracy-aware result, a separate eigensolve/correlation contraction, and an explicit global-spin-inversion-even-sector solve agree in chi to better than 1e-8 for all five (sample, field) references. The even-sector residuals are <=1.4e-12. The initial absolute-eigenvector finite-time auxiliary construction at Gamma=.05 had residual ~1.4e-8 because of the nearly degenerate doublet; the separate positive even-sector solution resolves that numerical ambiguity. The diagnostic finite-time conclusions above use Gamma=.2 and were independently reproduced in the even-sector calculation.

### L=6 evidence and its limits

- For all eight populations, requests4 and8 produce **identical q2 cycle arrays and achieved times**. All 64 cycles truncate, actual times1.137953–1.570618.
- Doubling walkers to10,000 still truncates all64 cycles at1.236163–1.571386. Doubling equilibration to80 blocks also truncates all64 at1.083280–1.565761.
- Susceptibility changes with those variants have large population uncertainty: the paired walker increase is +4.599561 +/-2.764560; the paired equilibration increase is -.041128 +/- .561199. These do **not** quantify the bias of the production L=6 disorder average.
- Energy is also not stable in this L=6 diagnostic. With 5000 walkers, doubling equilibration changes mean energy from -128.737929 to -130.906777; the paired shift is **-2.168848 +/- .209437**. Doubling walkers at40 equilibration blocks shifts energy by **-2.194078 +/- .689676**. There is no exact L=6 reference here, but these shifts warn against treating the initial state as fully equilibrated or population-size converged.

### Matched-population comparisons

Differences below are second setting minus first, using only shared child indices (16 for L=2 central sweeps, 8 for L=6). In particular, comparisons to the 32-population baseline use its matched first16, not the unmatched 32-population mean.

| First | Second | Matched pops | Paired chi shift +/- SE | Identical cycle q2? |
|---|---|---:|---:|---|
| L2_g02_t2 | L2_g02_t4 | 16 | +0.178538542 +/- 0.017933750 | no |
| L2_g02_t4 | L2_g02_t8 | 16 | +0.023269792 +/- 0.013314385 | no |
| L2_g02_t8 | L2_g02_t16 | 16 | +0.000000000 +/- 0.000000000 | yes |
| L2_g02_t4 | L2_g02_w10000_t4 | 16 | -0.003521094 +/- 0.015464738 | no |
| L2_g02_t8 | L2_g02_w10000_t8 | 16 | -0.012960937 +/- 0.016868525 | no |
| L2_g02_w10000_t8 | L2_g02_w20000_t8 | 16 | +0.010598958 +/- 0.012117580 | no |
| L2_g02_t4 | L2_g02_eq80 | 16 | +0.006536979 +/- 0.017880642 | no |
| L2_g02_t4 | L2_g02_eq160 | 16 | +0.021508854 +/- 0.017554601 | no |
| L2_s1_g02_t4 | L2_s1_g02_t8 | 16 | +0.003227604 +/- 0.026587580 | no |
| L2_g1_t4 | L2_g1_t8 | 8 | +0.000000000 +/- 0.000000000 | yes |
| L6_g005_t4 | L6_g005_t8 | 8 | +0.000000000 +/- 0.000000000 | yes |
| L6_g005_t4 | L6_g005_w10000_t4 | 8 | +4.599561285 +/- 2.764560359 | no |
| L6_g005_t4 | L6_g005_eq80_t4 | 8 | -0.041128009 +/- 0.561199118 | no |

### Recommended next work

1. **Do not spend a production budget merely raising nominal forward time or independent-population count.** The measured cutoff prevents longer projection, and more populations only shrink statistical uncertainty around the same biased/truncated estimator.
2. **Develop and benchmark a way to obtain sufficiently long descendant propagation with controlled lineage loss.** The current flat-guide / fixed-fraction ancestor-floor combination is the immediate obstruction. Any changed stopping/resampling/guiding or alternative pure-estimator approach is an experimental implementation change and must earn validation; simply deleting the floor is not a demonstrated solution.
3. **Use L=2 as the acceptance test:** require actual achieved-time extension, susceptibility approaching the exact reference, and walker/equilibration stability across samples and fields. The deterministic calculation supplies useful targets (time8 much closer, time16 closer still for sample0/Gamma=.2), but not a universal time prescription for L=6.
4. **Then repeat L=6 equilibration and population-size studies with the validated estimator.** The current L=6 energy shifts and lineage collapse both need resolution before production qualification.

### Artifacts and operational status

- Remote: `/root/autodl-tmp/issue9-convergence-20261025-cb4421c/`.
- Local, hash-verified copy: `data/kagome/validation/issue9-convergence-20261025-cb4421c/`.
- Preserved: `study.py`, immutable source snapshot, `source_manifest.json`, `manifest.json`, `study.log`, `exact.json`, `deterministic_reference.json`, `final.json`, 256 per-population records, per-setting summaries, and `artifact_hashes.json`.
- Reproduce the main campaign in an isolated staged directory with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/yjye_project_env/bin/python -B -u study.py`.
- All population means, SEs, susceptibility scaling, and truncation totals were independently recomputed from the downloaded raw records.
- At **22:10:53 UTC+8 on 2026-10-25**, no benchmark processes or production scan processes remained. GPU0%, memory0MiB. The instance remains on.

**Issue #9 remains open; production HOLD remains.** No production scan was restarted and no solver source files were modified.
