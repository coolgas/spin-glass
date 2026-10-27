# Issue #9: rollback fixes and bounded flat-sampler validation

Report date: 2026-10-26. **Correctness checks pass; the 6×6 pilot is not ground-state converged. Keep issue #9 open and production on hold.**

## Implemented fixes

- `code/kagome_qea/pqmc.py`: rejected forward steps now retain the deep pre-attempt backup in both the live and saved populations and restore the shift. This prevents the cycle-end assignment from reinstalling in-place-mutated rejected parents. Rejected RNG draws remain consumed; rejected time and energy contributions remain excluded.
- Weight ESS is evaluated after rescaling by the largest weight, preserving scale invariance for extreme finite weights and rejecting invalid inputs.
- Floor inputs are validated before population initialization: finite, feasible effective counts and exactly integral, feasible unique counts. Existing defaults remain descendant Kish floor 8 and unique-ancestor floor 2.
- `code/kagome_qea/trial_overlap.py`: archive validation now checks integral metadata, feasible ancestry/ESS/fraction diagnostics, stop codes and floor failures, rejected-step bounds, ordered checkpoints with coherent padding and matching endpoints, and zero-time forward-energy sentinels.
- `code/test_reviewed_fixes.py`: forced first- and second-step rejection regressions check physical state, shift, RNG consumption, accepted time/energy and next-cycle continuation, alongside numerical and malformed-archive cases.

The flat sampler remains the sampler used for this pilot. Guided sampling is deferred.

## Correctness evidence

| Check | Result | Evidence |
| --- | --- | --- |
| Regression suite, local | 40/40 pass | `local_regression.log` |
| Regression suite, frozen AutoDL source | 40/40 pass | `regression.log` |
| Real-archive save/load roundtrips | 5/5 pass | `archive_roundtrips.json` |
| Malformed-real-archive rejections | 8/8 pass | `archive_corruption_checks.json` |
| Actual L6 child-0 rejected-state restoration | 8/8 cycles match backups, shifts and unit weights | `rollback_trace.json` |
| Actual L6 child-0 next-cycle continuation | 7/7 transitions match backups exactly | `rollback_trace.json` |
| Completed L2 regression controls | 2/2 reproduce prior records exactly; 16/16 cycles complete | `final.json`, `runs/` |

The eight malformed-real-archive checks cover negative rejected ESS, excessive weight ESS, infeasible descendant fraction, fractional unique floor, excessive rejected time, checkpoint endpoint mismatch, negative checkpoint diagnostics and reversed checkpoint times.

In all eight matched old/new L6 child comparisons, first-cycle q2 is unchanged and later cycles change. This is consistent with removal of the inter-cycle rejected-state leak. Old truncated susceptibility values are bug reproductions, not correctness targets.

## Bounded 6×6 pilot

- L=6 Kagome means 6×6 unit cells, **108 spins**, with PBC; sample 0, sigma=1, h=0, Gamma=0.05.
- 5,000 walkers, block duration 0.2, 40 measurement blocks and 8 forward cycles per population. Equilibration is varied below.
- Eight populations per L6 setting; seed 20261025 with matched `SeedSequence(seed).spawn(32)[child]` streams. Settings are paired, not independent cohorts. The two eq40 projection targets give identical trajectories.
- Total saved populations: 34 (32 L6 populations across four settings, plus two completed L2 regression controls).

**All +/- values below are across-population standard errors for this one disorder sample. They do not estimate ground-state accuracy, equilibration bias, population bias or disorder-ensemble uncertainty. Energies are total energies, not per-spin energies.**

| Equilibration blocks | Requested forward time | chi_SG +/- SE | Fixed-window energy +/- SE | Attained-time range | Truncated cycles |
| --- | --- | --- | --- | --- | --- |
| 40 | 8 | 8.491252 +/- 4.420771 | -124.709026 +/- 1.006776 | 1.168267–4.045397 | 64/64 |
| 40 | 16 | 8.491252 +/- 4.420771 | -124.709026 +/- 1.006776 | 1.168267–4.045397 | 64/64 |
| 80 | 8 | 10.571207 +/- 6.074845 | -129.509645 +/- 1.102357 | 1.186095–4.175202 | 64/64 |
| 160 | 8 | 7.392915 +/- 4.170931 | -132.929675 +/- 0.781042 | 1.458717–5.216361 | 64/64 |

Increasing requested projection time from 8 to 16 reproduces exactly the same stopped trajectories: all cycles stop before either target. This equality is **not evidence of projection convergence**.

Paired fixed-window energy changes relative to eq40/time8:

- Eq80/time8: -4.800619 +/- 0.482260.
- Eq160/time8: -8.220649 +/- 0.598671.

Measurement-window first-/second-half energy means also retain drift:

| Equilibration blocks | First half | Second half |
| --- | --- | --- |
| 40 | -122.910141 | -126.507911 |
| 80 | -128.901696 | -130.117595 |
| 160 | -132.715105 | -133.144245 |

These fixed-window comparisons use `energy_measure_trace`. `energy_forward_trace` is one arithmetic mean per forward cycle, not a time-resolved forward-step trace; it is not substituted for the fixed-window energy diagnostic.

The two L2 controls use sample0, Gamma=0.2, seed0, 40 equilibration blocks and requested time8. Their unchanged values demonstrate regression preservation only; this two-population subset is not a new exact-reference accuracy campaign.

## Convergence decision

Rollback and the confirmed diagnostic/cache bugs are fixed and directly tested. **Ground-state susceptibility remains unqualified:** every L6 cycle truncates, fixed-window energy depends materially on equilibration, and susceptibility has substantial population-to-population variation. No absolute/relative susceptibility accuracy target has been set.

Further bounded flat-sampler work must demonstrate equilibration and attained-projection-time convergence and quantify walker-population dependence before production can be trusted. Longer requested times alone do not resolve an ancestry-limited stop. Guided sampling remains deferred as requested.

## Source identity and evidence preservation

- Base HEAD: `6f130ae809215bdcfc579239b0f6d2fdad540ac9`.
- Solver state: **uncommitted and unpushed three-file working-tree patch**; not a deployed commit.
- Frozen patch SHA256: `93d0501e69348867a532cb316bc5243d08e5be27bdb3c5d9c17492c8ee126c0b`.
- `solver.patch` and solver file hashes in `source_manifest.json` match the working-tree fix; `git diff --check` passes. Existing unrelated bytecode modifications are preserved.
- Local evidence: `data/kagome/validation/issue9-rollback-20261025/`. Remote frozen campaign: `/root/autodl-tmp/issue9-rollback-20261025/`.
- The downloaded evidence set contains **98 SHA256-verified artifacts, totaling 1,581,602 bytes**, enumerated in `artifact_hashes.json`. This count describes the downloaded campaign set before adding this report, the local regression log and GitHub receipts.
- Original `campaign.json` SHA256: `27abb9c0a902cb4f1c000bebf667c8a4bf4d10d9937bbf31deca84d05b8d2e71`. It and all 34 population archives are unchanged by resumed postprocessing (`resume.json`).

## Harness repair, timing and operational status

The first postprocessing attempt failed on a NumPy boolean that the JSON encoder did not serialize. All populations were already saved. The harness now uses an explicit `bool(...)` and a NumPy-scalar serializer and supports `--resume`. Resuming reused all 34 archives, reran zero populations, and preserved original campaign metadata. The original harness, manifest, failure and validation log remain as `original_*`; the manifest records the harness-only hash revision.

`failure.txt` deliberately retains the original, resolved serialization failure. Successful completion is recorded in `resume.log` and `final.json` (`failures=[]`), rather than erasing that evidence.

- Directory label `20261025` is the campaign identifier. Source staging was 2026-10-26 00:00:44 UTC+8 (2026-10-25 16:00:44 UTC); population campaign start was 2026-10-25 16:01:08 UTC.
- Resumed postprocessing completed at **2026-10-26 00:34:18 UTC**. `final.json.elapsed=28.35807240754366` seconds measures resumed postprocessing and the state trace, **not total population-campaign runtime**.
- Completion status checked at **2026-10-26 00:34:46 UTC**: no validation or production-scan processes, GPU utilization 0%, GPU memory 0 MiB; AutoDL remains powered on.
- Production deployment `/root/autodl-tmp/kagome-qea/` was not overwritten or restarted; production archives are preserved. Issue #9 remains open pending convergence qualification.
