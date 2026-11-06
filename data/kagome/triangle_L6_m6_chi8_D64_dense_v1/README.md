# Matched L6 chi8 scan, MPS D64 cap

User authorization: "good, so resubmit the job with D64".
Restart verified on 2026-11-05 at 05:17:22 UTC. Controller PID: 556387.
Remote root: `/root/autodl-tmp/triangle-L6-m6-c8-D64-dense-20261105-v1`.

## Scope and solver

- L=6, m=6, chi=8; Gamma=0.100:0.025:1.500, disorders 60000–60007:
  456 cases matched to the completed chi3 baseline.
- D32 warmup followed by a hard maximum D64, with independent starts 0 and 1.
  No D128/D256/D512 stage is dispatched. Original sweep settings and numerical
  thresholds remain intact. Both-start and D32-to-D64 diagnostics are retained.
- K is uniform on [0.5,1.5], h=0.15, lambda=1. Disorder entropy, bonds,
  Hamiltonian construction, and physical-moment extraction are unchanged.
- The scan runs all cases, even when another case fails strict numerical
  qualification. Such results have `qualified=false` and are exploratory
  finite-D outputs, not certified ground states. Controller `excluded` counts
  exclusions from the strictly qualified subset, not discarded calculations.
- qEA, connected/raw chiSG and raw q2 come from the same saved states. The
  lower-energy endpoint supplies the selected observables; both starts remain.

## Restart and resources

The old controller and its three D256 workers were stopped after command-line
identity checks. All 85 old case files were rehashed unchanged. D32/D64
checkpoints and models were copied and hash-verified into this separate run.
`checkpoint_reuse.json` identifies their original sources. Reused stage timings
retain their old cumulative values; a new result's `seconds` is its current
invocation time and is not a from-scratch runtime estimate.

Four workers use four BLAS/OpenMP threads each, with 3 GiB reserved per worker
and 16 GiB shared memory headroom. Disk admission retains a 3 GiB reserve, a
1.25 archive projection factor, and 0.05 GiB temporary growth per worker. The
D64 structural archive estimate is 10.444 MiB per case, rounded up to 10.5 MiB
as a projection floor; larger measured completed archives override the floor.
Per-stage resource guards remain active. The first launch held on an 11 MiB
floor; the live relaunch uses the rounded structural estimate and is verified
running. No reserve or safety factor was reduced.

At verification: 1/456 completed and qualified (reused), four live workers,
9.293 GiB free, projected remaining archives 5.832 GiB including the safety
factor. The finite queue has no deadline and no automatic failed-case retries.
The unrelated overlap controller remains active.

## Evidence and delivery

`restart_verification.json`, `prelaunch_validation.json`, `controller_launch.json`,
`manifest.json`, and `status.json` are small local evidence snapshots. The L4
exact validation and completed low-field D64 case passed capped replay; a
wrong-cap cached-result reuse was explicitly rejected. Raw data, models, states,
checkpoints, logs, and further analysis remain on AutoDL. There is no local
persistent fetcher. Any delivered notebook belongs directly in `notebook/`.

Disorder-only uncertainty applies to aggregate reports; individual-disorder
curves have no error bars. This campaign does not establish physical-retention
convergence or a thermodynamic critical point.
