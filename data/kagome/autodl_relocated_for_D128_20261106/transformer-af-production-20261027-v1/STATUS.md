# Production pipeline execution status

Snapshot audited at 2026-10-28T01:33:52.700339+00:00.

**State:** running_notebook_stage_b_escalation

**Scientific status:** qualification-stage artifacts only. Final-data collection and ground-state qualification are pending.

Anchor sizes L=4,6,8,10 retain all 49 contexts, initially 64 fresh disorders, four repeats and the same 16-disorder unguided comparison subset. L=5,7,9 retain matched transition cuts.

## Active: notebook Section 5, Stage B

The specified L10 task completed. The earlier tiny-probe queue is paused and superseded; it does not count toward Stage B.

L=4 (48 spins), PBC, W=64 four-replica groups; equilibration 2, energy-measurement time 1, forward time 1, four cycles and branch interval 0.1. Both methods use float64.

Four development disorders, Gamma=0.5,1,2, h=0,0.15,0.3, and four repeats: 288 population runs.

| Method | Completed | Expected | Ancestry alarms |
|---|---:|---:|---:|
| network | 144 | 144 | 87 |
| constant | 144 | 144 | 144 |

Stage B active failures: 0. Structural checks pass; this does not certify convergence.

Next notebook steps: assess runtime/ancestry, escalate representative cases to W=256 and 1024 and longer times (including h=0.05), then evaluate cost feasibility before Stage C. Stage C and final-disorder collection are not auto-promoted.

## Section 5 escalation schedule

State: running_notebook_stage_b_escalation. Completed 114/144.

Fixed disorder 60000; Gamma=0.5,2; h=0,0.05; both guides, float64 and four repeats. W64 small-field baseline, W256/W1024, then separate equilibration 4 and forward time 2 at W1024. Measurement 1, four cycles and branch interval 0.1 throughout.

The controller requires the full base pilot and holds the shared supervisor lock. The active status records its operational deadline; separately documented continuations preserve the original launch and frozen scientific identities. This is representative feasibility evidence; Stage C still requires eight disorders and substantive convergence ladders.

Scientific assessments: [STAGE_B_REVIEW.md](STAGE_B_REVIEW.md) and [PHYSICAL_CONVERGENCE.md](PHYSICAL_CONVERGENCE.md).

Active escalation deadline UTC: 2026-10-29T01:23:56.846418+00:00.

## Archived preliminary queue (paused)

| Item | Completed | Expected |
|---|---:|---:|
| Direct-guide checks | 392 | 392 |
| Exact reference contexts | 392 | 392 |
| Qualification/feasibility runs | 11267 | 27328 |

| L | Completed runs | Direct network | Unguided | Ancestry alarms |
|---:|---:|---:|---:|---:|
| 2 | 3136 | 0 | 0 | 1 |
| 4 | 1162 | 10 | 1152 | 196 |
| 5 | 1162 | 10 | 1152 | 306 |
| 6 | 1162 | 10 | 1152 | 382 |
| 7 | 1162 | 10 | 1152 | 477 |
| 8 | 1162 | 10 | 1152 | 567 |
| 9 | 1161 | 9 | 1152 | 626 |
| 10 | 1160 | 8 | 1152 | 687 |

L=2 uses the cached direct-rate precision ensemble. Larger-size CPU and GPU lanes progress at different rates; total completion does not imply the direct-network lane is complete.

All alarm cases are retained. Completed short runs are not certified final measurements.

## Throughput and recovery

Selected GPU workers: 1. CPU workers: 4.

| GPU workers | Matched 8-task wave seconds |
|---:|---:|
| 1 | 58.67 |
| 2 | 59.87 |
| 4 | 60.43 |

## Short-setting precision assessment

784 finite-forward reference contexts completed for 49 fields, eight disorders and two precisions.

40 of 147 approximate paired-disorder 95% bounds meet the declared tolerances; 107 remain unresolved.

These are comparisons at one short setting with four independent repeats, not simultaneous confidence guarantees. Unresolved bounds do not establish a float32 bias. Float64 remains the comparison implementation; neither long-time float32 accuracy nor ground-state convergence is qualified.

GPU mean over the latest up-to-30 telemetry samples: 97.43333333333334%.

Each bounded process window checkpoints and pauses workers at expiry. The original 12-hour launch record and separate continuation windows are retained.

Source hashes, task identities, finite outputs, cycle means and spectral normalization pass the structural audit. Precision equivalence and projection convergence remain separate gates.

Active failures: 0. Reviewed incidents: 1.

Larger-size development IDs: 60000:60032. Final IDs 61000:61256 remain unused.

Next: ensemble precision, population/time/cadence convergence, alternative starts, crossing-error targets and a measured final-campaign budget. This qualification queue is separate from the 62,720-run final anchor survey.
