## Corrected campaign: compute allocation and progress

**Production dispatch paused at 2026-11-03T10:01:10.713605+00:00 for convergence diagnosis.** Production processes stopped; saved checkpoints are preserved. Compute is devoted to the L6 convergence study.

**Snapshot: 2026-11-03T10:02:30.984631+00:00 (UTC).** This is a saved status, not a live monitor.

| Work | Status | Completed training fits | Completed GFMC runs | Compute allocation |
|---|---|---:|---:|---|
| disorder-scaling-20261103-v1 | Retired; preserved legacy output | 320 | 480 | Zero workers |
| disorder-scaling-fine-20261103-v1 | Retired; preserved legacy output | 351 | 154 | Zero workers |
| Corrected main campaign | dispatch_paused_for_convergence_diagnosis | 33/1152 | 50/2016 | 0 training + 0 GFMC workers |

**Recorded failures:** corrected campaign 0; retired campaigns disorder-scaling-20261103-v1: 16, disorder-scaling-fine-20261103-v1: 0. All are retained.

**Early GFMC measurement check:** 9 completed records pass saved raw-hash, normalization, overlap-moment and finite-diagnostic checks; 2 have ancestry alarms. These remain in the data and are not qualified as converged results.
- Ancestry alarm: L=4, Gamma=0.55, disorder 60000, jastrow, start, repeat 1.
- Ancestry alarm: L=4, Gamma=0.55, disorder 60000, jastrow, time, repeat 1.

**One-shot process/source check (2026-11-03T08:03:39.066712+00:00):** 0 active legacy processes; 199 production source files checked, 0 mismatches.

| Active GFMC checkpoint | Size / field / disorder | Repeat | Saved steps / total | Completed measurement cycles |
|---|---|---:|---:|---:|

Checkpoint steps show trajectory progress only; they do not measure convergence.

**Configured numerical protocol:** repaired large-RBM learning rate (0.001), log-domain GFMC, and frozen random/up/down sampling checks before guide selection. The standalone weak-field pilot has been merged into this campaign; its completed fits and partial checkpoint are reused with matching source hashes.

**Priority and coverage:** fixed h=0.15; periodic kagome, N=3L². Ascending Gamma across L=4 and L=6. L=4: Gamma=0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.85, 1.00; L=6: Gamma=0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.85, 1.00. Disorders 60000–60007, both Jastrow/RBM guides, and both training seeds remain in the plan. No disorder is selected by overlap width.

**Resource budget:** at most 0 projector futures are submitted at once; 53 are deferred at this snapshot. No wall-clock cutoff, but the convergence hold overrides dispatch. The earlier completion ETA is suspended pending diagnosis.

**Validated so far:** the repair controls contain 24 strong-field projection runs with no final projection failure or ancestry alarm, and the earlier 30-test regression suite passed. Sixteen targeted restart checks also passed, including the repaired histogram checkpoint replay and numerical-provenance mismatch rejection. These are implementation checks, not a ground-state convergence certificate.

**Physical conclusions still pending:** corrected weak-field disorder averages, population/forward-time sensitivity, and connected-correlation size checks. Passing frozen-start diagnostics alone does not establish equilibrium. The existing physical figures retain the frozen 112-run broader snapshot; the retired counts above are not corrected ensemble results.

Remote campaign: `/root/autodl-tmp/disorder-scaling-repaired-20261103-v2`. Raw stop audit, launch configuration, source hashes, and imported-checkpoint identities are preserved with this snapshot.
