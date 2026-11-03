## Corrected campaign: compute allocation and progress

**Snapshot: 2026-11-03T07:32:54.982215+00:00 (UTC).** This is a saved status, not a live monitor.

| Work | Status | Completed training fits | Completed GFMC runs | Compute allocation |
|---|---|---:|---:|---|
| disorder-scaling-20261103-v1 | Retired; preserved legacy output | 320 | 480 | Zero workers |
| disorder-scaling-fine-20261103-v1 | Retired; preserved legacy output | 351 | 154 | Zero workers |
| Corrected main campaign | running | 3/896 | 0/1568 | 4 training + 6 GFMC workers |

**Recorded failures:** corrected campaign 0; retired campaigns disorder-scaling-20261103-v1: 16, disorder-scaling-fine-20261103-v1: 0. All are retained.

**One-shot process/source check (2026-11-03T07:31:55.675354+00:00):** 0 active legacy processes; 199 production source files checked, 0 mismatches.

| Active GFMC checkpoint | Size / field / disorder | Repeat | Saved steps / total | Completed measurement cycles |
|---|---|---:|---:|---:|
| rbm_2 / time | L=6, Gamma=0.55, J=60000 | 0 | 320/2880 | 0 |
| rbm_2 / main | L=6, Gamma=0.55, J=60000 | 1 | 320/1440 | 1 |
| rbm_2 / main | L=6, Gamma=0.55, J=60000 | 0 | 320/1440 | 1 |
| rbm_2 / start | L=6, Gamma=0.55, J=60000 | 0 | 320/1440 | 1 |
| rbm_2 / population | L=6, Gamma=0.55, J=60000 | 1 | 160/1440 | 0 |
| rbm_2 / population | L=6, Gamma=0.55, J=60000 | 0 | 160/1440 | 0 |

Checkpoint steps show trajectory progress only; they do not measure convergence.

**What is running:** repaired large-RBM learning rate (0.001), log-domain GFMC, and frozen random/up/down sampling checks before guide selection. The standalone weak-field pilot has been merged into this campaign; its completed fits and partial checkpoint are reused with matching source hashes.

**Priority and coverage:** fixed h=0.15; periodic kagome, N=3L². Ascending Gamma across L=4 and L=6. L=4 retains Gamma=0.55,0.56,...,0.70 plus 0.85 and 1.00; L=6 retains 0.55,...,0.60 plus 0.65,0.70,0.85,1.00. Disorders 60000–60007, both Jastrow/RBM guides, and both training seeds remain in the plan. No disorder is selected by overlap width.

**Resource budget:** at most 6 projector futures are submitted at once; 2 are deferred at this snapshot. New dispatch stops at 2026-11-03T12:59:37.507694+00:00; already-running GFMC jobs finish their finite trajectories. The complete 896-fit plan is not a promise of completion within this budget.

**Validated so far:** the repair controls contain 24 strong-field projection runs with no final projection failure or ancestry alarm, and the earlier 30-test regression suite passed. Sixteen targeted restart checks also passed, including the repaired histogram checkpoint replay and numerical-provenance mismatch rejection. These are implementation checks, not a ground-state convergence certificate.

**Physical conclusions still pending:** corrected weak-field disorder averages, population/forward-time sensitivity, and connected-correlation size checks. Passing frozen-start diagnostics alone does not establish equilibrium. The existing physical figures retain the frozen 112-run broader snapshot; the retired counts above are not corrected ensemble results.

Remote campaign: `/root/autodl-tmp/disorder-scaling-repaired-20261103-v2`. Raw stop audit, launch configuration, source hashes, and imported-checkpoint identities are preserved with this snapshot.
