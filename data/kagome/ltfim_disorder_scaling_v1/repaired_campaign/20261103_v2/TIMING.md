> Suspended forecast: broad production is paused for the L6 convergence study. The dates below are historical scenarios, not an active completion ETA.

## Recalibrated campaign runtime

**Snapshot: 2026-11-03T09:51:10.509579+00:00.** Supersedes the earlier 4–6 day forecast. All times below are remaining time from this snapshot.

Production has 31/1152 completed fits and 46/2016 completed GFMC records. Its live allocation is **two GPU training workers and two CPU GFMC workers**, with no deadline. Diagnostics have separate workers. Finishing diagnostics does not automatically enlarge the production pools.

| Milestone | Planning estimate |
|---|---|
| Both RBM seeds: +1200 updates and frozen checks | Approximately 20–35 minutes |
| Both RBM seeds: +3600 cumulative updates and frozen checks | Approximately 1–2 hours |
| Original versus extended guides: all matched GFMC controls | Approximately 5–8 hours, if guides pass frozen checks |
| Ten field/forward-time/population controls | Approximately 5–8 hours |
| Gamma .55, both sizes/guides, all eight disorders and planned sensitivities | Approximately 8–12 hours |
| Gamma .55–.60, both sizes, including sensitivities | Approximately 4–6 days |
| All standard production training | Approximately 9–12 days |
| Entire prescribed L4/L6 campaign, current two/two allocation | **Approximately 13–18 days (November 16–21)** |
| Conditional: restore four trainers/six projectors after diagnostics | Approximately 5–7 days; requires an explicit scheduler change |

**Measured production rates:**

| Size / guide | Fits sampled | Median minutes per fit | Observed range |
|---|---:|---:|---:|
| L4 / jastrow | 8 | 11.3 | 10.5–16.2 |
| L4 / rbm_2 | 8 | 11.3 | 11.2–16.8 |
| L6 / jastrow | 8 | 26.5 | 22.8–31.2 |
| L6 / rbm_2 | 4 | 29.4 | 26.2–32.5 |

| Size / guide / GFMC setting | Runs sampled | Median minutes per run |
|---|---:|---:|
| L4 / jastrow / main | 6 | 0.7 |
| L4 / jastrow / population | 2 | 1.1 |
| L4 / jastrow / start | 2 | 0.3 |
| L4 / jastrow / time | 2 | 0.9 |
| L4 / rbm_2 / main | 6 | 4.8 |
| L4 / rbm_2 / population | 2 | 8.4 |
| L4 / rbm_2 / start | 2 | 3.9 |
| L4 / rbm_2 / time | 2 | 8.0 |
| L6 / jastrow / main | 6 | 2.7 |
| L6 / jastrow / population | 2 | 6.7 |
| L6 / jastrow / start | 2 | 1.1 |
| L6 / jastrow / time | 2 | 5.2 |
| L6 / rbm_2 / main | 4 | 39.5 |
| L6 / rbm_2 / population | 2 | 91.1 |
| L6 / rbm_2 / start | 2 | 46.2 |
| L6 / rbm_2 / time | 2 | 85.6 |

Remaining workload is **366.5 training worker-hours** and **527.2 GFMC worker-hours**, calculated by multiplying each size/guide/audit median by its unfinished task count. Two workers give 183.2 training wall-hours and 263.6 projection wall-hours. Training and projection overlap; these are not added. The idealized bottleneck is 11.0 days. Applying a stated 15–60% allowance for pipeline delays, guide-histogram sampling and throughput variation gives 12.6–17.6 days, rounded to 13–18 days.

The extension checkpoints measured approximately 0.94 seconds per update per seed under the current split. Completed L6 fits indicate about eight additional minutes for frozen diagnostics per milestone. Extension GFMC starts after both training milestones; up to twelve runs share two CPU workers. Diagnostic and first-field milestones are planning judgments from these rates and queue sizes, not simulated schedules.

The conditional four/six scenario has an idealized bottleneck of 3.8 days, plus the diagnostic interval and throughput allowance. It is not the active configuration and was not enacted by this recalibration.

**Limits:** Timing samples are predominantly Gamma=.55, with few disorders and changing concurrency. Higher fields, particularly .85/1.00, may change rates beyond this range. The 15–60% allowance is a planning scenario, not a statistical confidence interval. Imported L6 RBM pilot timings are excluded from the rate estimate, but their completed fits are credited. Active partial work is conservatively counted as unfinished. Projection timings omit subsequent guide-histogram sampling, covered by the allowance. Failed qualification can leave a prescribed context without valid results; no failed disorder is replaced. The schedule covers standard fits and the already launched finite diagnostics, not extending every RBM fit or additional convergence remedies. Scientific convergence and L8/L10 expansion have no defensible completion date yet.

**Runtime policy:** no wall-clock cutoff; the finite authorized campaign continues.
