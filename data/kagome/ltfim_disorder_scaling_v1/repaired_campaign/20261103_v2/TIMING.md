## Expanded campaign: scope and runtime estimate

**Measured at 2026-11-03T07:44:25.854853+00:00.** Runtime scenarios, not a completion guarantee or a convergence certificate.

Both L=4 (N=48) and L=6 (N=108) target Gamma=0.55,...,0.70 by 0.01, plus 0.85 and 1.00, at h=0.15. Eight fixed disorders, both guide families, two training seeds, and two projection repeats are retained.

**Full plan:** 1152 fits; 1152 main GFMC runs + 864 population/time/start checks = 2016 runs.

**Saved progress:** 7 completed fits, 7 completed GFMC records; 0 recorded failures.

| Milestone | Planning time from this snapshot |
|---|---|
| First L=6, Gamma=0.55 RBM main repeats | Approximately 20–40 minutes; individual-disorder runs |
| Gamma=0.55, both sizes/guides, all eight disorders: main ensemble | Approximately 4–6 hours |
| Gamma=0.55 including all planned sensitivities | Approximately 6–10 hours |
| Gamma=0.55–0.60, both sizes, including sensitivities | Approximately 1–2 days |
| Entire expanded plan including 0.85 and 1.00 | Approximately 4–6 days of continued operation |

**Measured rates and assumptions:**
- L=4 jastrow: 11.2 minutes per fit; 2 repaired completed fits; scenario -10%/+20%.
- L=4 rbm_2: 11.2 minutes per fit; 2 repaired completed fits; scenario -10%/+20%.
- L=6 jastrow: 22.5 minutes per fit; explicit 15–30 minute planning assumption; partial L6 Jastrow optimizer timing is about 15 minutes before frozen diagnostics.
- L=6 rbm_2: 22.5 minutes per fit; explicit 15–30 minute planning assumption; partial L6 Jastrow optimizer timing is about 15 minutes before frozen diagnostics.
- L=6 RBM GFMC main: 39.3 minutes per run; repaired partial-trajectory extrapolation (2 observations).
- L=6 RBM GFMC population: 82.6 minutes per run; repaired partial-trajectory extrapolation (2 observations).
- L=6 RBM GFMC time: 80.3 minutes per run; repaired partial-trajectory extrapolation (1 observations).
- L=6 RBM GFMC start: 38.4 minutes per run; repaired partial-trajectory extrapolation (1 observations).

At 4 training workers, central full-plan training workload is 80.9 wall-hours; the explicit training-rate scenario spans 60.2–104.3 hours.
At 6 GFMC workers, central full-plan projection workload is 85.3 wall-hours. Training and projection overlap, so these totals are not added.
With completed work credited, the idealized remaining-work bottleneck is 84.9 hours; the planning scenario is 97.7–135.9 hours, allowing 15–60% pipeline/throughput overhead and the upper training-rate scenario.

**Limitations:** This is a saved forecast for the initial expanded-grid restart, not an adaptive scheduler or live ETA. The milestone ranges are planning judgments based on these throughput estimates. L6 RBM projection timing is extrapolated from saved partial repaired trajectories. L4 estimates use repaired completions/checkpoints when available. L6 Jastrow projection timings use retired-campaign timing metadata only, not its physics results. Field, disorder, guide and CPU contention can change runtime. Before-projection guide sampling is covered by the overhead scenario. Training samples are still few; imported pilot timing is not treated as representative of four-worker throughput. Failed sampling checks can leave contexts unqualified rather than supply a complete ensemble. The 4–6 day range covers the prescribed runs only; further convergence extensions have no defensible fixed ETA yet.

**Runtime policy:** no wall-clock cutoff. The finite campaign is authorized to continue until all planned tasks have been attempted. The throughput estimates above retain their original measurement timestamp.
