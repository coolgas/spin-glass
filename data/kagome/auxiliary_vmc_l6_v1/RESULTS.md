# L6 auxiliary-field pilot

Verified snapshot: 2026-11-04T04:32:08.032080+00:00

One periodic disorder (60000), one training seed (0), N=108, Gamma=0.6, h=0.15.
Every listed energy and variance uses the original Hamiltonian.

Latest checkpoint completed by all seven branches: 2400 updates.

| Branch | Target energy/site | Variance/N | Energy difference from baseline | Conditional paired SE | Variance ratio |
|---|---:|---:|---:|---:|---:|
| fixed_r0 | -1.1262843 | 0.024205 | +0.0125165 | 0.0000655 | 1.146 |
| fixed_r1 | -1.1373498 | 0.017038 | +0.0014510 | 0.0000576 | 0.806 |
| fixed_r2 | -1.1306452 | 0.019623 | +0.0081557 | 0.0000599 | 0.929 |
| release_r0 | -1.1393042 | 0.023098 | -0.0005033 | 0.0000630 | 1.093 |
| release_r1 | -1.1291532 | 0.032842 | +0.0096476 | 0.0000692 | 1.554 |
| release_r2 | -1.1374772 | 0.026569 | +0.0013236 | 0.0000664 | 1.258 |
| unpinned | -1.1388009 | 0.021128 | 0 | — | 1 |

Negative energy differences favor auxiliary training; variance ratios below one favor it.
Conditional SEs preserve temporal correlation within independent chain columns and paired streams.
They assume equilibration and do not measure disorder or training-seed variability.
Update 400 is fully pinned; update 1600 follows epsilon=0.02 training;
updates 2000/2400 have 400/800 zero-pinning refinement updates.

After 800 updates with pinning removed, 1/3 release branches have lower sampled target energy and 0/3 have lower local-energy variance than the baseline.
This pilot does not show a reliable benefit across the three classical references.
The references were selected before training; choosing the best branch afterward adds selection uncertainty.

| Projection branch | Repeat | Energy/site | Descendant ESS minimum | Largest lineage | Alarm |
|---|---:|---:|---:|---:|---|
| fixed_r0 | 0 | -1.1426385 | 1.00 | 1.000 | True |
| fixed_r0 | 1 | -1.1427573 | 1.00 | 1.000 | True |
| fixed_r1 | 1 | -1.1462578 | 1.03 | 0.984 | True |
| fixed_r1 | 0 | -1.1463178 | 1.01 | 0.996 | True |
| fixed_r2 | 1 | -1.1396127 | 1.01 | 0.996 | True |
| fixed_r2 | 0 | -1.1400613 | 1.02 | 0.992 | True |
| release_r0 | 1 | -1.1466067 | 1.91 | 0.695 | True |
| release_r0 | 0 | -1.1465957 | 1.34 | 0.861 | True |
| release_r1 | 0 | -1.1402744 | 1.03 | 0.986 | True |
| release_r1 | 1 | -1.1403137 | 1.00 | 1.000 | True |
| release_r2 | 0 | -1.1448256 | 1.77 | 0.742 | True |
| release_r2 | 1 | -1.1448607 | 1.16 | 0.928 | True |
| unpinned | 0 | -1.1461901 | 1.49 | 0.793 | True |
| unpinned | 1 | -1.1463276 | 1.03 | 0.984 | True |

Completed projection repeats: 14/14; genealogy alarms: 14.
Recorded failed branches/tasks: 0.

Projection records are independently reconstructed from raw arrays before inclusion.
Reference dependence and start checks are retained in analysis_results.json and the executed results notebook.
Ground-state convergence is unestablished.
