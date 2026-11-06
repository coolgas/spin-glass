## Scan scope and timing

Queued scan: Gamma=0.100,0.125,...,1.500, with 57 fields shared by L=4,6,8,10;
eight matched disorder IDs (60000–60007) at each size, m=6, chi=2,3, and two
solver starts. K~Uniform(0.5,1.5), lambda=1, h=0.15. There are 3,648 cases
before reuse. The graph above contains 16 two-disorder pilot points, at
Gamma=0.1,0.5,1.0 where available; L10 is unfinished.

At the status check on 2026-11-04 23:21 UTC, the D1024 continuation of
L8, Gamma=1.0, disorder 60000, chi=3 was active after 1 hour 56 minutes.
Neither seed had produced its result. The scan waits for both starts to pass
and for the remaining L8/L10 frontier checks.

Measured total time per case, including both starts: L4 about 9–10 seconds;
L6 about 36–100 seconds; passing L8 about 99–206 seconds. The unresolved
L8 case took 5.2 hours through D512, followed by the ongoing D1024 diagnosis.
L10 has no timing benchmark yet.

Conditional estimates for the serial scan use about two days for L4/L6/L8
at typical passing-case rates, plus 912 L10 cases:

| Assumed average L10 time per case, including both starts | Approximate scan time |
| --- | --- |
| 10 minutes | 8–9 days |
| 1 hour | 40 days |
| 5 hours | 6–7 months |

These are throughput scenarios, not a completion forecast. They exclude
additional convergence investigations, further refinement, and retention
qualification. The current diagnostic has no reliable remaining-time estimate.
Periodic local fetching is stopped; this notebook is a synced snapshot.
