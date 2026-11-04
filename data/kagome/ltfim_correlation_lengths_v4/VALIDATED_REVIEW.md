# Validated v4 convergence review

Development only. Exact agreement and small contrasts do not override ancestry alarms; stronger plateaus, alternative starts, held-out checks, and larger-size convergence remain required.

Fixed-disorder repeat bootstrap; approximate pointwise 95% bounds; four repeats have limited tail resolution. Independent settings resampled independently; cycle halves paired within runs.

Validated 103/384 runs; 1/6 complete groups.

| Γ | Forward | Walkers | GFMC ξ | Exact ξ | MC SE | Alarmed runs | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 64 | 8192 | 0.46117 | 0.53404 | 0.01824 | 23/64 | unresolved |

| Contrast (second − first) | Δξ | MC SE | Shift within tolerance |
|---|---:|---:|---|

Per-disorder ancestry and paired early/late-cycle diagnostics are in `validated_review.json`. All runs are retained.
