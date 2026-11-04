# Low-field GFMC remedy review

L2 development only. No exact labels in guide fitting. Held-out disorders and larger sizes require separate validation; no run is removed for an ancestry alarm.

Fixed-disorder whole-run bootstrap; approximate pointwise 95% bounds. Cycle halves paired within runs; settings independent. Four repeats give limited tail resolution.

Validated 408/408 runs and 24/24 guides.

| Γ | Setting | GFMC ξ | Exact ξ | MC SE | Alarms /32 | Exact check | Half-drift check |
|---:|---|---:|---:|---:|---:|---|---|
| 0.25 | base | 0.08003 | 0.08000 | 0.00031 | 0 | True | True |
| 0.25 | population | 0.08003 | 0.08000 | 0.00026 | 0 | True | True |
| 0.25 | time | 0.07906 | 0.08000 | 0.00035 | 0 | True | True |
| 0.25 | start | 0.08023 | 0.08000 | 0.00034 | 0 | True | True |
| 0.5 | base | 0.36183 | 0.36573 | 0.00108 | 0 | True | True |
| 0.5 | population | 0.36343 | 0.36573 | 0.00036 | 0 | True | True |
| 0.5 | time | 0.36470 | 0.36573 | 0.00110 | 0 | True | True |
| 0.5 | start | 0.36680 | 0.36573 | 0.00103 | 0 | True | True |
| 0.75 | base | 0.42295 | 0.42241 | 0.00048 | 0 | True | True |
| 0.75 | population | 0.42257 | 0.42241 | 0.00027 | 0 | True | True |
| 0.75 | time | 0.42313 | 0.42241 | 0.00049 | 0 | True | True |
| 0.75 | start | 0.42286 | 0.42241 | 0.00059 | 0 | True | True |

| Γ | Base versus | Δξ | MC SE | Contrast passes |
|---:|---|---:|---:|---|
| 0.25 | population | 0.00001 | 0.00040 | True |
| 0.25 | time | -0.00097 | 0.00047 | True |
| 0.25 | start | 0.00020 | 0.00046 | True |
| 0.5 | population | 0.00160 | 0.00114 | True |
| 0.5 | time | 0.00287 | 0.00156 | True |
| 0.5 | start | 0.00497 | 0.00150 | True |
| 0.75 | population | -0.00038 | 0.00055 | True |
| 0.75 | time | 0.00018 | 0.00069 | True |
| 0.75 | start | -0.00008 | 0.00077 | True |

## Overall development checks

- Γ=0.25: passed
- Γ=0.5: passed
- Γ=0.75: passed
