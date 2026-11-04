# L=10 (300-spin PBC kagome) measured feasibility

## Scope and status

Bounded AutoDL H800 probes using the frozen seed501 checkpoint, one disorder (ID 10000), Gamma=0.5, and h=0 or 0.3. These are execution and cost measurements, not converged ground-state or phase-boundary results. Probe source snapshots are retained and their recorded hashes verified by this report generator.

- v1: **completed**, 127.1 s; 2026-10-27T05:11:39Z; precision: float64.
- v2: **completed**, 319.2 s; 2026-10-27T05:14:53Z; precision: float64.
- v3: **completed**, 47.1 s; 2026-10-27T05:20:49Z; precision: float64 versus float32.

## Current float64 inference

| L | Spins | h | Configurations | All-flip seconds | Peak allocated GiB |
|---:|---:|---:|---:|---:|---:|
| 8 | 192 | 0.3 | 128 | 1.731 | 0.924 |
| 8 | 192 | 0.0 | 128 | 3.461 | 0.927 |
| 10 | 300 | 0.3 | 128 | 5.477 | 1.925 |
| 10 | 300 | 0.0 | 128 | 10.960 | 1.930 |

All-flip queries evaluate the ratios needed to choose a single-spin event. The current projector repeatedly requests these ratios during propagation; its exact state cache avoids repeated queries when states recur. Peak allocated memory is PyTorch allocation, not total device memory or a production memory estimate.
The v1 128-configuration float64 L=10 batches allocate about 1.93 GiB. The longer v2 probe also completed with 128 four-replica groups (512 configurations), using about 2.18 GiB peak allocated memory. Production populations and cache storage need their own memory measurements.

## End-to-end short projection

| Probe | L | h | Inference | Four-replica walker groups | Total imaginary time | Wall seconds | Inference seconds | Minimum descendant ESS |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| v1 | 10 | 0.3 | float64 | 8 | 0.025 | 4.735 | 4.187 | 8.00 |
| v1 | 10 | 0.0 | float64 | 8 | 0.025 | 9.279 | 8.733 | 8.00 |
| v1 | 8 | 0.3 | float64 | 8 | 0.025 | 1.149 | 0.930 | 8.00 |
| v1 | 10 | 0.3 | float64 | 32 | 0.025 | 16.851 | 15.127 | 30.12 |
| v2 | 10 | 0.3 | float64 | 32 | 0.200 | 84.875 | 82.849 | 21.33 |
| v2 | 10 | 0.0 | float64 | 32 | 0.200 | 163.647 | 161.856 | 25.60 |
| v2 | 10 | 0.3 | float64 | 128 | 0.025 | 68.358 | 61.566 | 122.27 |
| v3 | 10 | 0.3 | float32 | 32 | 0.200 | 4.238 | 4.008 | 22.26 |
| v3 | 10 | 0.0 | float32 | 32 | 0.200 | 8.159 | 7.929 | 21.33 |

Healthy ancestry over these very short forward times does not predict ancestry at production forward times. Each projector context has one seed; no disorder average or confidence interval on accuracy is inferred.

## Float32 speed/error probe (candidate only)

The precision probe uses full float32 matmul (highest setting), 128 guide-drawn and 32 independent random configurations. Errors compare the same configurations against float64. The projector adapter retains double-precision NumPy rate/weight arithmetic. Changing inference precision is not yet qualified for production.

| h | Chunk | Float64 seconds | Float32 seconds | Speed ratio | Max relative rate error | Max local-energy/site error |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 512 | 6.834 | 0.338 | 20.2x | 3.24e-05 | 2.04e-05 |
| 0.3 | 2048 | 6.834 | 0.320 | 21.4x | 3.24e-05 | 2.04e-05 |
| 0.0 | 512 | 13.664 | 0.678 | 20.1x | 3.82e-05 | 2.19e-05 |
| 0.0 | 2048 | 13.664 | 0.641 | 21.3x | 3.82e-05 | 2.19e-05 |

These finite sample discrepancies do not bound errors for other disorders, fields, states or long projections. Precision changes require exact-reference checks at L=2 and independent-ensemble comparisons of energy, overlap moments and connected correlations at larger sizes. A matching random seed need not produce matching trajectories.

## Cost stress test, not a production forecast

For scale only, extrapolate the W=32, total-time=0.2 probe to W=2048 and total time T=32+8+16*8=168 (the old 12-spin main settings). These settings are NOT qualified at L=10. Assume propagation cost linear in walker count and time, subtract measured sampling setup, and add setup scaled with walker count:

`estimated_seconds = (seconds - sample_seconds)*(2048/32)*(168/0.2) + sample_seconds*(2048/32)`

| Inference | h | Illustrative days per single population run |
|---|---:|---:|
| float64 | 0.3 | 51.61 |
| float64 | 0.0 | 100.77 |
| float32 | 0.3 | 2.55 |
| float32 | 0.0 | 4.99 |

Cache hit rates, state distributions, event rates, long-time ancestry, batch efficiency and concurrency can all change these costs; no billed GPU-hour prediction is established. The purpose is to show why successful execution alone does not make the planned multi-disorder campaign affordable.

## Decision and next gate

- The existing architecture/checkpoint can execute 300-spin PBC inference and short GFMC.
- The current float64 path is not a practical basis for the full proposed campaign at the example W=2048/time=168 settings; inference dominates the measured runtime.
- The isolated float32 candidate gives about 20x faster short propagation on both tested fields. It is a promising optimization, not a production accuracy certificate; the conditional large-run cost remains substantial.
- Production-scope decision, 27 October 2026: include L=10 under the same full 49-context survey and computation/acceptance protocol as L=4,6,8. Initially use 64 fresh disorders, four repeats and the same 16-disorder unguided comparison subset; qualification and stronger audits are additional.
- Use the same observables, precision targets, convergence ladders and release checks at every size. Converged populations and times may differ by size/context; successful short execution does not establish qualification.
- Optimize measured inference cost and validate numerical precision before scaling populations.
- Benchmark larger populations and longer forward times on multiple disorders and fields; then price the common production design using converged correlation observables. Schedule matched crossing cuts first while retaining full anchor-grid coverage as required work; budget-limited unfinished contexts remain pending.
- Report a failure to qualify or an unaffordable configuration explicitly rather than substituting many short biased runs.

Source: `results/benchmark.json` and `results/gpu.csv` under the v1/v2/v3 feasibility directories. Run `PYTHONPATH=code python -m transformer_qmc.report_l10_feasibility` to regenerate.

Precision reference: PyTorch, Numerical accuracy (official documentation, `https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html`).
