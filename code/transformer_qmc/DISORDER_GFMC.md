# K=1 disorder-averaged GFMC comparison

The executed report is `notebook/disorder_averaged_guided_gfmc.ipynb`.
Its first output records the population coverage of that execution. A running
report is partial; blank panels mean that matched results are pending.

The campaign compares a constant trial state with the selected frozen seed501
transformer on 32 fresh disorders, 12 spins, two longitudinal fields (0, 0.3)
and seven transverse fields (0.25, 0.5, 0.75, 1, 1.5, 2, 3). Four independent
population streams are used at each point. Seed labels are reused across
disorders and methods. Disorder SE and bootstrap intervals condition on those
streams; the separate Monte Carlo uncertainty is calculated from disorder
averages for each stream, retaining their covariance across disorders.

Main settings are 2,048 four-replica walkers, equilibration time 32, energy
measurement time 8, sixteen forward cycles of duration 8, and branch interval
0.1. The first eight disorders receive four additional checks, each changing
one parameter: 8,192 walkers, forward time 16, equilibration time 64, or branch
interval 0.05. This gives 7,168 population records in total.

Energy uses mixed estimation. The overlap mean estimates q_EA; the second
overlap moment and both equal-time susceptibilities have separate plots. At
finite size and zero longitudinal field, q_EA vanishes exactly. The stochastic
overlap estimator can be negative. All forward cycles are retained, including
cycles with poor ancestry. Equal-time susceptibilities are not imaginary-time
response functions.

`reference_disorder_gfmc.py` calculates infinite-population finite-forward
targets using the actual production tag times. Its exact Hamiltonian
projection is an evaluation reference. Exact amplitudes are never used by
`table_projector.py` for propagation. The error decomposition is additive
before taking RMSE; the two component RMSEs do not add in general.

Artifacts live under `data/kagome/transformer_af_disorder_gfmc_v1/`:

- `manifest.json`: checkpoint, settings and production source identities.
- `source_snapshot/`: immutable sources copied from the remote launch.
- `cache/`: 448 exact references and frozen-network amplitude tables.
- `populations/`: resumable independent population records and diagnostics.
- `finite_forward/`: deterministic finite-forward targets for 448 contexts.
- `status.json`, `reference_status.json`: remote execution status.
- `mirror_status.json`: local mirror/report status.
- `audit.json`: checked coverage and provenance.
- `improvement.json`, `disorder_*.png`: paired error ratios and report figures.

Regenerate the report and audit with:

```sh
PYTHONPATH=code OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m transformer_qmc.report_disorder_gfmc --results data/kagome/transformer_af_disorder_gfmc_v1 --notebook notebook/disorder_averaged_guided_gfmc.ipynb
PYTHONPATH=code OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m transformer_qmc.audit_disorder_gfmc --results data/kagome/transformer_af_disorder_gfmc_v1
```

The local watcher `watch_disorder_gfmc.py` mirrors remote results and executes
the report every minute. The report performs a final audit once both remote
stages report completion. A complete result requires 448 caches, 7,168 unique
population records and 448 finite-forward contexts. The production manifest
hashes every Python file in its staged transformer package, so resume against
the preserved remote sources. Analysis additions are staged outside that
package to preserve its identity.

Remote root:
`/root/autodl-tmp/transformer-af-ltfim-20261027-disorder-gfmc-v1/`.
The remote supervisor has a six-hour dispatch cap. If it reaches the cap,
resume its population stage using the same staged sources, checkpoint and
output directory. The AutoDL instance must remain powered on.
