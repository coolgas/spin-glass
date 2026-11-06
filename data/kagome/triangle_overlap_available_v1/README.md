# Available overlap results, 2026-11-05

Open `notebook/triangle_overlap_available.ipynb` (relative to the repository root)
for five embedded figures, observable
definitions, qualification criteria, coverage, and interpretation. All scientific
extraction and plotting ran on AutoDL using an isolated analysis source snapshot.

The report contains 217 distinct extracted cases, 214 qualified cases, and three
excluded original solver failures. Views overlap: 23 requested m=8, chi=3/16
cases; 33 qualified L8 retention cases; 144 archived exact retained-space cases.
All seven available chi=3 to 16 pairs (six L4 and one L6) exceed at least one
prospective retention tolerance. Solver convergence does not establish retention convergence.
The current coverage does not support a critical-field or exponent fit.

The L6 archive spans Gamma=0.1, 0.25, 0.5, 0.75, 1.0 with eight matched disorders
for each of (m,chi)=(4,2),(6,2),(6,3). Intervals use 5,000 whole-disorder
bootstrap resamples (seed 20261105); no bars appear on individual curves or
the fixed two-disorder size means. Size ensembles are independent.

`brief.json` records counts, requested coverage and paired shifts. `delivery.json`
records notebook/figure hashes and source/dependency hashes. `hold.json` records
the dispatch hold; `worker_snapshot.json` records the subsequent live audit.
Controller status files remain historical while dispatch is held.
The delivery manifest preserves remote artifact basenames. Verify its notebook
entry against `notebook/triangle_overlap_available.ipynb`; the other delivered
files reside in this data directory.

## Local report placement

All notebooks belong in `notebook/`, never under `data/`. Fetch this notebook
directly to `notebook/triangle_overlap_available.ipynb`. For subsequent metadata
and figure transfers into this directory, exclude `*.ipynb`; do not stage a
notebook here before copying it.

Remote analysis root:
`/root/autodl-tmp/triangle-overlap-available-20261105-v1/`

Full case extraction, summary, archive index, saved bootstrap indices, PDFs,
physical moments, states, checkpoints, isometries and campaign sources remain
on AutoDL. Only the concise notebook (~1.2 MB), five small plot previews and
small evidence were fetched. No persistent fetcher was started.

Requested and original retention dispatch remain held pending further direction
after the resource upgrade. Active solves, the D1024 diagnostic and unrelated
overlap work were preserved. The full matched L=4,6,8 plan remains documented
in `code/transformer_qmc/CORRELATION_LENGTH_SCALING_OUTLINE.md`.
