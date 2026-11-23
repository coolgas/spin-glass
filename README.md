# Rydberg kagome: simulations, observables and figures

The project covers multiple lattice sizes: the historical L4 dataset, current
L6 adaptive production, and future L8 campaigns. Use `rydberg-kagome` for the
shared project and identify lattice size in individual campaign names.

## L6 adaptive feature branch

`pilot/adaptive-d-20261123` adds the compact MPO, adaptive two-start case path
and verified process restart. See [ADAPTIVE_CASE_USAGE.md](ADAPTIVE_CASE_USAGE.md)
for invocation and restart semantics, [PREPRODUCTION_REVIEW.md](PREPRODUCTION_REVIEW.md)
for validation, and [MEMORY_UPDATE_HANDOFF.md](MEMORY_UPDATE_HANDOFF.md) for Ali
outputs and ownership. L6 production is running from the immutable `9ffa6bb`
implementation; see the handoff for current paths and launch evidence. Source
files on this feature branch differ from the historical frozen L4 manifest; the snapshot verifier below
applies to the baseline release. Keep the L4 manifest and dataset intact.

## Historical L4 deployment

Versioned deployment of `rydberg_kagome_L4_Pq_progress.ipynb`, its complete
simulation source, production configuration, reduced dataset and provenance.
Production source: **rydberg-kagome-20261117-v5**. Figure snapshot:
**20261119T005408Z**, 128 disorders × 3 models × 2 detunings × 30 drives.

## Placement on Ali

Project root: `/mnt/ali/yty/spin-glasses/rydberg-kagome/`.
The former `rydberg-kagome-l4-pq` path is a compatibility symlink. Existing jobs,
Git worktree metadata and historical provenance can continue to use it. New
campaigns and documentation should use the canonical project root above.

| Path | Purpose |
|---|---|
| `repository/` | Git-managed simulation source, notebooks, scripts and configurations |
| `data/snapshot-20261119T005408Z/` | Verified reduced arrays and per-case provenance |
| `releases/` | Immutable deployment archives, hashes and Git bundle |
| `outputs/` | New figures, execution logs and future explicit outputs |
| `environments/` | Project-specific environments, when installed |

The notebook is self-contained and includes its plotting arrays. The external
snapshot is supplied for independent checks and reuse. `code/rydberg_kagome/`
matches every hash in `provenance/production-release/manifest.json`; current local
code and frozen v5 were identical at deployment. The original source was
uncommitted in its parent repository; this standalone repository establishes a
tracked baseline without pretending that the parent Git revision contained it.

## Verify and reproduce plots

From `repository/`, with NumPy installed:

```sh
python scripts/verify_snapshot.py
```

Install `requirements/plotting.txt` in a project environment and provide LaTeX,
`dvipng`, `amsmath`, `type1cm`, and `cm-super`. Then either open the notebook in
Jupyter and run all cells, or export all figures directly:

```sh
python scripts/render_notebook.py --output ../outputs/figures-20261120
```

The output directory must be new. Figures use the exact notebook code, retain
the six flags and uncertainty bands, and export vector PDFs plus PNG previews.
Missing LaTeX is an explicit error; there is no typography fallback.

## Recompute simulations and regenerate the snapshot

Use the production dependency versions in `requirements/simulation.txt` with
Python 3.12. Read `code/rydberg_kagome/README.md` and the CLI help:

```sh
PYTHONPATH=code python -B -m rydberg_kagome --help
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B -m unittest discover -s code/rydberg_kagome/tests -v
```

`configs/production-tasks.json` and `configs/production-spec.json` preserve exact
inputs and seeds. New simulations require fresh output directories and appropriate
resource planning. Export an independently completed campaign using:

```sh
python scripts/export_snapshot.py --run /absolute/completed/run --output /absolute/new/snapshot
```

The exported arrays can be embedded in a future notebook under a new snapshot ID.
The notebook supplied here is the exact current saved version, including outputs.
Original notebook-building/editing scripts are retained under
`provenance/notebook-history/` for lineage; their old absolute paths are historical
and should not be run blindly on Ali.

## Scientific and checkpoint provenance

These are fixed chi_RG=12, D_MPS=128 results with unresolved RG sensitivity and
six flagged solver cases. The histograms use 20,000 replica pairs per case and
equal disorder weighting. Derivatives, Binder estimator, bootstrap seeds and
flag propagation are defined in the notebook.

Wavefunction/model checkpoints remain on AutoDL. They are not required to redraw
these figures. Original result files and checkpoint cleanup records distinguish
retained selected states from intentionally retired alternates. The historical
README in the frozen source describes the original two-start checkpoint layout;
the later retention receipts in `provenance/` document the current layout.

## Code management

`main` holds the verified deployment baseline, tagged `ali-l4-pq-20261120`.
Use descriptive feature branches for changes. Keep release artifacts immutable,
record hashes and environment versions, and place new outputs outside Git.
The repository has no configured public remote. Its matching Git bundle and
deployment archive are retained in the sibling `releases/` directory.
