# Reproducing the research campaigns

The current implementations are under `code/`. Historical Python sources,
shell scripts, protocols, launch manifests, task grids, source hashes and
available environment records are retained at their original paths under
`data/kagome/`. Those source/configuration files are intentionally versioned
even when the corresponding raw outputs are omitted.

`source_inventory.json` lists every preserved archive source/configuration file
and its SHA-256. `code/export_reproduction_sources.py` rebuilds that inventory
from a full local archive. Historical snapshots matter: later changes to seeds,
solvers and reporting can change a rerun. Use a campaign's own `code/`,
`reused_source/` or `original_reused_source/` with its launch manifest when
matching historical results. Never replace a historical snapshot with current
sources and assume that its hashes still match.

## Environment and execution

Use a Linux scientific Python environment (the recorded bond-scan environment
used Python 3.12.3 and NumPy 2.4.6). Install the dependencies with
`python -m pip install -r reproduction/requirements.txt`. Select a suitable
PyTorch build for GPU training. This dependency list is not a historical lock
file; available campaign environment records take precedence.

Run scientific calculations and figure generation on AutoDL. From the
repository root, set:

```sh
export PYTHONPATH="$PWD/code"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Use fresh output directories: campaign drivers resume existing cases, and some
check source hashes before resuming. The commands below are explicit compute
launches, not commands required merely to view the embedded notebooks.

## Original-bond L4/L6 and L6 Gamma scan

```sh
python -m transformer_qmc.triangle_effective_l6_uniform --output runs/l6_uniform/results
python -m transformer_qmc.triangle_effective_l6_report --results runs/l6_uniform/results --output runs/l6_uniform/report
python -m transformer_qmc.triangle_effective_l6_gamma_scan --output runs/l6_gamma/results --reuse runs/l6_uniform/results
```

The first campaign produces 48 cases: L4 and L6, Gamma=0.5, disorders
60000–60007 and (m,chi)=(4,2),(6,2),(6,3). The scan produces 120 L6 cases at
Gamma=0.1,0.25,0.5,0.75,1.0, including 24 reusable fixed-field cases. The scan
can also run independently by omitting `--reuse`.

All bonds have K~Uniform(0.5,1.5), lambda=1 and h=0.15. Disorder entropy is
`SeedSequence([20261027,L,sample_id,1])`. Both solver starts (0 and 1) and 4096
physical replica pairs are retained. Sampling seeds are set in
`triangle_effective_l6_adaptive.run_case`; historical reused cases may have
older sampling seeds, preserved in their source snapshots. Thus a fresh run
can reproduce the experiment without necessarily matching every old sampled
histogram bin byte for byte.

The driver writes cases, states, isometries, spectra, diagnostics, summaries,
figures and an embedded notebook. For figure-only regeneration after a run:

```sh
python -m transformer_qmc.triangle_effective_l6_gamma_scan --output runs/l6_gamma/results --report-only
```

Production plots show only disorder SE. Histograms are normalized per disorder
and averaged equally; paired changes retain covariance. Solver agreement does
not establish physical convergence. The saved L6 results remain sensitive to
retention and have no exact full-spin L6 reference.

## Earlier triangle RG studies

```sh
python -m transformer_qmc.triangle_effective_interaction_scan --output runs/interaction/results
python -m transformer_qmc.triangle_effective_l2_fixed_target --results runs/interaction/results --output runs/l2_fixed_target
python -m transformer_qmc.triangle_effective_bond_scan --output runs/bond_scan/results
```

These are distinct historical experiments: the interaction scan changes lambda;
the bond scan changes the bond range. Their parameters must not be substituted
for the original-bond L6 experiments.

The earlier L4 benchmark is `triangle_effective_benchmark --output PATH
--baseline BASELINE`, with `triangle_effective_report --results PATH`.
The higher-field comparison is `triangle_effective_field_scan --output PATH
--reference REFERENCE_JSON`. Their comparison inputs come from earlier
campaigns; regenerate those first or restore the archived inputs. Their launch
manifests record input locations/hashes. The code-only delivery does not supply
every measured comparison input.

## Earlier QMC, variational and projection campaigns

The existing guides in `code/transformer_qmc/` document the launchers and
prerequisites:

| Campaign family | Guide / entry points |
| --- | --- |
| Transformer training and benchmarks | `CAMPAIGNS.md`, `run_campaigns.py`, `experiment.py` |
| GFMC production | `PRODUCTION_GFMC.md`, `production_protocol.py`, `production_runner.py`, `production_dispatch.py` |
| Disorder and overlap scans | `DISORDER_GFMC.md`, `FINE_OVERLAP_SCAN.md`, `disorder_scaling.py`, `fine_overlap_campaign.py` |
| Simple guides and sampled training | `SIMPLE_GUIDES.md`, `simple_guide_benchmark.py`, `sampled_simple_guides.py` |
| Triangle projection | `TRIANGLE_METHOD.md`, `triangle_campaign.py` |
| Worldline checks | `WORLDLINE_METHOD.md`, `worldline_campaign.py`, `worldline_accuracy.py` |
| L6 convergence and learned guides | `L6_CONVERGENCE.md`, `L6_PCD_CURRICULUM.md`, `SELF_LEARNING.md` |
| Auxiliary VMC | `auxiliary_vmc_pilot.py`, `auxiliary_vmc_report_completion.py` |

Read each launcher's `--help`, its guide and the matching campaign manifest.
Retain the recorded task grid, disorder IDs, seeds, precision, checkpoint
selection and solver settings. Training-dependent experiments require retraining
or restoring their prerequisite checkpoints; omitted checkpoints cannot be
reconstructed from their hashes alone. Some historical launchers contain
absolute AutoDL paths that must be mapped to the new working directory.

## Verification and limits

The published notebooks already embed their figures. Reproducing the numerical
outputs requires rerunning the simulations and, for learned guides, training.
This delivery preserves all tracked source/configuration snapshots found in the
local archives, but it is not evidence that every historical campaign has been
rerun successfully from a clean environment. GPU nondeterminism, eigenspace
choices, library versions and historical sampling seeds can affect exact output
bytes. Compare physical measurements and reported uncertainties as well as
source hashes.

The existing `verify_triangle_bond_delivery.py` and
`verify_triangle_l6_gamma_delivery.py` validate full downloaded archive layouts,
including reused source snapshots. They are not validators for an arbitrary
fresh output-directory layout. Preserve a run's full layout when using them.

Raw outputs remain in the original local archive and on its local Git branch
`archive/raw-results-20261104`; this branch is not part of the code-only push.
