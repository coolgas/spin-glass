# Fixed-h disorder-average finite-size scaling pilot

Scientific endpoint: at fixed h=0.15, tune Gamma across sizes L=4,6,8,10,
test crossings of xi_R/L and eventually a common finite-size scaling collapse.
No critical point or exponent is assumed. The field window must be numerically
workable for disorder averages; individual hard samples remain equally weighted.

## Launched pilot

Remote workspace: `/root/autodl-tmp/disorder-scaling-20261103-v1/`.
All training, projection, analysis and tests run on AutoDL. Local use is editing
and transfers. Campaign PID and command are in `launch.json`.

The follow-up scan is in `/root/autodl-tmp/disorder-scaling-fine-20261103-v1/`.
At fixed h=0.15, its requested L=4 grid is Gamma=0.55–0.70 with spacing 0.01
(16 points). Existing 0.60/0.65/0.70 results supply the comparison fields;
13 additional fields are queued. The L=6 follow-up grid is 0.55–0.60 with
spacing 0.01. The manifest records explicit `fields_by_size` and reuse sources.
The extended launcher retains the original training deadline and two trainers /
two projector workers; the expanded queue may require a later continuation.
All new projections retain raw pure histograms, width, negative mass and uniform
magnetization. See `code/transformer_qmc/FINE_OVERLAP_SCAN.md` for the protocol.

- Sizes L=4,6; Gamma=0.60,0.65,0.70,0.85,1.00; h=0.15.
- Eight development disorders 60000–60007, shared across fields at each size.
  Disorder generation depends on L, so equal IDs across sizes are not paired.
- Jastrow and alpha=2 RBM; two continuation seeds per context; 1200 updates
  per fit, 256 chains, eight tempering levels. No basis enumeration in training.
- Choose the seed with lower frozen sampled energy averaged across random,
  up and down starts. All seed diagnostics/checkpoints remain available.
- Four independently branching physical GFMC populations; two runs per setting.
  Main setting: 512 walkers per population, block 0.1, equilibration and forward
  times 16, eight measurement cycles.
- Fixed disorders 60000,60001 also test 1024 walkers, doubled times, and uniform
  starts. These checks compare paired disorder subsets, not unrelated averages.
- Total plan: 320 fits and 560 GFMC runs. Training has a 12-hour checkpointed
  budget; submitted GFMC jobs drain afterward. A paused campaign is resumable
  with the same command, manifest and source snapshot.

The pipeline trains on the H800 and dispatches frozen guides to eight CPU
workers. L8/L10 are the next stage after cost and numerical sensitivity are
assessed at L4/L6. The cancelled worldline campaign remains stopped.

### Training throughput update

The pilot now uses `parallel_disorder_scaling.py` with four independent CUDA
trainers sharing the H800 through NVIDIA MPS, alongside eight CPU projector
workers. `parallel_launch.json` records the active launch; `parallel_status.json`
records aggregate progress and `training/worker_status/` records live tasks.
The original scientific source snapshot and training signatures are reused.
Each fit retains its own model, optimizer, chains, RNG and checkpoint. The
original 12-hour deadline, L4/L6 grid and per-fit budgets are retained.

Before switching, the old executor was stopped and all results archived under
remote `pre_parallel_20261103T092326/`: 24 completed fits and 44 projector runs.
The parallel executor reuses completed records and resumes unfinished tasks.
It starts only after every old worker has exited, preventing concurrent writes.

Production-step benchmarks on AutoDL measured four-versus-one MPS trainer
throughput gains of 3.70x/2.53x for L4 Jastrow/RBM, and 3.38x/1.77x for L6.
These short benchmarks exclude frozen diagnostics, checkpoints and scheduling
overhead; they are not end-to-end completion-time guarantees. A CUDA test
confirmed serial/concurrent parameters and diagnostics agree to 1e-12, completed
checkpoints can be reused, and live status writes are isolated per worker.
Standalone benchmark records are in `training_concurrency_mps.json`.

## Observables and uncertainty

q = N^-1 sum_i <Zi>^2, including the field-induced polarization background;
q2 = N^-2 sum_ij <ZiZj>^2. Products use separate physical populations.
Connected spectra use the four-population cross-covariance estimator. Retain
S(0) and three shortest reciprocal directions. Average spectra equally across
disorders before forming xi_R/L, with |kmin|=2pi/(sqrt(3)L). Invalid ratios and
ancestry alarms are retained, never clipped or filtered out of an ensemble.

Exploratory intervals resample whole disorders after averaging two independent
runs. D=8 pilot intervals and D=2 sensitivity subsets do not certify precision.
Partial summaries explicitly list included disorder IDs. Final analysis needs
paired-field resampling and sampling uncertainty checks before collapse fitting.

All points initially carry ground_state_qualified=false. The report generates
q and xi_R/L versus Gamma for each guide and size, plus paired sensitivity
differences. No collapse or critical-exponent fit is performed automatically.

## Validation and artifacts

Three tests passed on AutoDL: moments against an independent exact contraction,
physical-population permutation invariance, and exact interrupted/resumed
projector replay with raw-moment reconstruction.

Remote `results/` contains manifest, training checkpoints and diagnostics,
selection records, raw GFMC moments, run checkpoints, progress, review.json,
and completion status. At campaign completion the report writer produces
REPORT.md, sensitivity.json and standalone PNG/PDF figures. No polling monitor
or instance-shutdown action is installed. The completed overlap milestone is
documented in `MILESTONE_20261103.md`; active campaigns continue separately.

## Preliminary overlap notebook

Open `notebook/preliminary_disorder_overlap.ipynb` from the repository root for
executed plots and tables, or its `.html` companion for a browser view. The
notebook is executed on AutoDL and copied back with its outputs embedded.
It uses frozen production records under `notebook_snapshot/`. Dedicated later
histogram measurements use those same frozen guide checkpoints and are recorded
separately in `histogram_protocol.json`.

The compact notebook has two figures. The L2 figure overlays exact Jastrow/RBM
guide distributions, actual guide MCMC, pure GFMC and the exact ground state
on common axes, with one panel per Gamma=0.60/0.65. The L4 figure uses
before/after projection columns and Gamma=0.60/0.65 rows, showing actual guide
MCMC before and pure GFMC after projection. Every curve averages eight
disorders equally.
Individual samples and diagnostics remain in the saved data. Error bars show
one standard error across eight disorder means, s_J/sqrt(8), with caps and no
shaded bands. Starts and repeats are averaged within each disorder first.

The pure histogram estimator retains tagged spin configurations and their
descendant counts separately for four independently branching populations.
It pools equal forward cycles within each population, crosses the weighted
configuration measures only between distinct populations, and averages the six
population pairs. Two repeats are averaged before the equal disorder average.
For L4 the additional measurement deterministically replays the original main
tasks; q, q2 and connected spectra must match the original run records to 1e-12.
Raw tags, weights, RNG checkpoints and ancestry alarms are retained. Alarmed
samples are never removed from the average.

L2 uses disorders 50000–50007 and the previously energy-selected full-basis
guide fits. Its new before-projection MC uses 256 independent chain columns,
eight tempering levels, 500 burn-in sweeps and 100 retained draws separated by
five sweeps, pairing disjoint 128-column halves at each draw; random/up/down
starts receive equal weight. The after-projection setting matches the L4 main
setting: 512 walkers per population, equilibration/forward times 16, eight
cycles and two repeats. The exact L2 reference uses a different ensemble from
L4. All distributions show probability mass at allowed discrete q values,
without smoothing. Error bars exclude mixing and projection-bias bounds.

Freeze/sample on AutoDL with `python -m transformer_qmc.preliminary_overlap_notebook`
and its explicit production, bundle-root and exact-scan arguments. Measure pure
histograms with `python -m transformer_qmc.projected_overlap_histograms`, passing
`--snapshot`, `--production`, `--exact-scan`, and `--workers 8`. This reuses
completed tasks and resumes checkpoints. Then rebuild the compact display with
`python -m transformer_qmc.compact_overlap_notebook --snapshot ... --output ...`.
The main notebook generator delegates to the compact display once these
histogram artifacts exist. The older diagnostic layout remains available as
`make_diagnostic_notebook`; all original diagnostics remain on disk.

Completed compact snapshot: `20261103T031619Z_compact_complete`. It contains
128/128 projected histogram records and 32/32 L2 guide-MCMC records, together
with L4 guide samples, exact references, raw archives, provenance and figures.
The earlier production capture is retained separately on disk. Histogram
sampling uses the captured guide checkpoint hashes; deterministic L4 replay
must reproduce production moments and spectra to tolerance 1e-12.

## Width-scan progress notebook

`notebook/overlap_scan_progress.ipynb` and its `.html` companion were executed
on AutoDL using frozen records in `scan_progress/20261103T032310Z/`.
Only complete eight-disorder, two-repeat main ensembles are plotted. The
measured width is sqrt([<q^2>]_J - [<q>]_J^2); width error bars use a
delete-one-disorder jackknife and mean-overlap bars use s_J/sqrt(8).
Raw hashes, moment reconstruction and the within/between-disorder variance
identity are checked during execution. Full P(q) shapes are not inferred
from moments.

For L4, widths increase from Gamma=0.60 to 0.85: Jastrow 0.1567 to 0.2025,
RBM 0.1453 to 0.2119. The corresponding within-disorder variances also
increase. These remain preliminary because of ancestry alarms and incomplete
sensitivity checks. Full histogram shapes at 0.70/0.85, complete fine-grid
averages and L6 projected ensembles remain pending at this snapshot.

To refresh on AutoDL, use `python -m transformer_qmc.overlap_scan_progress`
with `--original /root/autodl-tmp/disorder-scaling-20261103-v1`,
`--fine /root/autodl-tmp/disorder-scaling-fine-20261103-v1` and a separate
`--root` bundle directory. Each execution freezes a new dated snapshot.

All 64 guide histogram tasks completed (32 selected-seed contexts). The
eight-disorder, three-start averaged guide distributions have largest bins at:

| Guide | Gamma | Largest bin q | Mean q | P(q<0) |
| --- | --- | --- | --- | --- |
| Jastrow | 0.60 | 0.6250 | 0.602107 | 0.00015625 |
| Jastrow | 0.65 | 0.5833 | 0.565232 | 0.00037760 |
| RBM alpha=2 | 0.60 | 0.6667 | 0.638860 | 0.00028646 |
| RBM alpha=2 | 0.65 | 0.6250 | 0.605550 | 0.00026042 |

These are finite-MCMC unprojected guide estimates. Maximum within-disorder
total-variation differences between starts are 0.047–0.056 across these four
groups. Temperature transport and short-start agreement do not certify valley
equilibration. The exact L2 difficult disorder 50043 retains maxima at q=-0.5
and q=+0.5 at both fields, while its negative mass increases from 0.19946 to
0.26650. The exact eight-disorder average has one maximum at q=2/3 and negative
mass 0.08848 and 0.11911, respectively.
