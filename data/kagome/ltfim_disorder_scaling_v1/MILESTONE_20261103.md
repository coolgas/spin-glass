# Executed overlap milestone, 2026-11-03

At fixed h=0.15, the completed compact notebook shows equal-weight disorder
averages of measured P(q) at Gamma=0.60 and 0.65. L2 overlays exact guides,
actual guide MCMC, exact ground states and pure GFMC. L4 compares guide MCMC
before projection with pure GFMC afterward. Capped error bars show one
standard error across eight disorders, after averaging starts and repeats
within each disorder. No disorder is selected by its observed width.

The periodic kagome Hamiltonian is H=sum K_ij ZiZj -0.15 sum Zi -Gamma sum Xi.
Independent AF K_ij are Uniform(0.5,1.5), stored as J_ij=-K_ij. Bonds use
NumPy default_rng(SeedSequence([20261027,L,sample_id,1])) in bond-list order.
Quenched bonds are reused across fields, guides and optimization seeds.

## Frozen deliverables in this commit

- `notebook/preliminary_disorder_overlap.ipynb` and `.html`, executed on AutoDL.
- `notebook_snapshot/20261103T031619Z_compact_complete/`: complete 128-run
  projected histogram ensemble, 32 L2 guide-MCMC tasks, L4 guide samples,
  exact L2 references, raw tags/counts, provenance and PNG/PDF figures.
- `notebook/overlap_scan_progress.ipynb` and `.html`, executed on AutoDL.
- `scan_progress/20261103T032310Z/`: original/fine run JSON and raw NPZ,
  status, manifests, historical failures, summary and width-scan PNG/PDF.
- Research source, production plan, existing research notebooks and figures
  are saved together. These earlier notebooks can require campaign mirrors
  beyond the two self-contained frozen overlap snapshots above.
- `validation_20261103.log`: 33 AutoDL regression tests passed, covering
  checkpoint replay, moments, core physics, sampled guides, overlap
  distributions and finite-size crossing logic.
- `histogram_validation_20261103.log`: the weighted pure histogram estimator
  agrees with exact convolution and independent first/second moments.
- Root `compact_complete_delivery_sha256.json` verifies the complete compact
  delivery. Root `overlap_milestone_sha256.json` covers both overlap notebooks,
  their HTML exports, the frozen snapshots and both validation logs.

The large ongoing training/production mirrors and earlier intermediate
snapshots remain on local/AutoDL disks. They are not part of this frozen
delivery. All seeds, selection diagnostics, invalid ratios, ancestry alarms,
failed predecessors, checkpoints and raw measurements remain in their
campaign directories; the milestone does not delete or overwrite them.

## Broader distributions and remaining measurements

The complete L4 moment ensembles show widths at Gamma=0.60,0.65,0.70,0.85
of 0.1567,0.1625,0.1703,0.2025 for Jastrow and
0.1453,0.1532,0.1619,0.2119 for RBM. These are widths of the disorder-averaged
distribution, with disorder jackknife errors. They do not supply unmeasured
histogram shapes. Much of the variance increase is within disorders.

The width snapshot has only one complete two-repeat disorder per guide at
the new fine-grid field Gamma=0.55. A later live check found 34/576 fits and
18 completed projection runs in the fine campaign, including four Jastrow
and two RBM main runs there. The original campaign had 250/320 fits and
260 projection runs. Neither active scheduler reported current failures;
historical setup failures remain retained. Live progress will advance beyond
these dated counts.

Full P(q) histograms at 0.70/0.85 require additional deterministic replay.
Fine-grid averages and L6 projected ensembles are incomplete. The original
L6 fields 0.60,0.65,0.70,0.85,1.00 remain scheduled alongside the fine low-field
extension. L8/L10 remain deferred, cancelled worldline work remains stopped,
and no polling monitor is installed.

Ancestry alarms and limited start/population/time sensitivity checks keep
ground_state_qualified=false. Disorder error bars do not bound mixing or
projection bias. Broadening does not yet establish glassiness, finite-size
drift, a critical point or a scaling collapse.

## Measured broader-distribution follow-up

`notebook/broader_overlap_distributions.ipynb` and its HTML companion add
measured higher-field histograms, executed on AutoDL. The separate campaign
`broader_histograms/20261103T034000Z/` freezes all complete original main
ensembles, reuses the 64 verified 0.60/0.65 histogram records, and replays 80
higher-field runs with four CPU workers at reduced priority. It retains models,
source hashes, raw measurements, production checkpoints and replay checkpoints.
The earlier paragraph describes the state before this follow-up.

The preview freezes only complete measured groups into a separate snapshot;
pending RBM histogram ensembles are labeled on the plots. Jastrow at Gamma=0.85
has peak q=0.375, mean q=0.32064, width 0.20248, and negative mass 0.05365,
compared with peak q=0.625, mean q=0.58095, width 0.15670 and negative mass
0.000735 at Gamma=0.60. These are actual histogram shapes, with no smoothing
or selection by broadness. The report includes capped pointwise disorder bars,
width jackknife errors, paired field differences, magnetization and a variance
decomposition. All ancestry alarms remain equally weighted.

Delivered preview: `broader_histograms/20261103T035100Z_preview/`, captured at
2026-11-03T03:49:58Z. It contains 112 verified histogram runs: complete Jastrow
ensembles at 0.60,0.65,0.70,0.85,1.00 and complete RBM at 0.60,0.65.
At Gamma=1.00, Jastrow has peak q=1/6, mean q=0.13456, width
0.21726 +/- 0.00687 and P(q<0)=0.23566. The histogram remains single-peaked
while broadening and shifting toward zero. Higher-field RBM replays continue
in the separate campaign. `broader_overlap_delivery_sha256.json` at the
repository root covers the frozen preview and executed report.

Five focused AutoDL regression tests passed. The notebook audits frozen input
hashes and raw measurement hashes, reconstructs moments and connected spectra
from descendant measurements, and compares every displayed run against frozen
production to 1e-12. Ground-state qualification remains unresolved.
