# Monte Carlo failure diagnostics — 2026-11-05

Notebook: `notebook/monte_carlo_failure_L6_glassy_diagnostics.ipynb`

This compact offline bundle contains 320 unique saved measurement records and
384 feedback-training stints, extracted and validated on AutoDL. The scientific
snapshot is **2026-11-05T08:40:14.071530+00:00**. It is not a live monitor.

## Evidence and limits

- L6 low-field fixed-guide budget grid: 96/96 runs have ancestry alarms.
- L6 Γ=.85/1.00 matched-measurement-budget reference grid: 48/48 alarm.
- Final robust feedback: 6/8 alarm; final curriculum PCD: 5/8 alarm.
- L2 exact-reference feedback: 0/4 alarm; strong-field Γ=4 controls: 0/24.
- L8/L10 were user-stopped before physical measurements in the audited scaling
  campaign. Missing data cannot establish a universal L≥6 failure threshold.
- These finite-budget checks establish unresolved sampling reliability, not
  a deep-glass phase or a transition location. Strong-field projection settings
  differ from the lower-field settings; the notebook states those differences.

Ancestry alarms mean descendant ESS<8 or maximum lineage fraction>.5. They are
distinct from execution errors. All selected runs, seeds, disorders, and alarms
are retained. The main study has two prespecified diagnostic disorders; no
eight-disorder ensemble or pooled disorder uncertainty is claimed. Individual
disorder plots have no error bars.

## Files

- `report.json.gz`: lossless compact report, approximately 1.04 MiB. It includes
  full record paths/hashes, tasks, provenance, histogram masses, connected
  spectra, ancestry arrays, training summaries, campaign statuses, retained
  failure artifacts, explicit control reuse, and bond-reconstruction hashes.
- `sha256.json`: SHA256 of compressed report and uncompressed JSON. The notebook
  checks both and matches the report hash to its recorded execution metadata.
- `l4_contrast/`: separately hashed four-run L4 Γ=.60 contrast, training/selection
  provenance and updated notebook verification. Projection settings match L6;
  Adam learning rate and measured-seed coverage do not. See its README.
- `l4_two_seed_contrast/`: expanded L4 comparison with all eight runs: original
  seed0 plus four new seed1 projections of archived guides. Both seeds are now
  retained; the L4/L6 Adam learning-rate difference remains.

Raw tags/counts, models, checkpoints, and other large scientific artifacts stay
in their AutoDL campaign directories. The original report was extracted without
new simulations. The later two-seed L4 supplement adds exactly four seed1
projection measurements, documented in its separate finite-run manifest.

## Reproduction

Open the notebook locally and use **Run All** from the repository or `notebook/`.
Requirements: NumPy, Matplotlib, IPython, and a Python Jupyter kernel; PyTorch
and remote access are not required to render the archived evidence.

The scientific extraction and initial figure execution took place under:

`/root/autodl-tmp/monte-carlo-failure-report-20261105-v1/`

Reproduction scripts in the repository:

1. `code/transformer_qmc/monte_carlo_failure_report.py` — run **on AutoDL** with
   `--base /root/autodl-tmp --output <report-directory>`. It performs read-only
   analysis of the enumerated campaigns and writes a new derived report. Use
   one BLAS thread. It does not submit simulation jobs.
2. `code/transformer_qmc/build_monte_carlo_failure_notebook.py` — use
   `--output <notebook-path>`; it builds the evidence notebook. Set
   `MC_FAILURE_REPORT_DIR=<report-directory>` for remote execution.
3. To reproduce the historical seed0-only notebook, populate `<report-directory>/l4_contrast`
   with `code/transformer_qmc/monte_carlo_failure_l4_contrast.py` on AutoDL; pass
   `--baseline <report-directory>/report.json.gz --output <report-directory>/l4_contrast`.
   This verifies saved measurements only and does not run simulations.
4. The current notebook uses `<report-directory>/l4_two_seed_contrast`; pass
   `--seed1-root /root/autodl-tmp/l4-seed1-contrast-20261105-v1` and
   `--output <report-directory>/l4_two_seed_contrast` to the same extractor after
   the four seed1 cases complete. See that directory's README for the runner.

The original `verification.json` describes the notebook before the L4 addition.
The 32-cell, 13-figure L4 addition is verified in `l4_contrast/verification.json`.
The current 34-cell, 14-figure notebook additionally gives L2 training seed1
its own panel beside seed0 and the exact L2 reference, adjacent to the size
contrast. Both seed1 projection repeats are existing records in the original
320-run evidence. Their projection settings match the L4/L6 panels, while
their final self-learning guide preparation differs. Only disorder60000 is
represented in this L2 validation subset. Current execution verification is
in `l2_seed1_verification.json` for that historical notebook version. The later
L4 seed1 completion keeps 34 cells/14 figures and updates the existing L4 panels
to show both seeds; current verification is `l4_two_seed_contrast/verification.json`.
The original evidence remains unchanged.

The extractor checks raw SHA256, descendant counts and alarms, finite
diagnostics, histogram normalization/moments, and raw q/q²/connected-spectrum
reconstruction where recorded. Training input/output models and raw target/fit
hashes are checked; measurement guide hashes match their training milestones.
Bond arrays are reconstructed from archived source and seed entropy, not
claimed as independently stored raw-GFMC Hamiltonian arrays. Original
`ground_state_qualified` flags remain false.
