# Saved L4 contrast supplement

Added adjacent to the L6 Γ=.60 distribution panels in
`notebook/monte_carlo_failure_L6_glassy_diagnostics.ipynb`.

## Scope

- Four repaired L4 RBM α=2 measurements: disorders60000/60001, selected training
  seed0, projection repeats0/1, Γ=.60, h=.15, K∼U(.5,1.5).
- Same projection settings as L6: four512-walker populations, guide start,
  equilibration16, forward16, δ=.1, eight cycles. Numerical source hashes match.
- Same training curriculum1.50→.75→.60, 400 updates per stage, 256 chains,
  eight temperatures, two sweeps/update. **Adam differs: L4 .01, L6 .001.**
- L4 seed0 was chosen by lower frozen energy. Seed1 checkpoints exist, but
  these contexts lack seed1 projection histograms. The plots test repeat
  agreement, not L4 training-seed robustness.
- Disorder RNG includes L; equal IDs do not identify equal bonds across sizes.
- All four runs are shown, including the alarmed disorder60000/repeat1.
  No single-disorder error bars or ground-state qualification are asserted.
- One figure shows native probability masses. A second shows L4/L6 on shared
  axes after dividing mass by each size's native Δq=2/(3L²), without smoothing.

L4 repeat TV is .011339/.022496 and absolute q² difference .003606/.008002
for disorders60000/60001. Ancestry alarms are1/4, versus8/8 for the displayed
L6 two-seed measurements. Learning-rate and seed-selection differences prevent
attributing this contrast to size alone or establishing L4 convergence.

## Evidence

`report.json.gz` stores the four measurements, raw paths/SHA256, replayed
ancestry diagnostics, training and seed-selection records, all four L4 guide
hashes, L6 baseline guide signatures/hashes, and within-seed repeat comparisons.
`sha256.json` hashes both compressed and uncompressed evidence. The supplement
is linked to the unchanged original report by `baseline_report_sha256`.
`verification.json` records execution of the updated notebook.

Raw arrays and checkpoints stay in `/root/autodl-tmp/l4-production-20261103-v1/`
on AutoDL. Extraction verifies raw hashes, descendant counts/alarms, histogram
normalization/moments, raw q/q²/connected spectra, guide checkpoint linkage,
energy-based selection, and matching measurement/source settings. It does not
claim independent reconstruction of every histogram bin.

Reproduce **on AutoDL**, with one BLAS thread:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/bin/python code/transformer_qmc/monte_carlo_failure_l4_contrast.py \
  --base /root/autodl-tmp \
  --baseline <evidence-directory>/report.json.gz \
  --output <evidence-directory>/l4_contrast
```

Then generate/execute the notebook with
`MC_FAILURE_REPORT_DIR=<evidence-directory>`. Local Run All needs NumPy,
Matplotlib, IPython and a Python kernel; no PyTorch or SSH is required.
No simulation was launched or restarted for this supplement.
