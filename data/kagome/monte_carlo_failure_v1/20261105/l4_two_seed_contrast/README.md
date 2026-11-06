# L4 two-training-seed contrast

This supplement completes the missing **training seed1** curves in the L4 P(q)
panels of `notebook/monte_carlo_failure_L6_glassy_diagnostics.ipynb`.
It combines four original seed0 measurements with four newly measured seed1
projections: Γ=.60, disorders60000/60001, projection repeats0/1.

## Matching and limits

- All four L4 guides were already trained in the original production campaign.
  Seed1 is measured from its own archived checkpoint, without retraining.
- Within L4, the two seeds have the same Hamiltonian realization per disorder,
  RBM α=2, curriculum1.50→.75→.60, 400 updates/stage, Adam .01, 256 chains,
  eight temperatures, and two sweeps/update.
- All eight projections use four512-walker populations, guide initialization,
  equilibration16, forward16, δ=.1 and eight cycles. The task/RNG convention and
  repaired `log_weight_v1` implementation are unchanged.
- Blue/red identify training seeds0/1; solid/dashed identify projection
  repeats0/1. All runs and alarms are retained; no individual-disorder error bars.
- L6 uses the same projection settings but Adam .001 rather than .01.
  Equal disorder IDs across L do not imply identical bonds. This is not a
  size-only causal comparison or an assertion of ground-state convergence.
- Original production selected seed0 by frozen energy. The updated comparison
  includes both seeds regardless of energy and ancestry outcome.

## Evidence and checks

`report.json.gz` and `sha256.json` store/check compact evidence for all eight
measurements, guide provenance, historical selections, within-seed repeat
differences, between-seed differences of repeat means, and the seed1 run manifest.
`verification.json` records notebook execution and delivery checks.

Raw tags/counts, checkpoints and logs remain on AutoDL:

- Seed0: `/root/autodl-tmp/l4-production-20261103-v1/results/`
- Seed1: `/root/autodl-tmp/l4-seed1-contrast-20261105-v1/`

The extraction validates raw SHA256, descendant counts/ancestry, q/q² and
connected spectra, guide checkpoint linkage, matching numerical source hashes,
and **every histogram bin reconstructed from raw tags/counts** for all eight
L4 runs. Source artifacts and original seed0 records are hash-preserved.
All original qualification flags remain visible.

The original `report.json.gz`, the prior `l4_contrast/` seed0-only supplement,
and their historical verification files remain unchanged. The notebook renderer
loads the expanded supplement explicitly and does not launch simulations.

## Reproduction on AutoDL

The finite measurement runner is
`code/transformer_qmc/complete_l4_seed1_contrast.py`. It uses one single-thread
worker, saved checkpoints, exactly four cases, no deadline and no automatic
retries. It does not alter other campaigns. A completed run is reused only
after its model/source/raw hashes pass validation.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/bin/python code/transformer_qmc/complete_l4_seed1_contrast.py \
  --production /root/autodl-tmp/l4-production-20261103-v1 \
  --output /root/autodl-tmp/l4-seed1-contrast-20261105-v1

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/bin/python code/transformer_qmc/monte_carlo_failure_l4_contrast.py \
  --base /root/autodl-tmp \
  --baseline <evidence-directory>/report.json.gz \
  --seed1-root /root/autodl-tmp/l4-seed1-contrast-20261105-v1 \
  --output <evidence-directory>/l4_two_seed_contrast
```

Use the notebook generator with `MC_FAILURE_REPORT_DIR=<evidence-directory>`.
Local Run All needs NumPy, Matplotlib, IPython and a Python Jupyter kernel,
without PyTorch or SSH.
