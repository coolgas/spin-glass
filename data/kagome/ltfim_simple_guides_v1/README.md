# Jastrow and RBM guide comparison

The completed screen contains 36 full-basis L2 energy-only fits, and the
projector comparison contains 96 GFMC runs. Three optimization seeds per ansatz
and disorder are retained. Guide selection uses minimum variational energy.
Exact eigenvectors enter only post-training evaluation.

The original screen and first 88 projector records were computed locally before
the user required all computation on AutoDL. Remaining projector runs resumed
from transferred checkpoints on AutoDL. Timing totals therefore mix machines;
they are not a controlled hardware speed comparison. The remote driver checks
source hashes and raw moments before reduction.

Remote workspace: `/root/autodl-tmp/simple-guides-20261103-v1/`.

* `screen/`: all fits, histories, parameters, exact evaluation and projection
  diagnostics. These are explicitly full-basis L2 diagnostics.
* `projector/`: energy-based selection manifest, raw moments, checkpoints, and
  completed 96-run review. Four independent physical populations per run.
* `audit/`: population, time, initialization and triangle-step sensitivity
  checks for Jastrow and alpha=2 RBM. Exact CT forward-walk targets are offline.
* `sampled/`: CUDA sampled-VMC direct/continuation fits at L2 and L4. Frozen
  random/up/down chain diagnostics are separate from adaptive training data.

The latter two campaigns may be incomplete in a retrieved snapshot; consult
their status and launch records. Neither selected L2 comparisons nor L4
feasibility fits qualify the L4–10 finite-size correlation-length target.

The explanatory notebook is `notebook/simple_guiding_wavefunctions.ipynb` and
the method details are `code/transformer_qmc/SIMPLE_GUIDES.md`. Notebook execution
and all follow-up compute run on AutoDL. Raw failures remain in their archives.

## Completed main comparison

Maximum relative connected-spectrum error, continuous-time GFMC, four repeats:

| Gamma | Disorder | Jastrow | RBM alpha=1 | RBM alpha=2 |
|---:|---:|---:|---:|---:|
| 0.25 | 50025 | 0.522% | 0.162% | 0.366% |
| 0.25 | 50059 | 0.078% | 153.028% | 0.707% |
| 0.5 | 50043 | 23.390% | 22.860% | 22.884% |
| 0.75 | 50004 | 0.717% | 0.437% | 0.198% |

These are errors of the mean spectra, not confidence bounds or qualification
thresholds. The alpha=1 RBM fails on disorder 50059 even after selecting the
lowest energy among its three seeds. The Gamma=0.5 case remains unresolved;
both RBM CT groups have an ancestry alarm in one of four runs. Low-field
successes support simpler guides without establishing universal reliability
or proving an ALiBi-specific failure.

`sampled_snapshot.json` preserves the 31 completed sampled fits used in the
executed notebook snapshot. Later files under `sampled/` may contain more fits.
The snapshot's raw NPZ hashes were checked on AutoDL. On disorder 50025,
continuation improves several failed direct fits, but uses more optimization
updates; a matched-compute schedule comparison is still required.

All 64 sensitivity-audit runs completed on AutoDL. At Gamma=0.5, alpha=2 RBM
CT errors are 16.55% with 4096 walkers and 18.79% with doubled times, compared
with 22.88% in the main setting. The uniform-start mean error is 5.23%, but both
runs raise ancestry alarms. This context remains unqualified. The exact CT
finite-time references reveal simultaneous finite-projection and sampling
errors; full details and invalid length fractions are in `audit/review.json`.
