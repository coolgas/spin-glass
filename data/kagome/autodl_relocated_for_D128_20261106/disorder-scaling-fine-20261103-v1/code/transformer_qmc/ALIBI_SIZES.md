# Larger-size ALiBi refinement

Remote root: `/root/autodl-tmp/transformer-af-ltfim-20261026-alibi-sizes-v1`.
Local results: `data/kagome/transformer_af_alibi_sizes_v1/`.

The campaign specializes the original shared ALiBi checkpoint to periodic
Kagome L=4,6,8 (48,108,192 spins), at Γ=0.5,h=0.3 and disorder ID 0. Two
independent optimization seeds (201,202) start from the same original shared
checkpoint. The L2-specialized checkpoint is not reused. Architecture remains
width 32, two causal ALiBi blocks, four heads, and one static-context block.

Each run uses 2,400 sampled VMC updates, batch 128, Adam learning rate
0.0005→0.00005, and gradient clipping at norm 10. The final checkpoint is
predeclared; there is no checkpoint or seed selection on noisy energy estimates.
Final evaluation uses 32 batches × 256 independent energy configurations and
independent four-replica samples, with streams separate from training and
interim validation. No exact-state labels or enumerated guides are used.

The batch is smaller than the 12-site refinement's 512; the larger-state
all-spin-flip cost is measured as part of this bounded pilot. These runs test
improvement and seed sensitivity, not a predeclared exact-accuracy threshold.

Artifacts per run:

- `settings.json`, `disorder.npz`, `history.json`, `validation.json`.
- Named `__latest.pt`, initial/final snapshots, and `checkpoint_index.json`.
- `evaluation.json` with energy, local-energy variance, q², connected χ,
  batch values, standard errors, evaluation RNG seed, and latest-checkpoint hash.
- Final filename such as
  `alibi__L08__N192__g0p5__h0p3__dis000__init-shared__seed201__step002400__final.pt`.

`summary.json` retains all six outcomes. `status.json` reports current execution
state. The detached process writes `train.log` at the remote root. A lock
prevents overlapping supervisors for the same output; rerunning the same
command against the frozen source resumes incomplete runs from saved state.

Load a final checkpoint:

```python
import json
from pathlib import Path
from transformer_qmc.physics import make_af_sample
from transformer_qmc.unified import UnifiedTransformer
from transformer_qmc.experiment import load_checkpoint
from transformer_qmc.model import TransformerGuide

root = Path('data/kagome/transformer_af_alibi_sizes_v1')
rows = json.loads((root / 'summary.json').read_text())
run = next(r for r in rows if r['L'] == 8 and r['seed'] == 201)
state = load_checkpoint(root / run['final_checkpoint'], 'cpu')
ham = make_af_sample(run['L'], 0).hamiltonian(.3)
model = UnifiedTransformer(ham, .5, **state['metadata']['architecture'])
model.load_state_dict(state['model'])
guide = TransformerGuide(model)
```

After downloading artifacts, run
`PYTHONPATH=code python3 -B -m transformer_qmc.report_sizes`
to generate `notebook/alibi_larger_sizes.ipynb` and standalone plots.

Interpretation: lower energy at fixed Hamiltonian is a variational improvement
when resolved over sampling error. Lower variance and seed agreement are useful
but do not certify the ground state or pure overlap convergence. Compare both
seeds' observables; independent projector/population tests remain necessary.
Every final checkpoint is Hamiltonian-specific, even though its architecture
accepts other sizes.

The frozen remote runner hashes `checkpoint.pt` (the named latest archive),
while recording the relative final-snapshot path. Separate PyTorch saves can
have different byte hashes despite identical model weights. The report checks
the latest hash, verifies exact final/latest weight equality, reloads each
final model, and exercises direct ratios. `checkpoint_verification.json`
records both file hashes explicitly. The local runner also records the hash's
path for future campaigns; frozen remote source remains preserved.

An additional frozen-model observable audit uses 64 independent batches of
512 four-replica groups per checkpoint (32,768 groups total), including raw
overlap, connected susceptibility, and magnetization. It uses fresh RNG
streams and no gradients. `observable_audit.json` records these estimates;
the report compares the seeds at this greater precision. It supplements the
final energy evaluation and does not retune any checkpoint.

## Completed outcome

All six 2,400-update runs and all six independent observable audits completed
on AutoDL. Both seeds improve energy relative to the original shared checkpoint
at every size. Final E/N values (seeds 201,202) are:

| Spins | Seed 201 | Seed 202 |
|---|---|---|
| 48 | -1.097286 ± 0.000124 | -1.097254 ± 0.000099 |
| 108 | -1.116020 ± 0.000127 | -1.116066 ± 0.000131 |
| 192 | -1.069442 ± 0.000114 | -1.069249 ± 0.000129 |

Errors are sampling SE only. Variance decreases by approximately 5×, 3.2×,
and 3.3–3.5× relative to the original shared model. The higher-statistics
audit resolves L8 seed differences in q² (0.48053 vs 0.47289) and connected
χ (0.22159 vs 0.23322), despite close energies. L4/L6 q² and connected χ
are consistent within about 1.3 combined SE, but magnetization differs
significantly at every size. These are improved variational guides, not
qualified exact-ground-state observable estimates.
