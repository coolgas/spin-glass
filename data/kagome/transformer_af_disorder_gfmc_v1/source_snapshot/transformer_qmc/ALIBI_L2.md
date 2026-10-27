# ALiBi refinement on the 12-spin Hamiltonian

Remote campaign: `/root/autodl-tmp/transformer-af-ltfim-20261026-alibi-l2-v1/`.
Local artifacts: `data/kagome/transformer_af_alibi_l2_v1/`.

Hamiltonian: periodic L=2 Kagome, disorder ID 0 from `make_af_sample(2,0)`,
K∈[0.5,1.5], Γ=0.5, h=0.3. Geometry and couplings are saved in `disorder.npz`.
The previous shared model is an immutable input from the dual-v1 campaign.

`refine_l2.py` runs the same width-32, two-causal-block ALiBi architecture with
shared and fresh initialization, two seeds each (101,102), 4,000 updates/run,
batch 512, Adam learning rate 0.001→0.0001. Training uses sampled VMC energy
gradients. Enumeration is used only for checkpoint validation and selection.

Predeclared targets:

- Energy error <1e-4 per spin.
- Relative raw q² error <1%.
- Relative connected susceptibility error <5%.

Each run saves latest, final, and best checkpoints. "Best" means lowest
enumerated variational energy among reporting checkpoints, not lowest noisy
sampled energy or best agreement with a selected observable. Both the best
step and all observable errors are recorded. `selected.json` identifies the
lowest-energy checkpoint across the four runs, without overwriting any run.
`summary.json` retains all four outcomes. This is validation on the trained
Hamiltonian, not independent evidence of generalization.

Example best filename:

```text
alibi__L02__N012__g0p5__h0p3__dis000__init-shared__seed101__best.pt
```

Load the selected guide:

```python
import json
from pathlib import Path
from transformer_qmc.physics import make_af_sample
from transformer_qmc.unified import UnifiedTransformer
from transformer_qmc.experiment import load_checkpoint
from transformer_qmc.model import TransformerGuide

root = Path('data/kagome/transformer_af_alibi_l2_v1')
selected = json.loads((root / 'selected.json').read_text())
sample = make_af_sample(2, 0)
ham = sample.hamiltonian(.3)
state = load_checkpoint(root / selected['best_checkpoint'], 'cpu')
model = UnifiedTransformer(ham, .5, **state['metadata']['architecture'])
model.load_state_dict(state['model'])
guide = TransformerGuide(model)
```

The model retains the ALiBi architecture's ability to accept other sizes, but
this checkpoint is specialized to the stated Hamiltonian. Do not assume that
fine-tuning preserves shared-model accuracy elsewhere.

Generate the compact executed report after downloading results:

```bash
PYTHONPATH=code python3 -B -m transformer_qmc.report_l2
```

The output is `notebook/alibi_12site_accuracy.ipynb`, with standalone PNG/PDF
accuracy traces in the local results directory.

## Completed results

All four runs completed on AutoDL and passed the predeclared targets. The
selected shared-initialized seed-101 checkpoint at step 4,000 has energy error
9.7263e-6/spin, raw q² relative error 0.07066%, connected susceptibility relative
error 0.34318%, and fidelity 0.99997676. The other shared seed reaches
9.7486e-6/spin; fresh starts reach 6.5440e-5 and 4.9170e-5/spin. These are
enumerated variational energies. Fresh starts can be better on individual
observables despite larger energy errors. The selected checkpoint was reloaded
locally and independently evaluated in double precision; normalization differs
from one by less than 1e-15 and energy agrees within 1e-8 total.
