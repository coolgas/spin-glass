# Shared ALiBi correction: symmetry, conditioning, and disorder coverage

The previous 24-context guide failed two different tests: zero-field parity
and generalization to new disorder/equivalent lattice translations. The new
implementation targets both while keeping one parameter state across sizes.

## Implemented guide

`symmetric.py` wraps one ALiBi decoder in a normalized probability model.

* The lexicographically smallest bond vector over all cell translations defines
  a canonical origin. Spins are permuted into that frame for evaluation and
  back to the original frame after generation. All tied origins are averaged
  in probability, so clean and partially periodic samples retain exact symmetry.
  This construction enforces joint spin/bond translation covariance; it does
  not impose a translation symmetry on fixed generic random bonds.
* At exactly h=0, the probability is the equal mixture of base probabilities
  for s and -s. Sampling chooses a fair global flip; amplitudes, local energies,
  gradients, and GFMC ratios all use the square root of that same mixture.
* Canonicalization fixes translations only. Rotations and reflections are not
  claimed. It is discrete and can change discontinuously with bond inputs.
  Tied-origin averaging can cost L² base evaluations in a clean system; for
  continuous random bonds there is almost surely only one canonical origin.
* `physical_decoder.py` adds the bond-weighted field from earlier cells as
  a causal input, and a learnable positive local-energy logit prior. Intracell
  bonds and the longitudinal field contribute to this prior. Current/future
  spins cannot enter a cell's conditional, verified by a causality test.
  The main network remains ALiBi, width 64, four causal blocks, four heads.

## Training and qualification

`train_symmetric.py` performs sampled physical VMC without exact-state labels.
Each optimizer step sums energy-per-spin gradients from L=2,4,6,8. Field pairs
cycle through (Γ,h)=(0.5,0),(0.5,0.3),(2,0.3); disorder is randomly selected
from IDs 0,1 and 100–355 (258 samples per size). Each size/ID uses its own
deterministic quenched bond draw. All contexts share one parameter set.

Validation IDs 10000 and 10001 and final-test IDs 20000–20007 are excluded
from training. IDs 2–9 from the previous investigation are also excluded and
used for retrospective comparisons. IDs inspected during pilot development
are development data thereafter, even if they were initially called test IDs.
A later confirmatory evaluation needs another untouched set.

The first run is a 600-update pilot, batch 128 per size, started from scratch
because canonicalization changes the inputs and the old checkpoint overfits
its two training bond realizations. It tests optimization and held-out learning;
600 updates are not an accuracy qualification. A longer run is conditional on
the measured pilot behavior. GPU training is detached and checkpoints contain
optimizer and both sampling/context RNG states. Source and settings hashes
identify the run, and a filesystem lock prevents simultaneous writers.

Remote root:
`/root/autodl-tmp/transformer-af-ltfim-20261026-shared-symmetric-v2`.
Pilot output: `pilot/`, `pilot.log`, `pilot.pid`.

The completed pilot reduced mean energy error/spin on retrospective held-out
IDs 2–9 at L2, Γ=0.5,h=0.3 from approximately 0.266 to 0.059. Fidelity and
connected observables remain insufficient, and original training-context
accuracy regressed. See the executed `notebook/shared_alibi_symmetric_pilot.ipynb`.
The trained checkpoint passed the translation/parity audit across all four
sizes (full enumeration at L2, sampled configurations at larger sizes).

The main continuation runs 6,000 additional updates, batch 256 per size,
initialized from that pilot, under
`/root/autodl-tmp/transformer-af-ltfim-20261026-shared-symmetric-v2-main`.
It reserves new test IDs 30000–30007 and evaluates all three field pairs at
larger sizes too. `monitor_symmetric.py` retrieves artifacts, refreshes
`notebook/shared_alibi_symmetric_training.ipynb`, and audits the final checkpoint
when finished. Pilot and main frozen source trees are separate. An SSH failure
stops the monitor with its error while remote training remains detached.

## Verification completed before training

`python -m transformer_qmc.test_symmetric` checks normalized full-state
probabilities, exact parity at zero field, non-imposed parity at nonzero field,
joint translation covariance (including tied origins), size switching with
unchanged weights, the VMC gradient against the differentiated exact energy,
direct samples against enumerated moments, and GFMC ratios against a lookup
table. These tests passed locally and on AutoDL. The existing five core tests
also pass locally. These establish mathematical consistency, not learned accuracy.

No projector run is part of this correction campaign. The final same-weight
guide must be evaluated on all target sizes, both zero/nonzero longitudinal
field, and unseen disorder before interpreting it as a successful replacement.
