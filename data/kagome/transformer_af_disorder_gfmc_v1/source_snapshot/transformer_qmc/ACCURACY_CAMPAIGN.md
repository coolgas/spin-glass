# Extended ALiBi accuracy campaign

Remote root: `/root/autodl-tmp/transformer-af-ltfim-20261026-alibi-accuracy-v1`.
Local artifact directory: `data/kagome/transformer_af_alibi_accuracy_v1`.
Runner: `python -m transformer_qmc.push_accuracy`.

The campaign targets the same disorder-0 Hamiltonians at Γ=0.5,h=0.3 and
L=8,6,4,2, in that order, with independent branches seeded 301 and 302.
Each branch starts from its corresponding previously refined ALiBi guide.
No projector is run as part of this campaign.

Each branch:

1. Expands capacity from width 32/two causal blocks to width 64/four blocks,
   retaining four ALiBi heads and the static-context block.
2. Transfers teacher conditional probabilities by 1,000 distillation updates
   on teacher-sampled configurations. This only initializes the new model;
   teacher and student energies are checked independently afterward. There
   are no exact-state training labels.
3. Runs 8,000 physical sampled-VMC updates, batch 512, cosine learning-rate
   decay from 0.0003 to 0.000015.
4. Runs 4,000 further VMC updates, batch 1,024, cosine decay from 0.00005
   to 0.0000025, initialized from the first stage's validation-selected model.
5. Saves candidates every 1,000 updates; latest optimizer/RNG checkpoints
   every 200 updates. Candidates and latest files have distinct names.
6. Selects candidates using independent interim energy evaluations. The
   original parent remains eligible, preventing an automatic replacement by
   a worse expanded model. Final evaluation independently assesses all three
   candidates (parent, main-stage winner, low-rate-stage winner), each with
   16,384 energy samples and independent four-replica measurements.
7. At L2, full-state validation records energy/overlap/connected susceptibility
   errors and fidelity against the exact solution.

Selection uses noisy interim energies and therefore still needs interpretation.
Final samples are not reused to optimize or select candidates. All seed outcomes
and rejected-stage outcomes are retained. A selected candidate is not certified
as a converged ground state. Compare changes across the last training windows,
capacity, stages, and independent seeds before calling a plateau useful.

Each candidate filename includes L, N, width, depth, fields, disorder ID, seed,
stage, and update count. `evaluation.json` records actual checkpoint paths,
SHA-256 hashes, and separate final sampling results. `campaign.json` freezes
parent/source hashes and settings. Source is staged before launch and not
overwritten while the process runs. A campaign lock prevents concurrent runs
against the same output. Resuming restores stage optimizer and sampling RNG.

The previous guides are immutable inputs. The expanded checkpoint is still
Hamiltonian-specific, even though its architecture accepts other sizes.

Monitor `train.log`, `results/status.json`, per-run `vmc_history.json` /
`polish_history.json`, and `results/summary.json`. There is no shutdown action.
The first measured H800 throughput for width64/depth4 at N192 was approximately
0.9 seconds per update at batch1024; these short measurements include startup
effects and are not a full-campaign time guarantee. Plan for several hours.

Completion is only the end of this scheduled campaign, not a proof of maximal
possible accuracy. Further training/capacity or an optimizer change should be
driven by unresolved validation improvements and observable sensitivity.

`monitor_accuracy.py --watch` runs locally with credentials only in SSHPASS.
It checks the remote PID and status, retrieves artifacts every ten minutes,
and refreshes the executed `notebook/alibi_accuracy_push.ipynb`. On completion,
it retrieves final artifacts and runs `qualify_accuracy.py`, recording energy
improvements and seed sensitivities in `qualification.json`. A training crash
or SSH error is reported rather than retried indefinitely. The monitor does
not launch the projector or change any checkpoint. Its completion wakes the
agent session for review of the actual results and further warranted work.
