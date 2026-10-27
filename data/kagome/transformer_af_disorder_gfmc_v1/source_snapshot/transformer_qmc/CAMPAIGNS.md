# AutoDL fixed-size and unified transformer campaigns

Remote root: `/root/autodl-tmp/transformer-af-ltfim-20261026-dual-v1`.
The frozen source is under `code/`; training artifacts are under `results/`.

The detached `transformer_qmc.run_campaigns` supervisor runs these sequentially:

| Output directory | Architecture | Training |
|---|---|---|
| `results/fixed_size/<task>/` | Per-instance transformer with learned positions | Seven tasks, 800 updates each, batch 128 |
| `results/unified_alibi/` | One shared ALiBi transformer | 24 Hamiltonian contexts, 4,800 updates total, batch 128 |

Fixed tasks cover L=2,4,6,8 at Γ=0.5,h=0.3, plus L6 at Γ=2,h=0.3,
L6 at Γ=0.5,h=0, and a second L6 low-field training seed. All use disorder ID 0.
Unified contexts cover L=2,4,6,8, disorder IDs 0 and 1, and field pairs
(0.5,0.3), (2,0.3), (0.5,0). Unified seed is 42.
The fixed and unified runs have different optimization budgets and objectives;
this is an initial baseline comparison, not a controlled ALiBi ablation.

Checkpoint examples:

```text
fixed__L06__N108__g0p5__h0p3__dis000__seed021__latest.pt
fixed__L06__N108__g0p5__h0p3__dis000__seed021__step000800__final.pt
unified_alibi__L02-04-06-08__dis000-001__fields3__seed042__latest.pt
unified_alibi__L02-04-06-08__dis000-001__fields3__seed042__step004800__final.pt
```

`checkpoint.pt` is a relative compatibility symlink to the named latest file.
`checkpoint_index.json` lists snapshots, current step, and metadata. Each saved
state includes model, optimizer, sampler RNG, history, step, and model metadata;
the unified state also saves the context order and its RNG. Fixed latest files
update at the first step and every reporting interval; numbered snapshots are
saved at step 1, every 200 steps, and completion. Unified latest files update at
step 1 and every 120 steps; numbered intermediate snapshots occur every 600 steps
(the intersection of report and snapshot intervals), plus step 1 and completion.

`campaign.json` records configuration and source hashes. Fixed tasks also have
`settings.json` and `disorder.npz`. Unified bonds are stored in
`disorder_L<L>_s<ID>.npz`. Each run writes history and fresh-sample evaluation JSON.

Monitor with the user's helpers:

```bash
autodl-log /root/autodl-tmp/transformer-af-ltfim-20261026-dual-v1/results/fixed_size.log 20
autodl-log /root/autodl-tmp/transformer-af-ltfim-20261026-dual-v1/results/unified_alibi.log 20
```

Supervisor status: `results/status.json`; errors: `supervisor.log` and the relevant
campaign log. A failure stops the sequence and preserves checkpoints. Rerunning
the supervisor against the same frozen source/output resumes incomplete work;
its process lock prevents concurrent supervisors for that output. The instance
stays powered on after completion.

Download `results/` to `data/kagome/transformer_af_dual_v1/` to use the notebook's
default local path. The initial training runners evaluate VMC; projector
qualification is a separate subsequent campaign.
