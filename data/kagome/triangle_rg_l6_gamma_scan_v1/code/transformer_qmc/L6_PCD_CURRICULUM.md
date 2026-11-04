# L6 persistent chains and field curriculum — 2026-11-03

The finite comparison launched on AutoDL at **2026-11-03T16:29:00Z**, PID396825,
under `/root/autodl-tmp/l6-pcd-curriculum-20261103-v1/`. The worker uses its own
source snapshot in `source/code`; launch arguments and the validation digest
are recorded in `launch.json`. All scientific computation and notebook execution
take place on AutoDL. The existing L6/L4 campaigns continue separately. L8 and
L10 were stopped at the user's request; their artifacts are retained.

## Hypotheses and research basis

The existing likelihood fits already kept negative chains alive during each
1,000-update fit. This intervention carries the cold bank, full tempering ladder,
transport labels and RNG across feedback stints as well. The guide and Adam
optimizer also continue. Fresh training populations equilibrate each stint;
physical tags and measurement averages never cross guide adaptation.

[Tieleman (2008)](https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf) introduced PCD
to approximate likelihood gradients with chains that persist as the model
changes. Its usefulness depends on sufficiently slow parameter changes and
adequate mixing. [Desjardins et al. (2010)](https://proceedings.mlr.press/v9/desjardins10a.html)
showed that ordinary Gibbs chains can miss RBM modes and studied tempering as
a remedy. Persistent chains can also remain trapped, so transport diagnostics
alone cannot establish equilibrium.

[Pilati, Inack and Pieri (2019)](https://arxiv.org/abs/1907.00907) proposed
iterative mixed-walker-density fitting. With normalized density `psi_theta^2`
fitted to `f_s ∝ psi_s psi_0`, an ideal fit gives
`psi_(s+1) ∝ sqrt(psi_s psi_0)`, halving the log-amplitude error at a fixed
Hamiltonian. Their ferromagnetic-chain examples used 10,000 walkers, 20–50
stints and 50,000–100,000 updates per stint. This kagome diagnostic uses a much
smaller budget. [Brodoloni and Pilati (2024)](https://arxiv.org/abs/2407.05978)
applied self-learned guides to frustrated Edwards–Anderson systems with explicit
population and ansatz checks. Their square lattice at zero longitudinal field
differs from this model.

A high transverse field is a heuristic easier starting point. Continuing the
guide while reducing that field may reduce abrupt target changes. Γ=4 has not
been proved paramagnetic here, and the fixed-Hamiltonian error-halving argument
does not directly apply while the field changes.

## Fixed design

Periodic kagome L6 has108 sites, `K_ij ~ U(.5,1.5)`, and
`H = sum K_ij Z_i Z_j - .15 sum Z_i - Gamma sum X_i`. Stored bonds use `J=-K`;
disorder RNG is `default_rng(SeedSequence([20261027,6,sample_id,1]))`.
Disorders60000/60001 and seeds0/1 are retained in every arm.

| Arm | Negative chains between stints | Field history |
|---|---|---|
| `direct_reset` | Reset each stint | Γ=.6 for24 stints |
| `direct_pcd` | Carry full ladder and RNG | Γ=.6 for24 stints |
| `curriculum_pcd` | Carry full ladder and RNG | Two stints each at4,3,2,1.5,1.2,1,.85,.75,.65; six at.6 |

Each disorder/seed shares one Γ=4 VMC400 initialization across all arms, with
Adam learning rate.001. Each arm then starts a fresh likelihood optimizer.
All arms use four independent1,024-walker training islands,16 negative
temperatures, equilibration8, collection12, delta.1,64 saved configurations per
block total,1,000 updates/stint, batch256, ten negative sweeps/update and
learning rate.001. Islands have equal target mass.

The negative marginal is
`log q_beta(s) = 2 beta a.s + 2 sum log[2cosh(beta(sW+b))]`.
Its cold endpoint equals `psi^2`; intermediate marginals differ from
`psi^(2 beta)`. Empirical backtracking caps centered `2 log psi` step RMS at.025
and maximum at.15, with eight halvings then parameter restoration. Adam moments
persist. These empirical bounds do not imply a global distributional bound.

There are12 contexts,288 stints and288,000 likelihood updates. One sequential
worker dispatches one stint per context/arm in round-robin order. It has no
deadline or retry loop. Failed contexts remain in `failures.json` and the report.
Direct reset versus direct PCD isolates persistence between stints. Curriculum
versus direct PCD matches likelihood updates, but changes propagation cost and
target-field training: six versus24 stints atΓ=.6.

## Measurements and interpretation

At stints21 and24, each guide receives independent frozen-start diagnostics
and two GFMC repeats: four512-walker populations, equilibration16, forward16,
eight cycles, atΓ=.6. There are48 planned new measurements. Raw tags and
descendant counts reconstruct overlap distributions, q2, connected spectra,
energy and genealogy. Model, raw-data and source hashes preserve provenance.
Persistence audits compare actual cold-bank and full-ladder arrays at stint
boundaries. Training audits check raw files and guide continuity.

Ancestry alarms remain descendant ESS<8 or largest lineage>.5. All alarmed
measurements and frozen-start alarms are retained. Reports compare repeats,
seeds, arms and training budgets descriptively. Two repeats do not reliably
bound systematic error. Population/time/start dependence and training
saturation remain unresolved; no result is automatically ground-state qualified.

## Validation and delivery

The frozen launch sources passed25 tests on AutoDL with CUDA, zero skips,
failures or errors. Checks include likelihood conventions, sampler invariance,
CPU/CUDA interrupted continuation, persistent chains through field changes,
initial-ladder checkpoint interruption, reset controls and raw GFMC
reconstruction. `validation/regressions.json` hashes every package Python source
and `qmc_numerics.py`; the worker verifies these hashes before dispatch.

Notebook execution was checked on empty and explicitly synthetic populated
display fixtures; each executed all three code cells without errors. Synthetic
fixtures live only under `validation/`. AutoDL needed pandas installed for
tables; the dependency log is retained. The real notebook embeds its dated
report, worker status and validation JSON and can be read without remote files.

After each disorder/seed dispatch group, the worker writes
`results/research_report.json`, executes `results/l6_pcd_curriculum.ipynb`,
and exports matching HTML. Terminal status is saved before the final notebook.
The initial notebook has no completed measurements; physical conclusions remain
pending. Live progress is in `results/status.json` and per-arm training status;
the worker log is `worker.log`.

The local delivery is `data/kagome/l6_pcd_curriculum_v1/`, with a dated snapshot,
validation, launch and L8/L10 stop records. Its SHA256 inventory verifies copied
artifacts. The readable notebook and HTML are also under `notebook/`.
