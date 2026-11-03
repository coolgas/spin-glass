# L6 convergence first — 2026-11-03

The combined L4/L6 runner is paused at saved checkpoints. At the user's direction,
L4 production continues in a separate L4-only runner; broad L6 production is paused.
Compute priority is L6 reliability in Gamma=.55–.70 at h=.15. Existing artifacts,
including L2 exact comparisons, remain preserved. No larger-size production or
automatic broad-campaign restart follows from completion of this study.

## Stage 1: finite training-budget screen

Use periodic kagome L6, N108, Hamiltonian sum Kij ZiZj - .15 sum Zi - Gamma sum Xi,
iid AF K~U(.5,1.5), stored J=-K. Bond RNG is
default_rng(SeedSequence([20261027,6,sample_id,1])). Prespecify fields
.55,.60,.65,.70; disorders 60000,60001; RBM alpha2 and seeds0,1. These two
disorders are diagnostic cases, not the planned eight-disorder ensemble.

Compare 400,1600,4000 target-field updates. Baselines use 400 updates each at
1.50,.75,target, Adam .001, 256 chains, eight temperatures and two sweeps/update.
Continuation preserves model, Adam, chains and RNG. Every checkpoint receives
frozen random/up/down checks. Both seeds remain visible; no energy-based seed
selection enters these comparisons. Sampling alarms exclude that checkpoint
from GFMC and remain explicit; longer training can still proceed.

The existing .55/disorder60000 extension owns two contexts. The new runner owns
the other fourteen, comprising 42 training milestones and at most 84 GFMC runs.
At each eligible milestone, two repeats use four independent populations of
512 walkers, equilibration16, forward16 and eight cycles. Task/RNG identities
are matched across budgets, with separate output directories and model hashes.
Only worker-sized batches of projector tasks are submitted. Milestone results feed GFMC
while subsequent training runs; the whole training grid need not finish first.
All raw overlaps, counts, weights, energies and model provenance are retained.

Allocation after utilization review: new study two GPU trainers/eight CPU projectors; existing extension
two GPU trainers then two CPU projectors; existing field controls two CPU
projectors. L4 additionally uses one GPU trainer and two CPU projectors.
At most five GPU trainers and fourteen CPU projectors across these finite jobs.
The extension's two trainers and two projectors run in separate phases, so the
combined active compute-worker maximum is seventeen, within the 20-core quota.
No deadline. The added L4 workload may affect throughput; the L6 ETA remains provisional.

## Completion budget

Stage 1 is finite and must be assessed before adding Stage 2 jobs. Budget roughly
30–48 hours from the November 3 launch was the original two-projector estimate.
The new eight-projector allocation reduces the idealized projection component
to about seven hours; training readiness and shared-machine throughput now need
measurement before giving a revised full-study completion promise. At the
measured approximately 40 minutes per standard L6 RBM GFMC run, 84 runs require
56 worker-hours, or seven wall-hours on eight workers. Training overlaps projection;
allowance covers pipeline gaps and field/throughput variation. This is a planning
range, not a confidence interval or convergence promise. Higher-field timing is
not yet measured. Earlier .55 extension and field controls should report sooner.
No automatic expansion, retries or broad production restart follows completion.
The authorized finite plan runs to completion without an artificial cutoff.

## Screening and qualification

Stage 1 cannot certify ground-state convergence. Before judging the new anchor
results, use these provisional screening tolerances: last-two-budget and
between-seed differences in energy/site <=.001, q2 <=.01, histogram total
variation <=.05 and each connected-spectrum component <=10% relative to the
larger magnitude (absolute floor .001). These are engineering screening targets,
not yet precision requirements for a finite-size critical fit. Uncertainty must
be narrow enough to resolve each tolerance; a noisy agreement is inconclusive.
Two repeats give limited precision. Use independent-repeat units and blocked
cycle analysis; walkers and descendant copies are not independent samples.
Cross-budget RNG matching also means budgets are not independent replicates.

Retain the existing ancestry alarm (descendant ESS<8 or largest lineage>.5),
report the full distributions, and require resolution of alarms before promotion.
Ancestry improvement or energy agreement alone cannot pass qualification.

Stage 2 is designed from the screen: for each anchor and both fixed disorders,
test population512/1024/2048 as needed, independent equilibration16/32 and
forward8/16/32 variations, guide/uniform starts, and learning-rate sensitivity
(.001 versus .0005) with matched training budgets. Retain Jastrow comparisons
as an independent guide-family check. Do not launch the full Cartesian product
before identifying which discrepancies need resolution. Record this next finite
plan and any revised tolerances before inspecting its new outcomes.

Qualification must address stable P(q), energy, connected spectra, repeat/seed
spread and ancestry jointly at every anchor. Intermediate fields .56–.59,
.61–.64,.66–.69 remain unqualified by interpolation: after anchors pass, check
their frozen fits and prespecified GFMC sensitivities before broad production.
Retain .85/1.00 references and L4 work for the later resumed plan. Report any
disorder summaries with equal prespecified weights and capped uncertainty bars;
never replace a difficult disorder. No fixed ETA for scientific convergence.
