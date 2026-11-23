# Adaptive-D demonstration, 2026-11-23

## Iteration 2 (completed)

V2 exited normally (code 0) at its soft time limit in 549.46 seconds;
supervisor completed in 555.09 seconds. Peak worker RSS was 1.272 GiB.
Both starts triggered persistent-truncation growth at sweep 20, D16 -> D32.
Start 0 completed 32 sweeps and start 1 completed 31 sweeps; the soft deadline
fell between their turns. Neither reached final confirmation; result correctly
reports state=time_limit, solver_pass=false and production_qualified=false.
This is a time-limited demonstration, not a solver-convergence failure.

Post-run trace verification confirmed contiguous sweep counters, D16 through
sweep 20, D32 thereafter, and mixer activity only at sweeps 21..25 following
growth. Both D16 checkpoint hashes verified and restored magnetizations
matched exactly (maximum difference 0). The transition was actually executed,
followed by 12 and 11 additional sweeps for starts 0 and 1 respectively.
Energy decreased by 1.69036e-4 and 1.68714e-4 per spin. Discarded weights fell
from about 2.74e-4 to 9.74e-5 and 9.33e-5, respectively. No code error occurred.
The bounded watcher completed; no pilot jobs remain scheduled by this session.

Scope of verification: real L6 first transition and checkpoint restoration,
plus small exact-reference integration tests for the later decision schedule
and complete 120-sweep confirmation. Actual D128+ transitions, production
memory admission, full restart/resume and physical bond-dimension convergence
are not established by this short demo. No production launch is authorized.

The user authorized iteration after the first short run failed. Original v1
output and source are preserved. V1's reduced probe reached the first trigger
at sweep 20, then failed on redundant mixer deactivation; its target probe
was terminated by the 900-second limit without recording a completed sweep.

V2 remote worktree: `worktrees/adaptive-demo-20261123-v2`, branch
`pilot/adaptive-demo-20261123-v2`, same baseline commit as v1. Six new scripts
were SHA-256 verified on transfer. Production package files remain unchanged.
Fixes include guarded mixer shutdown plus environment-consistent cleanup,
in-place dimension growth, actual sweep-count assertions, magnetization
stability at both disagreement decision points, nonfinite-value rejection,
phase-specific progress, diagnostic flushing every five sweeps, checkpoint
restore/moment checks, and terminal supervisor sidecars.

Ten tests passed on Ali: nine policy tests and a real TeNPy integration test
on a six-site transverse-field Ising chain. The latter verifies state and
sweep-count preservation across growth, actual bond growth from 2 to 8,
canonical error below 1e-8 and energy agreement with an independently built
64x64 exact Hamiltonian to nine decimal places.

An additional real-TeNPy test was subsequently executed over stdin on Ali
without modifying the active worktree. It forces increases at sweeps 20,
40 and 60, verifies exactly 120 total sweeps and successful final-window
acceptance within that budget. Both integration tests passed in 3.8 seconds;
the validated suite now comprises eleven tests. The updated integration-test
source is retained in the local checkout.

V2 output: `outputs/adaptive-demo-20261123-v2`. Supervisor PID 65766 owns a
single reduced-D probe, with a hard limit of 600 seconds and soft deadline
540 seconds. No target-D probe was relaunched. Session-owned watcher
`sh_205f0e502001WNsrYuCmb2JoFD` verifies worker identity/threads, checks every
60 seconds and expires after 660 seconds. Await its completion notification;
do not duplicate or poll it. Next: verify actual L6 transitions, checkpoint
receipts and whether exit was a timeout or a code failure.

## Iteration 1 (historical)

Authorized scope: a short pilot only; production remains paused.

Host: `ali`. Project root:
`/mnt/ali/yty/spin-glasses/rydberg-kagome-l4-pq`.
New worktree: `worktrees/adaptive-demo-20261123`, baseline commit
`aeb225549a75c48942a4f162343578c9bbed5818`, branch
`pilot/adaptive-demo-20261123`. Four added scripts were transferred with
SHA-256 verification. Six policy tests passed locally and on Ali.

Output: `outputs/adaptive-demo-20261123-v1`.
Supervisor PID 64361; reduced probe PID 64363, target probe PID 64364;
both worker birth markers were `23671826`. Supervisor owns its two workers
and terminates them at a total 900-second wall-clock deadline. Each worker
has one BLAS/OpenMP thread and a 48-GiB address-space ceiling. Preflight
found 121.5 GiB available memory, negligible load and 932.5 GiB disk free.
Both workers were verified alive with one thread after launch.

Session-owned completion watcher: shell `sh_205de47ce001ZarfAPwJhmO6oH`.
It checks the supervisor every 60 seconds, expires after 1,020 seconds,
and returns through the tool notification. Do not duplicate or poll it.

Both probes use the hash-verified cached Hamiltonian/model from
`outputs/rydberg-L6-pilot-20261120-v1/cases/L6_s000_full_d3.15_g17_c12_D128`.
They initialize two fresh MPS starts, retain full interactions and chi=12,
and use the actual nine-block L6 system. The reduced ladder 16/32/64/128
tests transitions cheaply; the target ladder is 128/256/384/512. Both use
decision sweeps 20/40/60 and at most 120 total sweeps per start. Reduced
results cannot establish production accuracy. Neither performs replica
sampling or qualifies RG convergence.

The worker aims to finish diagnostics by 840 seconds, leaving a margin
before the supervisor's hard stop. Long individual sweeps may be interrupted;
trace.jsonl retains completed sweeps, and stage checkpoints occur at decision
points. An early timeout is diagnostic evidence, not a solver failure.
All old pilot files remain untouched; cancelled retries remain cancelled.

Next action on completion: inspect compact final outcomes and transition
records; distinguish wall-clock timeout, actual solver failure, and converged
results. Report measured sweep costs and limits on what the short run tested.
Do not automatically extend or restart expensive probes.
