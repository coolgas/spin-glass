# Memory work handoff — 2026-11-23

## Latest integration (v6)

The user selected production-case integration, complete resume and bounded
production-path validation, and explicitly requested a code push. Shared-controller
reservation coordination is deferred. Production and the cancelled retries remain
stopped.

`campaign.run_case(..., adaptive=True, validate=False)` and CLI
`case --adaptive --no-exact` now use the compact MPO and paired adaptive starts,
followed by final moments, replica sampling and separate numerical/scientific flags.
The 120-sweep budget is cumulative per start across growth and process restarts.
Immutable checkpoint generations contain both engines, histories, total counts,
dimension, mixer, settings and pending growth. Atomic `resume.json` selects a
complete generation. Source/config/environment and payload hashes must match;
a kernel case lock prevents simultaneous writers. Operational invocation budgets
can change on resume. Completed results are reused only after hash checks.

All 38 tests passed on the v6 worktree: the full 35-test suite (23 script tests,
12 inherited scientific tests), followed by all three case tests after adding
the final-sampling interruption regression. That new test verifies no repeated
optimization, preserved final checkpoints and rejection of a corrupt completed
artifact. New coverage compares paused/resumed and uninterrupted small-case final
observables and sampling, exercises active-mixer/growth restoration, interrupted
generation writes, corrupt payloads, identity mismatch and memory deferral.
The three restart tests also passed (~0.6 seconds) after adding an injected
second-payload write failure: the existing pointer stays byte-identical and
restores the previous complete pair despite an orphan new first payload.
The full-case supervisor test passes success, failed sampling and worker-failure
fixtures. Run its fixtures with TMPDIR on the Ali data disk: its real 64-GiB
free-space preflight deliberately rejects the small root filesystem.

Tested source: Ali `worktrees/adaptive-integration-20261123-v6`.
Bounded validation: `outputs/adaptive-integration-validation-20261123-v6`.
Controller PID 75737 completed both sequential processes normally in 621.295 s;
counts advanced [1,0] -> [1,1] at D128 across process restart. Sweep times were
309.773 and 307.126 s. Each process used one thread, 8-GiB RLIMIT_AS and a
420-second hard timeout. All four generation manifests and eight payload hashes
verified, current pointer matches its immutable manifest, and package hashes match
the local source. There is correctly no completed result. Output: 25 files,
47,949,403 bytes. Completion watcher `sh_206341dff001OUAD8LeJIFkdpL` has finished.

## Authorized full-case convergence test

### Threading switch completed (supersedes one-thread execution below)

All four matched checkpoint probes exited 0. Times for the same second sweep
of start 0: 1 thread 155.910 s; 2 threads 105.822 s; 4 threads 75.019 s;
8 threads 63.341 s. Eight threads were 2.46x faster than one in this shared-load
test. Total energy spread was <7e-13; peak RSS ranged 5.53–5.56 GiB. Later-sweep
single-thread time differs from the ~310-s first sweeps; compare matched sweeps.

The original case resumed with eight threads, controller PID 79487, worker
PID 79488. Live worker verified at eight threads and ~603% CPU, with all four
numerical thread environment variables set to 8. Counts [1,1] and D128 were
preserved. The old -15 exit is an intentional checkpoint-boundary switch.
Original deadline is retained. Owned completion watcher:
`sh_2064a687d001y5aNeQfeFoOxK7` (120-s interval, maximum 73 h). Prior watchers
and the switch controller have finished. Do not duplicate the active watcher.

Production should compare aggregate throughput as well as case latency: eight
threads minimize measured latency here, but have lower per-core efficiency than
two/four threads. Omega concurrency, external workload and memory determine the
production choice; no universal eight-thread recommendation is established.

Historical switch procedure:

The user requires multithreaded cases and omega-parallel production with multiple
CPU cores per omega. The current case is switching through an owned boundary
controller, shell `sh_206474b280019KvNaUPOxJMeHG`, executing
`scripts/switch_adaptive_threads.py`. It waits up to 600 s for the next committed
checkpoint before signalling only the verified owned worker, then waits for the
old supervisor to finish. The old completion watcher may consequently report
`failed`/SIGTERM: this is an intentional resource switch, not a scientific failure.

Four independent same-checkpoint sweep probes use 1/2/4/8 threads concurrently
(15 total), each capped at 16 GiB, for at most 900 s. The script checks 80 GiB
live headroom before dispatch. It verifies consistent energies, chooses the
fastest successful multithreaded probe and resumes the original case with the
same scientific source and counters. Probe outputs are under
`outputs/adaptive-full-case-20261123-v6/thread-benchmark` and never write the
case restart. Initial worker logs/status/resources are timestamp-archived on
resume. The 72-hour deadline is retained relative to the original launch.
New status and an owned watcher will be verified after switch completion.
Boundary verified at [1,1], generation `2e0deae9402f4349ad48bffd8201bf3b`.
Probe PIDs 78875/78876/78877/78878 used 1/2/4/8 live threads respectively,
with measured CPU use ~100/186/342/638 percent at 46 seconds. Switch controller
PID 78852. Multiple cores are demonstrably active. Parallel probes share memory
bandwidth, so timings are conditional on this shared load and should not be
presented as isolated per-thread scaling guarantees.

Production design is now explicitly omega-parallel plus multithreaded per omega;
see `PROTOCOL_DECISIONS.md`. The current case still alternates its two starts.
Shared reservations and a validated production scheduler remain outstanding.

Live worker PID 77262 was verified at one thread, ~100% of one CPU, ~2.7 GiB
RSS while its first D128 sweep was running. Completion watcher
`sh_2063f2934001bWACbOVhrkwWAQ` is owned by this session, samples every 120 s,
and ends on terminal status/controller exit or after 73 hours. Do not duplicate.

The user now requires at least one full-case convergence before production and
authorized this test. One separate case was launched, supervisor PID 77260,
source `worktrees/adaptive-integration-20261123-v6`, output
`outputs/adaptive-full-case-20261123-v6`. It starts fresh, preserving the bounded
resume demonstration separately. L6, sample 0, full interaction, delta=3.15,
omega=0.9037538790536807, chi=12, D ladder 128/256/384/512, two starts,
120 cumulative sweeps/start, 20,000 replica pairs. No accuracy threshold changes.

One compute thread; RLIMIT_AS=64 GiB, live 16-GiB headroom reserve and peak-aware
growth admission. Memory cap is not predicted usage or exclusive allocation.
Preflight observed ~117 GiB available and another user's active compute job.
Data volume had ~932 GiB free and >65 million free inodes. Supervisor has a
72-hour hard deadline; this is an operational bound, not an expected runtime.
Both solver convergence and sampling consistency must pass for the full-case
gate. Memory deferral, timeout or reaching the sweep limit without convergence
leaves the gate unmet. Do not relabel completion as convergence.

All checkpoint generations are retained; source and old pilots remain unchanged.
At this stage, results are pending. Production remains paused regardless of
the test outcome until the remaining requirements and launch authorization are
resolved. Do not duplicate this case, restart cancelled retries or change package
source in its active worktree. The code push can proceed with this pending gate
documented; it does not authorize production.

Checkpoint retention is deliberately preservation-first for these bounded tests:
all generations remain on Ali. No automatic pruning is implemented. At the maximum
budget a case can write 240 sweep generations, 120 paired-decision generations,
up to three growth generations and the initial generation (364 generations,
1,092 restart files, before case products). Do not scale this cadence unchanged
to 7,680 cases. Measure bytes and plan retention/archive cadence before production;
preserve latest verified resume points and required milestones. No bulk data is
being transferred to Dropbox.

## Historical review (v5)

Read `PREPRODUCTION_REVIEW.md` for the latest state. Final tested source is
`worktrees/adaptive-review-20261123-v5` on Ali. All 31 tests passed, including
complete nine-block L6 chi=2 operator comparison, mismatched-matrix rejection,
compact-site checkpoint restoration and supervisor exit paths. The final
chi=12 model passed pairwise operator checks and its random-state energies are
unchanged. No new DMRG campaign ran during v5 review. Production integration,
complete resume and coordinated shared-host reservations remain explicit gates.
The v3/v4 entries below are historical; their measured runs remain preserved.

## Authorization and scope

The user authorized short adaptive-D pilot iterations and requested lower memory
use because Ali is shared with other jobs. Production remains paused. Do not
launch the 7,680-case L6 full-model ensemble. The three old interrupted pilot
retries were explicitly cancelled; do not restart them.

Local working checkout:
`/Users/tianyiyan/ResearchLocal/spin-glasses/repository/rydberg-l6-production-20261122`

Local branch: `pilot/adaptive-d-20261123`. Baseline:
`aeb225549a75c48942a4f162343578c9bbed5818`.
Feature changes are being prepared for the authorized push to the existing Ali
origin; the final verification entry records the outcome. Preserve unrelated work.
Read `AGENTS.md`, `ADAPTIVE_PILOT.md` and this file before continuing.

## Verified adaptive-D work

The corrected reduced-D pilot on Ali exited normally in 549.46 seconds, peak
RSS 1.272 GiB. Both independent starts advanced from D16 to D32 at sweep 20
and continued for 12 and 11 further sweeps. Mixer activity after growth was
exactly sweeps 21–25. Sweep counts did not reset. Both saved checkpoint hashes
verified; restored magnetizations matched exactly. This was a time-limited
control-logic demonstration, not a production-convergence result.

Eleven adaptive tests passed, including exact-reference TeNPy integration and
a test of the complete 120-sweep schedule. See `ADAPTIVE_PILOT.md` for details.
Those test results predate the compact-MPO changes below.

## New memory changes: bounded DMRG pilot completed

Latest outcome: v3 supervisor finished normally in 320.05 seconds. Both
workers exited 0 with `time_limit` outcomes, before the six-minute hard limit.
No watcher or worker from this pilot remains scheduled.

- Reduced probe: D16 -> D32 at sweep 20 and D32 -> D64 at sweep 40;
  completed 42/41 sweeps, peak RSS 1.522 GiB. All four stage checkpoint hashes
  verified; restored magnetizations matched exactly. The earlier legacy probe
  reached only D32, so its 1.272-GiB peak is not a like-for-like memory baseline.
- Target probe: one real D128 sweep completed in 312.49 seconds, whole-worker
  peak RSS 5.506 GiB. The second start had no completed sweep before timeout.
  This is not a full-case peak or proof of D128+ convergence. Prior legacy
  probe recorded zero sweeps under its 15-minute limit; prior complete legacy
  full-model cases peaked near 17.6 GiB. Do not claim a precise percentage
  reduction from these unmatched workloads.
- Matched D32 mixer-free sweeps 26..31 (12 start/sweep observations each)
  averaged 12.076 seconds with legacy MPO versus 3.659 seconds with compact
  MPO (about 3.3x faster in these measurements, not a production guarantee).

Post-profile hardening: growth forecasts now use max(current RSS, process peak
RSS), because idle between-sweep RSS substantially underestimates tensor
allocation peaks. Two new admission regression tests cover this and shared
headroom accounting. This change was made after v3; v3's recorded admission
figures used instantaneous RSS. Final source is deployed with verified hashes
to `worktrees/adaptive-memory-20261123-v4`, branch
`pilot/adaptive-memory-20261123-v4`; the measured v3 source remains preserved.
Both new admission tests passed on v4, bringing validation to 27 passing tests
across this iteration. No v4 DMRG pilot was launched; the peak-aware admission
change has regression coverage but no new full-case memory measurement.

- `code/rydberg_kagome/dmrg.py`: `build_mpo(model, interactions=None)` preserves
  the original builder by default. Supplying the original masked physical
  interaction matrix selects shared outgoing projected-Z channels. It replaces
  pair-specific channels with reused channels and is intended to preserve all
  interactions without truncation. Algebraic and numerical equivalence must
  pass before making claims of accuracy or memory savings.
- `scripts/adaptive_memory.py`: live host/cgroup headroom checks and a
  conservative growth forecast proportional to max(current RSS, peak RSS)
  times `(next_D/current_D)^2`
  with a 50% margin. The forecast is a policy heuristic, not a measured bound.
- `scripts/demo_adaptive_d.py`: opts into the compact MPO using hash-verified
  `geometry.npz`; records the builder and source hashes; defaults to an 8-GiB
  address-space ceiling; reserves 16 GiB of live available memory and defers
  growth when the forecast exceeds the budget. These are pilot defaults, not
  an allocation grant or validated production resource settings.
- `scripts/test_compact_mpo.py`: new tests compare the full small-system dense
  Hamiltonian, independent operator action, moments and admission boundaries.
  Both tests passed on Ali. The complete small Hamiltonian matches both the
  original MPO and independent operator action. All eleven adaptive tests
  also passed on the new worktree.

The memory patch was transferred with SHA-256 verification to
`worktrees/adaptive-memory-20261123-v3`, branch
`pilot/adaptive-memory-20261123-v3`. Existing v1/v2 source and output remain
immutable. Two bounded representation profiles (legacy and compact) completed
in separate one-thread processes, with 8-GiB address-space limits
and 180-second subprocess timeouts. They constructed the actual L6 MPO and
compared expectations on two identically initialized random MPS states.
Output: `outputs/adaptive-memory-profile-20261123-v3`.
Profile shell `sh_2061366e5001zXom8Y1PaFfnIn` has completed. Maximum MPO bond
dimension fell 242 -> 98. Setup times were 1.575 -> 1.071 seconds. Setup/random
expectation peak RSS was 0.1704 -> 0.1543 GiB; this is not a DMRG peak-memory
comparison. Random-state energy differences were 8.53e-14 and 7.11e-14.
The twelve inherited scientific tests also passed: 25 tests total including
the thirteen new policy/runtime/equivalence/admission tests.

A six-minute DMRG pilot completed at
`outputs/adaptive-memory-demo-20261123-v3`, supervisor PID 69012. It owns two
one-thread probes (reduced and target ladders), each with an 8-GiB address-space
ceiling. Prelaunch headroom was 121.5 GiB; the admission check required at least
32 GiB available for these two caps plus 16 GiB host headroom. The worker also
checks memory before sweeps and dimension increases. Owner watcher:
`sh_2061444090013mlkGwk18E8umD`, now completed. Do not poll or duplicate.
These process limits and live checks
are not cross-controller reservations or guaranteed RSS bounds.

## Next authorized work

1. Completed: equivalence, adaptive and inherited tests (25 total) on Ali.
2. Completed: bounded L6 representation profile and random-state comparisons.
3. Completed: bounded DMRG pilot and four checkpoint verifications. See above.
4. Preserve
   checkpoints, source hashes, failures and logs. Do not automatically extend
   the previously bounded probes or revive cancelled retries.
5. Update this handoff with actual test results and measured memory changes.

Known implementation limits to resolve/review: the pilot's memory check is not
a cross-controller reservation lock; concurrent controllers can race. A check
before a sweep cannot predict sudden external allocations. An address-space
limit is not an RSS limit. A memory-deferred pilot is terminal and does not yet
implement production resume/admission. Do not advertise a complete production
scheduler. Do not relax physical tolerances to fit memory.

## Remote paths and ownership

Host: `ali`; project root:
`/mnt/ali/yty/spin-glasses/rydberg-kagome-l4-pq`.

- Original pilot: `outputs/rydberg-L6-pilot-20261120-v1` (21 immutable results).
- Failed first demo: `outputs/adaptive-demo-20261123-v1`.
- Successful bounded demo: `outputs/adaptive-demo-20261123-v2`.
- Its source: `worktrees/adaptive-demo-20261123-v2`.
- V2 supervisor PID 65766 and worker PID 65768 have finished. The completion
  watcher `sh_205f0e502001WNsrYuCmb2JoFD` has finished. These are historical IDs;
  never signal them without fresh identity/ownership verification.

Load `ali-hpc` for remote work and `research-storage-safety` for new checkpoints
or transfers. Inspect current shared-host resources before launching. Keep
new outputs outside Git and synced roots. Do not modify shared environments,
other agents' jobs, or the frozen L4 source. The latest available-memory
snapshot was about 121 GiB; it is not an exclusive allocation.
