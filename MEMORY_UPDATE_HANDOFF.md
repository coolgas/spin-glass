# Memory work handoff — 2026-11-23

## Active CPU increase — 2026-11-23 (supersedes launch thread/controller settings)

The user requested more CPU resources per running case after the resource report.
Target: **six numerical-library threads per case, four concurrent cases** (24
threads total), leaving room on the shared 32-CPU host. This supersedes the prior
four-thread choice. Memory/storage admission, physical inputs, seeds, scientific
source and task manifest are preserved. Original invocation deadlines are retained.

Operational runtime: `scripts/production_thread_runtime.py`, deployed as a
separate wrapper under campaign `thread-tuning-20261123-v1/`. It imports the
unchanged production implementation `9ffa6bb`; no scientific source files or
existing manifest entries were edited. Runtime SHA-256:
`fa5319e4ba425cbbae339270e660106f1a105aa42363bcce0697e7407e69518b`.
Remote smoke tests passed six-thread MKL verification, pause/resume with identical
scientific identity, final sampling checks, and controller launch/admission.
Evidence: `thread-tuning-20261123-v1/smoke-verification.json`.

Controller **471658**, birth **29019551**, adopted all four original workers;
the previous controller 343925 was intentionally retired. Handover process
**471661**, birth **29019612**, is owned by this authorized resource change.
Do not duplicate either. It waits at most three hours for each original worker's
next committed paired checkpoint, validates both payload hashes while the worker
is stopped, records its old launch/checkpoint, terminates only that verified
identity, and lets the controller resume the same case with six threads. No
additional checkpoint stream is introduced. At Unix time `1795448251.997` all
four original workers remained live at four threads, D128, sweeps [4,3]–[4,4];
the handover was `awaiting_checkpoints`, not yet a completed six-thread switch.

Read `thread-tuning-20261123-v1/switch-status.json`, per-case handover receipts
and `cases/<id>/thread-runtime-<pid>.json` for the actual transition outcome.
The bounded handover verifies that resumed cases advance beyond their saved
counts. Six-thread throughput has not yet been measured on the production cases.
A read-only completion observer is attached to the existing handover: local
background shell `sh_209111b78001XGWfX4RjRMmyNK`, sampling every 60 seconds and
ending on completion, timeout, handover exit or ten minutes after its three-hour
deadline. Its ownership/result files are `completion-observer-owner.json` and
`completion-observer-result.json` in the tuning directory. Do not duplicate it;
the local shell provides an automatic completion notification. Latest pre-observer
snapshot at Unix `1795448391.917`: four running, 7,676 pending, original workers
at four threads and D128 with sweeps [4,4]–[5,4]. Manifest and runtime hashes match.

If the controller needs recovery, use the following command from the immutable
production source, with its `code/` on `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`,
the campaign `tmp/` as `TMPDIR` and numerical thread environment variables 6:

```bash
/opt/miniconda3/envs/spin-glass/bin/python -B -u \
  /mnt/ali/yty/spin-glasses/rydberg-kagome/outputs/l6-full-adaptive-production-20261123-v1/thread-tuning-20261123-v1/production_thread_runtime.py \
  controller \
  --root /mnt/ali/yty/spin-glasses/rydberg-kagome/outputs/l6-full-adaptive-production-20261123-v1 \
  --plan /mnt/ali/yty/spin-glasses/rydberg-kagome/outputs/l6-full-adaptive-production-20261123-v1/thread-tuning-20261123-v1/plan.json \
  --threads 6
```

The campaign exclusive lock prevents duplicate control. Use this operational
restart command in preference to the historical launch command below. Preserve
`plan.json`, `controller-adoption.json`, smoke evidence and handover receipts.

## Project naming and canonical path — 2026-11-23

The shared Ali project is now **`rydberg-kagome`**, independent of lattice size,
covering L4, current L6, and future L8 campaigns. Canonical base:
`/mnt/ali/yty/spin-glasses/rydberg-kagome`.
The old `rydberg-kagome-l4-pq` path is a compatibility symlink to this directory.
An atomic directory/symlink exchange preserved the directory inode and continuous
old-path access. All five production processes were verified live afterward;
the manifest hash was unchanged. Receipt: `project-path-rename-20261123.json`
at the canonical project root. Existing commands, Git worktree metadata and
immutable launch records below remain valid through the compatibility link.
Use the canonical base for new runs and references. Keep that link while any
running case, checkpoint or historical provenance depends on the old path.
Local Git `origin` uses the canonical path:
`ali:/mnt/ali/yty/spin-glasses/rydberg-kagome/worktrees/l6-pilot-20261120`.
The post-rename verification found a fresh controller snapshot (about five
seconds old), four running cases, 7,676 pending and the same manifest SHA-256.
Git access through the canonical root resolves the production checkout to
`9ffa6bb22cf4f1dec5e8c421d16c722ed2663f94`.

## Current override: compact production and 64-GiB admission budget

The user authorized production and diagnostic-only canonical residuals. They
subsequently changed restart cadence to **30 sweeps per start**. The local solver,
CLI, case wrappers, task manifest and queue fallback defaults now use 30. A
paired boundary writes once, with forced growth/controlled-exit/completion saves.
Only the latest committed pair is retained for unfinished new cases; preserve
the original completed test and cancelled retries. Production is authorized.
The endpoint and window policies now treat canonical residuals as diagnostics.

All cases keep models, geometry, both starts' final moments, full optimization
history, signed histograms, flags and provenance. Only six audit cases retain
final wavefunctions (sid 0, drive indices 0/14/29, both detunings); other successful
cases retire verified restart tensors after analysis commits. Later resampling
or extension requires rerunning those cases. Sampling-check failures retain
restart tensors and are reported separately; sampling exceptions remain resumable.

See `PRODUCTION_STORAGE_ESTIMATE.md`: expected retained data **35–45 GiB**,
campaign admission budget **64 GiB**, currently 917 GiB free. This replaces all
older all-wavefunction capacity estimates. New writes are bounded per tensor,
case and log; current allocated bytes, failure evidence, active write reserves
and pending outputs are included at every admission cycle. This is an application
budget, not a filesystem quota. Shared admission and growth reservations use PID
identities and a common lock; each case runs four numerical-library threads.

Validation v2 passed 36 tests in 81.949 s; v3 passed 37 in 82.418 s. The final v4
passed **39 tests in 82.020 s**, including hard address-space/deadline protection,
missing-artifact classification and cgroup ancestry admission. Evidence is Ali
`outputs/compact-production-tests-20261123-v4/tests.log`; validated source is
`worktrees/compact-production-validation-20261123-v4`. These are operational changes;
scientific inputs and acceptance thresholds are unchanged. The worker uses a
64-GiB address-space ceiling, 72-hour graceful optimization budget, and 73-hour
hard invocation deadline. Production launched from the published implementation;
the launch record below is authoritative.
Older paused-authorization, storage and active-process statements below are
historical; the original full-case worker/watcher is finished.

## Production launch — 2026-11-23

Implementation revision **`9ffa6bb22cf4f1dec5e8c421d16c722ed2663f94`** was pushed
to Ali `pilot/adaptive-d-20261123` and GitHub `coolgas/spin-glass:ali`; both refs
were verified. Production uses a clean detached checkout at Ali
`/mnt/ali/yty/spin-glasses/rydberg-kagome/worktrees/l6-compact-production-20261123-v1`.
All 57 Python/JSON files under `code/` and `scripts/` matched the successful v4
validation snapshot by SHA-256. Receipt:
`outputs/l6-compact-source-verification-20261123-v1.json`.

Campaign root (all output paths absolute and outside synced storage):
`/mnt/ali/yty/spin-glasses/rydberg-kagome/outputs/l6-full-adaptive-production-20261123-v1`.
Its `tasks.json` has 7,680 unique cases, including six full-retention audits;
SHA-256 **`12716ed784d2f69a6bd96b1a0442c5d57f48953da9630e1e5762295e553f4e07`**.
`launch-provenance.json` records the code revision, manifest hash, physical
conventions/seeds, environment, storage policy, command and launch identity.
`initial-worker-verification.json` and `environment-verification.json` preserve
live limits, thread settings and package versions. Four launch/source/environment
receipts were downloaded and individually hash-verified into local
`/Users/tianyiyan/ResearchLocal/spin-glasses/releases/l6-production-launch-20261123-v1`;
its `transfer-record.json` records source and hashes. This is a launch record,
not a backup of live campaign outputs.
The controller PID is **343925**, `/proc` birth identity **28881825**.
Owner: this authorized OpenCode L6 production launch; other sessions should
observe or adopt at a natural boundary, without duplicating the controller.

Launch command, from that source with `PYTHONPATH=<source>/code`,
`PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=<campaign>/tmp` and numerical thread variables 4:

```bash
/opt/miniconda3/envs/spin-glass/bin/python -B -u -m rydberg_kagome.queue \
  --manifest /mnt/ali/yty/spin-glasses/rydberg-kagome-l4-pq/outputs/l6-full-adaptive-production-20261123-v1/tasks.json \
  --output /mnt/ali/yty/spin-glasses/rydberg-kagome-l4-pq/outputs/l6-full-adaptive-production-20261123-v1 \
  --workers 8
```

The ceiling is eight; actual admission currently permits **four cases**, owing
to 24-GiB initial peak reservations and 16-GiB host headroom. Initial workers:
343929/343930/343931/343932, sample 0, drive indices 0/1, both detunings. All four
were live with four OS threads, BLAS/OpenMP variables 4, and `RLIMIT_AS=68719476736`
bytes. Each reached `sweeping`, D128, next sweep 1. Initial allocated campaign size
was 22,335,488 bytes; filesystem free 984,648,478,720 bytes. No completed production
case or convergence claim follows from launch.

The owned bounded verification watcher (`sh_208f964bd001l37fNeCWHdEyi2`) finished
successfully at Unix time **1795447072.660321**: all four initial workers advanced
to **[1,0]**, D128, and were sweeping start 1. Counts were **4 running / 7,676
pending**, no completed or failed cases. Receipts: `launch-verification-watcher.json`
and `launch-progress-verification.json` in the campaign. No watcher remains active.
At that snapshot load1 was 31.43 and available memory 120,931,766,272 bytes;
CPU and peak-memory admission both allowed no additional workers. The controller
continues to evaluate shared resource headroom without changing four threads/case.
The controller continuously schedules and adopts live worker identities on
restart. If the controller dies, rerun the exact command/environment from the
same immutable source and manifest; its exclusive lock prevents duplicate control.
Do not modify this source or change scientific inputs while the campaign runs.

Higher-D memory is forecast-based; hard limits may block or fail individual
cases. Failed sampling/checkpoint evidence remains on disk and may pause
application-budget admission. No filesystem quota was installed. The original
full test and the three cancelled retries remain untouched and stopped.

## Canonical-residual diagnostics update

The user requested that canonical residuals be saved every `x` rounds in
production. Implemented on the feature source as a **5-sweep cadence**, aligned
with paired convergence checks. For each start, each sweep-5 moment record now
stores `canonical_error = numpy.linalg.norm(psi.norm_test())` alongside `D`,
`mixer`, `m`, and `qea`. Completed `result.json` writes the cadence, definition,
threshold and full compact per-start history under `canonical_diagnostics`.
Final endpoint canonical error remains a separate measurement after final
canonicalization. Missing diagnostics from older checkpoints/source versions
are not interpreted as zero.

Updated `ADAPTIVE_CASE_USAGE.md` and the case integration test to verify values
persist across pause/resume, match uninterrupted execution, and appear in final
results. Focused remote validation passed: 6 tests (`test_adaptive_case` and
`test_adaptive_resume`) in 30.141 s, on Ali in isolated worktree
`worktrees/canonical-diagnostics-20261123-v1`, outputs in
`outputs/canonical-diagnostics-tests-20261123-v1`.

**The active full-case test was not modified or resumed with this source.** Its
checkpoint/source identity is locked to the earlier revision; do not hot-patch
or migrate it. The diagnostic also must not be retroactively inferred for that
run. Use this revision for new production cases after the existing convergence
gate and other production prerequisites are met. Publication commits and remote
refs should be recorded below when completed.

## Latest monitoring update

## Full-case convergence-gate outcome

The single authorized Ali full case finished at its 120-sweep-per-start limit:
`outputs/adaptive-full-case-20261123-v6`, L6/sample 0/full model, delta=3.15,
Omega/V0=0.9037538790536807, chi=12, final D=256. Both starts completed 120
sweeps. Supervisor status is `completed_gate_failed`; optimization ended
`sweep_limit`, `solver_pass=false`, and `full_case_gate_passed=false`. Production
remains paused; the required convergence gate was not met.

Finalization did complete. The 20,000-pair sampling-moment check passed. Endpoint
canonical residuals **after final `psi.canonical_form()`** are 1.422650209431156e-14
and 1.4172788008575924e-14; these are not the missing periodic pre-finalization
residuals and do not explain the failed convergence check. Final energies are
-159.98693668278554 and -159.98693668278560. Bond-dimension verification and RG
convergence remain false; production qualification is false. Solver elapsed
28,137.29 s; peak RSS was 20,052,260 KiB. Preserve checkpoints and result files.
No further sweeps or retry are authorized by this outcome; review diagnostics and
agree a new validation protocol before scheduling more computation.

Watcher SSH `sh_2064a687d001y5aNeQfeFoOxK7` disconnected with a broken pipe.
The detached solver was unaffected: controller 80189 and four-thread worker
80191 verified live at saved counts [45,44], D256; status age ~15 s. Its orphan
monitor PID 79688 was verified and terminated; no solver process was signalled.
Replacement watcher: `sh_206d6df34001gGwMzREXoFA4Wt`, SSH keepalive every 30 s,
120-s sampling interval, deadline original launch +73 h. Ownership receipt:
`outputs/adaptive-full-case-20261123-v6/completion-watcher-replacement.json`.
This supersedes watcher references below. Do not duplicate the replacement.

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

### User selection: four threads per omega

Latest observed timing snapshot (2026-11-23; observation, not a live counter):
counts [6,5], D128, four threads, ~22 minutes since original launch including
benchmarking and switches. Completed four-thread sweeps took 63.24–63.79 s.
At unchanged D128 and this rate, remaining optimization was ~2.3 h to the
earliest acceptance near 70 sweeps/start, or ~4.0 h to the 120-sweep cap.
The user-facing planning range was roughly 2.5–4.5 h remaining, but final
observables/sampling cost is unmeasured and that allowance is provisional.
Larger D can substantially extend runtime. The first growth decision at sweep
20/start was ~30 minutes away at this snapshot. This is not a convergence promise.

Switch completed: saved counts [4,3], D128, generation
`c2d579f971c2480b8aef433dbd5ebed4`. Controller PID 80189, worker PID 80191.
Verified four live threads, ~336% CPU and OMP/OPENBLAS/MKL/NUMEXPR all set to 4.
Existing watcher `sh_2064a687d001y5aNeQfeFoOxK7` (remote PID 79688) survived the
quick handover and follows the current status; retain it, do not create another.
Original 72-hour deadline and per-start sweep counts remain intact. The switch
shell has finished successfully. Full convergence remains pending.
The following transition description is historical; live verification is complete.

The user selected four threads per omega, superseding the eight-thread benchmark
choice. Owned switch `sh_2064daa2c001QZzrbuDUzWr4W6` waits for the next committed
checkpoint, stops only verified worker 79488, and resumes this same case with
`--threads 4`. Last snapshot before switch: D128 and counts [3,3]. The existing
eight-thread watcher may report intentional -15 termination during the transition.
No repeat benchmark is needed. Production uses four cores per omega, with omega
concurrency chosen from shared CPU/memory headroom; parallelizing starts later
must split the per-omega budget rather than double it. Live verification completed;
the existing watcher was retained across the handover as recorded above.

### Historical eight-thread benchmark and transition

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
The switch and live verification completed; the subsequent four-thread selection
and current ownership are recorded above.
Boundary verified at [1,1], generation `2e0deae9402f4349ad48bffd8201bf3b`.
Probe PIDs 78875/78876/78877/78878 used 1/2/4/8 live threads respectively,
with measured CPU use ~100/186/342/638 percent at 46 seconds. Switch controller
PID 78852. Multiple cores are demonstrably active. Parallel probes share memory
bandwidth, so timings are conditional on this shared load and should not be
presented as isolated per-thread scaling guarantees.

Production design is now explicitly omega-parallel plus multithreaded per omega;
see `PROTOCOL_DECISIONS.md`. The current case still alternates its two starts.
Shared reservations and a validated production scheduler remain outstanding.

Historical initial launch: worker PID 77262 was verified at one thread, ~100% of one CPU, ~2.7 GiB
RSS while its first D128 sweep was running. Completion watcher
`sh_2063f2934001bWACbOVhrkwWAQ` finished during the first intentional switch.
It is not the current watcher; use the four-thread ownership record above.

The user now requires at least one full-case convergence before production and
authorized this test. One separate case was launched, supervisor PID 77260,
source `worktrees/adaptive-integration-20261123-v6`, output
`outputs/adaptive-full-case-20261123-v6`. It starts fresh, preserving the bounded
resume demonstration separately. L6, sample 0, full interaction, delta=3.15,
omega=0.9037538790536807, chi=12, D ladder 128/256/384/512, two starts,
120 cumulative sweeps/start, 20,000 replica pairs. No accuracy threshold changes.

Initially one compute thread, now four; RLIMIT_AS=64 GiB, live 16-GiB headroom reserve and peak-aware
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
