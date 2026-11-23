# Agreed L6 protocol and unresolved validation

Status 2026-11-23: production is authorized, with canonical residuals retained as
diagnostics. Production launched as `l6-full-adaptive-production-20261123-v1`
from published implementation `9ffa6bb`; all 39 final v4 tests passed in 82.020 s.
Four concurrent cases were initially admitted with four threads each. The user
selected thirty-sweep restart saves and substantial
SSD capacity remaining afterward. The implementation uses a 64-GiB admission
budget and six final-wavefunction audit cases; see `PRODUCTION_STORAGE_ESTIMATE.md`.

## User decisions

- L6 uses the full interaction model only. Match L4 physical conventions,
  geometry-generation rules, 128 disorder IDs, two detunings (2 and 3.15),
  and 30 drive points from 0.03 to 10. Target size is 7,680 cases.
- Retain chi_RG=12, two independent starts, initially 20,000 replica pairs,
  equal disorder weighting and the L4 observable definitions.
- Start production-like adaptation at D128, with proposed ladder
  128/256/384/512. The reduced 16/32/64/128 ladder is a short demo only.
- Save one paired restart every 30 cumulative sweeps per start, plus forced
  growth/controlled-exit/completion saves; retain the latest committed pair.
- Hard limit is 120 total sweeps PER START, not 120 per D. No extra 40-sweep
  extension. Decision points are 20, 40 and 60; no later D increase.
- Both starts retain their own states; if either triggers growth, grow both.
- Early growth uses persistent truncation at the cap, stalled energy or
  magnetization, or persistent disagreement of locally stable starts.
- Proposed early thresholds (including discarded weight 1e-6) remain
  provisional controls. They are not physical error bounds.

## Final acceptance

Require 50 consecutive energy differences below 1e-8 per physical spin, all at
the final D with the mixer off; magnetization drift below 1e-4; and two-start
agreement within 1e-7 energy per spin and 1e-3 q_EA. Canonical residuals are
recorded every five sweeps and at final endpoints as representation diagnostics,
not a solver convergence condition. A final `canonical_form()` is applied before
endpoint observables and the endpoint residual is retained in results.
Do not count earlier-D/mixer-active sweeps toward final confirmation. Keep
sampling consistency, solver convergence, bond-dimension verification and RG
convergence as separate flags. Completion or a timeout never implies accuracy.

The user's accuracy goal is at least the L4 standard. Matching settings does
not prove matching physical error at L6. No completed full-model D256 comparison
establishes production bond-dimension error. Chi sensitivity is unresolved at
both sizes. Preserve these limitations in reports and future launch decisions.

## Engineering validation

User requirement (2026-11-23): production must parallelize independent omega
cases and assign multiple CPU workers/threads to each omega. Do not deploy a
single-thread-per-case production configuration as the default. For the current
solver, multiple BLAS/OpenMP threads accelerate each sequential DMRG sweep;
the two independent starts are another possible process-level parallel unit.
Final user selection: **four numerical-library threads per omega case**. This
supersedes automatic selection of eight threads for the active validation case.
Treat four cores as the total concurrent compute budget per omega; if starts are
later parallelized, split that budget between them instead of silently doubling
it. Set OMP/OPENBLAS/MKL/NUMEXPR thread counts consistently to prevent nesting.
Tune the number of concurrent omega cases within shared CPU and memory headroom;
do not assume all 32 host CPUs belong to this project.
Separate processes cannot independently advance the same MPS without an explicit
parallel algorithm. The present adaptive case still alternates its two starts.

Choose concurrent omega count and cores per omega jointly from measured sweep
throughput, available CPU allocation, external workload, peak memory at the active
D, and checkpoint I/O. CPU demand is concurrent cases times simultaneous start
workers per case times threads per start, including nested numerical libraries.
Memory admission must budget whole cases and transient D-growth peaks. Implement
shared reservations before overlapping production controllers. The new queue
implements a common admission lock, PID/start-time identities, peak-memory
reservations updated before growth, and adoption of live workers on restart.
Each new case initially reserves 24 GiB, based on the measured 19.1-GiB D256
peak with headroom; higher-D growth reserves its own measured forecast. Operational
memory pauses retain restart data and do not extend the scientific sweep budget.

See `ADAPTIVE_PILOT.md` for verified pilot behavior and
`MEMORY_UPDATE_HANDOFF.md` for memory changes and unfinished work. Do not use
older queue revisions for L6. Use the validated adaptive manifest and source
identity recorded in the handoff, with explicit output and four threads per case.
That handoff and the campaign's `launch-provenance.json` record ownership and
the exact command for controller adoption; do not duplicate a live controller.
