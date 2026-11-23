# Preproduction code review — 2026-11-23, v6

## Current integration review

Threading update: the authorized full case has resumed with eight threads after
matched same-checkpoint probes at 1/2/4/8 threads took 155.910/105.822/75.019/63.341 s.
Energy spread <7e-13; peak RSS ~5.53–5.56 GiB. Actual eight-thread CPU use was
verified. This measures one D128 sweep under shared load, not whole-case scaling
or the optimal production throughput configuration. The full convergence gate
remains pending. See the primary handoff for current process/watcher ownership.

The adaptive solver is integrated into the production case API and CLI, including
final moments, sampling and immutable completed-result reuse. Both independent
starts resume from hash-verified immutable checkpoint generations with source,
environment and scientific-input identity checks. Mixer state, solver settings,
total sweeps, growth stage, diagnostics and pending growth are preserved. A kernel
lock provides exclusive case ownership and releases on process death.

All 38 tests passed on Ali: the full 35-test suite (23 script tests, ~38 seconds;
12 inherited scientific tests, ~1.8 seconds), followed by the three case tests
(~21.6 seconds) after adding a final-sampling interruption regression. That test
confirms retry performs no more DMRG sweeps, preserves final checkpoint hashes
and rejects a corrupt completed artifact. The small end-to-end fixture pauses at [2,1]
completed sweeps and resumes through final moments and sampling, matching an
uninterrupted run. Other new tests cover active-mixer and post-growth continuation,
incomplete generations, checkpoint corruption, identity mismatch and memory
deferral. TeNPy reports benign unused-option warnings for partially exercised
engines. No acceptance tolerance or cumulative 120-sweep budget was relaxed.
All three restart tests passed (~0.6 seconds) after adding an injected failure
while writing the second payload; the previous pointer stays byte-identical and
restores the earlier complete pair. Package source was unchanged during these
two additional test updates and the live L6 validation.
The additional full-case supervisor regression exercises success, failed sampling
and subprocess failure. It passes with TMPDIR on the project data volume; the
initial root-filesystem fixture was correctly rejected by the storage preflight.

Source is staged and hash verified in Ali
`worktrees/adaptive-integration-20261123-v6`. The actual L6 full-interaction,
chi=12, D128 production-path validation completed in 621.295 s at
`outputs/adaptive-integration-validation-20261123-v6`: two fresh processes exited
0 with counts [1,0] then [1,1], demonstrating resume. Eight engine payload hashes
across four immutable generations verified, as did the manifest pointer and source
identity. Both processes used an 8-GiB address-space cap, one thread and a
420-second hard timeout. This is a restart demonstration, not full convergence.

The user subsequently required one complete case to converge before production.
That test is now running separately in `outputs/adaptive-full-case-20261123-v6`,
supervisor PID 77260, with the same parameters, two independent starts and the
128/256/384/512 ladder. The unchanged 120-sweep cap applies per start. The single
worker has RLIMIT_AS=64 GiB and a 72-hour hard deadline; final sampling uses
20,000 pairs. Solver and sampling must both pass. See the primary handoff for
ownership and retention. This necessary gate remains pending; production is paused.

Remaining production requirements are shared-controller memory reservations
(explicitly deferred), full-case D128+ convergence/resource characterization,
RG and D-sensitivity qualification, and a measured checkpoint retention plan.
The current immutable-generation cadence retains all data and can create up to
1,092 restart files per 120-sweep case. It is suitable for bounded validation;
campaign-scale storage and checkpoint overhead have not been qualified.
Time budgets are checked between sweeps; a sweep or final sampling can overrun
the soft `--seconds` budget. The validation wrapper provides the hard timeout
and address-space ceiling; the general case API's memory checks are advisory.
Atomic rename protects against process interruption; no power-loss durability
claim is made. Source changes intentionally require a separate case/release,
because source identity is part of restart compatibility.

The frozen L4 `scripts/verify_snapshot.py` checks its historical manifest and
dataset; it is not a verifier for modified adaptive feature sources. Preserve
that manifest and baseline. Verify this feature deployment using the v6 source
hashes and tests, then verify the pushed Git ref.

## Historical v5 review

Production remains paused. This round reviewed and tested code; it did not
launch another DMRG pilot or production campaign.

## Source and reproducibility

Local working checkout: this directory, branch `pilot/adaptive-d-20261123`.
Remote tested worktree:
`/mnt/ali/yty/spin-glasses/rydberg-kagome-l4-pq/worktrees/adaptive-review-20261123-v5`
Remote branch: `pilot/adaptive-review-20261123-v5`.
Both derive from `aeb225549a75c48942a4f162343578c9bbed5818` with uncommitted
review changes. Preserve all v1–v4 sources and outputs for provenance.

## Changes

- Compact MPO now checks each projected two-block operator against the original
  model before using the supplied interactions. A mismatched matrix raises an
  error. Only one pair's chi^4 comparison is materialized at a time. Onsite
  terms are copied verbatim. No coupling or singular-value cutoff was added.
- Pilot rejects invalid time/memory budgets and nonfinite sweep diagnostics.
- Restored checkpoint test objects are released after checking, avoiding a
  retained redundant MPS copy during later optimization.
- Trace records distinguish a growth request from its memory-admitted outcome.
- Memory helper rejects invalid dimensions and avoids traversing outside the
  cgroup mount when a namespace-relative path contains `..`.

## Validation on Ali

All 31 tests passed in one final run: 19 script tests (~24.8 seconds) and 12
inherited scientific tests (~1.8 seconds), one BLAS/OpenMP thread.

New coverage includes:

- Complete nine-block L6 operator at chi=2 (512-dimensional Hilbert space)
  against independent projected-operator action, including snake ordering and
  Hermiticity. This is not full dense chi=12 validation.
- Full/nn/nn_nnn representation comparisons and rejection of a deliberately
  incorrect interblock interaction matrix.
- Checkpoint restoration against freshly reconstructed compact MPO sites,
  with identical state vector and energy.
- Actual supervisor success/failure exits and terminal sidecars using harmless
  subprocess fixtures. Hard timeout behavior was exercised by the historical
  bounded pilot, not by this new fixture test.

The initial new dense-test helper used a site count rather than the final site
index for TeNPy's get_IdR; that test-helper error was fixed and the whole suite
rerun successfully. It was not a Hamiltonian mismatch.

The final chi=12 L6 cached-model profile passed the new pairwise consistency
check. Compact bond dimensions remain [2,14,26,38,50,62,74,86,98,2]. Two random
state energies remained -90.46296708501569 and -90.44611938708452, unchanged from
v3 and within 9e-14 of the legacy representation. Setup took 1.080 seconds,
peak RSS 0.155 GiB for setup/expectation only, not for DMRG.

## Remaining production gates

This review supports Hamiltonian equivalence and the tested adaptive logic.
It does not make the pilot a complete production controller:

1. The production case runner still defaults to legacy MPO and fixed-D logic.
   Integrate compact/adaptive behavior explicitly in a new release, retaining
   final moment and sampling calculations, independent starts and all flags.
2. Production restart must restore both starts, total sweep counts, dimension,
   mixer phase, diagnostics, checkpoint hashes and source/config identity.
   Existing pilot checkpoints and restore tests do not implement full resume.
3. Coordinate reservations between cooperating controllers and account for
   live external memory use. Current headroom checks are not atomic allocation
   guarantees; RLIMIT_AS limits address space, not RSS. Memory-deferred cases
   need a deliberate queue/resume policy.
4. Validate actual D128+ end-to-end behavior and resource forecasts with a
   bounded production-path check. A single D128 sweep is not a converged case
   or a measured full-case memory upper bound. Keep convergence and statistical
   uncertainty separate from code correctness.

No running workers or watchers were created in this review. No approval to
launch production is implied. Read MEMORY_UPDATE_HANDOFF.md and
PROTOCOL_DECISIONS.md before further work.
