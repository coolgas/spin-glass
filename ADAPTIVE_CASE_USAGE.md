# Adaptive case execution and resume

This feature is validated for bounded development runs. The full L6 campaign
remains paused. Read `PROTOCOL_DECISIONS.md`, `PREPRODUCTION_REVIEW.md` and
`MEMORY_UPDATE_HANDOFF.md` before launching further work.

## Bounded invocation

From this checkout, with the spin-glass Python environment and `PYTHONPATH=code`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python -B -m rydberg_kagome case \
  --adaptive --no-exact \
  --output /absolute/non-synced/path/to/case \
  --L 6 --sample 0 --model full --delta 3.15 --omega 0.9037538790536807 \
  --chi 12 --dimensions 128 256 384 512 --pairs 20000 \
  --seconds 360 --invocation-sweeps 1 --memory-gib 8
```

Always provide the explicit L6 ladder: generic CLI/API dimension defaults retain
their historical values. `--max-sweeps` must remain 120 for adaptive execution.
The example uses the user's selected four threads per omega case.
Production is intended to run concurrent omega cases with multiple CPU cores per
case. Account for both case concurrency and threads when sizing CPU/memory use.
`--no-exact` selects the production-scale path; dense exact validation belongs
to small scientific tests. Do not combine `--adaptive` with `--fixed-production`.

The invocation ends at the next completed-sweep boundary when its soft time or
start-sweep budget is exhausted. `--invocation-sweeps 1` allows one total sweep
across the two starts per process. `--seconds` and `--memory-gib` alone do not
provide an OS hard limit. The bounded validation supervisor
`scripts/validate_adaptive_case.py` additionally applies RLIMIT_AS=8 GiB and a
420-second hard timeout to each of two sequential processes. Do not use its
validation settings as a production memory estimate.

## Resume and ownership

Repeat the invocation with the same output and scientific parameters, using the
same package sources and Python/NumPy/SciPy/TeNPy versions. Operational time,
sweep-per-invocation and memory budgets may change. Total per-start sweep counts
remain cumulative and cannot exceed 120. `case.lock` uses a kernel advisory lock;
its file may remain after exit and must not be treated as evidence of a live
worker. Timestamped owner and failure records preserve execution history.

`resume.json` points to a complete immutable generation in `restart/`. Each
generation has two engine payloads and a manifest. A killed invocation can leave
an unreferenced partial generation; resume uses the last committed pointer.
Hash/identity failure is an error, not permission to reset the case. Preserve
failed and incomplete files for diagnosis. Do not hand-edit counters, identities
or tolerances to force a resume. A source update requires a deliberate migration
or a separate case path; no automatic state migration is implemented.

`--model-cache /absolute/path/to/previous/case` is optional for initial setup.
The cached model's scientific inputs, payload hash and physical interactions
are checked. Cache provenance is recorded in the new case. The source cache is
retained.

## Results and storage

Canonical residuals are measured and checkpointed every five cumulative sweeps
per start, together with the observable measurements. In `resume.json` and each
immutable restart manifest, read `state.moments[start][sweep].canonical_error`;
the record also contains `D` and `mixer`. The value is exactly
`numpy.linalg.norm(psi.norm_test())`, before final canonicalization, and is reused
by the paired acceptance check with threshold `<1e-8`. This avoids additional
norm-test contractions. The cadence is currently fixed at five sweeps.
Completed `result.json` includes the full compact history in
`canonical_diagnostics.starts`, indexed by start, with explicit sweep numbers,
definition, cadence and threshold. Endpoint `canonical_error` remains a separate
measurement after final canonicalization. Older source versions did not save
the periodic residual; missing historical measurements must not be treated as zero.
This source revision is for new cases; existing source-locked runs require an
explicit migration and must not be hot-patched to enable these diagnostics.

Paused and memory-deferred cases return `completed=false` and retain a restart
pointer; they do not create `result.json`. Completed optimization produces final
checkpoints, moments and replica-sampling output, including cases that reach
the sweep limit without satisfying convergence. Check `solver_pass`, sampling
consistency, bond-dimension verification, RG convergence and production
qualification separately. Existing complete outputs are hash-checked on reuse.

All restart generations are currently retained. This is the bounded validation
retention policy, not a campaign-scale storage policy. Keep outputs on local or
HPC storage outside Git and synced roots. Before production, measure checkpoint
bytes, write overhead and final sampling cost; define retention and stable
archives while preserving verified resume points and scientific milestones.

## Tests

Use one BLAS/OpenMP thread and `PYTHONPATH=code`:

```bash
python -B -m unittest discover -s scripts -p 'test_*.py'
python -B -m unittest discover -s code/rydberg_kagome/tests -p 'test_*.py'
```

For `test_full_case_supervisor.py`, set TMPDIR to a project-owned directory on
the data volume: its fixtures exercise the real 64-GiB storage preflight.

The single authorized full-case test uses `scripts/run_adaptive_full_case.py`.
Use `--threads N` to select per-case numerical-library threads. `--resume` reuses
the existing case; the selected production setting is `--threads 4` (the default).
For restart, `--resume` reuses
the original case after its previous controller has finished; source identity and
checkpoint checks remain enforced. Earlier logs/status/resource receipts are
timestamp-archived, and the original 72-hour deadline is retained.
It applies a 64-GiB address-space ceiling and 72-hour hard timeout while retaining
the scientific limit of 120 sweeps per start. Its `status.json` records
`full_case_gate_passed` only when both solver and sampling pass. A passing case
does not authorize the production campaign. Consult the primary handoff before
using this supervisor; an owned full-case test may already be active.

`benchmark_adaptive_threads.py` replays one start-sweep from a committed checkpoint
into a separate timing/energy report. It does not advance the original case.
`switch_adaptive_threads.py` is an owned one-time transition helper: waits for a
new committed checkpoint, stops the verified worker, benchmarks 1/2/4/8 threads,
and resumes with the fastest successful multithreaded setting. Its probe directory
must not exist, preventing accidental duplicate use. Parallel probes measure under
shared load and may contend for memory bandwidth; this is a first tuning result,
not a universal optimal thread count. Keep failure reports and checkpoint state
if the helper exits without a successful resume.

Tests create temporary small-system data. The actual L6 validation outputs stay
in the named Ali output directory in the handoff; they are not committed to Git.
