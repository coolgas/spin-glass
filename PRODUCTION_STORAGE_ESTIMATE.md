# L6 compact production storage — 2026-11-23

The user requires most of the SSD to remain available after L6 production.
The new campaign uses a **64-GiB admission budget**, including case outputs,
logs, checkpoints, partial writes and failure evidence. This supersedes the
earlier design retaining every final wavefunction (396–1,483 GiB).

## Retained science

All 7,680 cases retain inputs/source and environment hashes, projected model,
geometry, both starts' final moments and observables, full optimization history,
adaptive decisions, canonical diagnostics every five sweeps, all numerical flags,
and the complete signed 109-bin, 20,000-pair overlap histogram.

Six fixed audit cases retain both final wavefunctions: sample 0, drive indices
0, 14 and 29, at both detunings. The remaining 7,674 use `analysis` retention.
After sampling passes and all artifact hashes and the result commit succeed,
only the named, hash-verified current restart generation is retired. A crash
between result commit and retirement is repaired idempotently on reuse or queue
adoption. Unrelated orphans and historical test output are never glob-deleted.

Sampling exceptions leave unfinished resumable work. A completed sampling-check
failure retains the latest pair and is reported as `sampling_failed`; it is not
silently resampled or declared valid. Retained failures consume the budget and
can eventually pause dispatch. Ordinary successful analysis cases need a rerun
for later resampling or additional optimization; this is the storage tradeoff.

## Measured baseline and projection

The preserved historical L6 D256 case in
`outputs/adaptive-full-case-20261123-v6/case` contains:

- Projected model: 4,133,944 bytes.
- Final wavefunction pair: 50,686,636 bytes.
- All case-root artifacts: 55,301,847 bytes; excluding that pair, 4,615,211 bytes.
- Latest restart pair: 50,785,192 bytes.
- Historical restart directory: 16,448,867,748 bytes / 1,088 files, untouched.

Independent fresh model-size qualification at the two drive endpoints and
disorder IDs 0/127 measured 4,179,135; 4,110,829; 4,110,513; and 4,182,004 bytes.
The models and SHA-256 receipts remain in
`outputs/compact-model-size-audit-20261123-v1/sizes.json` on Ali. This supports
the model-size estimate, without bounding all disorder geometries.

The measured 4.62-MB overhead projects to about **33 GiB** across 7,680 cases.
Full history, allocation rounding, receipts and bounded logs add overhead; the
planning expectation is **35–45 GiB**, with six audit pairs adding approximately
0.3 GiB at D256 or 1.14 GiB under D-squared D512 extrapolation. Model-size
variation, failures and compression make this a forecast, not a proven bound.
Admission reserves **6 MiB for every pending case**, plus **512 MiB per pending
or active audit case**, and **2 GiB additional growth per active case**.

Ali's data volume had **917.07 GiB free** at preflight. A 64-GiB campaign would
leave about **853 GiB** of that currently free capacity, before other workloads.
The original historical test is already included in existing filesystem usage.
No old output is deleted to meet the new budget.

## Enforcement and restart bounds

The controller inventories allocated bytes with `du -sk` on each admission
cycle, including logs, partials and orphans. It reserves all pending compact
outputs, audit tensors and active growth before starting workers; it also keeps
128 GiB filesystem headroom. Shared controller admission uses a common file lock
and live PID/start-time receipts. This is an application budget, **not an ext4
quota**: external writes and filesystem metadata are outside its control.

Production workers bound compressed tensor files to 256 MiB and tensor payload
growth to a 2-GiB case allowance, leaving 16 MiB for metadata and final analysis.
The writer checks before each compressed write; an oversized replacement leaves
the previous committed generation intact and preserves the partial as evidence.
Retry admission blocks cases near the allowance. Logs retain at most 256 KiB per
case, about 1.88 GiB for the campaign; terminal failure JSON remains separate.
These are operational bounds, not scientific truncation: oversized artifacts
fail rather than change the state or convergence criteria.

Restart saves occur every **30 cumulative sweeps per start**, with one paired
generation at 30/60/90/120 and forced initialization, growth, controlled-exit and
completion saves. Only the latest committed pair remains after replacement;
both payloads commit before the pointer advances. Canonical diagnostics remain
every five sweeps and never determine acceptance. The sweep budget stays 120
per start across restart/growth.

Expect roughly 18–25 files per completed case, approximately 140,000–192,000
files plus failures and receipts. Keep these on Ali/local storage. Export only
selected stable snapshots; do not recursively mirror the campaign into Dropbox.

## Validation and dispatch

The compact-retention tests compare final energies, moments and histogram counts
against full retention. They inject sampling interruptions, sampling-check
failure and post-result/pre-cleanup crashes, preserve orphan evidence, and reject
corrupt retained artifacts. Storage tests inject oversized compressed writes and
check log bounds, forecasts and real-worker controller adoption. Export fixtures
check complete matched cohorts, all flags, signed bins and the L4 Binder
convention `0.5*(3-mean_dis(m4)/mean_dis(m2)**2)`.

Current test results and launch ownership are recorded in
`MEMORY_UPDATE_HANDOFF.md`. Preserve old validation snapshots and cancelled
retries; no historical result is reinterpreted under the new acceptance policy.

Each production worker has a 64-GiB address-space ceiling, a 72-hour graceful
optimization budget and a 73-hour hard invocation deadline covering finalization
and sampling. The controller checks that deadline for adopted workers too.
Growth forecasts must fit 85% of the case ceiling and atomic shared reservations;
otherwise the case pauses with its latest committed pair. Forecasts remain
estimates; exceeding a hard limit fails an invocation rather than changing D or
scientific acceptance thresholds. Host and visible cgroup ancestry limits apply.
