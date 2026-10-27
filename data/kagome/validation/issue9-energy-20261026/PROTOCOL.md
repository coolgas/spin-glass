# L6 flat-sampler energy stationarity check

Declared before this campaign runs, 2026-10-26.

## Question and settings

Does the energy continue to drift when imaginary-time evolution is extended well beyond the earlier equilibration windows?

Use the frozen, rollback-fixed source from `issue9-rollback-20261025`, identified by its manifest and patch SHA256, without changing the solver. L=6 means 108 spins, PBC, sample0, sigma=1, h=0, Gamma=0.05; flat sampler, 5,000 walkers, two replicas. Eight independent populations use children 0–7 of `SeedSequence(20261025).spawn(32)`, matching the preceding pilot. Each population evolves from the random initial population for **2,560 blocks of length 0.2, total imaginary time 512**.

All blocks are recorded as a raw diagnostic trajectory (`n_equilibrate=0`, `n_measure=2560`). This does **not** mean early transient blocks are valid equilibrium measurements: no full-trajectory average is used as a ground-state estimate. Disable forward-overlap cycles (`n_forward_cycles=0`); ancestry-limited forward walking is a separate susceptibility qualification question.

## Analysis declared in advance

- Plot the energy against cumulative imaginary time, including the initial transient. Report total energy and energy per spin.
- Report fixed, nonoverlapping 320-block windows (time width 64), and their paired changes.
- Primary late-time check: compare blocks 1280–1920 and 1920–2560 (time 256–384 versus 384–512) within each population, then average the eight paired differences. Also report the last two 320-block windows.
- Report 95% Student-t intervals with 7 degrees of freedom from across-population differences, not from treating correlated blocks or walkers as independent. Report per-population late-window means and paired differences. These intervals are descriptive; repeated window comparisons are not an omnibus significance test.
- Report per-population linear energy slopes over the final 1280 blocks, with across-population uncertainty.
- Reproduce the old eq40, eq80 and eq160 measurement-window energies from the matching prefixes as a trajectory/regression check.
- Do not invent an absolute/relative energy-accuracy tolerance from observed scatter. An interval excluding zero demonstrates resolved drift at that precision. An interval containing zero means no drift resolved at that precision, **not proof of equality or ground-state accuracy**.
- An energy plateau alone cannot certify the pure susceptibility or eliminate walker-population bias or metastable trapping. Preserve independent-population disagreement in the report.

## Execution and provenance

Run at most eight CPU workers on AutoDL. Save partial energy/time traces every 320 completed blocks using a harness-only observation wrapper around the existing branching-interval function; the wrapper returns its original value without modifying solver state or RNG. Save each complete raw population as NPZ. Preserve source hashes, logs, runtime, original pilot archives and production data. Keep production on hold and the instance powered on.
