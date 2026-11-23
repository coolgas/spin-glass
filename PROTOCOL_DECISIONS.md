# Agreed L6 protocol and unresolved validation

Status: design and bounded pilot validation only; production is paused.
Update 2026-11-23: the user authorized one complete adaptive L6 case and requires
at least one full-case convergence success before production. This is a necessary
gate, not blanket production authorization or ensemble-wide accuracy evidence.

## User decisions

- L6 uses the full interaction model only. Match L4 physical conventions,
  geometry-generation rules, 128 disorder IDs, two detunings (2 and 3.15),
  and 30 drive points from 0.03 to 10. Target size is 7,680 cases.
- Retain chi_RG=12, two independent starts, initially 20,000 replica pairs,
  equal disorder weighting and the L4 observable definitions.
- Start production-like adaptation at D128, with proposed ladder
  128/256/384/512. The reduced 16/32/64/128 ladder is a short demo only.
- Hard limit is 120 total sweeps PER START, not 120 per D. No extra 40-sweep
  extension. Decision points are 20, 40 and 60; no later D increase.
- Both starts retain their own states; if either triggers growth, grow both.
- Early growth uses persistent truncation at the cap, stalled energy or
  magnetization, or persistent disagreement of locally stable starts.
- Proposed early thresholds (including discarded weight 1e-6) remain
  provisional controls. They are not physical error bounds.

## Final acceptance

Require 50 consecutive energy differences below 1e-8 per physical spin, all at
the final D with the mixer off; magnetization drift below 1e-4; canonical error
below 1e-8; and two-start agreement within 1e-7 energy per spin and 1e-3 q_EA.
Do not count earlier-D/mixer-active sweeps toward final confirmation. Keep
sampling consistency, solver convergence, bond-dimension verification and RG
convergence as separate flags. Completion or a timeout never implies accuracy.

The user's accuracy goal is at least the L4 standard. Matching settings does
not prove matching physical error at L6. No completed full-model D256 comparison
establishes production bond-dimension error. Chi sensitivity is unresolved at
both sizes. Preserve these limitations in reports and future launch decisions.

## Engineering validation

See `ADAPTIVE_PILOT.md` for verified pilot behavior and
`MEMORY_UPDATE_HANDOFF.md` for memory changes and unfinished work. Do not use
the generic legacy queue for L6: it defaults to L4 and uses inadequate generic
memory forecasts. A shared-host production scheduler has not been validated.
