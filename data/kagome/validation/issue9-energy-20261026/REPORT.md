# Extended L6 energy-stability check

**Decision: resolved late-time energy drift. Ground-state qualification remains false.**

Flat sampler; L6/sample0 (108 spins), Gamma=0.05, 5,000 walkers, two replicas, eight independent populations. Each trajectory covers 2,560 blocks, total imaginary time 512. All early blocks are diagnostics, not equilibrated measurements.

## Energy versus imaginary time

Intervals below are 95% Student-t intervals across eight population means; energies are total single-replica energies.

| Time interval | Energy mean | SE | 95% interval |
| --- | ---: | ---: | --- |
| 0–64 | -127.435848 | 0.786578 | [-129.295809, -125.575886] |
| 64–128 | -136.211704 | 0.661538 | [-137.775992, -134.647415] |
| 128–192 | -136.945306 | 0.780191 | [-138.790165, -135.100447] |
| 192–256 | -137.109390 | 0.777750 | [-138.948476, -135.270304] |
| 256–320 | -137.333404 | 0.800336 | [-139.225898, -135.440910] |
| 320–384 | -137.554495 | 0.777942 | [-139.394036, -135.714955] |
| 384–448 | -137.774538 | 0.768927 | [-139.592761, -135.956315] |
| 448–512 | -138.101619 | 0.807394 | [-140.010804, -136.192435] |

## Primary late-window comparison

Time 256–384 versus 384–512: paired energy change **-0.494129 +/- 0.170501 SE**, 95% interval [-0.897299, -0.090959].
Per spin: change -0.00457527, 95% interval [-0.00830833, -0.00084221].

| Population | Energy, time 256–384 | Energy, time 384–512 | Paired change |
| --- | ---: | ---: | ---: |
| 0 | -133.580095 | -133.582477 | -0.002382 |
| 1 | -137.989250 | -138.298318 | -0.309068 |
| 2 | -139.476397 | -140.459988 | -0.983591 |
| 3 | -135.426258 | -136.375841 | -0.949583 |
| 4 | -138.962214 | -138.980198 | -0.017984 |
| 5 | -137.567120 | -137.994923 | -0.427803 |
| 6 | -136.353921 | -137.549201 | -1.195281 |
| 7 | -140.196342 | -140.263682 | -0.067340 |

Final short-window change (time 384–448 to 448–512): -0.327081 +/- 0.135546 SE, 95% interval [-0.647597, -0.006566].
Slope over time 256–512: -0.00399247 total-energy units per imaginary-time unit; 95% interval [-0.00711136, -0.00087358].

## Interpretation and provenance

An interval containing zero means no drift was resolved at this precision; it is not an equivalence test. No absolute/relative energy tolerance has been specified. Population disagreement, finite-walker bias and possible trapping remain relevant even if the energy looks stationary. These runs do not extend the pure-overlap forward-walking horizon and do not qualify susceptibility.

All 24 matched eq40/80/160 trajectory-prefix checks reproduce the previous pilot exactly. Frozen source identity and settings are in `campaign.json`; partial and complete traces are retained. No solver source change was required for this study. Production remains on hold.

See `energy_trace.svg`, `final.json`, `PROTOCOL.md`, and `runs/`.
