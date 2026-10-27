# L6 energy check through imaginary time 2048

**no primary late-time drift resolved at current precision; plateau not certified. Ground-state convergence is not established.**

Eight matched populations; 108 spins, flat sampling, 5,000 walkers, two replicas, Gamma=0.05. Energies refer to one replica. Intervals are descriptive 95% Student-t intervals across populations.

| Imaginary-time window | Mean energy | SE | 95% interval |
| --- | ---: | ---: | --- |
| 0–256 | -134.425562 | 0.712659 | [-136.110734, -132.740390] |
| 256–512 | -137.691014 | 0.780621 | [-139.536889, -135.845139] |
| 512–768 | -138.262239 | 0.837371 | [-140.242306, -136.282171] |
| 768–1024 | -138.512214 | 0.871932 | [-140.574005, -136.450424] |
| 1024–1280 | -138.815115 | 0.910326 | [-140.967693, -136.662536] |
| 1280–1536 | -138.821803 | 0.909298 | [-140.971952, -136.671655] |
| 1536–1792 | -138.829747 | 0.911638 | [-140.985429, -136.674066] |
| 1792–2048 | -138.852327 | 0.919063 | [-141.025566, -136.679088] |

Primary paired change, time 1024–1536 to 1536–2048: **-0.022578 ± 0.022260 SE**, 95% interval [-0.075215, 0.030058].
Per spin: -0.00020906, 95% interval [-0.00069643, 0.00027832].

| Population | Energy 1024–1536 | Energy 1536–2048 | Change |
| --- | ---: | ---: | ---: |
| 0 | -133.762559 | -133.743151 | 0.019408 |
| 1 | -139.551472 | -139.547184 | 0.004288 |
| 2 | -142.187966 | -142.204572 | -0.016606 |
| 3 | -137.146891 | -137.152342 | -0.005452 |
| 4 | -139.327586 | -139.503445 | -0.175859 |
| 5 | -140.452432 | -140.455316 | -0.002884 |
| 6 | -137.822233 | -137.814368 | 0.007864 |
| 7 | -140.296534 | -140.307920 | -0.011386 |

Final time-256 window change: -0.02257987 ± 0.03653251 SE; 95% interval [-0.10896554, 0.06380580].

Slope over time 1024–2048: -0.00005094 ± 0.00005336 SE; 95% interval [-0.00017711, 0.00007524].

All eight energy and time prefixes through time 512 reproduce the preceding campaign exactly. See campaign.json for frozen source and inherited-harness hashes.

This extension was selected after resolved drift in the previous campaign. Intervals do not control error across sequential studies. No equivalence tolerance is specified; an interval containing zero does not certify a plateau. Population disagreement, trapping and finite-walker bias remain relevant. Susceptibility is not qualified. Production remains on hold.

Artifacts: energy_trace.png, energy_trace.svg, final.json, PROTOCOL.md, runs/ and partial/.
