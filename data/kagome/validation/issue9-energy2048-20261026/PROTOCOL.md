# L6 flat-sampler energy check through imaginary time 2048

Declared after inspecting the time-512 campaign and before this extension runs.

The earlier paired change between time 256–384 and 384–512 was −0.494129 (SE 0.170501; 95% interval −0.897299 to −0.090959). All eight populations decreased; their late energies differed substantially. Energy stationarity was not demonstrated.

Retain L6/sample0, 108 spins, periodic boundaries, sigma=1, h=0, Gamma=0.05, flat sampling, 5,000 walkers, two replicas and the same eight RNG streams. Run 10,240 blocks of length 0.2, reaching time 2048. Use the identical frozen solver. The previous archives contain traces rather than restartable walker/RNG states, so replay each trajectory from its original seed and require an exact match of its first 2,560 energies and times against the time-512 archive. Preserve both campaigns.

Record every block as a diagnostic, with no forward-overlap cycles. Retain partial traces every 1,280 blocks (time 256). The full-trajectory energy average is not an equilibrium estimate.

Primary comparison: time 1024–1536 versus 1536–2048, paired within population. Also report time-256 windows, final time-256 window change, each population's late means and differences, and energy slopes over time 1024–2048. Use across-population SE and 95% Student-t intervals (7 degrees of freedom); do not treat correlated blocks or walkers as independent observations. This is a follow-up selected after resolved drift in the earlier campaign; intervals are descriptive and do not control error across sequential extensions or multiple comparisons.

No energy-accuracy tolerance is specified. A confidence interval containing zero is not evidence of equivalence; persistent population disagreement can prevent a convergence claim even when individual slopes are small. These energy-only runs cannot establish susceptibility convergence or remove finite-walker bias. Keep production on hold.

Use eight CPU workers, expected wall time about 13 minutes based on the completed campaign. Keep AutoDL powered on. Verify the inherited passive observation wrapper and source/harness/protocol hashes before launch. Do not change solver code.
