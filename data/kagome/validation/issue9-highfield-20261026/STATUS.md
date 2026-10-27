# Stronger-field screen

Launched 2026-10-26 at 02:08:42 UTC on AutoDL, PID 749677, with 16 workers. Runs are detached and survive SSH disconnection.

Gamma/sigma=3 and 5, sigma=1, L6/sample0, flat guide, two replicas. Eight independent populations per field and walker count (5,000 and 10,000), 64 blocks of duration 0.2, total projection time 12.8. All initial transients are retained. No forward cycles are run.

Both local and remote smoke checks passed exact repeated-result comparison across all 21 result fields and the uncoupled L6 energies -324 and -540. The remote Gamma=5 benchmark took 0.819 seconds for 128 walkers and eight blocks. Linear scaling suggests several minutes for the full screen, subject to larger-population and parallel overhead.

Completed successfully at 02:21:52 UTC in 788.47 seconds. Downloaded and SHA256-verified all 76 remote files (669,659 bytes); independently verified all 32 complete population archives, statistics and field-specific references. Local and remote SciPy Student-t intervals agree to 1e-10 absolute tolerance. The implementation notebook executed all 12 code cells successfully in a fresh kernel.

| Gamma/sigma | Walkers | Energy at time 9.6–12.8 | SE | Late-drift 95% interval |
|---|---:|---:|---:|---|
| 3 | 5,000 | -344.747539 | 0.080377 | [-0.484147, 0.181288] |
| 3 | 10,000 | -344.733091 | 0.038299 | [-0.283923, 0.125730] |
| 5 | 5,000 | -552.476478 | 0.019745 | [-0.114111, 0.065961] |
| 5 | 10,000 | -552.482694 | 0.011308 | [-0.053251, 0.005891] |

Energies are total per replica, not per spin. The drift compares time 6.4–9.6 with 9.6–12.8, using eight independent populations. At Gamma=5 and 10,000 walkers, both the drift interval and the independent walker contrast [-0.056217, 0.043784] fit inside the working ±0.108 tolerance. At Gamma=3, the drift intervals and walker contrast [-0.183874, 0.212769] remain too broad. A zero-containing interval alone does not qualify convergence.

Both fields rapidly settle visually, without the persistent population offsets of Gamma=0.05. Gamma=5 is the best candidate for longer projection and a larger-population check. The all-X variational bounds are -324 and -540; estimates lie below those upper bounds, which does not establish absolute accuracy. The second-order reference is approximate.

The original archive records production_allowed=false because the screen protocol did not make a production decision. That flag is a historical protocol decision, not evidence of a failed Gamma=5 convergence check.

On review, Gamma=5 with 10,000 walkers meets the tested energy stability criterion at 0.001 per spin. Energy production is justified for this tested L6/sample0/field setting, using 32 equilibration blocks followed by at least 32 measurement blocks of duration 0.2, eight independent populations, the fixed solver and the flat guide. With no forward cycles, splitting the tested trajectory into equilibration and measurement changes which blocks enter the estimator but does not change the dynamics. The terminal 32-block mean is -552.470854 ± 0.010837 SE, or -5.115471 ± 0.000100 per spin. Longer or larger-population runs would increase confidence; the present data do not demonstrate a Gamma=5 energy failure requiring them before energy production.

The energy-only archives contain no q2 or susceptibility measurements because n_forward_cycles=0. Consequently they provide no direct test of the susceptibility estimator's forward duration or lineage diagnostics. This is a specific observable-validation gap, not a demonstrated error in the stronger-field energies. A susceptibility production run must retain achieved forward times and lineage diagnostics and assess dependence on forward time. The fields remain candidate paramagnetic controls; energy stability has not established the phase. The Gamma=5 decision does not extend to the earlier weak-field regime or the entire historical scan.
