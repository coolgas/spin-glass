# Completed L6 auxiliary-field pilot

Temporary pinning shows no reliable benefit across the three classical
references in this pilot. One released reference has a small final guide
energy advantage, but all three released guides have higher local-energy
variance than the unpinned baseline. Projection screens do not establish
accurate projected observables or a consistent improvement across references.

All seven branches completed 2400 training updates, including 800 updates
without pinning for the release branches. All 28 frozen diagnostics and
fourteen scheduled projection runs completed without recorded task failures.
The worker finished at 2026-11-04T04:31:17Z and the audited final report at
04:32:11Z. This is one periodic L6 disorder (60000), N=108, one training
seed (0), Gamma=0.6 and h=0.15. All measurements use the original Hamiltonian.

| Released guide | Final VMC energy/site minus baseline | Final VMC variance ratio | Mean projection energy/site minus baseline |
|---|---:|---:|---:|
| Reference 0 | -0.0005033 | 1.093 | -0.0003424 |
| Reference 1 | +0.0096476 | 1.554 | +0.0059648 |
| Reference 2 | +0.0013236 | 1.258 | +0.0014157 |

Projection differences average the two independent runs for each guide and
the two baseline runs. They are finite-screen energy estimates with unresolved
population and projection-time bias. They are not converged ground-state
energy differences, and two repeats do not quantify systematic error.

Reference 0 has lower projection energy in both repeats: differences from
the corresponding baseline repeat are -0.0004057 and -0.0002791 per spin.
Its repeat agreement is better than the baseline on several observables,
but it narrowly fails the planned q-squared and histogram thresholds.
Its descendant ESS minima are 1.34 and 1.91 out of 512 walkers; its largest
lineage fractions reach 0.861 and 0.695. These remain severe genealogy alarms.

| Guide | Repeat energy/site difference | Repeat q-squared difference | Histogram total variation | Maximum relative spectrum difference | All repeat thresholds pass |
|---|---:|---:|---:|---:|---|
| Unpinned | 0.0001375 | 0.007642 | 0.069373 | 0.141750 | No |
| Release reference 0 | 0.0000110 | 0.010307 | 0.050232 | 0.039690 | No |
| Release reference 1 | 0.0000392 | 0.015872 | 0.091045 | 0.114585 | No |
| Release reference 2 | 0.0000351 | 0.016391 | 0.079570 | 0.076059 | No |

Thresholds are respectively 0.001, 0.01, 0.05 and 0.10, with the recorded
0.001 spectrum denominator floor. None of the seven guides, including the
three fixed-pinning controls, passes every repeat threshold. All fourteen
projection runs trigger genealogy alarms. Across runs, minimum descendant
ESS ranges from 1.00 to 1.91 and largest lineage fractions from 0.695 to 1.00.
Reproducible energy estimates therefore do not establish reproducible
ground-state overlap distributions or correlation spectra.

Frozen sampling from three start families gives final energy ranges at most
0.0002815 per spin and Born-histogram total variation at most 0.00614, while
the released guides retain reference-dependent Born histograms differing by
up to 0.1708 in total variation. These start checks support the internal VMC
comparison but cannot establish training saturation or missed-state coverage.

The selected-reference energy gain is a limited observation. Disorder and
training-seed variability, branch-selection uncertainty, longer refinement,
population/time convergence and ground-state accuracy remain unresolved.
No quantum-gap improvement has been measured. Raw arrays, checkpoints,
alarms, source provenance and independently reconstructed projection records
are retained with the final report and executed notebook.
