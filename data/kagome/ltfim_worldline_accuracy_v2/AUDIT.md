# Correctness audit during the accuracy campaign

The four worldline tests pass both locally (Python implementation) and on AutoDL (Numba implementation). The running sampling source and parameters were not changed.

Verified checks:

- Single-spin discrete-time partition function and magnetization agree with an independently constructed transfer matrix.
- Replica-exchange log acceptance equals the difference in target log weights.
- Temporal-cluster sampling reproduces a known periodic-chain distribution.
- The full sampler, including exchange, reproduces magnetizations, pair correlations and the thermodynamic energy estimator for a frustrated three-spin triangle. The reference enumerates4096 space-time configurations (three spins, four slices) only inside the test. Its energy is also checked against a numerical derivative of the independently computed log partition function.
- Previous finest L=2 spectra differ from evaluation-only finite-temperature spectral references by at most about2.74%,1.50%,0.90%,0.68% in the four selected contexts. These are observed biases, not guaranteed accuracy bounds. The reference omitted thermal weight is bounded using the last retained energy; eigenpair residuals are separately checked.

Unresolved:

- The colder study has not established equilibration, uncertainty coverage or ground-state convergence.
- The fixed80-temperature ladder was not tuned from pilot acceptance/transport measurements at the new cold settings. Endpoint round trips alone would not certify mixing.
- Four starts supply the disjoint-replica estimator; they do not supply four independent estimates of that complete estimator. Time-block bootstrap errors require block-length and independent-start scrutiny.
- The running implementation stores final blocks only at task completion. There are no per-task intermediate progress records or restart checkpoints; CPU activity alone cannot establish scientific progress.
- Longest-first scheduling improves the expected batch finishing time but delays early feedback. A staged approach would be better for algorithm development.
- All new accuracy runs are L=2. Their settings and cost cannot be assumed sufficient or affordable at L=4–10.
- The comparison uses enumerated-corrected guided GFMC. It is not an equal-cost benchmark of two validated scalable algorithms.

Conclusion: the sampled discrete-time target and implementation have meaningful independent checks. The full ground-state finite-size study remains unvalidated. Use the current batch to identify the least expensive settings that achieve the required accuracy; do not copy its largest β and finest time step indiscriminately to larger systems.
