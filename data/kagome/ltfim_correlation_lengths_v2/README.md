# Correlation-length comparisons

The executed analysis is in `notebook/ltfim_correlation_lengths.ipynb`.
Lengths use the connected overlap spectrum, the continuum smallest-momentum
denominator, and nearest-neighbor spacing as the length unit. The Kagome torus
has physical side `2L` and `N=3L²` sites.

Rebuild from the repository root:

```sh
PYTHONPATH=code python3 -m transformer_qmc.correlation_lengths
python3 code/transformer_qmc/report_correlation_lengths.py
```

The first command resumes the cached local L2 benchmark. It verifies source and
input identities before reusing runs. The second command validates the saved
exact references, reduces the raw spectra, and executes the notebook.

- `l2_gfmc/`: 96 independent population runs at L2, h=0.15,
  Γ=0.5/1/2, eight matched disorders, four repeats per disorder. The neural guide
  is evaluated through its archived exact lookup table. Settings are recorded in
  the manifest; these runs are not a general ground-state convergence qualification.
- `l2_comparison.csv`: ratios of disorder-averaged spectra, with paired nested
  bootstrap uncertainty for GFMC minus exact. Ancestry alarms remain in the data.
- `l2_raw_spectra.csv`: four raw spectral entries for each exact context or GFMC run.
- `l2_by_disorder.csv`: diagnostic lengths obtained separately for each disorder.
  These are not averaged to form the reported disorder-ensemble length.
- `size_comparison.csv` and `size_raw_spectra.csv`: saved L4–10 feasibility runs
  at Γ=0.5, h=0.15, one disorder and one float32 run per size, with identical settings
  across sizes. Their very short projection times are not qualified for physical
  size scaling. Sampling and disorder errors cannot be estimated from one run.
- `analysis_provenance.json`: settings, source hashes, and hashes of input records.
- `l2_finite_time_check.json`: supplementary infinite-population calculation at
  Γ=0.5 using sparse imaginary-time propagation of the archived neural guide.
  For each forward cycle the normalized single-replica probability is
  proportional to `exp(-t_tag H) psi_G * exp(-t_forward H) psi_G`. Connected
  spectra are computed for that probability, then averaged over cycles and
  disorders before reducing to a length. This isolates finite-time effects
  from population noise; it is not an additional plot in the notebook.

Invalid spectral ratios are retained as missing values. Bootstrap invalid
fractions are recorded; finite-value standard errors alone do not resolve an
invalid point estimate. The notebook contains two figures and does not infer a
phase from the preliminary size series.
