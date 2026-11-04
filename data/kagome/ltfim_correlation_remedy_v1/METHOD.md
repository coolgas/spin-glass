# Low-field guided-GFMC remedy

## Diagnosis

The archived seven-field benchmark used the same frozen neural guide with joint
four-replica branching. At Gamma=0.25, 25 of 32 runs had ancestry alarms and the
eight-disorder length was 2.71096 rather than the matched exact 0.08000.
The frozen guide's squared ground-state overlap ranged from 0.00656 to 0.66095
on these eight disorders. Initial energy variance, fidelity, and gaps are in
`initial_guide_diagnosis.json`. This was not resolved by simply filling the field
grid. The original data remain in `../ltfim_correlation_lengths_v2/`.

## What changes

1. Optimize a positive, per-Hamiltonian correction to the frozen neural guide:
   `log psi_corrected(s) = log psi_neural(s) + sum_A theta_A product_{i in A} s_i`,
   using all subsets with one through four sites (793 coefficients at N=12).
   Minimize the variational energy, successively adding degrees 2, 3, and 4.
   The implementation receives the Hamiltonian and neural amplitudes, **not exact
   eigenvectors, energies, or spectra**. Degree, iteration cap, and optimizer
   tolerances are frozen before held-out validation. Exact references are loaded
   or computed only after fitting, for evaluation. Optimizer termination details
   are saved, including any iteration-limit terminations.
2. Branch four physical replica populations independently using separate random
   streams. For each population a, descendant-weighted moments are
   `m_i^a=<s_i>` and `Q_ij^a=<s_i s_j>`. Define
   `A_ij^(ab)=(Q_ij^a+Q_ij^b-m_i^a m_j^b-m_i^b m_j^a)/2`.
   Average `A^(01) A^(23)`, `A^(02) A^(13)`, and `A^(03) A^(12)` before Fourier
   reduction. Each product uses disjoint independent populations. This avoids
   squaring a noisy covariance estimated from one population and averages all
   cross-population combinations rather than a single arbitrary pairing.
   Finite-population and projection biases still require checks.

## Evidence

- Development disorders: 50000–50007, Gamma=0.25, 0.5, 0.75.
- Corrected-guide tests: 384 runs; four settings, four repeats per disorder.
- Base: W1024 per replica, equilibration16, forward16, branch0.1, 16 cycles.
- Population: W2048; time: equilibration32/forward32; alternative start: uniform
  spins with equilibration32/forward16. All other settings match base.
- Exact comparisons, population/time/start contrasts, and paired early/late-half
  diagnostics all pass the predefined approximate pointwise 95% bounds:
  absolute 0.01 for xi/(2L), relative 5% for each of the four spectra.
- An additional 24 ablation runs compare the original-guide independent estimator
  and corrected-guide joint estimator on three difficult contexts. Those results
  are diagnostics, not replacement production points.
- Held-out disorders: 50016–50023, excluded from remedy selection. The same frozen
  optimization recipe and base/time settings give 192 further runs; exact, time,
  and paired-half checks all pass. No held-out exact reference is used to tune
  the correction.
- There are **zero ancestry alarms in all 576 corrected-guide validation runs**.
  The 24 ablations include 12 original-guide runs, one of which has an alarm;
  this run is retained.
- Bootstrap resampling uses whole independent runs within fixed disorders for
  numerical tests. Plot uncertainties additionally resample disorders. Four
  repeats give limited tail resolution; these are not simultaneous confidence
  guarantees.

## Displayed replacement

The notebook uses the **time setting for all three low fields**, uniformly:
W1024 per independent replica, equilibration32, forward32, branch0.1, 16 cycles,
four repeats, and the same eight matched disorders as the original plot.
`before_after.csv`, `corrected_comparison.csv`, and `selection.json` record the
choice and provenance. The original higher-field points retain their archived
settings and are separately labelled. Both requested plots are retained.

## Scope

This resolves the observed L2 low-field failure to the stated numerical
tolerances. Guide optimization uses all 4096 configurations as an L2 accelerator;
it is **not a demonstrated scalable solution at L4–10**. Neither the corrected
guide nor the larger-size feasibility points establish thermodynamic spin-glass
behavior. Larger systems need a sampled, scalable guide-refinement implementation
and their own population/projection/disorder validation.

## Reproduction

From the repository root:

```sh
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest transformer_qmc.test_correlation_remedy -v
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m transformer_qmc.assess_correlation_remedy data/kagome/ltfim_correlation_remedy_v1/results
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m transformer_qmc.validate_correlation_remedy --output data/kagome/ltfim_correlation_remedy_v1/heldout --assess-only
python3 code/transformer_qmc/report_correlation_lengths.py
```

Remote source snapshots:
`/root/autodl-tmp/correlation-remedy-20261102T070107Z` and
`/root/autodl-tmp/correlation-remedy-validation-20261102T070834Z`.
The older v4 escalation jobs were paused during these tests and resumed
automatically afterward. No instance power operations were performed.

Related methodological background: Boninsegni and Moroni, *Population size bias
in Diffusion Monte Carlo*, arXiv:1209.1663; Overy et al., *Unbiased Reduced Density
Matrices and Electronic Properties from Full Configuration Interaction Quantum
Monte Carlo*, arXiv:1410.6047. These motivate population checks and independent
replicas; the estimator implemented here is separately validated by explicit
four-replica enumeration and dense Fourier contractions.
