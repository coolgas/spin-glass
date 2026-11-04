# Scalability of the corrected guide

The current degree-four correction is an L=2 benchmark. **It is not scalable as implemented.** It minimizes Hamiltonian energy without exact-state training labels, but evaluates that objective over every spin configuration. Avoiding exact labels does not avoid exponential computation.

For N=3L² spins, the implementation enumerates 2^N configurations and fits every spin monomial of degrees one through four: P=Σ(k=1…4) binomial(N,k). The dense feature matrix contains 2^N × P numbers.

| L | N | Configurations | Correction coefficients |
|---:|---:|---:|---:|
| 2 | 12 | 4,096 | 793 |
| 4 | 48 | 281,474,976,710,656 | 213,052 |
| 6 | 108 | 2^108 | 5,569,137 |
| 8 | 192 | 2^192 | 56,050,288 |
| 10 | 300 | 2^300 | 335,291,425 |

At L=4, even one float64 amplitude per configuration needs about 2.25 PB; the present dense feature matrix would need about 480 EB. Sampling the energy would remove enumeration, but the unrestricted degree-four expansion still grows as N^4. Each disorder and field also requires its own optimization. This benchmark therefore does not demonstrate generalization by the frozen neural guide.

## Sampled neural-refinement pilot

A separate pilot refined the 169,673-parameter neural guide with independent autoregressive samples and a covariance energy gradient. Training used batch128 and Adam learning rate0.00015 on the AutoDL H800. Neither training run enumerated states or used exact labels. L=2 enumeration was performed only afterward for evaluation.

| Context | Steps | Seconds/step | Result |
|---|---:|---:|---|
| L=2, Γ=0.25, disorder50004 | 200 | 0.0404 | Post-training squared fidelity0.892691; variational energy bias+0.140530 |
| L=4, Γ=0.5, disorder50004 | 100 | 0.1406 | Sampled energy−50.6620±0.0410 → −51.4415±0.0513 |

The L=2 variational relative spectral biases (S0, Sk1, Sk2, Sk3) were approximately (22.13, 14.17, 15.39, 17.43), meaning errors of 1,417–2,213%. These are variational observables, not results from a GFMC projection of the refined guide. At L=4 the estimated local-energy variance increased from2.94494 to3.95666, and peak allocated GPU memory was262,472,192 bytes (250.3 MiB). Lower energy alone did not establish accurate correlations.

These short, single-disorder pilots establish computational feasibility at L=4 only. They do not determine asymptotic runtime, convergence with training length, or GFMC population requirements. The two contexts have different fields and training lengths, so their timings are not a controlled scaling fit.

## What remains to establish

A usable larger-system method needs sampled guide optimization, with shared neural parameters or a restricted correction whose cost remains manageable. It must first reproduce L=2 overlap spectra on difficult held-out disorders, then pass independent-population GFMC checks for population, equilibration, forward time, ancestry and statistical uncertainty at larger L. Disorder sampling and size dependence must then be tested independently. An accurate 64-disorder L=2 field curve cannot establish that the correlation length grows with size.

Raw pilot measurements and the coefficient counts are in [scalability_pilot](scalability_pilot/). The uniform seven-field campaign and its numerical audits use the enumerated correction throughout.

## Expanded-disorder limitation

The completed 64-disorder campaign exposes failures even at L=2. At Γ=0.25, disorder50025 has squared guide fidelity4.90×10^-10 and approximately−84% spectral bias after projection; disorder50059 has fidelity4.21×10^-11 and approximately+152% spectral bias. Neither has an ancestry alarm. The eight-disorder audit retains14–21% spectral error despite stable population/time contrasts. At Γ=0.5, individual spectral errors and one ancestry alarm also remain. The two low-field ensemble points are therefore unresolved. The earlier successful small-ensemble validation does not extend uniformly to the enlarged disorder set.
