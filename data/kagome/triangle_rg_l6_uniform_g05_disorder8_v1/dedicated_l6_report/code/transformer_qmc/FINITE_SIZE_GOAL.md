# Scientific goal: field-tuned finite-size scaling at fixed longitudinal field

## Updated execution agreement, 2026-11-03

At fixed h=0.15 scan Gamma across sizes L=4,6,8,10, seeking crossings and a
consistent finite-size collapse of disorder-averaged xi_R/L and appropriately
defined overlap observables. A critical point or collapse is a hypothesis to
test, not a required fit outcome. q = mean_i <Zi>^2 contains field polarization;
do not fit it as a vanishing glass order parameter without treating that background.

The immediate sampled-guide pilot uses L=4,6, Gamma=0.60,0.65,0.70,0.85,1.00,
and development disorders 60000–60007. Two continuation seeds for each of Jastrow
and alpha=2 RBM receive 1200 updates per fit. Select by frozen sampled energy,
retaining both seeds. Four independent CT GFMC populations measure cross-replica
q, q2 and connected spectra. Eight cycles, 512 walkers, equilibration/forward
times 16 are a starting budget, not a qualified ground-state setting. Fixed
disorders 60000,60001 additionally test 1024 walkers, times 32, and uniform starts.
All settings have two repeats. Compare paired disorder averages and numerical
sensitivity, rather than requiring each sample to pass a universal threshold.

The pipeline is `disorder_scaling.py`; all compute and tests run on AutoDL.
The active continuation uses `parallel_disorder_scaling.py`: four independent
CUDA/MPS trainers and eight CPU projector workers, reusing the original source
snapshot, task signatures, checkpoints and 12-hour deadline. The size scope
remains L4/L6 pending review; L8/L10 have not been launched.
Ground-state qualification and collapse fitting remain disabled until the
numerical sensitivity and disorder coverage support them. L8/L10 follow the
larger-size feasibility/precision assessment. The cancelled worldline accuracy
campaign remains stopped; the historical dependency below is superseded.

## Earlier design and retained conventions

At fixed longitudinal field h=0.15, measure how R(L,Γ)=ξ_R(L,Γ)/L behaves across the accessible sizes L=4,6,8,10 and transverse fields Γ=0.25,0.5,0.75,1,1.5,2,3. L=2 is an exact-validation anchor. The primary outcome is a quantitative size trend with controlled uncertainty over this range; asymptotic saturation and a thermodynamic critical point are not required outcomes.

## Observable and convention

Cij=〈ZiZj〉−〈Zi〉〈Zj〉; S_R(k)=N^-1 Σij Cij² exp[ik·(ri−rj)]. Average spectra equally over quenched disorders before forming ξ_R. Use the mean of the three shortest reciprocal-direction spectra and |k_min|=2π/(√3 L). Retain invalid ratios as unresolved.

Report ξ_R in nearest-neighbor spacings and R=ξ_R/L, as requested. The physical side is2L, so this R is twice the previously reported ξ_R/(2L). This constant changes values and absolute tolerances, not trends or crossing locations. Label the convention on every plot; do not silently mix normalizations.

## Grid and execution order

- Target grid: four sizes × seven fields at h=0.15, periodic Kagome, N=3L², Kij uniform on[0.5,1.5].
- First establish a workable accuracy protocol at L=4, then L=6,8,10. Run across the field range; do not select only points showing growth or apparent crossings.
- Start with a small disorder pilot to estimate mixing, temperature requirements and variance. A pilot is not the final disorder ensemble.
- Use32 disorders per size as an initial production target; extend in prespecified batches toward64 or more when needed for the precision target and affordable. These counts are planning targets, not evidence of sufficient precision.
- Reuse each disorder realization across fields at fixed L, enabling paired field comparisons. The existing generator changes the Hamiltonian with L; equal seed IDs across sizes do not establish a physically paired disorder ensemble. Treat size ensembles independently unless a coupling construction is explicitly introduced.
- Ground-state convergence may require different β and sampling effort at different sizes and fields. If a point remains thermally unconverged, label it as finite-temperature/unresolved and retain it in the record. Do not silently use it as a ground-state point or drop difficult disorders.

## Practical accuracy targets

Initial planning targets, to be assessed against actual costs:

1. Combined disorder-plus-sampling95% interval half-width at most0.02 in R=ξ_R/L. Use absolute precision because relative errors become misleading near zero. Report interval methods and invalid-bootstrap fractions.
2. For changes in β, Δτ and sampling length, aim for approximate95% bounds on |ΔR| below0.01 and relative changes in each spectrum below5%. Stable ratios alone are insufficient; inspect spectra separately. These checks are empirical controls, not rigorous bounds on all remaining bias.
3. Check independent starts, early/late agreement, uncertainty stability against block length, exchange diagnostics and difficult-disorder behavior. Good exchange statistics alone do not certify equilibration.
4. Keep Monte Carlo uncertainty smaller than disorder uncertainty where practical. Use independent-run/time-block resampling within each disorder and whole-disorder resampling outside it, retaining momentum and field covariance. Do not count walkers, time slices or adjacent sweeps as independent samples.
5. Infer a resolved size trend only from an interval for the size difference itself. Include the numerical sensitivity alongside statistical uncertainty. If0.02 point precision cannot resolve the scientifically relevant differences, tighten it selectively or report an unresolved trend.

## Outputs and interpretation

Keep two main single-panel figures:

1. R=ξ_R/L versus Γ, one curve per L, with uncertainty and explicit unresolved-point markers.
2. R=ξ_R/L versus L, one curve per Γ, with the same qualification labels.

Report adjacent-size differences ΔR=R(L₂,Γ)−R(L₁,Γ) and their intervals in a compact table. Negative differences support decreasing normalized range over the measured interval; positive differences support increasing normalized range. A narrow interval near zero bounds the observed change; an interval including zero does not by itself establish size independence. Interpret the magnitude and precision together.

Decreasing R is consistent with a finite correlation range but does not prove eventual saturation. Approximately constant R with tight bounds supports roughly proportional growth of ξ_R over accessible sizes; it does not prove criticality. Apparent crossings may motivate denser field sampling, but no critical field or exponent should be extracted from an unsupported scaling fit. This connected-overlap observable alone does not classify magnetic or glassy order.

## Current dependency

The80-run higher-accuracy worldline study is testing four L=2 contexts at colder temperatures, finer time steps and longer sampling, with matched guided-GFMC and exact-reference comparisons. It informs the numerical protocol; it does not itself qualify L=4–10 or complete this finite-size grid. The previous L=4,6,10 worldline timing pilots are cost measurements only.

Remote execution preference: short launch/retrieval commands are authorized; no background polling monitor. Preserve raw results and report completed versus qualified coverage separately.
