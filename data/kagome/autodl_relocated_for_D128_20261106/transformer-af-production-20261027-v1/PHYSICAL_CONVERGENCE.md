# Physical-observable convergence review

2026-10-28T19:22:29.570504+00:00

**Development evidence; ground-state qualification remains pending.**

The matched comparisons use L=4 (48 spins), PBC, disorder 60000 and four independent whole-population repeats at each setting. All ancestry alarms are retained. A repeat is not a new disorder; four forward cycles within a run are correlated and are not used as independent error samples.

`q_ea_finite` is the signed two-replica overlap mean, targeting N^-1 sum_i <Zi>^2 after pure projection. It is not sqrt(q2). The unique finite-size symmetric ground state has zero site magnetization at h=0; at h>0 this estimator includes field-induced polarization. No clipping or absolute-value conversion is applied.

The notebook prescribes |difference| + 1.96 SE <= tolerance for E/N (0.001), q2 (0.002), and chi_connected/N (0.002). Tables also retain Welch t intervals because four repeats give uncertain variance estimates. A large interval or insignificant difference does not demonstrate equivalence. No tolerance has been prespecified for q4, q_ea_finite or nonzero-momentum spectra; those remain descriptive.

Only complete sets of four repeats enter means and contrasts. Differences compare independent runs at two settings on the same Hamiltonian; no trajectory pairing is assumed. These pointwise comparisons measure MC uncertainty only, and do not estimate disorder variability.

## Guided coverage

| Gamma | h | Setting | Completed repeats | Alarms | Worst ESS | Worst ESS/W | Largest lineage share |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.5 | 0.0 | baseline | 4/4 | 3 | 3.16538 | 0.049459 | 0.546875 |
| 0.5 | 0.0 | population256 | 4/4 | 1 | 7.84111 | 0.0306293 | 0.320312 |
| 0.5 | 0.0 | population1024 | 4/4 | 0 | 19.5426 | 0.0190845 | 0.169922 |
| 0.5 | 0.0 | equilibration4 | 4/4 | 0 | 24.3889 | 0.0238173 | 0.173828 |
| 0.5 | 0.0 | forward2 | 4/4 | 1 | 7.47467 | 0.00729948 | 0.313477 |
| 0.5 | 0.05 | baseline | 4/4 | 4 | 2.7234 | 0.0425532 | 0.59375 |
| 0.5 | 0.05 | population256 | 4/4 | 2 | 1.61658 | 0.00631475 | 0.785156 |
| 0.5 | 0.05 | population1024 | 4/4 | 0 | 21.5783 | 0.0210726 | 0.179688 |
| 0.5 | 0.05 | equilibration4 | 4/4 | 0 | 13.9691 | 0.0136417 | 0.25 |
| 0.5 | 0.05 | forward2 | 4/4 | 2 | 7.1582 | 0.00699043 | 0.358398 |
| 2.0 | 0.0 | baseline | 4/4 | 0 | 15.876 | 0.248062 | 0.171875 |
| 2.0 | 0.0 | population256 | 4/4 | 0 | 75.156 | 0.293578 | 0.0429688 |
| 2.0 | 0.0 | population1024 | 4/4 | 0 | 312.448 | 0.305125 | 0.0126953 |
| 2.0 | 0.0 | equilibration4 | 4/4 | 0 | 299.936 | 0.292906 | 0.0175781 |
| 2.0 | 0.0 | forward2 | 4/4 | 0 | 171.056 | 0.167047 | 0.0195312 |
| 2.0 | 0.05 | baseline | 4/4 | 0 | 14.6286 | 0.228571 | 0.171875 |
| 2.0 | 0.05 | population256 | 4/4 | 0 | 63.0154 | 0.246154 | 0.0703125 |
| 2.0 | 0.05 | population1024 | 4/4 | 0 | 277.108 | 0.270613 | 0.015625 |
| 2.0 | 0.05 | equilibration4 | 4/4 | 0 | 272.357 | 0.265974 | 0.0166016 |
| 2.0 | 0.05 | forward2 | 4/4 | 0 | 149.583 | 0.146077 | 0.0322266 |

## Guided physical contrasts

| Gamma | h | Change | Observable | Stronger minus candidate | MC SE | Normal 95% absolute bound | Tolerance | Assessment |
|---:|---:|---|---|---:|---:|---:|---:|---|
| 0.5 | 0.0 | baseline → population256 | energy_per_site | -0.00297771 | 0.00303775 | 0.00893171 | 0.001 | unresolved_precision_or_shift |
| 0.5 | 0.0 | baseline → population256 | q2 | 0.0285208 | 0.0253565 | 0.0782195 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.0 | baseline → population256 | q4 | 0.00658719 | 0.0108035 | 0.0277621 | pending | no_prespecified_tolerance |
| 0.5 | 0.0 | baseline → population256 | q_ea_finite | 0.0804138 | 0.173566 | 0.420604 | pending | no_prespecified_tolerance |
| 0.5 | 0.0 | baseline → population256 | chi_connected_per_site | -0.00975037 | 0.127186 | 0.259035 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.0 | population256 → population1024 | energy_per_site | -0.000637088 | 0.00156355 | 0.00370165 | 0.001 | unresolved_precision_or_shift |
| 0.5 | 0.0 | population256 → population1024 | q2 | 0.074405 | 0.0643739 | 0.200578 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.0 | population256 → population1024 | q4 | 0.0392485 | 0.0334044 | 0.104721 | pending | no_prespecified_tolerance |
| 0.5 | 0.0 | population256 → population1024 | q_ea_finite | 0.174164 | 0.17956 | 0.526102 | pending | no_prespecified_tolerance |
| 0.5 | 0.0 | population256 → population1024 | chi_connected_per_site | -0.0741005 | 0.0966161 | 0.263468 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.0 | population1024 → equilibration4 | energy_per_site | -0.00332784 | 0.00162534 | 0.00651351 | 0.001 | unresolved_precision_or_shift |
| 0.5 | 0.0 | population1024 → equilibration4 | q2 | -0.105686 | 0.0658721 | 0.234795 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.0 | population1024 → equilibration4 | q4 | -0.047769 | 0.0334686 | 0.113368 | pending | no_prespecified_tolerance |
| 0.5 | 0.0 | population1024 → equilibration4 | q_ea_finite | -0.0546392 | 0.14041 | 0.329843 | pending | no_prespecified_tolerance |
| 0.5 | 0.0 | population1024 → equilibration4 | chi_connected_per_site | -0.0467792 | 0.0459024 | 0.136748 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.0 | population1024 → forward2 | energy_per_site | 0.00155288 | 0.00144084 | 0.00437692 | 0.001 | unresolved_precision_or_shift |
| 0.5 | 0.0 | population1024 → forward2 | q2 | -0.0193365 | 0.0765121 | 0.1693 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.0 | population1024 → forward2 | q4 | -0.0113935 | 0.0391575 | 0.0881421 | pending | no_prespecified_tolerance |
| 0.5 | 0.0 | population1024 → forward2 | q_ea_finite | -0.00316366 | 0.17269 | 0.341636 | pending | no_prespecified_tolerance |
| 0.5 | 0.0 | population1024 → forward2 | chi_connected_per_site | -0.0360207 | 0.0417161 | 0.117784 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.05 | baseline → population256 | energy_per_site | -0.00707417 | 0.00243232 | 0.0118415 | 0.001 | resolved_shift_beyond_tolerance |
| 0.5 | 0.05 | baseline → population256 | q2 | 0.0559586 | 0.0499061 | 0.153775 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.05 | baseline → population256 | q4 | 0.0160475 | 0.0156515 | 0.0467245 | pending | no_prespecified_tolerance |
| 0.5 | 0.05 | baseline → population256 | q_ea_finite | 0.241811 | 0.129707 | 0.496036 | pending | no_prespecified_tolerance |
| 0.5 | 0.05 | baseline → population256 | chi_connected_per_site | -0.0223596 | 0.0240147 | 0.0694285 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.05 | population256 → population1024 | energy_per_site | -7.05619e-05 | 0.00168032 | 0.003364 | 0.001 | unresolved_precision_or_shift |
| 0.5 | 0.05 | population256 → population1024 | q2 | 0.0144561 | 0.0531296 | 0.11859 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.05 | population256 → population1024 | q4 | 0.00772512 | 0.0170329 | 0.0411097 | pending | no_prespecified_tolerance |
| 0.5 | 0.05 | population256 → population1024 | q_ea_finite | -0.128395 | 0.150225 | 0.422835 | pending | no_prespecified_tolerance |
| 0.5 | 0.05 | population256 → population1024 | chi_connected_per_site | -0.00283358 | 0.0151331 | 0.0324945 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.05 | population1024 → equilibration4 | energy_per_site | -0.00514575 | 0.000923834 | 0.00695647 | 0.001 | resolved_shift_beyond_tolerance |
| 0.5 | 0.05 | population1024 → equilibration4 | q2 | 0.0143467 | 0.0370655 | 0.0869951 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.05 | population1024 → equilibration4 | q4 | 0.00539518 | 0.0136207 | 0.0320917 | pending | no_prespecified_tolerance |
| 0.5 | 0.05 | population1024 → equilibration4 | q_ea_finite | -0.0160065 | 0.208231 | 0.42414 | pending | no_prespecified_tolerance |
| 0.5 | 0.05 | population1024 → equilibration4 | chi_connected_per_site | 0.023306 | 0.019972 | 0.0624511 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.05 | population1024 → forward2 | energy_per_site | -0.00289644 | 0.00141169 | 0.00566336 | 0.001 | unresolved_precision_or_shift |
| 0.5 | 0.05 | population1024 → forward2 | q2 | 0.052156 | 0.0398399 | 0.130242 | 0.002 | unresolved_precision_or_shift |
| 0.5 | 0.05 | population1024 → forward2 | q4 | 0.0188442 | 0.0161859 | 0.0505685 | pending | no_prespecified_tolerance |
| 0.5 | 0.05 | population1024 → forward2 | q_ea_finite | 0.070816 | 0.210634 | 0.483659 | pending | no_prespecified_tolerance |
| 0.5 | 0.05 | population1024 → forward2 | chi_connected_per_site | 0.0780563 | 0.0817943 | 0.238373 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.0 | baseline → population256 | energy_per_site | -0.000802782 | 0.000124454 | 0.00104671 | 0.001 | unresolved_precision_or_shift |
| 2.0 | 0.0 | baseline → population256 | q2 | 0.00188658 | 0.0024119 | 0.0066139 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.0 | baseline → population256 | q4 | 0.000606474 | 0.000384662 | 0.00136041 | pending | no_prespecified_tolerance |
| 2.0 | 0.0 | baseline → population256 | q_ea_finite | -0.0139669 | 0.0127194 | 0.038897 | pending | no_prespecified_tolerance |
| 2.0 | 0.0 | baseline → population256 | chi_connected_per_site | 0.000747257 | 0.0034564 | 0.0075218 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.0 | population256 → population1024 | energy_per_site | 0.000366233 | 0.000154288 | 0.000668637 | 0.001 | within_pointwise_tolerance |
| 2.0 | 0.0 | population256 → population1024 | q2 | -0.00157324 | 0.00119908 | 0.00392344 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.0 | population256 → population1024 | q4 | -0.000429222 | 0.000308226 | 0.00103335 | pending | no_prespecified_tolerance |
| 2.0 | 0.0 | population256 → population1024 | q_ea_finite | -0.00226593 | 0.00237264 | 0.0069163 | pending | no_prespecified_tolerance |
| 2.0 | 0.0 | population256 → population1024 | chi_connected_per_site | -0.00101418 | 0.00111335 | 0.00319634 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.0 | population1024 → equilibration4 | energy_per_site | -0.00011956 | 0.000128223 | 0.000370878 | 0.001 | within_pointwise_tolerance |
| 2.0 | 0.0 | population1024 → equilibration4 | q2 | 0.000610458 | 0.000892917 | 0.00236058 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.0 | population1024 → equilibration4 | q4 | 0.000155707 | 0.000161914 | 0.000473059 | pending | no_prespecified_tolerance |
| 2.0 | 0.0 | population1024 → equilibration4 | q_ea_finite | -0.00041453 | 0.00158967 | 0.00353028 | pending | no_prespecified_tolerance |
| 2.0 | 0.0 | population1024 → equilibration4 | chi_connected_per_site | 0.000463274 | 0.000869809 | 0.0021681 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.0 | population1024 → forward2 | energy_per_site | -0.000289739 | 0.000147907 | 0.000579636 | 0.001 | within_pointwise_tolerance |
| 2.0 | 0.0 | population1024 → forward2 | q2 | 0.000529289 | 0.000481163 | 0.00147237 | 0.002 | within_pointwise_tolerance |
| 2.0 | 0.0 | population1024 → forward2 | q4 | 0.000163874 | 0.000103307 | 0.000366355 | pending | no_prespecified_tolerance |
| 2.0 | 0.0 | population1024 → forward2 | q_ea_finite | 0.00102997 | 0.00134277 | 0.0036618 | pending | no_prespecified_tolerance |
| 2.0 | 0.0 | population1024 → forward2 | chi_connected_per_site | 0.000307189 | 0.00104136 | 0.00234826 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.05 | baseline → population256 | energy_per_site | -0.000806126 | 0.000483408 | 0.0017536 | 0.001 | unresolved_precision_or_shift |
| 2.0 | 0.05 | baseline → population256 | q2 | -0.00126436 | 0.00178364 | 0.0047603 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.05 | baseline → population256 | q4 | -0.000262946 | 0.000333314 | 0.00091624 | pending | no_prespecified_tolerance |
| 2.0 | 0.05 | baseline → population256 | q_ea_finite | 0.0105082 | 0.00941675 | 0.0289651 | pending | no_prespecified_tolerance |
| 2.0 | 0.05 | baseline → population256 | chi_connected_per_site | -0.000592126 | 0.00223309 | 0.00496899 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.05 | population256 → population1024 | energy_per_site | 0.000125359 | 0.00020896 | 0.000534921 | 0.001 | within_pointwise_tolerance |
| 2.0 | 0.05 | population256 → population1024 | q2 | -4.04782e-05 | 0.00104688 | 0.00209237 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.05 | population256 → population1024 | q4 | 9.24516e-05 | 0.000148596 | 0.0003837 | pending | no_prespecified_tolerance |
| 2.0 | 0.05 | population256 → population1024 | q_ea_finite | -0.00987752 | 0.0030214 | 0.0157995 | pending | no_prespecified_tolerance |
| 2.0 | 0.05 | population256 → population1024 | chi_connected_per_site | 0.00182576 | 0.00107136 | 0.00392563 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.05 | population1024 → equilibration4 | energy_per_site | -4.21992e-05 | 0.000214 | 0.000461639 | 0.001 | within_pointwise_tolerance |
| 2.0 | 0.05 | population1024 → equilibration4 | q2 | -0.00028091 | 0.000811988 | 0.00187241 | 0.002 | within_pointwise_tolerance |
| 2.0 | 0.05 | population1024 → equilibration4 | q4 | 4.47495e-06 | 0.000108492 | 0.00021712 | pending | no_prespecified_tolerance |
| 2.0 | 0.05 | population1024 → equilibration4 | q_ea_finite | -0.000928243 | 0.00245205 | 0.00573426 | pending | no_prespecified_tolerance |
| 2.0 | 0.05 | population1024 → equilibration4 | chi_connected_per_site | -0.000390477 | 0.000769091 | 0.00189789 | 0.002 | within_pointwise_tolerance |
| 2.0 | 0.05 | population1024 → forward2 | energy_per_site | 1.22345e-05 | 0.000211481 | 0.000426738 | 0.001 | within_pointwise_tolerance |
| 2.0 | 0.05 | population1024 → forward2 | q2 | -0.0014338 | 0.000815741 | 0.00303265 | 0.002 | unresolved_precision_or_shift |
| 2.0 | 0.05 | population1024 → forward2 | q4 | -0.000228356 | 0.000133645 | 0.000490299 | pending | no_prespecified_tolerance |
| 2.0 | 0.05 | population1024 → forward2 | q_ea_finite | -0.00204722 | 0.00240887 | 0.0067686 | pending | no_prespecified_tolerance |
| 2.0 | 0.05 | population1024 → forward2 | chi_connected_per_site | 0.000283241 | 0.000936546 | 0.00211887 | 0.002 | unresolved_precision_or_shift |

## Next convergence work

Available guided contrasts with declared tolerances: 9 within the approximate pointwise bound, 2 with a resolved shift beyond tolerance, and 37 unresolved.

Finish incomplete time contrasts before choosing settings. Where a shift is resolved, strengthen that parameter and obtain a third level before claiming a plateau. Where uncertainty is too broad, add independent repeats or longer validated measurements. Longer forward time must be judged together with ESS/W and lineage concentration.

Formal qualification also needs the notebook development-disorder policy, third time levels, branching-cadence, alternative-start and combined checks, and a measured feasible budget. Current one-disorder data cannot establish robustness or qualify the full field grid. Define q4 and q_ea_finite tolerances prospectively before using them in an acceptance gate.

CSV exports include every connected-spectrum momentum divided by N, with original momentum ordering (my,mx); k=0 duplicates chi_connected/N. Nonzero-momentum shifts remain descriptive. Ratios and finite-size scaling need qualified disorder averages and covariance propagation.

## Continuation cost

Window: 24.0 hours. Initial work estimate: 17.3 GPU-hours. Newly completed pending tasks: 30/30.

Completed continuation task segments total 17.974 GPU-worker hours. This excludes unfinished segments and operational/analysis overhead; resumed task totals also include time before this window.
