# Uniform seven-field corrected-guide benchmark

L2 only: guide optimization enumerates all 4096 states; scalability is unproven

4000 hierarchical bootstrap draws for plotted errors; fixed-disorder whole-repeat bootstrap for approximate pointwise 95% numerical checks. Four repeats limit tail resolution.

Reference agreement on all 64 disorders per field; population/time audits on eight prespecified disorders per field. No large-size qualification.

Validated 2240/2240 runs; 7/7 complete 64-disorder fields.

| Γ | Exact ξ | GFMC ξ | Disorder + MC SE | Alarms /256 | Check |
|---:|---:|---:|---:|---:|---|
| 0.25 | 0.208550 | 0.202973 | 0.050616 | 0 | unresolved |
| 0.5 | 0.432746 | 0.429570 | 0.049411 | 1 | unresolved |
| 0.75 | 0.516459 | 0.515279 | 0.017422 | 0 | passes L2 ensemble and audit checks |
| 1.0 | 0.438570 | 0.438811 | 0.012147 | 0 | passes L2 ensemble and audit checks |
| 1.5 | 0.277351 | 0.277304 | 0.003103 | 0 | passes L2 ensemble and audit checks |
| 2.0 | 0.207594 | 0.207569 | 0.001580 | 0 | passes L2 ensemble and audit checks |
| 3.0 | 0.140578 | 0.140475 | 0.000977 | 0 | passes L2 ensemble and audit checks |

## Disorder-prefix stability

Each prefix uses the first n matched disorders, with spectra averaged before computing the length. Prefixes are nested; their errors are correlated.

| Γ | Disorders | Exact ξ ± SE | GFMC ξ ± SE |
|---:|---:|---:|---:|
| 0.25 | 8 | 0.079998 ± 0.014870 | 0.080494 ± 0.015296 |
| 0.25 | 16 | 0.165850 ± 0.064126 | 0.165971 ± 0.064054 |
| 0.25 | 32 | 0.179850 ± 0.032263 | 0.170392 ± 0.028782 |
| 0.25 | 64 | 0.208550 ± 0.048126 | 0.202973 ± 0.050616 |
| 0.5 | 8 | 0.365727 ± 0.065607 | 0.365711 ± 0.065400 |
| 0.5 | 16 | 0.534040 ± 0.116034 | 0.535299 ± 0.116575 |
| 0.5 | 32 | 0.450303 ± 0.078635 | 0.450893 ± 0.079143 |
| 0.5 | 64 | 0.432746 ± 0.048398 | 0.429570 ± 0.049411 |
| 0.75 | 8 | 0.422407 ± 0.032368 | 0.422710 ± 0.032653 |
| 0.75 | 16 | 0.464831 ± 0.022278 | 0.465315 ± 0.022352 |
| 0.75 | 32 | 0.470205 ± 0.017784 | 0.470274 ± 0.017784 |
| 0.75 | 64 | 0.516459 ± 0.017328 | 0.515279 ± 0.017422 |
| 1.0 | 8 | 0.415883 ± 0.016224 | 0.415966 ± 0.016397 |
| 1.0 | 16 | 0.429237 ± 0.012228 | 0.429698 ± 0.012407 |
| 1.0 | 32 | 0.429698 ± 0.008863 | 0.429799 ± 0.008912 |
| 1.0 | 64 | 0.438570 ± 0.012125 | 0.438811 ± 0.012147 |
| 1.5 | 8 | 0.282489 ± 0.006612 | 0.282526 ± 0.006599 |
| 1.5 | 16 | 0.284616 ± 0.004961 | 0.284805 ± 0.005012 |
| 1.5 | 32 | 0.281258 ± 0.003300 | 0.281328 ± 0.003290 |
| 1.5 | 64 | 0.277351 ± 0.003107 | 0.277304 ± 0.003103 |
| 2.0 | 8 | 0.211732 ± 0.003758 | 0.211547 ± 0.003750 |
| 2.0 | 16 | 0.213339 ± 0.002811 | 0.213365 ± 0.002853 |
| 2.0 | 32 | 0.211582 ± 0.001920 | 0.211623 ± 0.001923 |
| 2.0 | 64 | 0.207594 ± 0.001582 | 0.207569 ± 0.001580 |
| 3.0 | 8 | 0.142878 ± 0.002508 | 0.142980 ± 0.002562 |
| 3.0 | 16 | 0.143895 ± 0.001798 | 0.143887 ± 0.001815 |
| 3.0 | 32 | 0.143410 ± 0.001298 | 0.143273 ± 0.001286 |
| 3.0 | 64 | 0.140578 ± 0.000990 | 0.140475 ± 0.000977 |

## Population and time contrasts

Approximate pointwise 95% upper bounds on the absolute change, conditional on the eight prespecified audit disorders. Spectrum bounds are relative to the matched exact spectra. Passing these contrasts also requires the separate reference, drift and ancestry checks recorded in validated_review.json.

| Γ | Audit | Bound on Δξ/(2L) | Largest relative spectral bound | Pass |
|---:|---|---:|---:|---|
| 0.25 | population | 0.000298 | 0.003103 | True |
| 0.25 | time | 0.000395 | 0.003609 | True |
| 0.5 | population | 0.000482 | 0.006937 | True |
| 0.5 | time | 0.000490 | 0.006211 | True |
| 0.75 | population | 0.000492 | 0.006172 | True |
| 0.75 | time | 0.000903 | 0.010665 | True |
| 1.0 | population | 0.000314 | 0.003763 | True |
| 1.0 | time | 0.000431 | 0.003902 | True |
| 1.5 | population | 0.000252 | 0.001728 | True |
| 1.5 | time | 0.000237 | 0.001691 | True |
| 2.0 | population | 0.000196 | 0.001108 | True |
| 2.0 | time | 0.000239 | 0.001247 | True |
| 3.0 | population | 0.000174 | 0.000794 | True |
| 3.0 | time | 0.000234 | 0.001168 | True |
