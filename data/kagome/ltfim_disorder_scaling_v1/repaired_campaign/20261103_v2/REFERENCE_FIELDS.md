# L6 lower-field and reference comparison

Verified snapshot: 2026-11-03T13:38:07.202535+00:00. No campaign is qualified by this report.

| Campaign | Training milestones | GFMC runs | Ancestry alarms | Task failures |
|---|---:|---:|---:|---:|
| l6-convergence-20261103-v1 | 21 | 29 | 29 | 0 |
| extended-rbm-20261103-v1 | 4 | 7 | 7 | 0 |
| ancestry-diagnostic-20261103-v1 | — | 6 | 5 | 0 |
| l6-reference-fields-20261103-v1 | 0 | 0 | 0 | 0 |

Reference scope: Gamma .85/1.00; disorders60000/60001; both RBM seeds;
400/1600/4000 target updates; at most 48 GFMC runs. All other jobs continue.
Compare with lower-field .55/.60/.65/.70 at fixed h=.15. Existing Gamma4 controls
remain in the notebook. Reference fields are not assumed to be outside a glassy regime.

| Gamma | Disorder | Seed | Training budget path | Repeat | Energy/site | q2 | ESS min / median | Max lineage | Alarm |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.65 | 60000 | 0 | projector/L6_s60000_g0p65_rbm_2_anneal_seed0/extra_0 | 0 | -1.168243 | 0.33342 | 1.38 / 4.32 | 0.844 | True |
| 0.65 | 60000 | 0 | projector/L6_s60000_g0p65_rbm_2_anneal_seed0/extra_0 | 1 | -1.167742 | 0.35332 | 1.12 / 4.82 | 0.945 | True |
| 0.65 | 60000 | 1 | projector/L6_s60000_g0p65_rbm_2_anneal_seed1/extra_0 | 1 | -1.173130 | 0.33788 | 1.19 / 6.39 | 0.914 | True |
| 0.65 | 60000 | 1 | projector/L6_s60000_g0p65_rbm_2_anneal_seed1/extra_0 | 0 | -1.173171 | 0.33259 | 1.80 / 5.64 | 0.736 | True |
| 0.60 | 60000 | 0 | projector/L6_s60000_g0p6_rbm_2_anneal_seed0/extra_0 | 0 | -1.143040 | 0.42094 | 1.50 / 5.52 | 0.812 | True |
| 0.60 | 60000 | 0 | projector/L6_s60000_g0p6_rbm_2_anneal_seed0/extra_0 | 1 | -1.143028 | 0.43184 | 1.75 / 5.75 | 0.727 | True |
| 0.60 | 60000 | 1 | projector/L6_s60000_g0p6_rbm_2_anneal_seed1/extra_0 | 1 | -1.148507 | 0.39478 | 2.39 / 7.03 | 0.629 | True |
| 0.60 | 60000 | 1 | projector/L6_s60000_g0p6_rbm_2_anneal_seed1/extra_0 | 0 | -1.148503 | 0.40440 | 1.17 / 7.96 | 0.922 | True |
| 0.70 | 60000 | 0 | projector/L6_s60000_g0p7_rbm_2_anneal_seed0/extra_0 | 1 | -1.193767 | 0.28702 | 1.98 / 4.77 | 0.678 | True |
| 0.70 | 60000 | 0 | projector/L6_s60000_g0p7_rbm_2_anneal_seed0/extra_0 | 0 | -1.194800 | 0.28428 | 1.00 / 4.22 | 1.000 | True |
| 0.70 | 60000 | 1 | projector/L6_s60000_g0p7_rbm_2_anneal_seed1/extra_0 | 0 | -1.199166 | 0.29185 | 1.22 / 4.27 | 0.902 | True |
| 0.70 | 60000 | 1 | projector/L6_s60000_g0p7_rbm_2_anneal_seed1/extra_0 | 1 | -1.199068 | 0.29418 | 1.27 / 4.16 | 0.885 | True |
| 0.55 | 60001 | 0 | projector/L6_s60001_g0p55_rbm_2_anneal_seed0/extra_0 | 1 | -1.123191 | 0.41963 | 1.52 / 6.21 | 0.807 | True |
| 0.55 | 60001 | 0 | projector/L6_s60001_g0p55_rbm_2_anneal_seed0/extra_0 | 0 | -1.123313 | 0.42242 | 1.89 / 4.31 | 0.717 | True |
| 0.55 | 60001 | 1 | projector/L6_s60001_g0p55_rbm_2_anneal_seed1/extra_0 | 1 | -1.115300 | 0.36133 | 1.32 / 4.26 | 0.867 | True |
| 0.55 | 60001 | 1 | projector/L6_s60001_g0p55_rbm_2_anneal_seed1/extra_0 | 0 | -1.115368 | 0.32334 | 1.15 / 5.52 | 0.934 | True |
| 0.55 | 60001 | 1 | projector/L6_s60001_g0p55_rbm_2_anneal_seed1/extra_1200 | 1 | -1.115670 | 0.33059 | 1.06 / 4.72 | 0.971 | True |
| 0.55 | 60001 | 1 | projector/L6_s60001_g0p55_rbm_2_anneal_seed1/extra_1200 | 0 | -1.115357 | 0.34474 | 1.39 / 8.52 | 0.846 | True |
| 0.65 | 60001 | 0 | projector/L6_s60001_g0p65_rbm_2_anneal_seed0/extra_0 | 1 | -1.172417 | 0.32123 | 1.12 / 4.75 | 0.943 | True |
| 0.65 | 60001 | 0 | projector/L6_s60001_g0p65_rbm_2_anneal_seed0/extra_0 | 0 | -1.172388 | 0.33137 | 1.25 / 5.09 | 0.893 | True |
| 0.65 | 60001 | 1 | projector/L6_s60001_g0p65_rbm_2_anneal_seed1/extra_0 | 0 | -1.165920 | 0.25073 | 1.09 / 4.13 | 0.959 | True |
| 0.65 | 60001 | 1 | projector/L6_s60001_g0p65_rbm_2_anneal_seed1/extra_0 | 1 | -1.166063 | 0.24745 | 1.18 / 3.14 | 0.918 | True |
| 0.60 | 60001 | 0 | projector/L6_s60001_g0p6_rbm_2_anneal_seed0/extra_0 | 1 | -1.147044 | 0.35772 | 1.26 / 5.50 | 0.885 | True |
| 0.60 | 60001 | 0 | projector/L6_s60001_g0p6_rbm_2_anneal_seed0/extra_0 | 0 | -1.147028 | 0.37450 | 1.43 / 4.35 | 0.828 | True |
| 0.60 | 60001 | 1 | projector/L6_s60001_g0p6_rbm_2_anneal_seed1/extra_0 | 1 | -1.140087 | 0.30084 | 1.24 / 4.02 | 0.896 | True |
| 0.60 | 60001 | 1 | projector/L6_s60001_g0p6_rbm_2_anneal_seed1/extra_0 | 0 | -1.139928 | 0.30602 | 1.06 / 4.64 | 0.969 | True |
| 0.70 | 60001 | 0 | projector/L6_s60001_g0p7_rbm_2_anneal_seed0/extra_0 | 1 | -1.199064 | 0.27062 | 1.16 / 3.63 | 0.926 | True |
| 0.70 | 60001 | 0 | projector/L6_s60001_g0p7_rbm_2_anneal_seed0/extra_0 | 0 | -1.199210 | 0.27448 | 1.08 / 3.38 | 0.963 | True |
| 0.70 | 60001 | 1 | projector/L6_s60001_g0p7_rbm_2_anneal_seed1/extra_0 | 0 | -1.194242 | 0.19384 | 1.10 / 3.21 | 0.953 | True |
| 0.55 | 60000 | 0 | results/projector/seed0_extra0 | 0 | -1.120318 | 0.48144 | 1.71 / 6.39 | 0.760 | True |
| 0.55 | 60000 | 0 | results/projector/seed0_extra0 | 1 | -1.120198 | 0.49238 | 1.37 / 6.67 | 0.854 | True |
| 0.55 | 60000 | 0 | results/projector/seed0_extra1200 | 0 | -1.120315 | 0.48420 | 3.28 / 10.60 | 0.539 | True |
| 0.55 | 60000 | 0 | results/projector/seed0_extra1200 | 1 | -1.120239 | 0.48055 | 2.28 / 10.98 | 0.654 | True |
| 0.55 | 60000 | 0 | results/projector/seed0_extra3600 | 0 | -1.120376 | 0.46491 | 1.36 / 6.20 | 0.857 | True |
| 0.55 | 60000 | 1 | results/projector/seed1_extra0 | 0 | -1.125728 | 0.44994 | 1.45 / 10.43 | 0.824 | True |
| 0.55 | 60000 | 1 | results/projector/seed1_extra0 | 1 | -1.125734 | 0.44220 | 1.85 / 7.53 | 0.727 | True |
| 0.55 | 60000 | 0 | ancestry-diagnostic-20261103-v1/results/projector | 0 | -1.120311 | 0.48365 | 3.34 / 22.41 | 0.540 | True |
| 0.55 | 60000 | 0 | ancestry-diagnostic-20261103-v1/results/projector | 1 | -1.120309 | 0.47713 | 3.04 / 13.48 | 0.566 | True |
| 0.55 | 60000 | 0 | ancestry-diagnostic-20261103-v1/results/projector | 0 | -1.120299 | 0.47507 | 2.29 / 6.63 | 0.648 | True |
| 0.55 | 60000 | 0 | ancestry-diagnostic-20261103-v1/results/projector | 1 | -1.120429 | 0.46970 | 2.00 / 13.20 | 0.705 | True |
| 0.55 | 60000 | 0 | ancestry-diagnostic-20261103-v1/results/projector | 1 | -1.120293 | 0.48433 | 1.10 / 6.13 | 0.955 | True |
| 0.55 | 60000 | 0 | ancestry-diagnostic-20261103-v1/results/projector | 0 | -1.120141 | 0.49498 | 8.19 / 16.03 | 0.289 | False |

Raw SHA256, histogram normalization/moments, finite ancestry and alarm consistency
were checked for every listed run. JSON retains full histograms, connected spectra,
energy, provenance and ancestry quantiles/fractions; raw arrays remain in their campaign paths.
Alarms and seed/repeat differences prevent a physical interpretation until resolved.
Two repeats and two fixed disorders give limited precision. No automatic promotion,
phase label or transition estimate follows from higher-field improvement.
