## Ancestry diagnosis and matched field controls

Snapshot: 2026-11-03T09:32:03.276006+00:00. Broad production dispatch is paused; previously submitted tasks may finish. The previous 4–6 day completion estimate is suspended.

Completed production GFMC records: 44; records with ancestry alarms: 19. Alarms are retained, not excluded from the ensemble.

For L6, Gamma=0.55, disorder60000, the following values are computed from saved raw descendant counts. ESS near 1–2 refers to minima over populations/cycles, not every measurement.

| Guide | Audit | Repeat | Walkers | Forward time | Median weight ESS | Median descendant ESS | Minimum descendant ESS | Alarmed population/cycles |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| jastrow | main | 0 | 512 | 16 | 506.8 | 19.49 | 8.44 | 0/32 |
| jastrow | main | 1 | 512 | 16 | 506.8 | 19.86 | 10.75 | 0/32 |
| jastrow | population | 0 | 1024 | 16 | 1013.5 | 39.20 | 13.53 | 0/32 |
| jastrow | population | 1 | 1024 | 16 | 1013.5 | 36.96 | 32.59 | 0/32 |
| jastrow | start | 0 | 512 | 16 | 506.7 | 19.64 | 13.71 | 0/32 |
| jastrow | start | 1 | 512 | 16 | 506.8 | 20.20 | 4.78 | 1/32 |
| jastrow | time | 0 | 512 | 32 | 506.8 | 10.06 | 6.02 | 6/32 |
| jastrow | time | 1 | 512 | 32 | 506.8 | 11.71 | 5.03 | 6/32 |
| rbm_2 | main | 0 | 512 | 16 | 501.7 | 10.05 | 2.05 | 11/32 |
| rbm_2 | main | 1 | 512 | 16 | 501.7 | 10.73 | 2.35 | 11/32 |
| rbm_2 | population | 0 | 1024 | 16 | 1003.1 | 14.49 | 1.54 | 5/32 |
| rbm_2 | population | 1 | 1024 | 16 | 1002.9 | 15.18 | 6.67 | 6/32 |
| rbm_2 | start | 0 | 512 | 16 | 501.5 | 7.57 | 1.31 | 17/32 |
| rbm_2 | start | 1 | 512 | 16 | 501.6 | 9.63 | 1.32 | 12/32 |
| rbm_2 | time | 0 | 512 | 32 | 501.6 | 4.17 | 1.20 | 30/32 |

The RBM has high one-step weight ESS but substantial cumulative ancestry loss during forward walking. Increasing forward time worsens ancestry diversity; doubling the walker count has not removed all alarms. This supports a sampling-efficiency diagnosis, not a phase identification or a proof that every observable is wrong.

**Field hypothesis:** larger Gamma may improve tunnelling and mixing, but it also changes the physical state and requires a field-specific guide. Ancestry collapse alone cannot establish a deep spin-glass phase or locate a transition.

**Active finite diagnostic:** two CPU workers, ten prespecified runs, no wall-clock cutoff. Fixed L6, N108, h=.15, periodic kagome, disorder60000, RBM seed0 and LR .001. At each Gamma=.55,.60: 512 walkers, forward times 8 and 16, two repeats; additionally Gamma=.55 uses 2048 walkers at forward time16, two repeats. Equilibration16 and eight cycles are retained. Seed0 has completed repaired checkpoints at both fields. Existing selected-seed production controls are preserved separately.

Field-specific guides have different optimization histories and quality, so this is an operational field comparison, not isolation of the Hamiltonian effect. Shorter forward time can reduce ancestry loss while increasing projection bias. Compare P(q), energy, connected spectra, independent repeats and ancestry jointly; no automatic production promotion or convergence claim follows from this diagnostic.

L4/L6 training-budget and learning-rate sensitivity remain pending. L8/L10 expansion remains gated on demonstrated L6 reliability.

Remote diagnostic: `/root/autodl-tmp/ancestry-diagnostic-20261103-v1`.
