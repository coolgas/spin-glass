# Guided-GFMC overlap grid

N=12, L=2 kagome PBC, fixed disorder 50000. The 49 contexts use four independent
runs each: W=512 four-replica walkers, equilibration time 4, energy measurement time 1,
forward time 2, eight cycles and branch interval 0.1. All run and cycle histograms
are retained. Histogram normalization and first, second and fourth moments were
validated against scalar tags in all 1,568 cycles.

The comparison uses exact ground-state probabilities, the frozen unprojected guide,
guided GFMC with one-run-mean standard errors, and a deterministic infinite-population
reference at the same finite projection times. The same disorder is used for every
curve. There is no disorder averaging in this figure. No convergence claim is made.

2/196 runs triggered an ancestry alarm.
Local lookup-GFMC propagation took 245.61 seconds,
excluding deterministic references and plotting; this is not a direct-network GPU cost.

Figure: guided_gfmc_full_grid.png / guided_gfmc_full_grid.pdf.
Tables: guided_gfmc_grid_histograms.csv / guided_gfmc_grid_diagnostics.csv.
Notebook: ../../notebook/exact_guide_overlap_distributions.ipynb.
Frozen inputs and source: ../../data/kagome/guided_overlap_grid_v1/.
