# L4 interaction screen before finite-size scaling

Completed complementary diagnostic. All 64 raw archives and 48 L4 physical
histograms are verified locally; see `results/local_verification.json`.
The self-contained notebook is also installed at
`notebook/triangle_effective_interaction_scan.ipynb` in the repository.
The primary continuation now lowers both endpoints of every bond law;
see the sibling `triangle_rg_bond_ranges_v1/` delivery.

Goal: identify candidate weak-intertriangle-coupling intervals where spatial
RG is stable under retention changes and physical P(q) changes with transverse
field. The eventual targets are finite-size scaling of q_EA and physical
connected correlation length, after validating the candidate regime.

This 4-by-4 panel layout follows the selected comparison in
`notebook/exact_guide_overlap_distributions.ipynb`. That original figure is
L2/N12; this scan is L4/N48. Rows are Gamma=.25,.5,1,2, columns are
lambda=0,.1,.3,1; h=.15; disorder60000; periodic kagome; orientation0;
superblock origin(0,0). Only intertriangle bonds are multiplied by lambda.
Original disorder arrays and complete modified coupling arrays are saved.

Three retained schedules (4,8),(6,8),(6,16) give 48 L4 calculations, two
Lanczos starts each, and 8192 independent physical replica pairs each in two
batches. An exact L2 companion covers all 16 contexts at m=4,6,8, with
independent physical-action checks and exact recovery at m=8. Full-spin L4
P(q) is exact at lambda=0 using product-triangle convolution. At lambda>0
L4 has retention comparisons; matched full-spin benchmarks remain necessary.

Scientific computation and plotting run on AutoDL at
`/root/autodl-tmp/triangle-rg-interaction-20261104-v1`, one CPU worker with
BLAS/OMP threads limited to one. The source snapshot, launch manifest,
stdout, raw states, isometries, eigenspectra, seeds, physical samples and
failures accompany the delivery. The GPU is not used.

`results/assessment.json` contains per-panel exact L2 errors and L4 retention
sensitivity, provisional candidate flags, and paired field-shift measurements.
Candidate flags are a diagnostic screen, not a physical-accuracy certification.
The field-induced mean overlap is not automatically a thermodynamic q_EA.
P(q) can move even at the noncollective decoupled endpoint. No transition or
finite-size-scaling result follows from this scan alone.
