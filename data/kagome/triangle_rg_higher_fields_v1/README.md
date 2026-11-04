# Spatial RG comparison with the L4 higher-field figure

The target is the section “L=4, N=48: higher-field distributions” in
`notebook/jastrow_rbm_overlap_distributions.ipynb`. Its frozen source is
`data/kagome/ltfim_disorder_scaling_v1/broader_histograms/20261103T035100Z_preview`.
The comparison uses exactly the measured curves in that snapshot, including
its incomplete RBM coverage at higher fields.

The finite calculation uses five fields (.60, .65, .70, .85, 1.00), eight
disorders (60000–60007), h=.15, periodic kagome L4, orientation0, origin(0,0).
All 40 contexts use (m,chi)=(6,8). Five additional contexts at disorder60000
use (8,8) to check removal of triangle truncation. Each calculation uses two
eigensolver starts and 16,384 independent physical replica pairs. Physical
overlap samples are joint Born draws through the hierarchy. The original
notebook's 112 frozen reference raw histogram hashes were verified locally
before launching the comparison.

Computation and report generation run on AutoDL in
`/root/autodl-tmp/triangle-rg-higher-fields-20261104-v1`, with a single CPU
worker and BLAS/OMP threads limited to one. The GPU is not used.

`results/` contains the generated report and notebook, five-field comparison
figure, assessment JSON, launch and status, and raw case archives. Original
inputs, complete source snapshot and campaign stdout accompany the results.
GFMC uncertainties and alarms remain part of the comparison: the references
are not exact N48 ground truth. Retention/partition convergence and finite-size
scaling are not established by this calculation.
