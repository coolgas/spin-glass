# RBM/GFMC repair: numerical diagnosis

Run on AutoDL, 2026-11-03. Physical ensemble: periodic kagome, h=0.15, K iid U(0.5,1.5), J=-K. The retained disorder is 60000, fixed before inspection; L=6 means N=108.

The original failed uniform-start task is reproduced from the identity-derived RNG seed, original model checkpoint, and original arithmetic. Its initial energy shift is approximately -5.20e6. All terminal log weights lie around -519800, so ordinary double-precision weights round to zero. No negative weights or sign problem are needed to cause this failure.

Log-domain propagation completes that first block with ESS=3.65/512. This removes the arithmetic crash but does not repair the old guide. The very large transient mixed energy is retained, not interpreted as a ground-state estimate.

The optimizer ablation changes only Adam learning rate (.01 versus .001) on the same disorder, initialization, chain count, temperature ladder, and sweep budget. Strong-field validation is bounded to one prespecified disorder and is not a disorder-averaged physical conclusion. No legacy production output is relabeled as corrected.
