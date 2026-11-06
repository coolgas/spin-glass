# Dense L6 field scan — confirmed 2026-11-05

**Completed at 04:09:31 UTC: all 456 cases qualified, no failures or
exclusions, all 57 fields have eight qualified matched disorders.**
The controller took 562.85 seconds. New compressed raw files total about
0.734 GiB and remain remote. Verification details are in `verification.json`.

Selected scope: L=6, m=6, chi=3, Gamma=0.100:0.025:1.500, eight matched
disorders 60000–60007. There are 456 field/disorder cases: 40 hash-verified
archived points reused by reference, and 416 new solves. K~Uniform(0.5,1.5),
h=0.15 and lambda=1 remain unchanged.

Remote campaign root:
`/root/autodl-tmp/triangle-L6-m6-c3-dense-20261105-v1/`

Local notebook (repository-relative):
`notebook/triangle_L6_m6_chi3_dense.ipynb`

Never download or stage notebooks in this data directory. Keep raw states,
isometries, original bonds, physical moments, solver outputs and failures
remote. Download only small evidence and plot previews here.

The original two-start retained-space Lanczos solver and diagnostic thresholds
are unchanged. Both-start qEA, connected susceptibility and raw q2 must agree
within 0.001 absolute / 1% relative. Failed diagnostics remain excluded.
Means require all eight qualified disorders at a field. Uncertainty is solely
whole-disorder bootstrap; individual curves have no error bars.

The costly L8 workers and D1024 diagnostic were stopped by user direction
at 03:55:10 UTC and their exit verified. Saved checkpoints and failures remain
remote. The previous three-size/high-retention campaign must not resume
without fresh authorization. The unrelated overlap campaign was preserved.
