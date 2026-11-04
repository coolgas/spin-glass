# L4 triangle RG diagnostic pilot

Five schedules completed on AutoDL on 4 November 2026. The tested fixed
low-energy schedule fails the physical accuracy screen; expansion is stopped.
See [assessment](results/ASSESSMENT.md) and
[results notebook](../../../notebook/triangle_effective_l4_results.ipynb).

`results/` contains the original launch/status records, initial validation,
post-pilot hierarchy validation, five case records and compressed raw states,
isometries, spectra, physical samples, and baseline comparison snapshots.
`campaign.log` preserves pilot stdout. Baseline raw files were verified on
AutoDL; their JSON records and regenerated physical bonds are archived here.
The full baseline raw campaign remains on AutoDL.

`provenance/pilot/code/` preserves the two pilot modules matching the original
`results/launch.json` hashes exactly. The benchmark module was reconstructed
from the later source by removing only the added hierarchy-validation functions
and their call; its SHA-256 matches the launch record. The RG module is unchanged.
`provenance/current/code/` preserves the later benchmark with additional
hierarchy validation, the unchanged RG module, and the report generator.
The hierarchy check ran separately after the pilot; it was not part of the
original launch. `delivery_sha256.json` records the delivered file hashes.

The pilot used a single CPU process with BLAS/OMP threading limited to one.
Scope: periodic N=48, disorder60000, Gamma=.55, h=.15, orientation0, origin(0,0).
Five schedules: (4,4), (4,8), (4,16), (6,8), (8,8); two eigensolver starts each;
16,384 independent physical-spin replica pairs per schedule.
The full validation grids, full L4 grid and L6 were not launched.

The .001 energy discrepancy per spin is a chosen screening tolerance. GFMC
references are estimates rather than exact N48 ground truth. Reported timings
do not establish a speedup at matched physical accuracy.
