# L6 convergence campaign status

Completed successfully on 2026-10-26 at **01:48:59 UTC**, after 1,539.80 seconds (25.66 minutes). Launched at 01:23:18 UTC on AutoDL, parent PID 748619, 16 workers. Completion and full outputs are recorded in `exit.json`, `final.json`, and `REPORT.md`.

- Increase projection: eight 5,000-walker populations, **20,480 blocks**, tau=4096, with exact prefix checks against the 10,240-block archives.
- Increase walkers separately: eight populations each at 10,000 and 20,000 walkers, 2,560 blocks, tau=512; compare identical time windows with the saved 5,000-walker campaign.
- Keep the guide flat, two replicas, Gamma=0.05, sample0, L6, sigma=1, and unchanged solver. Local and remote observation smoke checks reproduced all 21 output fields exactly, including an actual partial-write boundary.

Independent calculation completed: the classical ground-state energy is -143.53434217871674 (MILP primal/dual gap zero). A normalized product state has energy -143.63655720218088, so the quantum ground-state energy satisfies **E0 <= -143.63655720218088**. The quantum energy is also bounded below by -148.9343421787168 from the classical dual bound minus N*Gamma. These bounds have solver floating-point precision.

The previous time-2048 population means, -142.204572 to -133.743151, are all above this variational upper bound. The lowest plateau is about 1.43 above it, and the population average is about 4.80 above it. Thus stationarity of those traces does not establish an accurate ground-state energy. The bound is a reference calculation and is not used as a guiding function or to modify any trajectory.

All **113 remote artifacts (4,997,494 bytes)** have been downloaded and SHA256-verified locally. Local smoke outputs were preserved; differing remote smoke artifacts are under `remote_diagnostics/`. `download_manifest.json` maps each downloaded path to its remote path. `verify_archives.py` independently verified all 24 complete solver archives, all eight exact 10,240-block energy/time prefixes, all 24 saved partial traces, every late-window and walker contrast, and direct evaluation of the saved normalized product state. The reproducible result is in `local_verification.json`.

The final 5,000-walker mean over blocks 15,360–20,480 (time 3072–4096) is **-139.092153 ± 0.860096 SE**, with descriptive 95% interval [-141.125957, -137.058349]. The primary late drift is -0.128846 ± 0.074120 SE, interval [-0.304112, 0.046420], which extends outside the working ±0.108 total-energy tolerance. The final population means range from -142.190422 to -134.592748. The mean remains 4.544404 above the variational upper bound, and even the lowest population remains 1.446135 above it. Doubling the blocks lowers some trajectories further but does not qualify convergence.

At matched time 384–512, the mean energies at 5,000, 10,000 and 20,000 walkers are -137.938079, -137.252572 and -137.428585, with population SEs 0.785477, 0.864599 and 0.625156. Walker contrast intervals are broad and none fits inside ±0.108. There is no clear improvement at this time; the shorter larger-population runs do not establish their long-time behavior.

The explanation notebook now includes the completed block/walker comparisons, hash verification, exact prefixes, direct product-state evaluation and plots. All 11 code cells executed successfully in a fresh kernel, and the notebook passed format validation.

**Production remains unqualified:** `production_allowed=false`, `recommended_equilibration_blocks=null`. The next useful diagnostic is dependence on independently prepared initial populations while retaining the flat guide, with longer matched-time walker comparisons where warranted. Fresh-seed and representative disorder/field confirmation and separate susceptibility-forward-time checks remain necessary before production.
