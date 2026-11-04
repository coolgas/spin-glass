# Higher-accuracy worldline comparison

Cancelled at user request on 2026-11-02 at 15:41:49 UTC. The controller and all 18 workers exited; 0/80 tasks completed and no per-task NPZ results were saved. Logs, source, manifest and reference files are preserved remotely. The AutoDL instance remains running. See cancellation.json.

80 runs launched on AutoDL, 18 workers, without a background SSH monitor. See launch.json for the remote directory. Results and comparison plots are generated remotely at completion; retrieval is manual.

Four matched L=2 contexts: (Γ,disorder)=(0.25,50025),(0.25,50059),(0.5,50043),(0.75,50004). All use h=0.15, 80 exchange temperatures, and four independent ladders initialized random/up/down/random. This is targeted validation, not a disorder-averaged phase scan.

- Γ=0.25: β=64,128,256.
- Γ=0.5: β=128,256,512.
- Γ=0.75: β=16,32,64.
- Temperature scans: Δτ=0.0625,16,000 warmup sweeps and64,000 measurement sweeps.
- Finest-temperature discretization audit: Δτ=0.03125 with the same sampling length.
- Finest-setting sampling audit:128,000 measurement sweeps.
- 64 contiguous blocks saved; report bootstrap errors using1,2,4 merged blocks and early/late spectra. These remain diagnostics, not automatic convergence certification.

The comparison uses the existing corrected guided-GFMC production runs on precisely these disorders, including all repeats and alarms. Their guides enumerate states; agreement with them does not itself establish scalable accuracy. Finite-temperature and ground-state references are evaluation-only. They are built from64 low-lying eigenpairs with maximum residual below10^-9 and an explicit upper bound below10^-9 on omitted thermal probability, using the last retained eigenvalue and the number of omitted states. The worldline sampler never reads these references.

The previous finest worldline spectra deviate from their thermal targets by at most2.75%,1.50%,0.90%,0.68% in the four contexts, respectively. This explains why their ground-state discrepancies must not be interpreted directly as sampling failures. At Γ=0.5,disorder50043, the gap is0.02848; β=32 leaves a substantial first-excited-state population. These findings motivate the colder runs.

Pilot-based estimate: approximately350 CPU-hours, or20–30 wall-clock hours on18 workers. This is approximate and includes no guarantee of adequate mixing at the new temperatures. Cost comparisons must count all independent ladders, and distinguish GFMC projection time from guide-fitting time.
