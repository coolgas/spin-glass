# Convergence and disorder campaign

Authorized scope: expand disorder averaging and resolve GFMC convergence before
interpreting the size trend. AutoDL root:
`/root/autodl-tmp/correlation-lengths-20261103-v3`.

The existing H800 inference process belongs to another active task. This campaign
uses 16 single-thread CPU workers out of the 20-core allocation. Exact L2 solves
and GFMC with tabulated neural amplitudes are CPU workloads. The instance must
remain running; no power operations are included.

## Stage 1: exact disorder ensemble

128 fixed bond IDs, 50000–50127; L=2; h=0.15; Γ=0.25, 0.5, 0.75, 1, 1.5, 2, 3.
896 independent exact contexts. The initial eight IDs are a subset, allowing a
direct comparison with the previous notebook. Report bootstrap errors on ratios
of disorder-averaged spectra, prefix-size checks, and leave-one-disorder-out
sensitivity. No disorder is excluded because its correlations are unusually large.

## Stage 2: development convergence

16 fixed disorders, 50000–50015, including all previously difficult samples.
Γ=0.5, 0.75, 1 at h=0.15. Four independent population repeats per setting.
16 forward cycles. Six settings:

| Setting | Walkers | Equilibration | Forward time | Branch interval |
|---|---:|---:|---:|---:|
| base | 2048 | 16 | 16 | 0.1 |
| population | 4096 | 16 | 16 | 0.1 |
| equilibration | 2048 | 32 | 16 | 0.1 |
| forward | 2048 | 16 | 32 | 0.1 |
| cadence | 2048 | 16 | 16 | 0.05 |
| combined | 4096 | 32 | 32 | 0.05 |

Sparse deterministic propagation of the neural guide isolates finite-projection
bias for the same cycles and tag times. It is compared with exact ground-state
spectra before interpreting stochastic comparisons.

Initial development tolerances: absolute error bound 0.01 for ξ/(2L), and 5%
relative bounds for all four spectra. Use approximate pointwise 95% bounds,
keeping the momentum covariance. Resample independent repeats conditional on
the fixed disorders to assess numerical bias. Disorder uncertainty is reported
separately. Four repeats provide only limited tail precision; broad uncertainty
is unresolved, not acceptance. Ancestry alarms are retained and flagged.

These tolerances are initial research accuracy targets, not a phase-boundary
resolution claim. Passing a development comparison does not automatically qualify
production. Require stronger plateaus, alternative starts and held-out exact
checks, then verify convergence anew at larger sizes.

## Following stages

Review stage 2 before choosing stronger settings. Add third time/population
levels where needed, rather than silently accepting a noisy plateau. Validate on
additional exact disorders. Use L=4,6,8 convergence pilots with measured GPU cost
and currently available GPU capacity. Only then start at least 64 disorders per
size/context, expanding to 128 or more where uncertainty requires it. Cover the
intermediate-Γ peak and both flanks. No automatic promotion to production is made
by this initial campaign.

## Reproducibility

`manifest.json` fixes source hashes, checkpoint identity, settings, IDs and seeds.
The remote `code/` snapshot is immutable after launch. Completed records are
atomic and reused only when the signature matches; interrupted whole runs are
recomputed. `*_status.json` records progress. `exact_summary.json` contains the
expanded exact curve. `convergence_review.json` and `CONVERGENCE_REVIEW.md` are
produced by the separate assessment script when complete groups are available.
