# L6 pilot, 2026-11-20, v1

24 finite cases, sample 0, L=6 (108 physical spins, nine complete 12-spin blocks).
Three masks, detunings 2 and 3.15, and production-grid indices 0, 17 and 29.
18 cases use chi_RG=12 with direct D=128 and the same two-start acceptance protocol
as L4. Six intermediate-drive cases use the qualification D ladder 64,128,256;
the existing solver stops advancing a start when its D-comparison passes.
Every case uses 20,000 replica pairs and up to 120 sweeps with unchanged tolerances.
Projected full-vector exact diagonalization is disabled for nine chi12 blocks.

The fixed-production flag specifies the L4 solver protocol only; this is an L6
pilot, not a production ensemble or an RG convergence claim. One geometry cannot
establish disorder statistics. No RG-chi convergence is claimed.

The Ali validation environment is Python 3.15.2, NumPy 2.5.4, SciPy 1.18.1 and
TeNPy 1.1.1. It passed all 12 inherited tests but differs from L4's historical
environment. Record it with every pilot; do not attribute differences solely to L.

Keep all model/MPS checkpoints, both starts, D-ladder states, moments and failure
evidence during the pilot. Checkpoints occur at completed D stages. Plan up to
1 GiB output per case and keep at least 16 GiB disk headroom. Runs are outside Git.
No automatic trimming or retries. The detached controller can adopt live workers.

Initial admission reserves 8 GiB per worker for D256 growth plus 16 GiB system/other
job headroom, and accounts for CPU affinity, cgroup ancestry, memory availability,
other load and disk space. Use one BLAS/OpenMP thread per case. Reassess this
forecast from live RSS and completed-case peak memory; it is not a fixed pilot cap.
