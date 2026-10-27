# Frozen seed501 field/bond scan and GFMC

The shared symmetric physical ALiBi seed501 final remains the selected guide.
It won the predefined development criterion with score 0.291550, ahead of all
four overnight continuation finals. The checkpoint SHA-256 is
`5f9fa33b04102ddff2e43179440514021090ed3aaf228ce67dce53c2139e2544`.
All four sizes use exactly these parameters. Evaluation and GFMC construct no
optimizer and never use exact amplitudes as a guide.

## Frozen campaign

Remote source/results root:
`/root/autodl-tmp/transformer-af-ltfim-20261027-frozen-scan-v1/`.
The sources staged there are frozen. Local results are mirrored under
`data/kagome/transformer_af_frozen_scan_v1/`, and the executed report is
`notebook/shared_alibi_frozen_scan_gfmc.ipynb`.

The Hamiltonian is `sum K_ij ZiZj - Gamma sum Xi - h sum Zi`, with periodic
boundaries. K is the overall positive AF bond scale:
`K_ij = K*(1+0.5*u_ij)`, where each L/disorder ID has the same u across K.

| Parameter | Values |
|---|---|
| Gamma | 0.25, 0.5, 0.75, 1, 1.5, 2, 3 |
| h | 0, 0.15, 0.3, 0.6, 1 |
| K | 0.5, 1, 2 |
| L | 2, 4, 6, 8 |
| L2 disorder IDs | 10000–10007: 840 exact evaluations |
| L4/L6/L8 disorder IDs | 10000–10001: 630 sampled evaluations |

IDs 40000–40007 remain reserved. These are development scans; they do not
provide an independent final test. Only K=1 and the three anchor field pairs
(0.5,0), (0.5,0.3), (2,0.3) have direct training support.

Exact L2 records include energy, variance, fidelity, full connected correlation
matrices, susceptibility, magnetization and SG structure factors. Every exact
point checks normalization and eigensolver residual. The sampled larger-size
records include eight batches of 64 energy samples, independent four-replica
q2/connected-chi estimates, and additional sampled one/two-spin matrices.
The matrix estimates are descriptive; squared sample covariances are not used
as unbiased susceptibility estimates. Sample errors exclude guide bias and
two disorders do not establish disorder convergence.

## Completed exact results

The 840-point L2 stage and 14 exact projection references have completed.
Representative means over eight disorders:

| Gamma | h | K | Fidelity | Connected C RMS error | Absolute chi error |
|---|---|---|---:|---:|---:|
| 0.5 | 0 | 1 | 0.823343 | 0.239858 | 1.050146 |
| 0.5 | 0.3 | 1 | 0.701850 | 0.163106 | 0.687542 |
| 2 | 0.3 | 1 | 0.995486 | 0.016635 | 0.017438 |
| 0.25 | 0.3 | 1 | 0.255908 | 0.191504 | 0.659886 |
| 0.5 | 0.6 | 1 | 0.651330 | 0.154030 | 0.580631 |
| 0.5 | 0.3 | 0.5 | 0.802018 | 0.107446 | 0.434186 |
| 0.5 | 0.3 | 2 | 0.214381 | 0.191902 | 0.720846 |

At fixed Gamma/K and h/K, a common positive energy scaling leaves the exact
ground state invariant. All 240 available matched pairs pass: minimum exact
scale overlap 0.999999999999995; energies divided by K agree to the recorded
precision. Network scale overlap averages 0.939836 and falls to 0.647781.
The existing static network features include absolute energies, so scale
invariance is not enforced by architecture. This campaign measures that
transfer failure without changing the selected guide.

At K*tau=32, the exact projection references have fidelity above 0.99996 except
for disorder10001 at (Gamma,h,K)=(0.25,0.3,1), which reaches 0.997678 with
connected-C RMS error 0.013975. The reference stores pure projected energy and
the mixed energy appropriate to guide-weighted GFMC separately. Finite-size
exact projection does not establish stochastic or larger-size convergence.

## Selected-guide GFMC convergence study

The seven field/bond contexts are (Gamma,h,K):
(0.5,0,1), (0.5,0.3,1), (2,0.3,1), (0.25,0.3,1), (0.5,0.6,1),
(0.5,0.3,0.5), (0.5,0.3,2), on disorders10000 and10001.

For each context/disorder:

- W=512,2048,8192, crossed with forward K*tau=2,8,16.
- Branch K*dt=0.1, equilibration K*tau=16, energy measurement K*tau=4.
- At W8192 and forward16, check branch0.05 and separately equilibration8.
- Four independent populations, four physical replicas, 12 forward cycles.
- 154 settings and 616 population records in total.

The L2 lookup contains exactly the frozen network, never the exact reference
state. Time intervals are converted to physical time by dividing by K. All
cycles are retained. SE is across independent populations; low ancestry ESS,
unique ancestor counts and largest lineages are retained in every record.
Energy is a mixed estimator, while the diagonal replica observables use
forward walking. Stochastic energy is not variational.

The supervisor caps its campaign compute at eight hours and leaves completed
records resumable. This is a cap on processes, not instance power. Inspect
`supervisor.json` and the stage summaries to distinguish completion from a cap
or failure. The instance remains powered on.

## Larger-size direct inference

Float64 H800 measurements at Gamma0.5, K1:

| L | h | Configurations | All-spin flip-ratio wall seconds |
|---|---|---:|---:|
| 6 | 0.3 | 128 | 0.549 |
| 8 | 0.3 | 128 | 1.731 |
| 8 | 0 | 128 | 3.461 |

Actual continuous-time steps ran at L2/L4/L6/L8 with 16 four-replica walkers.
The h=0 probability mixture doubles network work. A separate direct L4 pilot
uses exact bounded memoization for repeated state queries, with a fresh cache
per Hamiltonian. Tests compare cached and direct ratios across repeated states,
cache eviction, and context changes.

The immediate preview at disorder10000, Gamma0.5,h0.3,K1 used W32,
equilibration1, energy measurement0.5, forward0.5, branch0.1 and four cycles.
Four independent populations completed in 35.45 aggregate seconds (with other
GPU work active): E/N=-1.123262 +/-0.003046, connected chi=1.3400 +/-0.4739.
Minimum ancestry ESS was2.44 and largest lineage0.625. These are execution
and cost results; short time and small population prevent an accuracy claim.

The fuller L4 pilot is queued after the sampled scan releases the GPU. Its
remote root is `/root/autodl-tmp/transformer-af-ltfim-20261027-direct-gfmc-v1/`:
two disorders, h=0/0.3, four independent populations per setting, W32/64,
equilibration1/2, forward0.5/1, branch0.1/0.05, four cycles. It has a separate
four-hour process cap. Pilot groups can indicate a need for stronger convergence
settings; they cannot qualify a ground-state phase diagram.

## Monitoring and reproducibility

`frozen_scan.py` provides separately locked exact, projection, benchmark,
sampled and gfmc stages. `run_frozen_scan.py` coordinates them and saves stage
logs. Each point/population is atomic and resumable. The manifest binds the
checkpoint, grid, sample allocation, estimator settings and source hashes.
Do not modify the staged campaign source during a run. Resuming a frozen
stage should use that original source tree and the same arguments.

`watch_frozen_scan.py` mirrors all three remote result trees every five minutes
and regenerates the executed notebook. Its state is in `mirror_status.json`;
errors are preserved in its log. It stops after final remote states or a
13-hour mirror cap. SSH credentials remain in SSHPASS loaded from Keychain.

The existing 15 physics/symmetry/SR/completion tests and the two added physical
scaling/cache correctness tests passed. The report cells execute from actual
downloaded records. Stages still running must be identified from current
summaries instead of inferring completion from this document.

A matched constant-guide baseline is documented in `PURE_VS_GUIDED_GFMC.md`.
It compares ordinary and learned-guide GFMC on identical L2 Hamiltonians,
populations, times, and estimators. Its independent campaign uses a separate
staged source tree and result directory.
