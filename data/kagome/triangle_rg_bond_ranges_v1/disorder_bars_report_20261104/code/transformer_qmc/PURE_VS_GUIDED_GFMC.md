# Ordinary versus frozen-network guided GFMC

This benchmark tests whether the selected shared seed501 guide improves on
ordinary GFMC before expanding neural training across field and bond scales.
“Pure GFMC” means a constant trial amplitude `psi_T(s)=1`, so every spin flip
has rate Gamma. Both methods use the existing continuous-time projector,
systematic population resampling, mixed energy, and four-replica forward
walking for q2 and connected susceptibility. Pure diagonal estimation is
common to both methods. No parameters are trained.

## First completed comparison

The local preview uses the 12-spin periodic Kagome sample10000, Gamma0.5,
h0.3, K1/2; W512, equilibration Ktau8, measurement Ktau2, forward Ktau2,
branch Kdt0.1, six forward cycles, and four independent populations per method.
All 16 populations completed. Errors below are SE across independent populations.

| K | Guide | Energy error per spin | Connected chi | Exact chi | Minimum ancestry ESS |
|---|---|---:|---:|---:|---:|
| 1 | Constant | 0.033446 +/-0.011523 | 1.48489 +/-0.41526 | 0.118463 | 1.13 |
| 1 | Network | 0.000932 +/-0.000520 | 0.29476 +/-0.00953 | 0.118463 | 97.23 |
| 2 | Constant | 0.089439 +/-0.005344 | 2.18275 +/-0.82184 | 0.036127 | 1.27 |
| 2 | Network | 0.036913 +/-0.004924 | 1.81323 +/-0.65644 | 0.036127 | 3.75 |

At K1, local-energy variance falls from28.3359 to0.45769 with the guide. At
K2 it falls from110.1035 to2.37280, but the guide's initial fidelity is only
0.03918 on this disorder. Better variance and energy do not establish converged
correlations: even the K1 guided susceptibility differs substantially from
the exact target. Both methods exhibit ancestry loss at K2. No cycles are
discarded because of poor ancestry.

The exact infinite-population finite-time susceptibility for this short K1
preview is0.273012 with the network and0.364378 with the constant guide.
Thus the guided result is close to its finite-time target, although still
far from the ground-state value0.118463. The constant result has a much larger
additional stochastic discrepancy. At K2 the respective finite-time targets
are0.379234 and0.123466; both stochastic estimates remain poor.

Results: `data/kagome/transformer_af_pure_vs_guided_preview_v1/`.
Executed report: `notebook/pure_vs_guided_gfmc_preview.ipynb`.

## Broader matched benchmark

`compare_pure_gfmc.py` runs four fixed (Gamma,h,K) contexts:
(0.5,0.3,1), (0.5,0.3,2), (2,0.3,1), (0.25,0.3,1).
Each uses development disorders10000/10001, W512/2048, forward Ktau2/8,
equilibration Ktau16, measurement Ktau4, branch Kdt0.1, 12 forward cycles,
four independent populations and both guides: 64 groups /256 populations.
These are initial population/time checks; branching-cadence and longer
equilibration checks are still needed to qualify convergence.

All64 groups /256 populations completed in385.98 wall seconds on AutoDL.
At W2048, forward Ktau8, selected per-disorder results are:

| Gamma | K | Disorder | Guide | Energy error/N +/-SE | Connected chi +/-SE | Exact finite-time chi | Exact ground chi | Min ancestry ESS |
|---|---|---|---|---:|---:|---:|---:|---:|
| 0.5 | 1 | 10000 | Constant | 0.017634 +/-0.001234 | 1.19584 +/-0.30104 | 0.13987 | 0.11846 | 1.00 |
| 0.5 | 1 | 10000 | Network | 0.000256 +/-0.000130 | 0.12763 +/-0.00789 | 0.13493 | 0.11846 | 84.02 |
| 0.5 | 1 | 10001 | Constant | 0.006970 +/-0.003392 | 1.46075 +/-0.36950 | 1.17133 | 1.05027 | 1.00 |
| 0.5 | 1 | 10001 | Network | 0.000125 +/-0.000115 | 1.09024 +/-0.03379 | 1.07401 | 1.05027 | 23.24 |
| 0.5 | 2 | 10000 | Constant | 0.017980 +/-0.005756 | 0.24019 +/-0.05331 | 0.03852 | 0.03613 | 1.00 |
| 0.5 | 2 | 10000 | Network | 0.019524 +/-0.004595 | 0.07703 +/-0.00540 | 0.04149 | 0.03613 | 1.94 |
| 2 | 1 | 10000 | Constant | -0.000383 +/-0.001020 | 1.55547 +/-0.20718 | 1.17511 | 1.17511 | 2.33 |
| 2 | 1 | 10000 | Network | 0.000029 +/-0.000044 | 1.18787 +/-0.01186 | 1.17511 | 1.17511 | 364.60 |

Guidance strongly improves the K1 cases in this benchmark. It is not uniformly
better in energy at K2, and the lowest-field disorder10001 still has poor
ancestry (minimum ESS2.03) and broad susceptibility uncertainty. These are
per-disorder findings; two disorders do not establish ensemble convergence.

The exact L2 scan supplies evaluation targets and a lookup of the frozen
network. Exact amplitudes are never used as a stochastic guide. The checkpoint
SHA-256 is `5f9fa33b04102ddff2e43179440514021090ed3aaf228ce67dce53c2139e2544`.
The manifest fixes settings, source hashes, and all input cache hashes;
per-population records are atomic and resumable. Initial samples follow each
guide's squared amplitude. This measures the complete trial-state setup,
including its effect on initial overlap and equilibration.

Remote root:
`/root/autodl-tmp/transformer-af-ltfim-20261027-pure-vs-guided-v1/`.
It runs four CPU workers in a separate staged source tree, with a two-hour
dispatch cap. At the cap, in-flight populations finish and remaining work is
resumable. Instance power stays on. The ongoing frozen scan and GFMC sources
are unchanged.

Local mirror: `data/kagome/transformer_af_pure_vs_guided_v1/`.
Executed report: `notebook/pure_vs_guided_gfmc.ipynb`.
`watch_pure_gfmc.py` mirrors every60 seconds and executes the report until a
terminal state or a three-hour mirror cap. Check `status.json`, `summary.json`,
and `mirror_status.json` for actual current coverage.

Wall times include initial sampling and propagation, and exclude input loading,
lookup construction, and direct neural inference. Timing/variance efficiency
ratios are descriptive, noisy with four populations, and apply to L2 lookup
rates. They do not establish larger-size end-to-end speedups. Accuracy, ancestry,
and population/time stability must be checked together.

`reference_pure_gfmc.py` projects both trial states deterministically at
Ktau=0,0.5,1,2,4,8,16,32,64. It records fidelity, full connected-matrix RMS
error, susceptibility, pure energy, and mixed energy against exact targets.
The executed report displays these trajectories when
`projection_reference.json` is present. They describe pure projected states;
finite-forward-time stochastic estimators have different boundary factors.
The reference also computes their exact infinite-population targets. A tag at
time t, followed for f, has ancestor distribution proportional to
`[exp(-tH) psi_T](s) * [exp(-fH) psi_T](s)`. The report averages this distribution's
four-replica observables over the same tag times as the stochastic cycles, so
finite-forward-time and finite-population errors can be assessed separately.

## Validation

The new constant-guide adapter passes an exact imaginary-time kernel comparison
on an interacting three-spin model and an independent-transverse-spin zero
variance/constant-weight check. All five existing core tests also passed.
The two propagator checks passed again on AutoDL. An additional exact kernel
identity verifies the finite-forward ancestor distribution for constant and
nonconstant guides. All eight core/constant-guide tests passed locally. Both
notebooks include six executed code cells with zero errors. All16 deterministic
trajectories pass monotonic fidelity/energy checks; the smallest fidelity at
Ktau64 is0.99977055. Campaign sources and input hashes were verified locally.

Example local commands, from the repository root:

```bash
PYTHONPATH=code python3 -m unittest transformer_qmc.test_pure_gfmc transformer_qmc.test_core -v
PYTHONPATH=code python3 -m transformer_qmc.compare_pure_gfmc \
  --scan data/kagome/transformer_af_frozen_scan_v1 \
  --output data/kagome/transformer_af_pure_vs_guided_preview_v1 --preview --workers 4
PYTHONPATH=code python3 -m transformer_qmc.report_pure_gfmc \
  --results data/kagome/transformer_af_pure_vs_guided_preview_v1 \
  --notebook notebook/pure_vs_guided_gfmc_preview.ipynb
```
