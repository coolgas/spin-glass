# Kagome quantum Edwards–Anderson, phase 1

Round 1 is the Gaussian nearest-neighbor ensemble at zero longitudinal field.
The flat-guide GFMC notes are in `notebook/tfim_gfmc.ipynb`.
`notebook/kagome_6x6_gfmc.ipynb` runs one periodic Gaussian sample. `L` is the number of cells on a side and is set to 8.
`notebook/kagome_qea_scan.ipynb` averages the forward-walked overlap of 32 periodic coupling draws. It starts at $L = 4$.
`notebook/kagome_qea_variance_plan.ipynb` is the plan for repeating that curve at Gaussian widths $\sigma = 0.5, 1, 2$.
This package is the Kagome solver side.

The operator is the longitudinal-transverse Ising model

$$
H = -\sum_b J_b \sigma_i^z \sigma_j^z - \Gamma \sum_i \sigma_i^x - h \sum_i \sigma_i^z.
$$

with `s_i = ±1` in the `σ^z` basis. A single-spin flip has matrix element `-Γ`.
Round 1 draws `J ~ N(0,1)` on the nearest-neighbor bonds and keeps `h = 0`.
`h` is a run argument, not part of the stored disorder.

## Layout

| Module | Role |
|---|---|
| `lattice.py` | `build_kagome(L, periodic)`. Re-exported by `generate_kagome_4x4.py`. |
| `hamiltonian.py` | Re-exports `ising.Hamiltonian`. |
| `sample.py` | Disorder records in `data/kagome/{ensemble}/{obc\|pbc}/L{L}_nr{N}.npz`. |
| `observables.py` | Overlap moments, Binder ratio, jackknife, bootstrap index, crossings. |
| `exact.py` | Sparse diagonalization (`N ≤ 16`). Moments (`N ≤ 12`) are optional. A gap outside the residual keeps the lowest state. `mix_window` mixes a requested window. |
| `pqmc.py` | Continuous-time projector QMC. Flat guide, forward-walked overlap, autocorrelation-aware errors. |
| `pimc.py` | Continuous-time path-integral Monte Carlo and a `Γ` ladder. |
| `scan.py` | Loop order over samples, `Γ`, and `h`. |
| `checks.py` | Validation against exact results. |

Solvers read `bonds` and `J`. They do not draw couplings and they do not assume
coordination 4. A later ensemble is a new drawer in `draw_couplings`.
Antiferromagnetic couplings are stored as negative `J`, because the operator
is `-∑ J σ^z σ^z`.

## Disorder

A sample is `(L, periodic, sample_id)`. The stream is PCG64 seeded by

```text
[1000, L, sample_id, int(periodic)]
```

Open and periodic realizations are independent. Debug files use 32 samples
and production files use 128, for `L = 2, 3, 4` and both boundary conditions.
Solver random streams are separate from this stream.

## Checks

From the repository root:

```text
python code/run_phase1_checks.py
python code/run_phase1_checks.py --fast
```

`--fast` skips the `L = 2` quantum Monte Carlo comparison. The full command
checks one `sample_id = 0` at `Γ = 1` for each boundary condition. It compares
projector estimates with the ground state and path-integral estimates with the
thermal state at `β = 3`. The projector overlap is a forward walk. A flat
guide collapses lineages, so that comparison uses several thousand walkers
and many cycles, and the error is the cycle scatter. This is an algorithm
check. It does not average over disorder and it does not report a crossing.

## Still open

Section 9 ensembles, the `(Γ, h)` production map, a Jastrow or RBM guide,
cluster updates, and the disorder-averaged `L = 3, 4` crossings are not run
here. `iter_jobs` is the production loop. Connected Binder moments are ready
for a nonzero field. The round-1 window is `Γ ∈ [0.4, 3.5]`.

Dependencies: NumPy and SciPy, Python 3.9 or newer.
