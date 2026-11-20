# Physical Rydberg Kagome

Independent package for the ordered-pair Rydberg Hamiltonian, direct 12-spin
superblock projection, TeNPy DMRG and reconstructed physical observables.
There are no imports from `transformer_qmc`.

Use Python >=3.10, NumPy, SciPy and physics-tenpy 1.1.0. Tests use unittest.
Run from the repository with `PYTHONPATH=code` and `PYTHONDONTWRITEBYTECODE=1`.
Set BLAS/OpenMP thread counts before importing NumPy. Store all outputs outside
Dropbox; the CLI requires an explicit output location.

```
python -B -m unittest discover -s code/rydberg_kagome/tests -v
python -B -m rydberg_kagome spec --output /absolute/local/path/spec.json
python -B -m rydberg_kagome case --output /absolute/local/path/case \
  --sample 0 --model full --delta 3.15 --omega 1 --chi 8
```

## Physics

Coordinates are in R units, energies in V0. n=(1+Z)/2 and the transverse
coefficient is +Omega/2. Symmetric matrices contain the physical V_ij.
The occupation interaction is sum_(i!=j) V_ij n_i n_j /2; the Pauli term
is sum_(i!=j) V_ij Z_i Z_j /8. The effective field uses the masked row sum
Delta/2-sum_j V_ij/4. Constants are retained in reported energies.
The MPO combines the two orientations exactly once at its once-per-bond
interface, giving V_ij/4. Crossing block coupling SVD retains all components.

Geometry uses verified Euclidean shortest images on the oblique torus.
Historical fractional-rounding data remain a separate convention. Rejection
uses fresh whole-array Gaussian proposals from the perfect positions. Seeds
are SeedSequence([20261116,L,sample_id,attempt]).

Four complete triangles form each 12-spin block. No intermediate triangle
projection occurs. The complete 4096-dimensional block is diagonalized using
a sparse eigensolver, retaining chi states. Physical ZiZj within one block
is projected as a product, not multiplied after separate projection.
Joint Born draws condition the MPS across blocks using physical decoder POVMs.

## Qualification and resume

Qualification runs two independent solver starts and a D ladder, compares physical
moments, performs mixer-free confirmation and optionally benchmarks the
projected Hamiltonian exactly. Checkpoints and moment arrays are retained at
each D. Resume verifies source/configuration and checkpoint hashes. An owner
record prevents duplicate workers; after a hard interruption, inspect the PID
and preserve the stale owner record before explicitly recovering ownership.

`solver_pass` concerns the fixed projected space. `sampling_moment_check`
checks consistency, not statistical precision. `rg_converged` and
`production_qualified` remain false until paired retention/disorder/measurement
checks are assessed.

The user selected production at chi=12 and D=128 on 2026-11-17 despite measured
RG sensitivity. Production runs two independent starts directly at D128,
with sweep stability, start agreement and sampling checks. It is labeled
`production_selected=true`, while `rg_converged=false`. There is no per-case
D ladder, so `bond_dimension_verified=false`; selected cases retain projected
exact comparisons. Completion and fixed-setting acceptance are distinct.

Production retains the full decoder/model and both final MPS checkpoints in
gzip format. MPS checkpoints reference reconstructed MPO site objects rather
than repeating their operator matrices. Restore the model with
`load_checkpoint(model_path)`, call `dmrg.build_mpo(model)`, then restore each
state with `load_checkpoint(state_path, mpo.lat.mps_sites())`. Tensor entries
and singular values are stored losslessly. Use the matching frozen release.
The fixed production cohort is 128 geometries, two detunings, 30 log-uniform
drives from .03 to 10 and three masks. Qualification includes additional
diagnostic parameter values and must not be counted as extra disorders.

## Source provenance

MPO construction and physical moment interfaces adapt the reviewed
`code/transformer_qmc/triangle_effective_mps.py`; conditional decoder sampling
and physical projections follow `triangle_effective_rg.py`. The new Rydberg
builder, exact fixtures and distance routine are independent implementations.
Release manifests record actual historical source hashes and all new source
hashes. See the research-plan notebook for tolerances and the deployment workflow.
