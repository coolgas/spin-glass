# GFMC

Continuous-time projector for one copy of the longitudinal-transverse Ising model

$$
H = -\sum_b J_b \sigma_i^z \sigma_j^z - \Gamma \sum_i \sigma_i^x - h \sum_i \sigma_i^z.
$$

$s_i = \pm 1$. It does not build a lattice and does not allocate a $2^N$ matrix.

```python
from gfmc import GFMC, Guide, ConstantGuide, save_run, load_run, iter_runs
```

## Operator

Any object with these attributes. `spins` has length `n_sites` and entries `±1`.

- `n_sites`: int. Number of spins.
- `diagonal_energy(spins)`: float. Diagonal entry of $H$.
- `flip_delta(spins, site)`: float. Energy after flipping `site`, minus the energy before.

## Guide

- `log_amplitude(spins)`: float. $\log \psi(s)$.
- `log_ratio(spins, site)`: float. $\log \psi(s \oplus \mathrm{site}) - \log \psi(s)$. The default calls `log_amplitude` twice.
- `ConstantGuide`: $\psi = 1$. Every `log_ratio` is 0. This is the default guide.

A ratio that is not a positive real raises `ValueError`.

## Class

`GFMC(hamiltonian, gamma, guide=None, *, h=None)`

- `hamiltonian`: the operator above. Its diagonal part may already carry a longitudinal field.
- `gamma`: float. Transverse field $\Gamma > 0$.
- `guide`: a `Guide`, or `None` for `ConstantGuide`.
- `h`: float or `None`. Longitudinal field in $-h \sum_i \sigma_i^z$. `None` keeps the field on `hamiltonian`. A number calls `with_field(h)` and leaves the bonds unchanged.

`run(n_walkers=2000, tau_block=0.25, n_equilibrate=40, n_measure=60, seed=0, configurations=None)`

- `n_walkers`: int, at least 2. Number of configurations kept together. One walker is one configuration and one positive weight.
- `tau_block`: float, positive. Imaginary time of one block.
- `n_equilibrate`: int, at least 0. Blocks projected and discarded.
- `n_measure`: int, at least 2. Blocks whose energies are returned.
- `seed`: int. Seed of the NumPy PCG64 stream.
- `configurations`: `None`, shape `(n_sites,)`, or shape `(n_walkers, n_sites)`. Entries `±1`. `None` draws independent random product states. A single vector is copied onto every walker.

Returned dict:

- `energy`: float. Mixed estimator, the weight-and-time average of the local energy.
- `energy_error`: float. Standard error of those block means.
- `energy_growth`: float. Growth estimator $E_T - \log(W / n_{\mathrm{walkers}}) / \tau_{\mathrm{block}}$.
- `energy_growth_error`: float. Standard error of those block means.
- `shift`: float. Final $E_T$.
- `n_walkers`: int. Population size.
- `h`: float. Longitudinal field used in the run.

## Runs on disk

`save_run(root, record)` writes one file and returns its path.

```text
{root}/tfim/gfmc/{lattice}/{boundary}/{size}/J{J}/gamma{gamma}/h{h}/seed{seed}.npz
```

In a directory name, a decimal point is `p` and a leading minus sign is `m`.

`record` fields:

- `lattice`: str. `chain` or `square`.
- `boundary`: str. `obc` or `pbc`.
- `size`: str. Label such as `n8` or `nx3_ny2`.
- `n_sites`: int.
- `J`: float. Coupling stored with the run.
- `gamma`: float.
- `h`: float.
- `seed`: int.
- `n_walkers`: int.
- `tau_block`: float.
- `n_equilibrate`: int.
- `n_measure`: int.
- `energy`: float.
- `energy_error`: float.
- `energy_growth`: float.
- `energy_growth_error`: float.
- `exact_energy`: float.
- `guide`: str. Name of the guiding function, `constant` for $\psi = 1$.

`load_run(path)` returns that record as a dict. `iter_runs(root)` returns every saved record under `{root}/tfim/gfmc`, in path order.
