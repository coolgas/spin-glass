# Ising operator

The longitudinal-transverse Ising model is

$$
H = -\sum_b J_b \sigma_i^z \sigma_j^z - \Gamma \sum_i \sigma_i^x - h \sum_i \sigma_i^z,
$$

with $s_i = \pm 1$. One coupling $J_b$ is stored per bond. This is the operator used at $N \sim 10^3$. The $2^N$ matrix is only `kagome_qea.exact.operator_matrix`, and that function stops at 16 sites. $\Gamma$ is not stored. `GFMC` takes $\Gamma$ and can replace $h$ through `with_field`.

```python
from ising import Hamiltonian, chain, square
```

## Hamiltonian

`Hamiltonian(n_sites, bonds, J, h=0.0)`

- `n_sites`: int, at least 1.
- `bonds`: integer array of shape `(n_bonds, 2)`. Site indices lie in `[0, n_sites)`.
- `J`: float array of length `n_bonds`. One coupling per bond.
- `h`: float. Longitudinal field. Default `0`.

Stored attributes: `n_sites`, `bonds`, `J`, `h`, and `neighbors`. `neighbors[i]` is a list of `(neighbor, J)`.

- `diagonal_energy(spins)`: float. `spins` has length `n_sites` and entries `±1`. Sums the stored bonds and the field.
- `flip_delta(spins, site)`: float. Uses only the neighbor list of `site`.
- `with_field(h)`: a new operator with the same bonds and `J`, and field `h`.
- `with_classical_scale(scale)`: a new operator with `J` and `h` multiplied by `scale`.

## Lattices

`chain(n, J=1.0, periodic=False, h=0.0)` returns a `Hamiltonian`.

- `n`: int, at least 1. Number of sites.
- `J`: float. Coupling on every bond.
- `periodic`: bool. `True` adds the bond `(0, n - 1)` and requires `n ≥ 3`.
- `h`: float. Longitudinal field.

Open bonds are `(i, i + 1)`.

`square(n_x, n_y, J=1.0, periodic=False, h=0.0)` returns a `Hamiltonian`. The site index is `x + n_x y`.

- `n_x`: int, at least 1. Number of columns.
- `n_y`: int, at least 1. Number of rows.
- `J`: float. Coupling on every bond.
- `periodic`: bool. `True` requires both sides at least 2.
- `h`: float. Longitudinal field.

Each unordered nearest-neighbor pair is stored once. On a side of length 2 the periodic image is that same pair, so the bond is not doubled.
