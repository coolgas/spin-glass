"""Open-boundary ALiBi guide with physical inputs and exact zero-field parity.

Separate architecture/version from the periodic guide. No torus translations,
wrapped geometry, or artificial boundary couplings are used.
"""
import math
import numpy as np
import torch
from torch import nn

from kagome_qea.lattice import build_kagome
from .physical_decoder import PhysicalDecoder
from .model import CellTransformer


class OpenPhysicalDecoder(PhysicalDecoder):
    def __init__(self, ham, gamma, width=64, layers=4, heads=4):
        # Parent calls set_hamiltonian dynamically; replace its 13-feature layer
        # after construction with the OBC feature dimension.
        super().__init__(ham, gamma, width=width, layers=layers, heads=heads)
        self.condition = nn.Linear(17, width)

    def set_hamiltonian(self, ham, gamma):
        L = math.isqrt(ham.n_sites // 3)
        if L < 2 or 3 * L * L != ham.n_sites or not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('OBC guide requires L>=2 kagome and positive finite gamma')
        _, expected, _, _ = build_kagome(L, periodic=False)
        actual = np.sort(np.asarray(ham.bonds), axis=1)
        bonds = {tuple(map(int, pair)): -float(value) for pair, value in zip(actual, ham.J)}
        if len(bonds) != len(actual) or set(bonds) != set(map(tuple, expected)):
            raise ValueError('OBC Hamiltonian must have exactly the open-patch bonds')
        if not np.isfinite(ham.J).all() or not np.isfinite(ham.h):
            raise ValueError('Nonfinite OBC Hamiltonian')

        def site(x, y, s):
            return 3 * (y * L + x) + s if 0 <= x < L and 0 <= y < L else None

        rows = []
        for y in range(L):
            for x in range(L):
                pairs = [(site(x,y,0),site(x,y,1)), (site(x,y,0),site(x,y,2)),
                         (site(x,y,1),site(x,y,2)), (site(x,y,0),site(x-1,y,1)),
                         (site(x,y,0),site(x,y-1,2)), (site(x,y,1),site(x+1,y-1,2))]
                mask = [float(a is not None and b is not None) for a, b in pairs]
                values = [bonds[tuple(sorted((a, b)))] if present else 0.
                          for (a, b), present in zip(pairs, mask)]
                rows.append(values + mask + [gamma, ham.h, 1/L, x/(L-1), y/(L-1)])
        self.n_sites, self.n_cells = ham.n_sites, L*L
        dtype, device = self.embedding.weight.dtype, self.device
        self.features = torch.tensor(rows, dtype=dtype, device=device)
        index = torch.arange(L*L, device=device)
        distance = index[:, None] - index[None, :]
        self.mask = -self.slopes[:, None, None].to(dtype) * distance.abs()
        self.mask = self.mask.masked_fill(distance[None] < 0, float('-inf'))
        dx = (index[:, None] % L - index[None, :] % L).abs()
        dy = (index[:, None] // L - index[None, :] // L).abs()
        self.geometry_bias = -self.slopes[:, None, None].to(dtype) * (dx + dy)
        if 'past_coupling' not in self._buffers:
            return self
        coupling = np.zeros((ham.n_sites, ham.n_sites))
        for (i, j), value in zip(ham.bonds, ham.J):
            coupling[i, j] = coupling[j, i] = -value
        idx = np.arange(ham.n_sites)
        past = coupling * (idx[:, None]//3 > idx[None, :]//3)
        options = 1-2*((np.arange(8)[:, None] >> np.arange(3)) & 1)
        energies = []
        for cell in range(self.n_cells):
            sub = coupling[3*cell:3*cell+3, 3*cell:3*cell+3]
            energies.append(.5*np.einsum('bi,ij,bj->b', options, sub, options)-ham.h*options.sum(1))
        self.past_coupling = torch.tensor(past, dtype=dtype, device=device)/gamma
        self.cell_energy = torch.tensor(np.stack(energies), dtype=dtype, device=device)/gamma
        self.spin_options = torch.tensor(options, dtype=dtype, device=device)
        return self


class OpenBoundaryTransformer(CellTransformer):
    """Normalized OBC probability; at h=0 use [p(s)+p(-s)]/2."""
    def __init__(self, ham, gamma, width=64, layers=4, heads=4):
        nn.Module.__init__(self)
        self.base = OpenPhysicalDecoder(ham, gamma, width=width, layers=layers, heads=heads)
        self.config = dict(width=width, layers=layers, heads=heads)
        self.set_hamiltonian(ham, gamma)

    @property
    def device(self):
        return self.base.device

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, '_hamiltonian'):
            self.set_hamiltonian(self._hamiltonian, self._gamma)
        return self

    def set_hamiltonian(self, ham, gamma):
        self.base.set_hamiltonian(ham, gamma)
        self._hamiltonian, self._gamma = ham, gamma
        self.n_sites, self.n_cells = ham.n_sites, ham.n_sites//3
        self.spin_flip_symmetric = ham.h == 0
        return self

    def log_prob(self, spins):
        direct = self.base.log_prob(spins)
        if self.spin_flip_symmetric:
            return torch.logaddexp(direct, self.base.log_prob(-spins)) - math.log(2)
        return direct

    @torch.no_grad()
    def sample(self, count, generator=None):
        spins = self.base.sample(count, generator)
        if self.spin_flip_symmetric:
            spins = spins * (1-2*torch.randint(2, (count, 1), device=self.device, generator=generator))
        return spins
