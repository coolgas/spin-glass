"""Normalized shared ALiBi guide with exact translation covariance and h=0 parity.

A lexicographically canonical bond record fixes the cell origin. All tied
origins are mixed, so clean/partially periodic disorder is supported too.
The extra origin average usually has one term for continuous random bonds.
"""

import math
import numpy as np
import torch
from torch import nn

from ising import Hamiltonian
from .model import CellTransformer
from .unified import UnifiedTransformer
from .physical_decoder import PhysicalDecoder
from .completion_decoder import CompletionDecoder


def translation_permutation(L, dx, dy):
    """Map old site indices to translated indices, preserving sublattice."""
    sites = np.arange(3*L*L)
    cells = sites//3
    return 3*((cells//L+dy)%L*L+(cells%L+dx)%L)+sites%3


def translate_hamiltonian(ham, permutation):
    bonds = np.sort(permutation[ham.bonds], axis=1)
    order = np.lexsort((bonds[:, 1], bonds[:, 0]))
    return Hamiltonian(ham.n_sites, bonds[order], ham.J[order], ham.h)


def canonical_context(ham):
    L = math.isqrt(ham.n_sites//3)
    if L < 2 or 3*L*L != ham.n_sites:
        raise ValueError('requires an L>=2 periodic Kagome lattice')
    best, canonical, permutations = None, None, []
    for dy in range(L):
        for dx in range(L):
            permutation = translation_permutation(L, dx, dy)
            translated = translate_hamiltonian(ham, permutation)
            # Compare original double-precision couplings before tensor casting.
            key = tuple(-translated.J)
            if best is None or key < best:
                best, canonical, permutations = key, translated, [permutation]
            elif key == best:
                permutations.append(permutation)
    return canonical, np.stack(permutations)


class SymmetricTransformer(CellTransformer):
    """Positive guide with one base parameter state for every size and context.

Symmetry is applied to probabilities. All inherited amplitude and spin-flip
ratio methods consequently use sqrt(the same normalized mixture) as sampling.
Canonicalization is discrete, so smoothness in the bond inputs is not promised.
"""

    def __init__(self, ham, gamma, width=64, layers=4, heads=4, physical=True, completion=False):
        nn.Module.__init__(self)
        if completion and not physical:
            raise ValueError('completion requires physical inputs')
        decoder = CompletionDecoder if completion else (PhysicalDecoder if physical else UnifiedTransformer)
        self.base = decoder(ham, gamma, width=width, layers=layers, heads=heads)
        self.config = dict(width=width, layers=layers, heads=heads, physical=physical)
        if completion:
            self.config['completion'] = True
        self.register_buffer('to_original', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('to_canonical', torch.empty(0, dtype=torch.long), persistent=False)
        self.set_hamiltonian(ham, gamma)

    @property
    def device(self):
        return self.base.device

    def _apply(self, fn):
        super()._apply(fn)
        # Rebuild conditioning at the new precision, rather than promoting
        # already rounded float32 fields during .double().
        if hasattr(self, '_hamiltonian'):
            self.set_hamiltonian(self._hamiltonian, self._gamma)
        return self

    def set_hamiltonian(self, ham, gamma):
        self._hamiltonian, self._gamma = ham, gamma
        canonical, permutations = canonical_context(ham)
        self.base.set_hamiltonian(canonical, gamma)
        self.n_sites, self.n_cells = ham.n_sites, ham.n_sites//3
        self.spin_flip_symmetric = ham.h == 0
        self.to_original = torch.as_tensor(permutations, device=self.device, dtype=torch.long)
        self.to_canonical = torch.argsort(self.to_original, dim=1)
        return self

    def log_prob(self, spins):
        terms = []
        for inverse in self.to_canonical:
            canonical = spins[:, inverse]
            terms.append(self.base.log_prob(canonical))
            if self.spin_flip_symmetric:
                terms.append(self.base.log_prob(-canonical))
        if len(terms) == 1:
            return terms[0]
        return torch.logsumexp(torch.stack(terms), dim=0)-math.log(len(terms))

    @torch.no_grad()
    def sample(self, count, generator=None):
        spins = self.base.sample(count, generator)
        if len(self.to_original) == 1:
            spins = spins[:, self.to_original[0]]
        else:
            choice = torch.randint(len(self.to_original), (count,), device=self.device,
                                   generator=generator)
            spins = torch.gather(spins, 1, self.to_original[choice])
        if self.spin_flip_symmetric:
            sign = 1-2*torch.randint(2, (count,1), device=self.device, generator=generator)
            spins = spins*sign
        return spins
