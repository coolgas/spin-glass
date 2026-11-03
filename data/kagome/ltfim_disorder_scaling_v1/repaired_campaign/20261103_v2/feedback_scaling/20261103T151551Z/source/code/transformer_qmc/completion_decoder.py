"""Causal one-step quantum completion prior for ungenerated neighboring spins.

The ordinary local prior counts only bonds whose second end has been generated.
Here each candidate cell also sees the independent-spin ground energy of its
future neighbors, -sqrt(gamma^2 + B^2). Future/future bonds remain the learned
network's responsibility; this is an input approximation, not an exact cavity
solution or a variational energy estimate.
"""

import numpy as np
import torch
from torch import nn

from .physical_decoder import PhysicalDecoder


class CompletionDecoder(PhysicalDecoder):
    def __init__(self, ham, gamma, width=64, layers=4, heads=4):
        super().__init__(ham, gamma, width=width, layers=layers, heads=heads)
        self.completion_scale = nn.Parameter(torch.tensor(0.))
        self.register_buffer('future_coupling', torch.empty(0), persistent=False)
        self.register_buffer('neighbor_indices', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('neighbor_mask', torch.empty(0), persistent=False)
        self.register_buffer('neighbor_response', torch.empty(0), persistent=False)
        self.register_buffer('prefix_mask', torch.empty(0), persistent=False)
        self.set_hamiltonian(ham, gamma)

    def set_hamiltonian(self, ham, gamma):
        super().set_hamiltonian(ham, gamma)
        if 'future_coupling' not in self._buffers:
            return self
        coupling = np.zeros((ham.n_sites, ham.n_sites))
        for (i, j), J in zip(ham.bonds, ham.J):
            coupling[i, j] = coupling[j, i] = -J/gamma
        # Each cell only needs its future nearest neighbors, at most six on
        # this graph. Keeping that dimension sparse avoids a batch*C*N*8 tensor.
        neighbors = []
        for cell in range(self.n_cells):
            candidates = np.flatnonzero(np.any(coupling[3*cell:3*cell+3] != 0, axis=0))
            neighbors.append(candidates[candidates//3 > cell])
        width = max(1, max(map(len, neighbors)))
        indices = np.zeros((self.n_cells, width), dtype=np.int64)
        mask = np.zeros_like(indices, dtype=float)
        response = np.zeros((self.n_cells, width, 8))
        options = self.spin_options.detach().cpu().numpy()
        for cell, sites in enumerate(neighbors):
            indices[cell, :len(sites)] = sites
            mask[cell, :len(sites)] = 1
            response[cell, :len(sites)] = coupling[sites, 3*cell:3*cell+3] @ options.T
        dtype = self.embedding.weight.dtype
        self.future_coupling = torch.tensor(coupling, dtype=dtype, device=self.device)
        self.neighbor_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self.neighbor_mask = torch.tensor(mask, dtype=dtype, device=self.device)
        self.neighbor_response = torch.tensor(response, dtype=dtype, device=self.device)
        self.prefix_mask = (torch.arange(self.n_sites, device=self.device)[None]//3 <
                            torch.arange(self.n_cells, device=self.device)[:, None]).to(dtype)
        self.longitudinal_ratio = ham.h/gamma
        return self

    def completion_energy(self, tokens):
        spins = self.spin_options[tokens].reshape(len(tokens), self.n_sites)
        # At cell c use only spins in cells <c, including when computing the
        # fields acting on future neighbors. No future token enters this input.
        prefix = spins[:, None, :] * self.prefix_mask[None]
        coupling = self.future_coupling[self.neighbor_indices]
        field = torch.einsum('bcn,ckn->bck', prefix, coupling)-self.longitudinal_ratio
        delta = (-torch.sqrt(1+(field[..., None]+self.neighbor_response[None]).square())
                 + torch.sqrt(1+field.square())[..., None])
        return (delta*self.neighbor_mask[None, :, :, None]).sum(2)

    def logits(self, tokens):
        return super().logits(tokens)-torch.tanh(self.completion_scale)*self.completion_energy(tokens)
