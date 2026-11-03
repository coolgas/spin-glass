"""Positive Jastrow and real RBM wavefunctions without sequence positions.

The RBM probability psi**2 admits a Gibbs sampler with TWO hidden copies.
Sampling is finite-length MCMC, not independent exact sampling. All models
are per-disorder; no shared-disorder generalization claim is made here.
"""
from itertools import combinations

import numpy as np
import torch
from torch import nn

from gfmc.guide import Guide


class Jastrow(nn.Module):
    def __init__(self, n, seed=0):
        super().__init__()
        self.n_sites = n
        pairs = torch.tensor(list(combinations(range(n), 2)), dtype=torch.long)
        self.register_buffer('pairs', pairs)
        self.a = nn.Parameter(torch.zeros(n, dtype=torch.float64))
        rng = np.random.default_rng(seed)
        self.j = nn.Parameter(torch.tensor(rng.normal(0, .01/n**.5, len(pairs)), dtype=torch.float64))

    def forward(self, spins):
        return spins @ self.a + (spins[:, self.pairs[:, 0]] * spins[:, self.pairs[:, 1]]) @ self.j

    def flip_log_ratios(self, spins):
        matrix = torch.zeros((self.n_sites, self.n_sites), dtype=spins.dtype, device=spins.device)
        matrix[self.pairs[:, 0], self.pairs[:, 1]] = self.j
        matrix = matrix + matrix.T
        return -2 * spins * (self.a + spins @ matrix)

    @torch.no_grad()
    def sweep(self, spins, generator):
        for site in torch.randperm(self.n_sites, device=spins.device, generator=generator):
            lr = self.flip_log_ratios(spins)[:, site]
            take = torch.log(torch.rand(len(spins), device=spins.device, generator=generator)) < 2*lr
            spins[take, site] *= -1
        # Symmetric global inversion is useful in the low-field landscape.
        lr = self(-spins)-self(spins)
        take = torch.log(torch.rand(len(spins), device=spins.device, generator=generator)) < 2*lr
        spins[take] *= -1
        return spins


class RBM(nn.Module):
    def __init__(self, n, alpha=2, seed=0):
        super().__init__()
        self.n_sites, self.alpha = n, alpha
        rng = np.random.default_rng(seed)
        self.a = nn.Parameter(torch.zeros(n, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(rng.normal(0, .01, alpha*n), dtype=torch.float64))
        self.w = nn.Parameter(torch.tensor(rng.normal(0, .1/n**.5, (n, alpha*n)), dtype=torch.float64))

    def forward(self, spins):
        field = spins @ self.w + self.b
        return spins @ self.a + torch.logaddexp(field, -field).sum(1)

    def flip_log_ratios(self, spins):
        field = spins @ self.w + self.b
        before = torch.logaddexp(field, -field).sum(1)
        # Site chunks bound peak memory for L=10.
        result = []
        for start in range(0, self.n_sites, 8):
            end = min(start+8, self.n_sites)
            new = field[:, None, :] - 2*spins[:, start:end, None]*self.w[None, start:end, :]
            result.append(-2*spins[:, start:end]*self.a[start:end]
                          + torch.logaddexp(new, -new).sum(2)-before[:, None])
        return torch.cat(result, dim=1)

    @torch.no_grad()
    def sweep(self, spins, generator):
        field = spins @ self.w + self.b
        probability = torch.sigmoid(2*field)
        hidden = []
        for _ in range(2):
            hidden.append(2*(torch.rand(probability.shape, device=spins.device, generator=generator)<probability).to(spins.dtype)-1)
        visible_field = 2*self.a + (hidden[0]+hidden[1]) @ self.w.T
        probability = torch.sigmoid(2*visible_field)
        spins = 2*(torch.rand(spins.shape, device=spins.device, generator=generator)<probability).to(spins.dtype)-1
        lr = self(-spins)-self(spins)
        take = torch.log(torch.rand(len(spins), device=spins.device, generator=generator)) < 2*lr
        spins[take] *= -1
        return spins


class SimpleGuide(Guide):
    """Batched GFMC adapter; sample() is explicitly finite-length MCMC."""
    def __init__(self, model, sweeps=200):
        self.model = model.eval()
        self.sweeps = sweeps
        self.n_sites = model.n_sites

    def tensor(self, states):
        return torch.as_tensor(states, dtype=torch.float64, device=next(self.model.parameters()).device)

    @torch.no_grad()
    def log_amplitudes(self, states):
        return self.model(self.tensor(states)).cpu().numpy()

    @torch.no_grad()
    def all_flip_log_ratios(self, states):
        return self.model.flip_log_ratios(self.tensor(states)).cpu().numpy()

    def log_amplitude(self, spins):
        return float(self.log_amplitudes(np.asarray(spins)[None])[0])

    @torch.no_grad()
    def sample(self, count, seed):
        device = next(self.model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(int(seed))
        spins = 2*torch.randint(2, (count,self.n_sites),device=device,generator=generator).double()-1
        for _ in range(self.sweeps):
            spins = self.model.sweep(spins, generator)
        return spins.cpu().numpy().astype(np.int8)
