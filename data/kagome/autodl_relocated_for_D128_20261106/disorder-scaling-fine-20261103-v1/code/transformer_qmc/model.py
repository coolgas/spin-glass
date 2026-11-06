"""Causal eight-state Kagome-cell transformer, with static bond conditioning."""

import numpy as np
import torch
from torch import nn

from gfmc.guide import Guide


class CellTransformer(nn.Module):
    def __init__(self, ham, gamma, width=32, layers=2, heads=4):
        super().__init__()
        if ham.n_sites % 3:
            raise ValueError("cell transformer requires a multiple of three sites")
        if gamma <= 0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and positive")
        self.n_sites = ham.n_sites
        self.n_cells = ham.n_sites // 3
        self.config = dict(width=width, layers=layers, heads=heads)
        self.embedding = nn.Embedding(9, width)  # symbol 8 is BOS
        self.position = nn.Parameter(torch.randn(self.n_cells, width) * 0.02)
        coupling = np.zeros((ham.n_sites, ham.n_sites), dtype=np.float32)
        for (i, j), value in zip(ham.bonds, ham.J):
            coupling[i, j] += -value
            coupling[j, i] += -value
        features = np.concatenate([coupling.reshape(self.n_cells, -1),
                                   np.tile([gamma, ham.h], (self.n_cells, 1))], axis=1)
        self.register_buffer("features", torch.tensor(features, dtype=torch.float32))
        self.condition = nn.Linear(features.shape[1], width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 8)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("mask", torch.triu(torch.ones(self.n_cells, self.n_cells,
                                                           dtype=torch.bool), diagonal=1))
        self.register_buffer("bits", torch.tensor([1, 2, 4], dtype=torch.long))

    @property
    def device(self):
        return self.position.device

    def tokens(self, spins):
        bits = ((1 - spins.long()) // 2).reshape(-1, self.n_cells, 3)
        return (bits * self.bits).sum(-1)

    def logits(self, tokens):
        previous = torch.cat([torch.full_like(tokens[:, :1], 8), tokens[:, :-1]], dim=1)
        x = self.embedding(previous) + self.position + self.condition(self.features)
        for block in self.blocks:
            x = block(x, src_mask=self.mask)
        return self.output(self.norm(x))

    def log_prob(self, spins):
        tokens = self.tokens(spins)
        return self.logits(tokens).log_softmax(-1).gather(-1, tokens[..., None]).sum((1, 2))

    @torch.no_grad()
    def sample(self, count, generator=None):
        tokens = torch.zeros((count, self.n_cells), device=self.device, dtype=torch.long)
        for i in range(self.n_cells):
            probabilities = self.logits(tokens)[:, i].softmax(-1)
            tokens[:, i] = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        bits = (tokens[..., None] & self.bits) != 0
        return (1 - 2 * bits.long()).reshape(count, self.n_sites)

    def log_amplitudes(self, spins, chunk=2048):
        return torch.cat([0.5 * self.log_prob(part) for part in spins.split(chunk)])

    @torch.no_grad()
    def all_flip_log_ratios(self, spins, chunk=2048):
        count, n = spins.shape
        flips = spins[:, None, :].expand(count, n, n).clone()
        idx = torch.arange(n, device=spins.device)
        flips[:, idx, idx] *= -1
        logs = self.log_amplitudes(flips.reshape(-1, n), chunk).reshape(count, n)
        return logs - 0.5 * self.log_prob(spins)[:, None]


class TransformerGuide(Guide):
    """Frozen batched adapter, also usable by the existing single-copy GFMC."""

    def __init__(self, model):
        self.model = model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def _states(self, states):
        return torch.as_tensor(states, dtype=torch.long, device=self.model.device)

    @torch.no_grad()
    def log_amplitudes(self, states):
        return self.model.log_amplitudes(self._states(states)).double().cpu().numpy()

    @torch.no_grad()
    def all_flip_log_ratios(self, states):
        return self.model.all_flip_log_ratios(self._states(states)).double().cpu().numpy()

    def log_amplitude(self, spins):
        return float(self.log_amplitudes(np.asarray(spins)[None])[0])

    def sample(self, count, seed):
        generator = torch.Generator(device=self.model.device).manual_seed(int(seed))
        return self.model.sample(count, generator).cpu().numpy().astype(np.int8)


class EnumeratedGuide(Guide):
    """N<=16 frozen-network lookup: exact caching, not exact-state training.

    The production direct adapter above needs no enumeration. This accelerator
    lets the small-system projector benchmark use many independent populations.
    """

    def __init__(self, log_amplitudes, n_sites):
        self.logs = np.asarray(log_amplitudes, dtype=float)
        if n_sites > 16 or self.logs.shape != (1 << n_sites,) or not np.isfinite(self.logs).all():
            raise ValueError("invalid small-system guide table")
        self.n_sites = n_sites

    def indexes(self, states):
        return ((1 - np.asarray(states, dtype=np.int64)) // 2) @ (1 << np.arange(self.n_sites))

    def log_amplitudes(self, states):
        return self.logs[self.indexes(states)]

    def all_flip_log_ratios(self, states):
        indexes = self.indexes(states)
        flipped = indexes[:, None] ^ (1 << np.arange(self.n_sites))
        return self.logs[flipped] - self.logs[indexes, None]

    def log_amplitude(self, spins):
        return float(self.log_amplitudes(np.asarray(spins)[None])[0])

    def sample(self, count, seed):
        p = np.exp(2 * (self.logs - self.logs.max()))
        indexes = np.random.default_rng(seed).choice(len(p), size=count, p=p / p.sum())
        return (1 - 2 * ((indexes[:, None] >> np.arange(self.n_sites)) & 1)).astype(np.int8)
