"""Size-independent Kagome conditioning and causal ALiBi wavefunction."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import CellTransformer


def alibi_slopes(heads):
    def power_two(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        return [start ** (i + 1) for i in range(n)]
    n = 2 ** int(math.log2(heads))
    return power_two(n) + (power_two(2*n)[::2][:heads-n] if n != heads else [])


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, x, bias):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(attention.transpose(1, 2).reshape(batch, length, width))
        return x + self.ff(self.norm2(x))


class UnifiedTransformer(CellTransformer):
    """One set of parameters for all L>=2 periodic nearest-neighbor Kagome tori.

    Static disorder has unrestricted attention; only spin tokens are causal.
    set_hamiltonian changes nonpersistent context, never trainable parameters.
    """
    def __init__(self, ham, gamma, width=32, layers=2, heads=4):
        nn.Module.__init__(self)
        if heads < 1 or width % heads:
            raise ValueError('width must be divisible by positive heads')
        self.config = dict(width=width, layers=layers, heads=heads)
        self.embedding = nn.Embedding(9, width)
        self.condition = nn.Linear(13, width)
        self.static_block = AttentionBlock(width, heads)
        self.blocks = nn.ModuleList([AttentionBlock(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 8)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('bits', torch.tensor([1, 2, 4]))
        self.register_buffer('slopes', torch.tensor(alibi_slopes(heads)), persistent=False)
        self.register_buffer('features', torch.empty(0), persistent=False)
        self.register_buffer('mask', torch.empty(0), persistent=False)
        self.register_buffer('geometry_bias', torch.empty(0), persistent=False)
        self.set_hamiltonian(ham, gamma)

    @property
    def device(self):
        return self.embedding.weight.device

    def set_hamiltonian(self, ham, gamma):
        L = math.isqrt(ham.n_sites//3)
        if L < 2 or 3*L*L != ham.n_sites or gamma <= 0 or not np.isfinite(gamma):
            raise ValueError('requires periodic L>=2 Kagome and positive finite gamma')
        bonds = {tuple(sorted((int(i), int(j)))): -float(v)
                 for (i, j), v in zip(ham.bonds, ham.J)}
        rows, used = [], set()
        def site(x, y, s):
            return 3*((y % L)*L + x % L)+s
        for y in range(L):
            for x in range(L):
                pairs = [(site(x,y,0),site(x,y,1)), (site(x,y,0),site(x,y,2)),
                         (site(x,y,1),site(x,y,2)), (site(x,y,0),site(x-1,y,1)),
                         (site(x,y,0),site(x,y-1,2)), (site(x,y,1),site(x+1,y-1,2))]
                keys = [tuple(sorted(pair)) for pair in pairs]
                if any(key not in bonds for key in keys):
                    raise ValueError('Hamiltonian does not have the required Kagome bonds')
                used.update(keys)
                rows.append([bonds[key] for key in keys] + [gamma, ham.h, 1/L,
                            np.sin(2*np.pi*x/L), np.cos(2*np.pi*x/L),
                            np.sin(2*np.pi*y/L), np.cos(2*np.pi*y/L)])
        if used != set(bonds) or len(bonds) != 6*L*L:
            raise ValueError('unexpected or duplicate Kagome bonds')
        self.n_sites, self.n_cells = ham.n_sites, L*L
        dtype = self.embedding.weight.dtype
        self.features = torch.tensor(rows, dtype=dtype, device=self.device)
        index = torch.arange(L*L, device=self.device)
        distance = index[:, None] - index[None, :]
        self.mask = -self.slopes[:, None, None].to(dtype)*distance.abs()
        self.mask = self.mask.masked_fill(distance[None] < 0, float('-inf'))
        dx = (index[:, None] % L - index[None, :] % L).abs()
        dy = (index[:, None] // L - index[None, :] // L).abs()
        # Periodic cell geometry supplements the one-dimensional causal ordering.
        torus = torch.minimum(dx, L-dx) + torch.minimum(dy, L-dy)
        self.geometry_bias = -self.slopes[:, None, None].to(dtype)*torus
        return self

    def logits(self, tokens):
        previous = torch.cat([torch.full_like(tokens[:, :1], 8), tokens[:, :-1]], dim=1)
        context = self.static_block(self.condition(self.features)[None], self.geometry_bias)
        x = self.embedding(previous) + context
        for block in self.blocks:
            x = block(x, self.mask)
        return self.output(self.norm(x))
