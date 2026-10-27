"""Causal bond-weighted spin inputs and a local-energy logit prior for ALiBi."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .unified import UnifiedTransformer


class PhysicalDecoder(UnifiedTransformer):
    def __init__(self, ham, gamma, width=64, layers=4, heads=4):
        super().__init__(ham,gamma,width=width,layers=layers,heads=heads)
        self.past_field = nn.Linear(3,width,bias=False)
        # softplus(-2) starts with a weak local-energy preference.
        self.energy_scale = nn.Parameter(torch.tensor(-2.))
        self.register_buffer('past_coupling',torch.empty(0),persistent=False)
        self.register_buffer('cell_energy',torch.empty(0),persistent=False)
        self.register_buffer('spin_options',torch.empty(0),persistent=False)
        self.set_hamiltonian(ham,gamma)

    def set_hamiltonian(self,ham,gamma):
        super().set_hamiltonian(ham,gamma)
        # Parent constructor calls this before the subclass buffers exist.
        if 'past_coupling' not in self._buffers:
            return self
        coupling=np.zeros((ham.n_sites,ham.n_sites))
        for (i,j),J in zip(ham.bonds,ham.J):
            coupling[i,j]=coupling[j,i]=-J
        idx=np.arange(ham.n_sites)
        past=coupling*(idx[:,None]//3 > idx[None,:]//3)
        options=1-2*((np.arange(8)[:,None] >> np.arange(3)) & 1)
        cell_energy=[]
        for cell in range(self.n_cells):
            sub=coupling[3*cell:3*cell+3,3*cell:3*cell+3]
            cell_energy.append(.5*np.einsum('bi,ij,bj->b',options,sub,options)-ham.h*options.sum(1))
        dtype=self.embedding.weight.dtype
        self.past_coupling=torch.tensor(past,dtype=dtype,device=self.device)/gamma
        self.cell_energy=torch.tensor(np.stack(cell_energy),dtype=dtype,device=self.device)/gamma
        self.spin_options=torch.tensor(options,dtype=dtype,device=self.device)
        return self

    def logits(self,tokens):
        previous=torch.cat([torch.full_like(tokens[:,:1],8),tokens[:,:-1]],dim=1)
        spins=self.spin_options[tokens].reshape(len(tokens),-1)
        # Strictly lower triangular in cell indices: current/future spin tokens
        # cannot leak into the conditional for this cell.
        fields=(spins @ self.past_coupling.T).reshape(len(tokens),self.n_cells,3)
        context=self.static_block(self.condition(self.features)[None],self.geometry_bias)
        x=self.embedding(previous)+context+self.past_field(fields)
        for block in self.blocks:
            x=block(x,self.mask)
        local=fields @ self.spin_options.T+self.cell_energy[None]
        return self.output(self.norm(x))-F.softplus(self.energy_scale)*local
