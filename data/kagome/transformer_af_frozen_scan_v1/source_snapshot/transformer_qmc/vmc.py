"""Sampled energy-gradient optimization; exact data are evaluation-only."""

import time
import numpy as np
import torch

from .physics import basis, diagonal, correlations


@torch.no_grad()
def local_energy(model, ham, gamma, spins):
    bonds = torch.as_tensor(ham.bonds.copy(), device=model.device)
    coupling = torch.as_tensor(ham.J.copy(), device=model.device, dtype=torch.float64)
    diag = -ham.h * spins.sum(-1).double()
    diag -= (spins[:, bonds[:, 0]] * spins[:, bonds[:, 1]]).double() @ coupling
    ratios = model.all_flip_log_ratios(spins).double().exp()
    energies = diag - gamma * ratios.sum(-1)
    if not torch.isfinite(energies).all():
        raise FloatingPointError("nonfinite local energy")
    return energies


def train(model, ham, gamma, steps=1000, batch=512, lr=0.002, seed=0, report_every=100,
          checkpoint=None, resume=None):
    if steps < 1 or batch < 2:
        raise ValueError("need positive steps and batch>=2")
    torch.manual_seed(seed)
    generator = torch.Generator(device=model.device).manual_seed(seed + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    started = time.monotonic()
    first_step = 0
    if resume is not None:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        generator.set_state(resume['generator'].cpu())
        history = resume['history']
        first_step = resume['step']
    model.train()
    for step in range(first_step, steps):
        spins = model.sample(batch, generator)
        energy = local_energy(model, ham, gamma, spins)
        # For psi=sqrt(p), 2*dlog(psi)=dlog(p). Correct finite-batch centering.
        centered = (energy - energy.mean()) * batch / (batch - 1)
        loss = (centered.detach() * model.log_prob(spins)).mean()
        optimizer.zero_grad()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = lr * (0.1 + 0.9 * (1 - (step + 1) / steps))
        if step % report_every == 0 or step == steps - 1:
            row = dict(step=step + 1, energy=float(energy.mean()),
                       variance=float(energy.var(unbiased=True)), gradient=float(gradient),
                       seconds=time.monotonic() - started)
            history.append(row)
            print(f"step {step+1}/{steps} E={row['energy']:.7f} var={row['variance']:.5f}", flush=True)
            if checkpoint is not None:
                checkpoint(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                generator=generator.get_state(), step=step+1, history=history))
    return history


@torch.no_grad()
def enumerate_model(model, ham, gamma, points=None, L=None):
    states = basis(ham.n_sites)
    spins = torch.as_tensor(states, device=model.device, dtype=torch.long)
    logs = model.log_amplitudes(spins).double().cpu().numpy()
    probability = np.exp(2 * logs)
    normalization = probability.sum()
    ratios = np.exp(logs[np.arange(len(logs))[:, None] ^ (1 << np.arange(ham.n_sites))]
                    - logs[:, None])
    energies = diagonal(ham, states) - gamma * ratios.sum(-1)
    p = probability / normalization
    mean = float(p @ energies)
    result = correlations(p, states, points, L)
    result.update(energy=mean, variance=float(p @ (energies - mean)**2),
                  normalization=float(normalization), log_amplitudes=logs, probability=p)
    return result
