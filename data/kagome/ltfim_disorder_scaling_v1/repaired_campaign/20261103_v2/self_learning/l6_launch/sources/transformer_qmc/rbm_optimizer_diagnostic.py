"""Bounded matched optimizer diagnostic; execute on AutoDL, not locally."""
import argparse
import json
from pathlib import Path
import time

import torch

from .collective_campaign import write
from .correlation_lengths import sha
from .physics import make_af_sample
from .sampled_simple_guides import TemperedChains, energy_loss, freeze_diagnostics, local_energy
from .simple_guides import RBM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ham = make_af_sample(6, 60000).hamiltonian(.15)
    sources = {name: sha(Path(__file__).with_name(name)) for name in
               ['rbm_optimizer_diagnostic.py', 'sampled_simple_guides.py', 'simple_guides.py', 'physics.py']}
    write(args.output/'protocol.json', dict(L=6, N=108, h=.15, periodic=True,
          sample_id=60000, seed=0, chains=256, temperatures=8, sweeps_per_step=2,
          steps=args.steps, sources=sources, purpose='Matched step-size diagnostic; not an ensemble'))
    for gamma, lr in [(1.5, .01), (1.5, .001), (4., .01), (4., .001)]:
        key = f'g{gamma:g}_lr{lr:g}'.replace('.', 'p')
        model = RBM(108, 2, 0).to('cuda')
        sampler = TemperedChains(model, 256, 8, 100)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = []
        started = time.monotonic()
        for step in range(args.steps):
            for _ in range(2):
                states = sampler.sweep()
            energy = local_energy(model, states, ham, gamma)
            optimizer.zero_grad()
            energy_loss(model(states), energy, 108).backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if (step+1) % 25 == 0:
                row = dict(step=step+1, energy=float(energy.mean()), variance=float(energy.var()),
                           gradient_norm=float(gradient), **sampler.diagnostics())
                history.append(row)
                write(args.output/(key+'_progress.json'), dict(gamma=gamma, lr=lr, history=history))
                print(json.dumps(dict(gamma=gamma, lr=lr, **row)), flush=True)
        diagnostics, arrays = freeze_diagnostics(model, ham, gamma, 128, 8, 50, 0)
        import numpy as np
        np.savez_compressed(args.output/(key+'.npz'), **arrays)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), args.output/(key+'.pt'))
        write(args.output/(key+'.json'), dict(gamma=gamma, lr=lr, history=history, diagnostics=diagnostics,
              seconds=time.monotonic()-started, ground_state_qualified=False))
    write(args.output/'complete.json', dict(configurations=4, ground_state_qualified=False))


if __name__ == '__main__':
    main()
