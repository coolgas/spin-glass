"""Run the prespecified two-seed L2 feedback pilot sequentially on AutoDL."""
import argparse
from pathlib import Path

import torch

from .collective_campaign import write
from .self_learning_rbm import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    for seed in [0, 1]:
        args = argparse.Namespace(output=a.output/f'seed{seed}', initial=None,
            L=2, gamma=.6, sample_id=60000, seed=seed, stints=6, walkers=1024,
            equil=8., collect=12., delta=.1, sample_walkers=64, updates=1000,
            batch=256, gibbs=10, initial_sweeps=400, lr=.001, device='cuda')
        write(a.output/'status.json', dict(state='running', seed=seed, planned_seeds=[0, 1]))
        try:
            run(args)
        except Exception as error:
            write(a.output/'failure.json', dict(seed=seed, error=repr(error)))
            raise
    write(a.output/'status.json', dict(state='completed_unqualified', completed_seeds=[0, 1],
        ground_state_qualified=False))


if __name__ == '__main__':
    main()
