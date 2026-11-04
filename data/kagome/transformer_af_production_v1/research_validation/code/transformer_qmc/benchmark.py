"""Bounded GPU throughput measurement for choosing campaign settings."""

import time
import torch
from .physics import make_af_sample
from .model import CellTransformer
from .vmc import train


def main():
    torch.set_num_threads(4)
    for L in (2,4,6,8):
        ham=make_af_sample(L).hamiltonian(.3)
        model=CellTransformer(ham,.5).cuda()
        start=time.monotonic()
        train(model,ham,.5,steps=3,batch=128,report_every=1)
        torch.cuda.synchronize()
        print('BENCH',L,'seconds',time.monotonic()-start,
              'peak_GB',torch.cuda.max_memory_allocated()/1e9,flush=True)
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
