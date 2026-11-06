"""Bounded CUDA throughput benchmark using the production sampled VMC step."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import torch

from .physics import make_af_sample
from .sampled_simple_guides import TemperedChains,energy_loss,local_energy
from .simple_guides import Jastrow,RBM


def worker(L,kind,steps,barrier):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ham=make_af_sample(L,60000).hamiltonian(.15)
    model=(Jastrow(ham.n_sites) if kind=='jastrow' else RBM(ham.n_sites,2)).cuda()
    sampler=TemperedChains(model,256,8,100)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    def update():
        for _ in range(2):states=sampler.sweep()
        energy=local_energy(model,states,ham,.7)
        optimizer.zero_grad();energy_loss(model(states),energy,ham.n_sites).backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
    for _ in range(5):update()
    torch.cuda.synchronize();barrier.wait(timeout=120)
    started=time.monotonic()
    for _ in range(steps):update()
    torch.cuda.synchronize()
    return dict(L=L,kind=kind,steps=steps,seconds=time.monotonic()-started,
                peak_memory_bytes=torch.cuda.max_memory_allocated())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--sizes',type=int,nargs='+',default=[4,6])
    p.add_argument('--trainers',type=int,nargs='+',default=[1,2,4])
    a=p.parse_args();ctx=mp.get_context('spawn');rows=[]
    with ctx.Manager() as manager:
        for L in a.sizes:
            for count in a.trainers:
                for kind in ['jastrow','rbm_2']:
                    barrier=manager.Barrier(count)
                    with ProcessPoolExecutor(max_workers=count,mp_context=ctx) as pool:
                        futures=[pool.submit(worker,L,kind,a.steps,barrier) for _ in range(count)]
                        results=[f.result() for f in futures]
                    seconds=max(r['seconds'] for r in results)
                    row=dict(L=L,kind=kind,trainers=count,results=results,
                             aggregate_updates_per_second=count*a.steps/seconds)
                    rows.append(row);print(json.dumps(row),flush=True)
                    a.output.parent.mkdir(parents=True,exist_ok=True)
                    a.output.write_text(json.dumps(dict(rows=rows,scope='Production paused during isolated MPS benchmark'),indent=2)+'\n')


if __name__=='__main__':main()
