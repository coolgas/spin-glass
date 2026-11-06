"""Bounded direct-neural-guide collective GFMC cost probes on CUDA.

No enumeration is used at larger sizes. Short trajectories measure execution
cost and weight stability, and do not establish ground-state convergence.
"""
import argparse
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .collective_campaign import write, source_hashes
from .collective_projector import SplitProposal
from .correlation_lengths import sha
from .frozen_scan import model_from, SELECTED_SHA256
from .model import TransformerGuide
from .physics import make_af_sample


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=7200)
    args=parser.parse_args()
    assert sha(args.checkpoint)==SELECTED_SHA256 and torch.cuda.is_available()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(20261103)
    record=dict(state='running',checkpoint_sha256=SELECTED_SHA256,
        source_sha256=source_hashes(),pilot_source_sha256=sha(Path(__file__)),rows=[],
        scope='Float64 direct neural guide, no enumeration. Short cost probes only.',
        ground_state_qualified=False)
    started=time.monotonic()
    def expired(signum,frame):raise TimeoutError('GPU probe budget reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.seconds)
    write(args.output/'results.json',record)
    model=model_from(args.checkpoint,'cuda')
    try:
        for L in [2,4,6,8,10]:
            ham=make_af_sample(L,50025).hamiltonian(.15)
            model.set_hamiltonian(ham,.25);guide=TransformerGuide(model)
            for walkers in [128,512,1024]:
                for method in ['natural','guided','broad']:
                    begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
                    write(args.output/'progress.json',dict(L=L,walkers=walkers,method=method,state='sampling',updated_unix=time.time()))
                    initial=guide.sample(4*walkers,20261103+L+walkers).reshape(4,walkers,ham.n_sites)
                    sample_seconds=time.monotonic()-begin
                    states=initial.copy();rng=np.random.default_rng(9900+L+walkers)
                    kernel=SplitProposal(ham,.25,guide,.05,.1 if method=='broad' else 0.,natural=method=='natural')
                    ancestors=np.tile(np.arange(walkers),(4,1));ess=[];durations=[]
                    proposed=np.zeros(ham.n_sites+1,dtype=np.int64);surviving=proposed.copy()
                    for step in range(16):
                        write(args.output/'progress.json',dict(L=L,walkers=walkers,method=method,step=step,state='projecting',updated_unix=time.time()))
                        begin=time.monotonic()
                        states,parents,stats=kernel.step(states,rng)
                        torch.cuda.synchronize();durations.append(time.monotonic()-begin)
                        ancestors=ancestors[np.arange(4)[:,None],parents]
                        ess.append(stats['ess']);proposed+=stats['proposed'];surviving+=stats['surviving']
                    name=f'L{L}_{method}_w{walkers}'
                    np.savez_compressed(args.output/(name+'.npz'),initial=initial,states=states,ancestors=ancestors,
                        weight_ess=ess,seconds_per_step=durations,proposed=proposed,surviving=surviving)
                    row=dict(L=L,walkers=walkers,method=method,steps=16,delta=.05,
                        sample_seconds=sample_seconds,median_step_seconds=float(np.median(durations)),
                        minimum_weight_ess=float(np.min(ess)),mean_weight_ess=float(np.mean(ess)),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(),npz_sha256=sha(args.output/(name+'.npz')),
                        ground_state_qualified=False)
                    record['rows'].append(row);write(args.output/'results.json',record)
                    print(row,flush=True)
        record['state']='completed'
    except TimeoutError:
        record['state']='stopped_time_cap'
    except Exception as error:
        record['state']='failed';record['error']=repr(error)
        raise
    finally:
        signal.alarm(0);record['seconds']=time.monotonic()-started
        write(args.output/'results.json',record)


if __name__=='__main__':main()
