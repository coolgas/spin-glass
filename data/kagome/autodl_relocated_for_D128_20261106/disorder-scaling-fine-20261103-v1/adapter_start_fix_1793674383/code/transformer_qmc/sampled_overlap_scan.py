"""Matched-update sampled VMC field follow-up on the difficult disorder.

Each fit uses 1200 updates. Direct and continuation use the same sampler and
diagnostic budgets; the number of field stages sets updates per stage. Exact
references are used only by train() after fitting and only at L2.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .collective_campaign import write,label
from .correlation_lengths import sha
from .physics import make_af_sample,exact_reference
from .sampled_simple_guides import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fields',type=float,nargs='+',default=[.5,.6,.7,.75])
    p.add_argument('--sizes',type=int,nargs='+',default=[2,4])
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--hours',type=float,default=3)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    manifest=dict(fields=args.fields,sizes=args.sizes,seeds=args.seeds,disorder=50043,
        updates_per_fit=1200,chains=256,temperatures=8,diagnostic_sweeps=200,
        sources={n:sha(Path(__file__).with_name(n)) for n in
            ['sampled_overlap_scan.py','sampled_simple_guides.py','simple_guides.py','physics.py']},
        scope='Difficult-disorder diagnostic, equal update budgets; no ensemble or phase qualification')
    path=args.output/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:write(path,manifest)
    inputs=args.output/'inputs';(inputs/'guides').mkdir(parents=True,exist_ok=True)
    for gamma in args.fields:
        path=inputs/'guides'/(label(gamma,50043)+'.npz')
        if not path.exists():
            reference=exact_reference(make_af_sample(2,50043).hamiltonian(.15),gamma)
            np.savez_compressed(path,exact_amplitude=reference['amplitude'])
    tasks=[dict(L=L,gamma=g,sample_id=50043,kind=k,seed=s,mode=m)
        for L in args.sizes for g in args.fields for k in ['jastrow','rbm_2']
        for s in range(args.seeds) for m in ['direct','anneal']]
    deadline=time.monotonic()+args.hours*3600;rows=[]
    for task in tasks:
        stages=len([g for g in sorted(set([1.5,.75,task['gamma']]),reverse=True)
                    if g>=task['gamma']]) if task['mode']=='anneal' else 1
        assert 1200%stages==0
        settings=argparse.Namespace(device='cuda',chains=256,temperatures=8,
            steps=1200//stages,sweeps_per_step=2,diagnostic_sweeps=200,inputs=inputs)
        try:record=train(task,settings,args.output,deadline)
        except Exception as error:
            write(args.output/'failure.json',dict(task=task,error=repr(error)));raise
        if record['status']!='completed':
            write(args.output/'status.json',dict(state=record['status'],completed=len(rows),total=len(tasks),task=task))
            return
        rows.append(record);write(args.output/'review.json',dict(rows=rows))
        write(args.output/'status.json',dict(state='running',completed=len(rows),total=len(tasks),task=task))
        print(json.dumps(dict(task=task,completed=len(rows),seconds=record['seconds'],
            evaluation=record.get('exact_evaluation'))),flush=True)
    write(args.output/'status.json',dict(state='completed',completed=len(rows),total=len(tasks)))


if __name__=='__main__':main()
