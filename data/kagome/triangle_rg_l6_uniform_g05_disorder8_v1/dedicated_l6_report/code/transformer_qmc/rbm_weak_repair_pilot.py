"""Bounded corrected weak-field L6 training pilot; execute on AutoDL."""
import argparse
import json
from pathlib import Path
import time

import torch

from .collective_campaign import write
from .correlation_lengths import sha
from .parallel_disorder_scaling import training_worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minutes',type=float,default=40)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tasks=[dict(L=6,gamma=g,sample_id=60000,kind='rbm_2',seed=s,mode='anneal')
           for g in [.55,.6] for s in [0,1]]
    settings=dict(device='cuda',chains=256,temperatures=8,steps=400,sweeps_per_step=2,
                  diagnostic_sweeps=200,learning_rate=.001,inputs=root/'inputs')
    sources={n:sha(Path(__file__).with_name(n)) for n in
             ['sampled_simple_guides.py','simple_guides.py','rbm_weak_repair_pilot.py']}
    write(root/'protocol.json',dict(tasks=tasks,settings={k:str(v) if isinstance(v,Path) else v for k,v in settings.items()},
          h=.15,N=108,boundary='periodic',sources=sources,budget_minutes=args.minutes,
          scope='One prespecified disorder; training repair pilot, not a disorder ensemble or GFMC result'))
    (root/'training').mkdir(exist_ok=True);deadline=time.monotonic()+args.minutes*60
    results=[]
    for task in tasks:
        if time.monotonic()>=deadline:break
        write(root/'status.json',dict(state='training',active=task,results=results))
        try:
            r=training_worker(task,settings,str(root/'training'),deadline)
            results.append(dict(task=task,status=r['status'],sampling_alarms=r.get('sampling_alarms')))
            if r['status']!='completed':break
        except Exception as error:
            results.append(dict(task=task,error=repr(error)))
        write(root/'review.json',dict(results=results))
    write(root/'status.json',dict(state='completed' if len(results)==len(tasks) and all(
          r.get('status')=='completed' for r in results) else 'incomplete',results=results,
          ground_state_qualified=False))


if __name__=='__main__':main()
