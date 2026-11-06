"""Controlled guide-probability mixing after the collective proposal benchmark.

Mixing uses frozen guide probabilities only, with no exact-state information.
This is an enumerated L2 diagnostic; scalable mixture sampling and amplitudes
would require two evaluable normalized guides.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from . import collective_campaign as campaign
from .correlation_lengths import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=18)
    parser.add_argument('--hours',type=float,default=6.)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    campaign.write(args.root/'support-status.json',dict(state='waiting_for_primary_cpu_batch',pid=os.getpid()))
    # One blocking lock acquisition queues this compute stage without polling.
    with (args.root/'overnight-results/controller.lock').open('a') as primary:
        fcntl.flock(primary,fcntl.LOCK_EX)
        started=time.monotonic()
        original_tasks=campaign.tasks
        campaign.tasks=lambda stage:[t for t in original_tasks(stage) if t['guide']=='corrected']
        for fraction in [.01,.1,.5]:
            name=f'mix{fraction:g}'.replace('.','p')
            root=args.root/name;inputs=root/'inputs';(inputs/'guides').mkdir(parents=True,exist_ok=True)
            definition=dict(original_probability_fraction=fraction,
                formula='psi_mix = sqrt((1-a) psi_corrected^2 + a psi_original^2), each component normalized',
                exact_labels_used=False,source_sha256=sha(Path(__file__)),parents={})
            for gamma,sid in campaign.CONTEXTS:
                filename=campaign.label(gamma,sid)+'.npz';parent=args.inputs/'guides'/filename
                with np.load(parent) as saved:arrays={key:saved[key].copy() for key in saved.files}
                original=2*arrays['original_logs'];original-=logsumexp(original)
                corrected=2*arrays['logs'];corrected-=logsumexp(corrected)
                arrays['logs']=.5*np.logaddexp(np.log1p(-fraction)+corrected,np.log(fraction)+original)
                assert abs(logsumexp(2*arrays['logs']))<1e-12
                np.savez_compressed(inputs/'guides'/filename,**arrays)
                definition['parents'][filename]=sha(parent)
            campaign.write(root/'guide-definition.json',definition)
            campaign.write(args.root/'support-status.json',dict(state='running',mixture=name,pid=os.getpid()))
            remaining=args.hours-(time.monotonic()-started)/3600
            if remaining<=0:break
            campaign.campaign(root/'results',inputs,args.workers,['pilot','main','audit'],remaining)
        campaign.write(args.root/'support-status.json',dict(state='completed_or_budget_reached',seconds=time.monotonic()-started))


if __name__=='__main__':main()
