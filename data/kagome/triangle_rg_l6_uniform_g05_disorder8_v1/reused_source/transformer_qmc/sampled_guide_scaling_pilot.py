"""GPU feasibility pilot of sampled neural-guide refinement without enumeration.

This measures feasibility and variational diagnostics, not GFMC convergence.
L2 exact evaluation is performed only after all training steps.
"""
import argparse
import json
from math import comb
from pathlib import Path
import time

import numpy as np
import torch

from .correlation_lengths import sha
from .correlation_remedy_campaign import write
from .frozen_scan import SELECTED_SHA256, model_from
from .physics import make_af_sample
from .vmc import local_energy, train, enumerate_model


def evaluate(model,ham,gamma,seed,batches=16,batch=128):
    generator=torch.Generator(device=model.device).manual_seed(seed)
    means=[];variances=[]
    with torch.no_grad():
        for _ in range(batches):
            spins=model.sample(batch,generator)
            energy=local_energy(model,ham,gamma,spins)
            means.append(float(energy.mean()));variances.append(float(energy.var(unbiased=True)))
    return dict(energy=float(np.mean(means)),energy_se=float(np.std(means,ddof=1)/np.sqrt(batches)),
                local_energy_variance=float(np.mean(variances)),samples=batches*batch)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    assert sha(a.checkpoint)==SELECTED_SHA256 and torch.cuda.is_available()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    contexts=[dict(L=2,gamma=.25,sample_id=50004,steps=200),dict(L=4,gamma=.5,sample_id=50004,steps=100)]
    manifest=dict(contexts=contexts,batch=128,learning_rate=.00015,checkpoint_sha256=SELECTED_SHA256,
        training='Autoregressive independent samples; covariance energy gradient; Adam',
        enumerated_training=False,exact_training_labels=False,quartic_correction=False,
        qualification='Feasibility pilot only; energy improvement does not qualify correlations or GFMC')
    write(a.output/'manifest.json',manifest)
    scaling=[]
    for L in [2,4,6,8,10]:
        N=3*L*L;count=sum(comb(N,k) for k in range(1,5))
        scaling.append(dict(L=L,N=N,enumerated_states=str(2**N),quartic_coefficients=count,
            dense_feature_matrix_float64_bytes=str(8*(2**N)*count)))
    write(a.output/'enumeration_scaling.json',scaling)
    for context in contexts:
        L,g,sid,steps=[context[k] for k in ['L','gamma','sample_id','steps']]
        name=f'L{L}_g{g:g}_s{sid}'.replace('.','p');dest=a.output/(name+'.json')
        if dest.exists():continue
        started=time.monotonic();model=model_from(a.checkpoint,'cuda');model.requires_grad_(True)
        ham=make_af_sample(L,sid).hamiltonian(.15);model.set_hamiltonian(ham,g)
        seed=20261102+L;torch.cuda.reset_peak_memory_stats()
        initial=evaluate(model,ham,g,seed+91)
        training_start=time.monotonic()
        history=train(model,ham,g,steps=steps,batch=128,lr=.00015,seed=seed,report_every=20)
        torch.cuda.synchronize();training_seconds=time.monotonic()-training_start
        final=evaluate(model,ham,g,seed+92)
        torch.save(dict(model=model.state_dict(),context=context,manifest=manifest),a.output/(name+'.pt'))
        record=dict(context=context,initial=initial,final=final,history=history,
            parameter_count=sum(p.numel() for p in model.parameters()),training_seconds=training_seconds,
            seconds_per_step=training_seconds/steps,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            exact_labels_used=False,training_enumeration=False,ground_state_qualified=False)
        if L==2:
            result=enumerate_model(model,ham,g)
            np.savez_compressed(a.output/(name+'.npz'),logs=result['log_amplitudes'])
            record['post_training_enumeration']=dict(energy=result['energy'],variance=result['variance'],
                                                     normalization=result['normalization'])
        record['seconds']=time.monotonic()-started;write(dest,record)
        print(json.dumps(record),flush=True)
        del model;torch.cuda.empty_cache()
    write(a.output/'status.json',dict(state='completed',contexts=len(contexts)))


if __name__=='__main__':main()
