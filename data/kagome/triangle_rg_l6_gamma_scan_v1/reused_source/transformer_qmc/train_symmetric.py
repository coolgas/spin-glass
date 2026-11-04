"""Resumable all-size VMC with exact symmetry and broader quenched disorder."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from kagome_qea.exact import operator_matrix
from .diagnose_shared import measure, reference
from .experiment import checkpoint_save, load_checkpoint, write_json, evaluate
from .physics import basis, make_af_sample
from .symmetric import SymmetricTransformer
from .vmc import local_energy


@torch.no_grad()
def validate(model, ids, *, full=False):
    rows=[]
    spins=torch.tensor(basis(12),device=model.device,dtype=torch.long)
    model.eval()
    for s in ids:
        sample=make_af_sample(2,s)
        for gamma,h in ((.5,0.),(.5,.3),(2.,.3)):
            ham=sample.hamiltonian(h)
            model.set_hamiltonian(ham,gamma)
            logs=model.log_amplitudes(spins).double().cpu().numpy()
            p=np.exp(2*logs)
            if abs(p.sum()-1)>2e-5:
                raise RuntimeError('enumerated normalization failed')
            matrix=operator_matrix(ham,gamma)
            energy,exact,sectors=reference(matrix,h)
            result=measure(np.sqrt(p),matrix,(energy,exact),spins.cpu().numpy())
            if h==0 and result['rms_site_magnetization']>1e-7:
                raise RuntimeError('spin flip symmetry failed')
            rows.append(dict(L=2,sample_id=s,gamma=gamma,h=h,**result,
                             exact_chi=measure(exact,matrix,(energy,exact),spins.cpu().numpy())['chi_connected']))
    for L in (4,6,8):
        for s in ([0,10000,10001] if full else [10000]):
            for gamma,h in ((.5,0.),(.5,.3),(2.,.3)):
                ham=make_af_sample(L,s).hamiltonian(h)
                model.set_hamiltonian(ham,gamma)
                result=evaluate(model,ham,gamma,batches=16 if full else 4,batch=128,
                                seed=200000+L*100+s+int(h*10))
                rows.append(dict(L=L,sample_id=s,gamma=gamma,h=h,sampled=result))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--validate-every',type=int,default=600)
    parser.add_argument('--seed',type=int,default=501)
    parser.add_argument('--initialize',type=Path)
    parser.add_argument('--test-start',type=int,default=20000)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for training')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    architecture=dict(width=64,layers=4,heads=4,physical=True)
    train_ids=[0,1]+list(range(100,356))
    manifest=dict(architecture=architecture,seed=args.seed,steps=args.steps,batch=args.batch,
                  validate_every=args.validate_every,train_ids=train_ids,
                  validation_ids=[10000,10001],final_test_ids=list(range(args.test_start,args.test_start+8)),
                  fields=[[.5,0.],[.5,.3],[2.,.3]],sizes=[2,4,6,8],
                  symmetry='canonical bond origin with tied-origin mixture; h=0 probability flip mixture',
                  objective='equal mean energy per spin across all four sizes in every optimizer step',
                  exact_labels_used_for_training=False,
                  initialization_sha256=(hashlib.sha256(args.initialize.read_bytes()).hexdigest()
                                         if args.initialize else None),
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*.py')})
    path=args.output/'campaign.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('campaign identity mismatch')
    write_json(path,manifest)
    if set(train_ids) & set(manifest['validation_ids']+manifest['final_test_ids']):
        raise ValueError('training and evaluation disorder IDs overlap')
    torch.manual_seed(args.seed)
    model=SymmetricTransformer(make_af_sample(2).hamiltonian(0.),.5,**architecture).cuda()
    if args.initialize:
        model.load_state_dict(load_checkpoint(args.initialize,'cuda')['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
    generator=torch.Generator(device='cuda').manual_seed(args.seed+1)
    rng=np.random.default_rng(args.seed+2)
    latest=args.output/f'shared_symmetric_alibi__L02-04-06-08__seed{args.seed}__latest.pt'
    history,first,elapsed=[],0,0.
    if latest.exists():
        saved=load_checkpoint(latest,'cuda')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'].cpu());rng.bit_generator.state=saved['numpy_rng']
        first,history,elapsed=saved['step'],saved['history'],saved['elapsed']
    start=time.monotonic()
    for step in range(first,args.steps):
        # Rotating field + randomized disorder, all sizes before updating weights.
        gamma,h=manifest['fields'][step%3]
        sample_id=int(rng.choice(train_ids))
        optimizer.zero_grad();model.train()
        metrics=[]
        for L in manifest['sizes']:
            ham=make_af_sample(L,sample_id).hamiltonian(h)
            model.set_hamiltonian(ham,gamma)
            spins=model.sample(args.batch,generator)
            energy=local_energy(model,ham,gamma,spins)
            centered=(energy-energy.mean())*args.batch/(args.batch-1)/ham.n_sites
            loss=(centered.detach()*model.log_prob(spins)).mean()/4
            loss.backward()
            metrics.append(dict(L=L,energy_per_spin=float(energy.mean())/ham.n_sites,
                                variance_per_spin=float(energy.var())/ham.n_sites))
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        if not torch.isfinite(gradient):
            raise RuntimeError('nonfinite gradient')
        optimizer.step()
        lr=.0005*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/args.steps)))
        for group in optimizer.param_groups:
            group['lr']=lr
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            elapsed_now=elapsed+time.monotonic()-start
            history.append(dict(step=step+1,seconds=elapsed_now,sample_id=sample_id,
                                gamma=gamma,h=h,metrics=metrics,gradient=float(gradient),lr=lr))
            saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                       generator=generator.get_state(),numpy_rng=rng.bit_generator.state,
                       step=step+1,history=history,elapsed=elapsed_now,metadata=manifest)
            checkpoint_save(latest,saved)
            write_json(args.output/'history.json',history)
            write_json(args.output/'status.json',dict(state='running',step=step+1,total=args.steps))
            print('TRAIN',step+1,round(elapsed_now,2),metrics,flush=True)
        if (step+1)%args.validate_every==0 or step+1==args.steps:
            snapshot=args.output/f'shared_symmetric_alibi__step{step+1:06d}.pt'
            # Create current state even when cadence is not a multiple of 100.
            checkpoint_save(snapshot,dict(model=model.state_dict(),metadata=manifest,step=step+1))
            write_json(args.output/'status.json',dict(state='validating',step=step+1,total=args.steps))
            rows=validate(model,[0,1,10000,10001])
            write_json(args.output/f'validation_{step+1:06d}.json',dict(checkpoint=snapshot.name,rows=rows))
            print('VALIDATE',step+1,[(r['sample_id'],r['h'],r['energy_error_per_spin'])
                                    for r in rows if r['L']==2 and r['gamma']==.5],flush=True)
    final=args.output/f'shared_symmetric_alibi__L02-04-06-08__seed{args.seed}__final.pt'
    checkpoint_save(final,dict(model=model.state_dict(),metadata=manifest,step=args.steps))
    write_json(args.output/'status.json',dict(state='final_evaluation',step=args.steps))
    rows=validate(model,[0,1,2,3,4,5,6,7,8,9,10000,10001]+manifest['final_test_ids'],full=True)
    write_json(args.output/'evaluation.json',dict(checkpoint=final.name,
                sha256=hashlib.sha256(final.read_bytes()).hexdigest(),rows=rows))
    write_json(args.output/'status.json',dict(state='completed',checkpoint=final.name))


if __name__=='__main__':
    main()
