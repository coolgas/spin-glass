"""One ALiBi checkpoint: balanced mixed-size gradients and all-context validation."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from .experiment import checkpoint_save, load_checkpoint, write_json, evaluate
from .physics import make_af_sample, exact_reference
from .unified import UnifiedTransformer
from .unified_experiment import tasks
from .vmc import local_energy, enumerate_model


def validate(model,contexts,samples,batches=8,batch=128,seed=910000):
    """Every record uses exactly the same frozen parameters, with rebuilt context."""
    rows=[]
    model.eval()
    for i,c in enumerate(contexts):
        sample=samples[c['L'],c['sample_id']]
        ham=sample.hamiltonian(c['h'])
        model.set_hamiltonian(ham,c['gamma'])
        row=dict(c,sampled=evaluate(model,ham,c['gamma'],batches=batches,batch=batch,seed=seed+i))
        if c['L']==2:
            exact=exact_reference(ham,c['gamma'],sample.points,2)
            measured=enumerate_model(model,ham,c['gamma'],sample.points,2)
            row['exact']={k:exact[k] for k in ('energy','q2','chi_connected')}
            row['enumerated']={k:measured[k] for k in ('energy','variance','q2','chi_connected','normalization')}
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dual',type=Path,required=True)
    parser.add_argument('--l2',type=Path,required=True)
    parser.add_argument('--sizes',type=Path,required=True)
    parser.add_argument('--expanded',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=6000)
    parser.add_argument('--distill',type=int,default=2400)
    parser.add_argument('--batch',type=int,default=256)
    args=parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contexts=tasks()
    samples={(L,s):make_af_sample(L,s) for L in (2,4,6,8) for s in (0,1,2)}
    for (L,s),sample in samples.items():
        np.savez_compressed(args.output/f'disorder_L{L}_s{s}.npz',K=-sample.J,bonds=sample.bonds,
                            points=sample.points,seed_entropy=sample.seed_entropy)
    shared=args.dual/'unified_alibi/checkpoint.pt'
    sources={L:shared for L in (2,4,6,8)}
    selected=json.loads((args.l2/'selected.json').read_text())
    sources[2]=args.l2/selected['best_checkpoint']
    for r in json.loads((args.sizes/'summary.json').read_text()):
        if r['seed']==201:
            sources[r['L']]=args.sizes/r['final_checkpoint']
    completed=json.loads((args.expanded/'summary.json').read_text())
    for r in completed:
        if r['L']==8 and r['seed']==301:
            sources[8]=Path(r['reports'][r['selected_stage']]['checkpoint'])
    architecture=dict(width=64,layers=4,heads=4)
    manifest=dict(architecture=architecture,contexts=contexts,rounds=args.rounds,batch=args.batch,
                  distillation_updates=args.distill,seed=401,
                  objective='uniform context mean energy per spin; all four sizes contribute before every optimizer step',
                  checkpoint_contract='one parameter state; no per-size weights or fine-tuning in evaluation/GFMC',
                  teachers={str(L):dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for L,p in sources.items()},
                  shared_teacher_sha256=hashlib.sha256(shared.read_bytes()).hexdigest(),
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    path=args.output/'campaign.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('campaign identity changed')
    write_json(path,manifest)
    torch.manual_seed(401)
    model=UnifiedTransformer(samples[2,0].hamiltonian(.3),.5,**architecture).cuda()
    teachers={}
    for p in set(sources.values())|{shared}:
        state=load_checkpoint(p,'cuda')
        config=state.get('metadata',{}).get('architecture',dict(width=32,layers=2,heads=4))
        teacher=UnifiedTransformer(samples[2,0].hamiltonian(.3),.5,**config).cuda()
        teacher.load_state_dict(state['model'])
        teacher.eval()
        teachers[p]=teacher
    generator=torch.Generator(device='cuda').manual_seed(402)
    latest=args.output/'shared_alibi__L02-04-06-08__w064__d04__contexts24__seed401__latest.pt'
    state=load_checkpoint(latest,'cuda') if latest.exists() else None
    if state:
        model.load_state_dict(state['model'])
        generator.set_state(state['generator'].cpu())
    history=state['history'] if state else []
    start=time.monotonic()
    for stage,total,lr in [('distill',args.distill,.001),('vmc',args.rounds,.0002)]:
        if state and state['stage']=='vmc' and stage=='distill':
            continue
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        first=0
        if state and state['stage']==stage:
            optimizer.load_state_dict(state['optimizer']);first=state['step']
        for step in range(first,total):
            model.train();optimizer.zero_grad()
            metrics=[]
            if stage=='distill':
                batch_contexts=[contexts[step%len(contexts)]]
            else:
                # Each of six (disorder,field) pairs gets all four sizes together.
                batch_contexts=[contexts[6*i+step%6] for i in range(4)]
            for c in batch_contexts:
                ham=samples[c['L'],c['sample_id']].hamiltonian(c['h'])
                model.set_hamiltonian(ham,c['gamma'])
                if stage=='distill':
                    p=sources[c['L']] if c['sample_id']==0 and c['gamma']==.5 and c['h']==.3 else shared
                    teacher=teachers[p].set_hamiltonian(ham,c['gamma'])
                    with torch.no_grad():
                        spins=teacher.sample(args.batch,generator)
                        tokens=teacher.tokens(spins)
                        target=teacher.logits(tokens).softmax(-1)
                    loss=-(target*model.logits(tokens).log_softmax(-1)).sum(-1).mean()
                    loss.backward()
                    metrics.append(dict(c,cross_entropy=float(loss.detach())))
                else:
                    spins=model.sample(args.batch,generator)
                    energy=local_energy(model,ham,c['gamma'],spins)
                    centered=(energy-energy.mean())*args.batch/(args.batch-1)/ham.n_sites
                    loss=(centered.detach()*model.log_prob(spins)).mean()/4
                    loss.backward()
                    metrics.append(dict(c,energy_per_spin=float(energy.mean())/ham.n_sites,
                                        variance_per_spin=float(energy.var())/ham.n_sites))
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            for group in optimizer.param_groups:
                group['lr']=lr*(.05+.95*.5*(1+math.cos(math.pi*(step+1)/total)))
            if step==0 or (step+1)%100==0 or step+1==total:
                history.append(dict(stage=stage,step=step+1,metrics=metrics,gradient=float(gradient),seconds=time.monotonic()-start))
                saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),generator=generator.get_state(),
                           stage=stage,step=step+1,history=history,metadata=manifest)
                checkpoint_save(latest,saved)
                write_json(args.output/'history.json',history)
                write_json(args.output/'status.json',dict(state='running',stage=stage,step=step+1,total=total))
                print('PROGRESS',stage,step+1,metrics,flush=True)
            if step+1==total or (stage=='vmc' and (step+1)%600==0):
                snapshot=args.output/f'shared_alibi__L02-04-06-08__w064__d04__contexts24__seed401__{stage}__step{step+1:06d}.pt'
                checkpoint_save(snapshot,saved)
                rows=validate(model,contexts,samples,seed=910000+step*32)
                write_json(args.output/f'{stage}_validation_{step+1:06d}.json',dict(
                    checkpoint=snapshot.name,sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    all_contexts_same_weights=True,rows=rows))
        state=None
    # One final artifact, evaluated everywhere without optimizer steps.
    final=args.output/'shared_alibi__L02-04-06-08__w064__d04__contexts24__seed401__final.pt'
    checkpoint_save(final,dict(model=model.state_dict(),metadata=manifest,stage='final',step=args.rounds))
    rows=validate(model,contexts,samples,batches=32,batch=256,seed=1700000)
    heldout=[dict(L=L,sample_id=2,gamma=.5,h=.3) for L in (2,4,6,8)]
    rows+=validate(model,heldout,samples,batches=32,batch=256,seed=1800000)
    write_json(args.output/'evaluation.json',dict(checkpoint=final.name,
                sha256=hashlib.sha256(final.read_bytes()).hexdigest(),all_contexts_same_weights=True,rows=rows))
    write_json(args.output/'status.json',dict(state='completed',shared_checkpoint=final.name))


if __name__=='__main__':
    main()
