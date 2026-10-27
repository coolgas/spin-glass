"""Capacity expansion, long sampled VMC, and independent qualification on AutoDL.

No projector is launched here. Exact states are used only to evaluate L2.
"""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .experiment import load_checkpoint, checkpoint_save, write_json, evaluate
from .physics import make_af_sample, exact_reference
from .refine_l2 import diagnostics
from .unified import UnifiedTransformer
from .vmc import local_energy


def load_alibi(path, ham, device='cuda'):
    state=load_checkpoint(path,device)
    model=UnifiedTransformer(ham,.5,**state['metadata']['architecture']).to(device)
    model.load_state_dict(state['model'])
    return model, state


def energy_validation(model,ham,seed,batches=24,batch=256):
    generator=torch.Generator(device=model.device).manual_seed(seed)
    means,variances=[],[]
    model.eval()
    for _ in range(batches):
        states=model.sample(batch,generator)
        energy=local_energy(model,ham,.5,states).cpu().numpy()
        means.append(float(energy.mean()))
        variances.append(float(energy.var(ddof=1)))
    return dict(mean=float(np.mean(means)),se=float(np.std(means,ddof=1)/np.sqrt(batches)),
                variance=float(np.mean(variances)),batches=means,seed=seed,samples=batch*batches)


def distilled_student(teacher,ham,seed,steps=1000,batch=512):
    """Transfer learned conditionals to a larger ALiBi model, without ED labels."""
    torch.manual_seed(seed)
    student=UnifiedTransformer(ham,.5,width=64,layers=4,heads=4).cuda()
    optimizer=torch.optim.Adam(student.parameters(),lr=.001)
    generator=torch.Generator(device='cuda').manual_seed(seed+1)
    teacher.eval()
    trace=[]
    for step in range(steps):
        with torch.no_grad():
            spins=teacher.sample(batch,generator)
            tokens=teacher.tokens(spins)
            target=teacher.logits(tokens).softmax(-1)
        logp=student.logits(tokens).log_softmax(-1)
        loss=-(target*logp).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),10.)
        optimizer.step()
        if (step+1)%200==0 or step+1==steps:
            kl=(target*(target.clamp_min(1e-30).log()-logp.detach())).sum(-1).mean()
            trace.append(dict(step=step+1,conditional_kl=float(kl)))
            print('DISTILL',seed,trace[-1],flush=True)
    return student,trace


def optimize(model,ham,directory,metadata,*,steps,batch,lr,seed,stage,validation_every=1000):
    """Resumable sampled VMC with variance diagnostics and frozen candidates."""
    directory.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    generator=torch.Generator(device='cuda').manual_seed(seed)
    latest=directory/f'{metadata["name"]}__{stage}__latest.pt'
    first,history,validation=0,[],[]
    if latest.exists():
        state=load_checkpoint(latest,'cuda')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['generator'].cpu())
        first,history,validation=state['step'],state['history'],state['validation']
    start=time.monotonic()
    for step in range(first,steps):
        model.train()
        spins=model.sample(batch,generator)
        energy=local_energy(model,ham,.5,spins)
        centered=(energy-energy.mean())*batch/(batch-1)
        optimizer.zero_grad()
        loss=(centered.detach()*model.log_prob(spins)).mean()
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr']=lr*(.05+.95*.5*(1+math.cos(math.pi*(step+1)/steps)))
        if (step+1)%100==0 or step==0:
            row=dict(step=step+1,energy=float(energy.mean()),variance=float(energy.var()),
                     gradient_norm=float(norm),lr=optimizer.param_groups[0]['lr'],
                     seconds=time.monotonic()-start)
            history.append(row)
            print('TRAIN',metadata['name'],stage,row,flush=True)
        if (step+1)%validation_every==0 or step+1==steps:
            measured=energy_validation(model,ham,seed+100000+step)
            candidate=directory/f'{metadata["name"]}__{stage}__step{step+1:06d}.pt'
            validation.append(dict(step=step+1,checkpoint=candidate.name,**measured))
            checkpoint_save(candidate,dict(model=model.state_dict(),metadata=metadata,stage=stage,
                                           step=step+1,validation_energy=measured))
            write_json(directory/f'{stage}_validation.json',validation)
            print('VALIDATE',metadata['name'],stage,step+1,measured['mean']/ham.n_sites,
                  measured['variance']/ham.n_sites,flush=True)
        if (step+1)%200==0 or step==0 or step+1==steps:
            checkpoint_save(latest,dict(model=model.state_dict(),metadata=metadata,stage=stage,
                                       step=step+1,optimizer=optimizer.state_dict(),
                                       generator=generator.get_state(),history=history,validation=validation))
            write_json(directory/f'{stage}_history.json',history)
    # Selection uses only interim energy validation, never the final evaluation.
    best=min(validation,key=lambda row:row['mean'])
    model.load_state_dict(load_checkpoint(directory/best['checkpoint'],'cuda')['model'])
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sizes-results',type=Path,required=True)
    parser.add_argument('--l2-results',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=8000)
    parser.add_argument('--polish-steps',type=int,default=4000)
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--sizes',type=int,nargs='+',default=[8,6,4,2])
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parents=json.loads((args.sizes_results/'summary.json').read_text())
    l2=json.loads((args.l2_results/'summary.json').read_text())
    tasks=[]
    for L in args.sizes:
        for branch in (0,1):
            if L==2:
                record=next(r for r in l2 if r['task']['initialization']=='shared' and r['task']['seed']==101+branch)
                parent=args.l2_results/record['best_checkpoint']
            else:
                record=next(r for r in parents if r['L']==L and r['seed']==201+branch)
                parent=args.sizes_results/record['final_checkpoint']
            tasks.append(dict(L=L,seed=301+branch,parent=str(parent),
                              parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest()))
    manifest=dict(tasks=tasks,architecture=dict(width=64,layers=4,heads=4),
                  steps=args.steps,polish_steps=args.polish_steps,batch=args.batch,
                  polish_batch=2*args.batch,lr=.0003,polish_lr=.00005,distillation_steps=1000,
                  selection='interim energy only; selected checkpoints independently evaluated',
                  gamma=.5,h=.3,sample_id=0,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                  torch=str(torch.__version__),gpu=torch.cuda.get_device_name())
    path=args.output/'campaign.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('campaign identity changed')
    write_json(path,manifest)
    summary=[]
    for task in tasks:
        L,seed=task['L'],task['seed']
        name=f'alibi__L{L:02d}__N{3*L*L:03d}__w064__d04__g0p5__h0p3__dis000__seed{seed}'
        directory=args.output/name
        directory.mkdir(exist_ok=True)
        if (directory/'evaluation.json').exists():
            summary.append(json.loads((directory/'evaluation.json').read_text()))
            continue
        metadata=dict(name=name,architecture=manifest['architecture'],L=L,N=3*L*L,
                      gamma=.5,h=.3,sample_id=0,task=task)
        write_json(directory/'settings.json',metadata)
        sample=make_af_sample(L,0)
        ham=sample.hamiltonian(.3)
        np.savez_compressed(directory/'disorder.npz',K=-sample.J,bonds=sample.bonds,points=sample.points)
        write_json(args.output/'status.json',dict(state='running',task=task,stage='distill',completed=len(summary)))
        teacher,_=load_alibi(task['parent'],ham)
        initial=directory/f'{name}__distilled.pt'
        if initial.exists():
            model,_=load_alibi(initial,ham)
        else:
            model,trace=distilled_student(teacher,ham,seed)
            checkpoint_save(initial,dict(model=model.state_dict(),metadata=metadata,step=1000,stage='distill'))
            write_json(directory/'distillation.json',trace)
        if not (directory/'initial_validation.json').exists():
            write_json(directory/'initial_validation.json',dict(
                parent=energy_validation(teacher,ham,50000+seed),
                distilled=energy_validation(model,ham,51000+seed)))
        del teacher
        write_json(args.output/'status.json',dict(state='running',task=task,stage='vmc',completed=len(summary)))
        best=optimize(model,ham,directory,metadata,steps=args.steps,batch=args.batch,lr=.0003,
                      seed=seed+1000,stage='vmc')
        write_json(directory/'vmc_selected.json',best)
        write_json(args.output/'status.json',dict(state='running',task=task,stage='polish',completed=len(summary)))
        polish=optimize(model,ham,directory,metadata,steps=args.polish_steps,batch=2*args.batch,
                        lr=.00005,seed=seed+2000,stage='polish')
        # Both stage winners get independent evaluations; retain both and parent.
        reports={}
        for label,candidate in [('vmc',directory/best['checkpoint']),('polish',directory/polish['checkpoint']),
                                ('parent',Path(task['parent']))]:
            evaluated,_=load_alibi(candidate,ham)
            report=dict(checkpoint=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                        sampled=evaluate(evaluated,ham,.5,batches=64,batch=256,seed=700000+seed+100*len(reports)))
            if L==2:
                reference=exact_reference(ham,.5,sample.points,2)
                report['exact_diagnostics']=diagnostics(evaluated,ham,sample,reference)
            reports[label]=report
            del evaluated
        # Candidate choice is made from interim validation. Final samples audit it.
        parent_validation=json.loads((directory/'initial_validation.json').read_text())['parent']
        selection_energies={'parent':parent_validation['mean'],'vmc':best['mean'],
                            'polish':polish['mean']}
        chosen=min(selection_energies,key=selection_energies.get)
        result=dict(name=name,L=L,N=ham.n_sites,seed=seed,selected_stage=chosen,
                    reports=reports,ground_state_convergence_established=False)
        write_json(directory/'evaluation.json',result)
        summary.append(result)
        write_json(args.output/'summary.json',summary)
        print('DONE',name,chosen,flush=True)
    write_json(args.output/'summary.json',summary)
    write_json(args.output/'status.json',dict(state='completed',completed=len(summary)))
    write_json(args.output/'finished.json',dict(finished=True,ground_state_convergence_established=False))


if __name__=='__main__':
    main()
