"""Independent ALiBi refinement at L4/L6/L8 with fresh-sample comparisons."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from .experiment import load_checkpoint, save_training_checkpoint, write_json, evaluate
from .physics import make_af_sample
from .unified import UnifiedTransformer
from .vmc import train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shared',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2400)
    parser.add_argument('--batch',type=int,default=128)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('AutoDL CUDA training required')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    architecture=dict(width=32,layers=2,heads=4)
    tasks=[dict(L=L,seed=seed) for L in (4,6,8) for seed in (201,202)]
    manifest=dict(tasks=tasks,gamma=.5,h=.3,sample_id=0,architecture=architecture,
                  steps=args.steps,batch=args.batch,lr=.0005,
                  selection='predeclared final step; both independent seeds retained',
                  initialization='original shared checkpoint, not L2-specialized weights',
                  shared_sha256=hashlib.sha256(args.shared.read_bytes()).hexdigest(),
                  torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                  sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in Path(__file__).parent.glob('*.py')})
    path=args.output/'campaign.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise RuntimeError('campaign identity changed')
    write_json(path,manifest)
    summary=[]
    for task in tasks:
        L,seed=task['L'],task['seed']
        name=f'alibi__L{L:02d}__N{3*L*L:03d}__g0p5__h0p3__dis000__init-shared__seed{seed}'
        directory=args.output/name
        directory.mkdir(exist_ok=True)
        if (directory/'evaluation.json').exists():
            summary.append(json.loads((directory/'evaluation.json').read_text()))
            continue
        write_json(args.output/'status.json',dict(state='running',task=task,completed=len(summary)))
        sample=make_af_sample(L,0)
        ham=sample.hamiltonian(.3)
        np.savez_compressed(directory/'disorder.npz',K=-sample.J,bonds=sample.bonds,points=sample.points)
        torch.manual_seed(seed)
        model=UnifiedTransformer(ham,.5,**architecture).cuda()
        model.load_state_dict(load_checkpoint(args.shared,'cuda')['model'])
        metadata=dict(manifest,task=task,name=name,model_type='alibi_hamiltonian_specialized')
        write_json(directory/'settings.json',metadata)
        latest=directory/'checkpoint.pt'
        resume=load_checkpoint(latest,'cuda') if latest.exists() else None
        validation_path=directory/'validation.json'
        validation=json.loads(validation_path.read_text()) if validation_path.exists() else []
        if not validation:
            validation.append(dict(step=0,sampled=evaluate(model,ham,.5,batches=16,batch=128,seed=30000+L)))
            write_json(validation_path,validation)
        started=time.monotonic()
        def checkpoint(state):
            save_training_checkpoint(directory,name,state,metadata,final=state['step']==args.steps)
            write_json(directory/'history.json',state['history'])
            if state['step'] in (801,1601,args.steps):
                result=evaluate(model,ham,.5,batches=16,batch=128,seed=40000+seed+state['step'])
                validation.append(dict(step=state['step'],sampled=result))
                write_json(validation_path,validation)
                model.train()
                print('VALIDATE',name,state['step'],'E/N',result['energy']['mean']/ham.n_sites,
                      'variance/N',result['local_energy_variance']/ham.n_sites,flush=True)
        print('START',name,flush=True)
        history=train(model,ham,.5,steps=args.steps,batch=args.batch,lr=.0005,seed=seed,
                      report_every=200,checkpoint=checkpoint,resume=resume)
        write_json(directory/'history.json',history)
        # Fresh evaluation, independent of interim validation and training streams.
        sampled=evaluate(model,ham,.5,batches=32,batch=256,seed=90000+100*L+seed)
        result=dict(name=name,L=L,N=ham.n_sites,seed=seed,step=args.steps,
                    final_checkpoint=str((directory/f'{name}__step{args.steps:06d}__final.pt').relative_to(args.output)),
                    checkpoint_sha256_path=str(latest.relative_to(args.output)),
                    checkpoint_sha256=hashlib.sha256(latest.read_bytes()).hexdigest(),
                    sampled=sampled,seconds=time.monotonic()-started)
        write_json(directory/'evaluation.json',result)
        summary.append(result)
        write_json(args.output/'summary.json',summary)
        print('DONE',name,'E/N',sampled['energy']['mean']/ham.n_sites,flush=True)
    write_json(args.output/'summary.json',summary)
    write_json(args.output/'status.json',dict(state='completed',completed=len(summary)))
    write_json(args.output/'finished.json',dict(finished=True,ground_state_convergence_established=False))


if __name__=='__main__':
    main()
