"""AutoDL ALiBi L2 refinement: sampled VMC, full-state validation only."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from .experiment import load_checkpoint, checkpoint_save, save_training_checkpoint, write_json, evaluate
from .physics import make_af_sample, exact_reference
from .unified import UnifiedTransformer
from .vmc import train, enumerate_model


def diagnostics(model, ham, sample, reference):
    measured = enumerate_model(model,ham,.5,sample.points,2)
    result = {k:measured[k] for k in ('energy','variance','q2','chi_connected','normalization')}
    result['energy_error_per_spin'] = (measured['energy']-reference['energy'])/12
    for key in ('q2','chi_connected'):
        result[key+'_relative_error'] = abs(measured[key]/reference[key]-1)
    result['fidelity'] = float(np.sum(np.sqrt(measured['probability']*reference['probability']))**2)
    result['qualified'] = bool(abs(result['energy_error_per_spin'])<1e-4
                               and result['q2_relative_error']<.01
                               and result['chi_connected_relative_error']<.05)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shared',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--batch',type=int,default=512)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('AutoDL CUDA training required')
    args.output.mkdir(parents=True,exist_ok=True)
    sample=make_af_sample(2,0)
    ham=sample.hamiltonian(.3)
    ref=exact_reference(ham,.5,sample.points,2)
    architecture=dict(width=32,layers=2,heads=4)
    tasks=[dict(initialization=init,seed=seed,lr=.001)
           for init in ('shared','fresh') for seed in (101,102)]
    manifest=dict(L=2,N=12,gamma=.5,h=.3,sample_id=0,architecture=architecture,
                  tasks=tasks,steps=args.steps,batch=args.batch,
                  selection='lowest enumerated variational energy among validation checkpoints',
                  training='sampled VMC energy gradients; no exact wavefunction labels',
                  targets=dict(energy_error_per_spin=1e-4,q2_relative_error=.01,chi_connected_relative_error=.05),
                  shared_sha256=hashlib.sha256(args.shared.read_bytes()).hexdigest(),
                  torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                  sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    manifest_path=args.output/'campaign.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise RuntimeError('campaign identity changed')
    write_json(manifest_path,manifest)
    write_json(args.output/'exact.json',{k:ref[k] for k in ('energy','q2','chi_connected','gap','residual')})
    np.savez_compressed(args.output/'disorder.npz',K=-sample.J,bonds=sample.bonds,points=sample.points)
    summary=[]
    for task in tasks:
        name=f"alibi__L02__N012__g0p5__h0p3__dis000__init-{task['initialization']}__seed{task['seed']}"
        directory=args.output/name
        directory.mkdir(exist_ok=True)
        if (directory/'evaluation.json').exists():
            summary.append(json.loads((directory/'evaluation.json').read_text()))
            continue
        torch.manual_seed(task['seed'])
        model=UnifiedTransformer(ham,.5,**architecture).cuda()
        if task['initialization']=='shared':
            model.load_state_dict(load_checkpoint(args.shared,'cuda')['model'])
        metadata=dict(manifest,task=task,name=name,model_type='alibi_L2_specialized')
        write_json(directory/'settings.json',metadata)
        latest=directory/'checkpoint.pt'
        resume=load_checkpoint(latest,'cuda') if latest.exists() else None
        validation_path=directory/'validation.json'
        validation=json.loads(validation_path.read_text()) if validation_path.exists() else []
        best_path=directory/f'{name}__best.pt'
        best_energy=float('inf')
        if best_path.exists():
            best_energy=load_checkpoint(best_path,'cpu')['validation']['energy']
        if not validation:
            validation.append(dict(step=0,**diagnostics(model,ham,sample,ref)))
            write_json(validation_path,validation)
        started=time.monotonic()
        def checkpoint(state):
            nonlocal best_energy
            measured=diagnostics(model,ham,sample,ref)
            # Validation consumes no sampling RNG and is excluded from gradients.
            row=dict(step=state['step'],**measured)
            validation.append(row)
            write_json(validation_path,validation)
            state=dict(state,validation=measured)
            save_training_checkpoint(directory,name,state,metadata,final=state['step']==args.steps)
            if measured['energy']<best_energy:
                best_energy=measured['energy']
                checkpoint_save(best_path,dict(state,metadata=metadata))
            print('VALIDATE',name,row,flush=True)
        print('START',name,flush=True)
        history=train(model,ham,.5,steps=args.steps,batch=args.batch,lr=task['lr'],seed=task['seed'],
                      report_every=200,checkpoint=checkpoint,resume=resume)
        write_json(directory/'history.json',history)
        best=load_checkpoint(best_path,'cuda')
        model.load_state_dict(best['model'])
        result=dict(name=name,task=task,best_step=best['step'],best_checkpoint=str(best_path.relative_to(args.output)),
                    diagnostics=diagnostics(model,ham,sample,ref),
                    sampled=evaluate(model,ham,.5,batches=32,batch=512,seed=20000+task['seed']),
                    seconds=time.monotonic()-started)
        write_json(directory/'evaluation.json',result)
        summary.append(result)
        write_json(args.output/'summary.json',summary)
    winner=min(summary,key=lambda r:r['diagnostics']['energy'])
    write_json(args.output/'selected.json',winner)
    write_json(args.output/'finished.json',dict(finished=True,all_targets_met=winner['diagnostics']['qualified']))
    print('SELECTED',winner,flush=True)


if __name__=='__main__':
    main()
