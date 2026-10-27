"""Resumable GPU VMC campaign; explicit frozen checkpoints and quenched records.

python -m transformer_qmc.experiment --output RESULTS --steps 800 --batch 128
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .physics import make_af_sample, correlations
from .model import CellTransformer, TransformerGuide
from .vmc import train, local_energy


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def checkpoint_save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def save_training_checkpoint(directory, prefix, state, metadata, final=False):
    """Keep resumable latest state plus explicitly named periodic/final snapshots."""
    directory = Path(directory)
    state = dict(state, metadata=metadata)
    latest = directory / f'{prefix}__latest.pt'
    checkpoint_save(latest, state)
    # Compatibility with both campaign loaders; atomic relative symlink update.
    link = directory / 'checkpoint.pt.tmp-link'
    if link.is_symlink():
        link.unlink()
    link.symlink_to(latest.name)
    link.replace(directory / 'checkpoint.pt')
    if state['step'] == 1 or state['step'] % 200 == 0 or final:
        suffix = 'final' if final else 'snapshot'
        snapshot = directory / f"{prefix}__step{state['step']:06d}__{suffix}.pt"
        checkpoint_save(snapshot, state)
    write_json(directory / 'checkpoint_index.json', dict(
        latest=latest.name, step=state['step'], final=final, metadata=metadata,
        snapshots=sorted(p.name for p in directory.glob(f'{prefix}__step*.pt'))))


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def load_model(directory, device='cpu'):
    directory = Path(directory)
    settings = json.loads((directory/'settings.json').read_text())
    sample = make_af_sample(settings['L'], settings['sample_id'], settings['disorder_width'])
    ham = sample.hamiltonian(settings['h'])
    model = CellTransformer(ham, settings['gamma'], **settings['architecture']).to(device)
    state = load_checkpoint(directory/'checkpoint.pt', device)
    model.load_state_dict(state['model'])
    return model.eval(), sample, ham, settings


def evaluate(model, ham, gamma, batches=16, batch=128, seed=10000):
    """Fresh direct variational samples; uncertainty across independent batches."""
    generator = torch.Generator(device=model.device).manual_seed(seed)
    means, variances, q2, connected = [], [], [], []
    model.eval()
    for _ in range(batches):
        spins = model.sample(batch, generator)
        energy = local_energy(model, ham, gamma, spins).cpu().numpy()
        means.append(float(energy.mean()))
        variances.append(float(energy.var(ddof=1)))
        four = model.sample(batch*4, generator).cpu().numpy().reshape(batch,4,ham.n_sites)
        q = (four[:,0]*four[:,1]).mean(1)
        d = .5*(four[:,0]-four[:,1])*(four[:,2]-four[:,3])
        q2.append(float(np.mean(q*q)))
        connected.append(float(np.mean(d.sum(1)**2/ham.n_sites)))
    def summary(values):
        return dict(mean=float(np.mean(values)), se=float(np.std(values,ddof=1)/np.sqrt(len(values))),
                    batches=values)
    return dict(energy=summary(means), local_energy_variance=float(np.mean(variances)),
                q2=summary(q2), chi_connected=summary(connected), seed=seed,
                evaluation_batches=batches, batch=batch)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=800)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--only',default=None)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU training requested but CUDA is unavailable')
    args.output.mkdir(parents=True,exist_ok=True)
    tasks=[dict(name='L2_g0p5_h0p3_s0_t0',L=2,gamma=.5,h=.3,train_seed=18),
           dict(name='L4_g0p5_h0p3_s0_t0',L=4,gamma=.5,h=.3,train_seed=19),
           dict(name='L6_g2_h0p3_s0_t0',L=6,gamma=2.,h=.3,train_seed=20),
           dict(name='L6_g0p5_h0p3_s0_t0',L=6,gamma=.5,h=.3,train_seed=21),
           dict(name='L6_g0p5_h0p3_s0_t1',L=6,gamma=.5,h=.3,train_seed=22),
           dict(name='L6_g0p5_h0_s0_t0',L=6,gamma=.5,h=0.,train_seed=23),
           dict(name='L8_g0p5_h0p3_s0_t0',L=8,gamma=.5,h=.3,train_seed=24)]
    manifest=dict(tasks=tasks,steps=args.steps,batch=args.batch,device=args.device,
                  torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
                  gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
                  source_sha256={str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob('*.py'))})
    manifest_path=args.output/'campaign.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old!=manifest:
            raise RuntimeError('campaign identity changed; use a new output directory')
    else:
        write_json(manifest_path,manifest)
    for task in tasks:
        if args.only and task['name']!=args.only:
            continue
        directory=args.output/task['name']
        directory.mkdir(exist_ok=True)
        if (directory/'evaluation.json').exists():
            print('completed',task['name'],flush=True)
            continue
        settings=dict(task,sample_id=0,disorder_width=.5,steps=args.steps,batch=args.batch,
                      architecture=dict(width=32,layers=2,heads=4),lr=.002)
        sample=make_af_sample(task['L'])
        ham=sample.hamiltonian(task['h'])
        write_json(directory/'settings.json',settings)
        np.savez_compressed(directory/'disorder.npz',K=-sample.J,J=sample.J,bonds=sample.bonds,
                            points=sample.points,seed_entropy=sample.seed_entropy,L=sample.L)
        torch.manual_seed(task['train_seed'])
        model=CellTransformer(ham,task['gamma'],**settings['architecture']).to(args.device)
        path=directory/'checkpoint.pt'
        resume=load_checkpoint(path,args.device) if path.exists() else None
        print('START',task['name'],'N=',ham.n_sites,flush=True)
        start=time.monotonic()
        prefix=(f"fixed__L{task['L']:02d}__N{ham.n_sites:03d}"
                f"__g{task['gamma']:g}__h{task['h']:g}__dis000__seed{task['train_seed']:03d}").replace('.', 'p')
        history=train(model,ham,task['gamma'],steps=args.steps,batch=args.batch,
                      seed=task['train_seed'],lr=settings['lr'],report_every=50,
                      checkpoint=lambda state:save_training_checkpoint(
                          directory,prefix,state,dict(settings,model_type='fixed_learned_position'),
                          final=state['step']==args.steps),resume=resume)
        write_json(directory/'history.json',history)
        evaluation=evaluate(model,ham,task['gamma'],seed=10000+task['train_seed'])
        evaluation.update(seconds=time.monotonic()-start,
                          checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        write_json(directory/'evaluation.json',evaluation)
        print('DONE',task['name'],evaluation['energy'],flush=True)
    write_json(args.output/'training_finished.json',dict(finished=True))


if __name__=='__main__':
    main()
