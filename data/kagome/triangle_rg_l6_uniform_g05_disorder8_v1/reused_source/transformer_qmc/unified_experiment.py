"""Shared-size sampled VMC campaign with balanced energy-per-spin gradients."""

import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import torch

from .physics import make_af_sample, exact_reference
from .unified import UnifiedTransformer
from .vmc import local_energy, enumerate_model
from .experiment import save_training_checkpoint, load_checkpoint, write_json, evaluate


def tasks():
    return [dict(L=L, sample_id=s, gamma=g, h=h)
            for L in (2,4,6,8) for s in (0,1)
            for g,h in ((.5,.3),(2.,.3),(.5,0.))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4800)
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    contexts = tasks()
    samples = {(L,s):make_af_sample(L,s) for L in (2,4,6,8) for s in (0,1,2)}
    for (L,s), sample in samples.items():
        np.savez_compressed(args.output/f'disorder_L{L}_s{s}.npz', K=-sample.J,
                            bonds=sample.bonds, points=sample.points, seed_entropy=sample.seed_entropy)
    architecture = dict(width=32, layers=2, heads=4)
    model = UnifiedTransformer(samples[2,0].hamiltonian(.3), .5, **architecture).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    generator = torch.Generator(device='cuda').manual_seed(43)
    rng = np.random.default_rng(44)
    manifest = dict(tasks=contexts, steps=args.steps, batch=args.batch, architecture=architecture,
                    seed=42, objective='uniform contexts; energy per spin', torch=str(torch.__version__),
                    gpu=torch.cuda.get_device_name(),
                    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.glob('*.py'))})
    import json
    manifest_path = args.output/'campaign.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('campaign identity changed; choose a fresh directory')
    write_json(manifest_path, manifest)
    path = args.output/'checkpoint.pt'
    first, history, order = 0, [], np.arange(len(contexts))
    if path.exists():
        state = load_checkpoint(path, 'cuda')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['generator'].cpu())
        rng.bit_generator.state = state['rng']
        first, history, order = state['step'], state['history'], state['order']
    started = time.monotonic()
    for step in range(first, args.steps):
        if step % len(contexts) == 0:
            order = rng.permutation(len(contexts))
        context = contexts[int(order[step % len(contexts)])]
        ham = samples[context['L'],context['sample_id']].hamiltonian(context['h'])
        model.set_hamiltonian(ham, context['gamma']).train()
        spins = model.sample(args.batch, generator)
        energy = local_energy(model, ham, context['gamma'], spins)
        centered = (energy-energy.mean()) * args.batch/(args.batch-1)/ham.n_sites
        loss = (centered.detach()*model.log_prob(spins)).mean()
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr'] = .002*(.1+.9*(1-(step+1)/args.steps))
        history.append(dict(context, step=step+1, energy=float(energy.mean()),
                            variance=float(energy.var()), gradient=float(norm),
                            seconds=time.monotonic()-started))
        if step == 0 or (step+1) % 120 == 0 or step+1 == args.steps:
            save_training_checkpoint(args.output,
                'unified_alibi__L02-04-06-08__dis000-001__fields3__seed042',
                dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                       generator=generator.get_state(), rng=rng.bit_generator.state,
                                       order=order, step=step+1, history=history),
                dict(manifest,model_type='unified_alibi'), final=step+1==args.steps)
            write_json(args.output/'history.json', history)
            print('TRAIN', step+1, history[-1], flush=True)
    results = []
    # Seen and held-out quenched samples, at every trained size.
    for L in (2,4,6,8):
        for s in (0,2):
            sample = samples[L,s]
            ham = sample.hamiltonian(.3)
            model.set_hamiltonian(ham,.5).eval()
            result = dict(L=L, sample_id=s, gamma=.5, h=.3, held_out_disorder=s==2,
                          vmc=evaluate(model,ham,.5,batches=16,seed=10000+10*L+s))
            if L == 2:
                exact = exact_reference(ham,.5,sample.points,L)
                enum = enumerate_model(model,ham,.5,sample.points,L)
                result['exact'] = {k:exact[k] for k in ('energy','q2','chi_connected','gap','residual')}
                result['enumerated_vmc'] = {k:enum[k] for k in ('energy','variance','q2','chi_connected','normalization')}
            results.append(result)
            write_json(args.output/'evaluation.json',results)
            print('EVAL', L, s, result['vmc']['energy']['mean'], flush=True)
    write_json(args.output/'training_finished.json', dict(finished=True))


if __name__ == '__main__':
    main()
