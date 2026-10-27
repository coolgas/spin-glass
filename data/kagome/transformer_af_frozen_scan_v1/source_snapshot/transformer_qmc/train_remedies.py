"""Controlled all-size Adam/SR continuations with optional continuous-field replay."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from .experiment import checkpoint_save, load_checkpoint, write_json
from .physics import make_af_sample
from .symmetric import SymmetricTransformer
from .train_symmetric import validate
from .vmc import local_energy
from .sr import score_matrix, context_rows, solve_direction, apply_direction


def choose_context(step, rng, ids, continuous):
    anchors = ((.5, 0.), (.5, .3), (2., .3))
    # Half the continuous campaign revisits the original physical targets.
    if not continuous:
        gamma, h = anchors[step % 3]
    elif step % 2 == 0:
        gamma, h = anchors[(step//2) % 3]
    else:
        gamma = float(np.exp(rng.uniform(np.log(.4), np.log(2.2))))
        h = float(rng.uniform(0., .4))
    return int(rng.choice(ids)), gamma, h


def update(model, generator, *, sample_id, gamma, h, batch, method,
           optimizer=None, damping=.01, rate=.05, trust=.05, score_chunk=16):
    metrics, blocks, targets, scores_all = [], [], [], []
    if optimizer is not None:
        optimizer.zero_grad()
    for L in (2, 4, 6, 8):
        ham = make_af_sample(L, sample_id).hamiltonian(h)
        model.set_hamiltonian(ham, gamma)
        spins = model.sample(batch, generator)
        energy = local_energy(model, ham, gamma, spins)
        metrics.append(dict(L=L, energy_per_spin=float(energy.mean())/ham.n_sites,
                            variance_per_spin=float(energy.var())/ham.n_sites))
        if method == 'sr':
            scores = score_matrix(model, spins, score_chunk)
            rows, force = context_rows(scores, energy, ham.n_sites, 4)
            blocks.append(rows); targets.append(force); scores_all.append(scores)
        else:
            centered = (energy-energy.mean())*batch/(batch-1)/ham.n_sites
            ((centered.detach()*model.log_prob(spins)).mean()/4).backward()
    if method == 'sr':
        rows, force = torch.cat(blocks), torch.cat(targets)
        direction, residual = solve_direction(rows, force, damping)
        diagnostics = apply_direction(model, direction, scores_all, rate, trust)
        diagnostics.update(solve_residual=residual, damping=damping)
    else:
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        if not torch.isfinite(gradient):
            raise FloatingPointError('nonfinite Adam gradient')
        optimizer.step()
        diagnostics = dict(gradient=float(gradient), rate=optimizer.param_groups[0]['lr'])
    return dict(sample_id=sample_id, gamma=gamma, h=h, batch=batch,
                metrics=metrics, **diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initialize', type=Path, required=True)
    parser.add_argument('--method', choices=['adam', 'sr'], required=True)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--completion', action='store_true')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--high-batch', type=int, default=256)
    parser.add_argument('--score-chunk', type=int, default=16)
    parser.add_argument('--damping', type=float, default=.01)
    parser.add_argument('--rate', type=float, default=.05)
    parser.add_argument('--trust', type=float, default=.05)
    parser.add_argument('--validate-every', type=int, default=200)
    parser.add_argument('--seed', type=int, default=601)
    parser.add_argument('--context-offset', type=int, default=0)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    if min(args.batch, args.high_batch) < 2 or args.steps < 1:
        raise ValueError('batch>=2 and steps>=1 required')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = load_checkpoint(args.initialize, 'cuda')
    architecture = dict(initial['metadata']['architecture'])
    if args.completion:
        architecture['completion'] = True
    train_ids = [0, 1]+list(range(100, 356))
    manifest = dict(architecture=architecture, seed=args.seed, steps=args.steps,
                    batch=args.batch, high_batch=args.high_batch, method=args.method,
                    continuous=args.continuous, score_chunk=args.score_chunk,
                    damping=args.damping, rate=args.rate, trust=args.trust,
                     validate_every=args.validate_every, train_ids=train_ids,
                     context_offset=args.context_offset,
                    validation_ids=[10000, 10001], final_test_ids=list(range(40000, 40008)),
                    fields=[[.5, 0.], [.5, .3], [2., .3]], sizes=[2, 4, 6, 8],
                    continuous_distribution='50% anchor replay; otherwise log-uniform gamma [.4,2.2], uniform h [0,.4]',
                    metric='equal contexts; covariance(d log psi)/N; damping times identity',
                    objective='equal mean energy per spin across all four sizes per update',
                    exact_labels_used_for_training=False,
                    initialization_sha256=hashlib.sha256(args.initialize.read_bytes()).hexdigest(),
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob('*.py')})
    path = args.output/'campaign.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('campaign identity mismatch')
    write_json(path, manifest)
    torch.manual_seed(args.seed)
    model = SymmetricTransformer(make_af_sample(2).hamiltonian(0.), .5, **architecture).cuda()
    incompatible = model.load_state_dict(initial['model'], strict=False)
    expected_missing = ['base.completion_scale'] if args.completion and 'base.completion_scale' not in initial['model'] else []
    if incompatible.missing_keys != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(f'incompatible initialization: {incompatible}')
    if expected_missing:
        # Zero gate preserves the exact starting guide while retaining a
        # nonzero derivative for learning the completion feature's strength.
        with torch.no_grad():
            model.base.completion_scale.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=.00015) if args.method == 'adam' else None
    generator = torch.Generator(device='cuda').manual_seed(args.seed+1)
    rng = np.random.default_rng(args.seed+2)
    prefix = f'shared_{args.method}_replay{int(args.continuous)}__L02-04-06-08__seed{args.seed}'
    if args.completion:
        prefix = prefix.replace('__L', '__completion1__L')
    latest = args.output/(prefix+'__latest.pt')
    first, history, elapsed = 0, [], 0.
    if latest.exists():
        saved = load_checkpoint(latest, 'cuda')
        model.load_state_dict(saved['model'])
        if optimizer is not None:
            optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'].cpu()); rng.bit_generator.state = saved['numpy_rng']
        first, history, elapsed = saved['step'], saved['history'], saved['elapsed']
    elif not args.benchmark:
        rows = validate(model, [0, 1, 10000, 10001])
        write_json(args.output/'validation_000000.json', dict(checkpoint=args.initialize.name, rows=rows))
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(first, args.steps):
        sample_id, gamma, h = choose_context(step+args.context_offset, rng, train_ids, args.continuous)
        batch = args.batch if gamma <= 1. else args.high_batch
        model.train()
        row = update(model, generator, sample_id=sample_id, gamma=gamma, h=h,
                     batch=batch, method=args.method, optimizer=optimizer,
                     damping=args.damping, rate=args.rate, trust=args.trust,
                     score_chunk=args.score_chunk)
        row.update(step=step+1, seconds=elapsed+time.monotonic()-start,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated())
        history.append(row)
        if optimizer is not None:
            lr = .00015*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/args.steps)))
            for group in optimizer.param_groups:
                group['lr'] = lr
        if step == first or (step+1) % 20 == 0 or step+1 == args.steps:
            saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
                         generator=generator.get_state(), numpy_rng=rng.bit_generator.state,
                         step=step+1, history=history, elapsed=row['seconds'], metadata=manifest)
            checkpoint_save(latest, saved)
            write_json(args.output/'history.json', history)
            write_json(args.output/'status.json', dict(state='running', step=step+1, total=args.steps))
            print('TRAIN', json.dumps(row), flush=True)
        if not args.benchmark and ((step+1) % args.validate_every == 0 or step+1 == args.steps):
            snapshot = args.output/f'{prefix}__step{step+1:06d}.pt'
            checkpoint_save(snapshot, dict(model=model.state_dict(), metadata=manifest, step=step+1))
            write_json(args.output/'status.json', dict(state='validating', step=step+1, total=args.steps))
            rows = validate(model, [0, 1, 10000, 10001])
            write_json(args.output/f'validation_{step+1:06d}.json', dict(checkpoint=snapshot.name, rows=rows))
            print('VALIDATE', step+1, [(r['sample_id'],r['h'],r['fidelity'])
                                      for r in rows if r['L']==2 and r['gamma']==.5], flush=True)
    final = args.output/(prefix+'__final.pt')
    checkpoint_save(final, dict(model=model.state_dict(), metadata=manifest, step=args.steps))
    # Reserve test40000+ for a selected experiment after validation comparisons.
    write_json(args.output/'status.json', dict(state='benchmark_completed' if args.benchmark else 'completed',
                                              checkpoint=final.name, step=args.steps))


if __name__ == '__main__':
    main()
