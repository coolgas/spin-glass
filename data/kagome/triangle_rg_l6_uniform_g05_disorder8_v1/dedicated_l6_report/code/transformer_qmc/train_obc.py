"""Isolated, restartable OBC VMC using the PBC pilot/main optimization recipe."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from kagome_qea.exact import operator_matrix
from .diagnose_shared import measure, reference
from .experiment import checkpoint_save, load_checkpoint, write_json, evaluate
from .physics import basis, make_af_sample
from .obc import OpenBoundaryTransformer
from .vmc import local_energy


TRAIN_IDS = list(range(72000, 72258))
VALIDATION_IDS = [73000, 73001]
FIELDS = [[.5, 0.], [.5, .3], [2., .3]]


def sample(L, sid):
    return make_af_sample(L, sid, periodic=False)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py'))}


@torch.no_grad()
def validate(model, ids, *, full=False):
    rows = []
    spins = torch.tensor(basis(12), device=model.device, dtype=torch.long)
    model.eval()
    for sid in ids:
        for gamma, h in FIELDS:
            ham = sample(2, sid).hamiltonian(h)
            model.set_hamiltonian(ham, gamma)
            logs = model.log_amplitudes(spins).double().cpu().numpy()
            p = np.exp(2*logs)
            if abs(p.sum()-1) > 2e-5:
                raise RuntimeError('OBC enumerated normalization failed')
            matrix = operator_matrix(ham, gamma)
            energy, exact, _ = reference(matrix, h)
            result = measure(np.sqrt(p), matrix, (energy, exact), spins.cpu().numpy())
            if h == 0 and result['rms_site_magnetization'] > 1e-7:
                raise RuntimeError('OBC zero-field parity failed')
            rows.append(dict(L=2, periodic=False, sample_id=sid, gamma=gamma, h=h,
                             normalization=float(p.sum()), **result,
                             exact_chi=measure(exact, matrix, (energy, exact), spins.cpu().numpy())['chi_connected']))
    for L in (4, 6, 8):
        for sid in ([TRAIN_IDS[0]] + VALIDATION_IDS if full else VALIDATION_IDS[:1]):
            for gamma, h in FIELDS:
                ham = sample(L, sid).hamiltonian(h)
                model.set_hamiltonian(ham, gamma)
                result = evaluate(model, ham, gamma, batches=16 if full else 4, batch=128,
                                  seed=280000+L*100+sid+int(h*10))
                rows.append(dict(L=L, periodic=False, sample_id=sid, gamma=gamma, h=h, sampled=result))
    return rows


def load_obc(path, device='cpu'):
    saved = load_checkpoint(path, device)
    meta = saved['metadata']
    if meta.get('boundary') != 'obc' or meta.get('model_class') != 'OpenBoundaryTransformer-v1':
        raise ValueError('Expected an explicitly identified OBC-v1 checkpoint')
    model = OpenBoundaryTransformer(sample(2, TRAIN_IDS[0]).hamiltonian(0.), .5,
                                    **meta['architecture']).to(device)
    model.load_state_dict(saved['model'])
    model.eval(); model.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'pilot', 'main'], required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--validate-every', type=int, default=600)
    parser.add_argument('--seed', type=int, default=501)
    parser.add_argument('--initialize', type=Path)
    parser.add_argument('--stop-after', type=int, help='Checkpoint and pause after this absolute step')
    parser.add_argument('--skip-evaluation', action='store_true', help='Smoke tests only')
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 2 or args.validate_every < 1:
        raise ValueError('Invalid training settings')
    if args.skip_evaluation and args.phase != 'smoke':
        raise ValueError('Evaluation may only be skipped for smoke tests')
    if (args.phase == 'main') != (args.initialize is not None):
        raise ValueError('Main initializes from the OBC pilot; pilot/smoke start fresh')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('OBC training requires CUDA')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'obc_training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    test_start = 75000 if args.phase == 'main' else 74000
    manifest = dict(boundary='obc', periodic=False, model_class='OpenBoundaryTransformer-v1',
        phase=args.phase, architecture=dict(width=64, layers=4, heads=4), seed=args.seed,
        steps=args.steps, batch=args.batch, validate_every=args.validate_every,
        train_ids=TRAIN_IDS, validation_ids=VALIDATION_IDS, final_test_ids=list(range(test_start,test_start+8)),
        fields=FIELDS, sizes=[2,4,6,8], optimizer='Adam lr=0.0005 cosine to 0.00005; grad clip 10',
        objective='equal mean VMC energy/site across all four sizes every optimizer step',
        symmetry='h=0 probability spin-flip mixture; fixed OBC origin; no translation mixture',
        conditioning='6 nonwrapping bond slots + 6 presence masks + gamma,h,1/L,x/(L-1),y/(L-1)',
        exact_labels_used_for_training=False, ground_state_qualified=False,
        skip_evaluation=args.skip_evaluation,
        initialization_sha256=hashlib.sha256(args.initialize.read_bytes()).hexdigest() if args.initialize else None,
        source_sha256=source_identity())
    path = args.output/'obc_campaign.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('OBC campaign identity mismatch; refusing overwrite')
    write_json(path, manifest)
    completed = args.output/'status.json'
    if completed.exists():
        previous = json.loads(completed.read_text())
        if previous.get('state') == 'completed':
            final_path = args.output/previous['checkpoint']
            if hashlib.sha256(final_path.read_bytes()).hexdigest() != previous['sha256']:
                raise RuntimeError('Completed OBC checkpoint hash mismatch')
            print('ALREADY_COMPLETED_OBC', final_path, flush=True)
            return
    if set(TRAIN_IDS) & set(VALIDATION_IDS + manifest['final_test_ids']):
        raise ValueError('Disorder splits overlap')
    torch.manual_seed(args.seed)
    model = OpenBoundaryTransformer(sample(2, TRAIN_IDS[0]).hamiltonian(0.), .5,
                                    **manifest['architecture']).cuda()
    if args.initialize:
        initial = load_checkpoint(args.initialize, 'cuda')
        if (initial['metadata'].get('boundary') != 'obc' or
            initial['metadata'].get('phase') != 'pilot' or initial['step'] != 600):
            raise ValueError('Main requires completed 600-step OBC pilot weights')
        model.load_state_dict(initial['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    generator = torch.Generator(device='cuda').manual_seed(args.seed+1)
    rng = np.random.default_rng(args.seed+2)
    prefix = f'obc_v1_shared_alibi__{args.phase}__L02-04-06-08__seed{args.seed}'
    latest = args.output/(prefix+'__latest.pt')
    first, history, elapsed = 0, [], 0.
    if latest.exists():
        saved = load_checkpoint(latest, 'cuda')
        if saved['metadata'] != manifest:
            raise RuntimeError('Resume metadata mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'].cpu()); rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu()); torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        first, history, elapsed = saved['step'], saved['history'], saved['elapsed']
    start = time.monotonic()

    def status(state, **extra):
        write_json(args.output/'status.json', dict(state=state, phase=args.phase, boundary='obc',
                   updated_unix=time.time(), total=args.steps, **extra))

    try:
        for step in range(first, args.steps):
            gamma, h = FIELDS[step % len(FIELDS)]
            sid = int(rng.choice(TRAIN_IDS))
            optimizer.zero_grad(); model.train(); metrics = []
            for L in manifest['sizes']:
                ham = sample(L, sid).hamiltonian(h)
                model.set_hamiltonian(ham, gamma)
                spins = model.sample(args.batch, generator)
                energy = local_energy(model, ham, gamma, spins)
                centered = (energy-energy.mean())*args.batch/(args.batch-1)/ham.n_sites
                loss = (centered.detach()*model.log_prob(spins)).mean()/len(manifest['sizes'])
                loss.backward()
                metrics.append(dict(L=L, bonds=len(ham.bonds), energy_per_spin=float(energy.mean())/ham.n_sites,
                                    variance_per_spin=float(energy.var())/ham.n_sites))
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            if not torch.isfinite(gradient):
                raise RuntimeError('Nonfinite OBC gradient')
            optimizer.step()
            lr = .0005*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/args.steps)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            stop = args.stop_after is not None and step+1 >= args.stop_after
            if step == 0 or (step+1)%100 == 0 or step+1 == args.steps or stop:
                history.append(dict(step=step+1, seconds=elapsed+time.monotonic()-start,
                                    sample_id=sid, gamma=gamma, h=h, metrics=metrics,
                                    gradient=float(gradient), lr=lr))
                checkpoint_save(latest, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    generator=generator.get_state(), numpy_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    step=step+1, history=history, elapsed=elapsed+time.monotonic()-start, metadata=manifest))
                write_json(args.output/'history.json', history)
                print('TRAIN_OBC', step+1, history[-1], flush=True)
            if step == 0 or (step+1)%10 == 0 or stop:
                status('training', step=step+1, elapsed_seconds=elapsed+time.monotonic()-start,
                       peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            if not args.skip_evaluation and ((step+1)%args.validate_every == 0 or step+1 == args.steps):
                snapshot = args.output/(prefix+f'__step{step+1:06d}.pt')
                checkpoint_save(snapshot, dict(model=model.state_dict(), metadata=manifest, step=step+1))
                status('validating', step=step+1)
                rows = validate(model, TRAIN_IDS[:2]+VALIDATION_IDS)
                write_json(args.output/f'validation_{step+1:06d}.json', dict(checkpoint=snapshot.name, rows=rows))
                print('VALIDATE_OBC', step+1, flush=True)
            if stop and step+1 < args.steps:
                status('paused_at_requested_step', step=step+1)
                return
        final = args.output/(prefix+'__final.pt')
        checkpoint_save(final, dict(model=model.state_dict(), metadata=manifest, step=args.steps))
        status('final_evaluation', step=args.steps)
        rows = [] if args.skip_evaluation else validate(model, TRAIN_IDS[:10]+VALIDATION_IDS+manifest['final_test_ids'], full=True)
        digest = hashlib.sha256(final.read_bytes()).hexdigest()
        write_json(args.output/'evaluation.json', dict(checkpoint=final.name, sha256=digest, rows=rows))
        status('completed', step=args.steps, checkpoint=final.name, sha256=digest,
               elapsed_seconds=elapsed+time.monotonic()-start,
               peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
