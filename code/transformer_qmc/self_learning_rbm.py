"""Finite PQMC -> mixed-walker likelihood -> RBM guide feedback pilot.

RBM.forward is log(psi), so the density fitted to walkers is exp(2*forward).
Each stint starts a fresh population and excludes equilibration. No forward
walking tags or physical measurement averages cross a guide update.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .collective_campaign import write
from .correlation_lengths import sha
from .physics import make_af_sample
from .projector import propagate, rates_and_local
from .simple_guides import RBM, SimpleGuide


def likelihood_loss(model, data, negative):
    """Gradient of KL(data || psi_theta**2); scalar is a gradient surrogate.

    Negative states must sample the current normalized psi_theta**2. Finite
    persistent Gibbs chains approximate that expectation; they do not make
    the gradient exact or certify equilibration.
    """
    return 2*(model(negative.detach()).mean()-model(data.detach()).mean())


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def exact_evaluation(model, ham, gamma, data=None):
    """Small-system evaluation only; never used in fitting or projection."""
    from .physics import basis, exact_reference
    from kagome_qea.exact import operator_matrix
    spins = basis(ham.n_sites)
    device = next(model.parameters()).device
    logs = model(torch.tensor(spins, dtype=torch.float64, device=device)).cpu().numpy()
    probability = np.exp(2*(logs-logs.max())); probability /= probability.sum()
    truth = exact_reference(ham, gamma)
    amplitude = np.sqrt(probability)
    energy = float(amplitude @ (operator_matrix(ham, gamma) @ amplitude))
    row = dict(energy=energy, exact_energy=truth['energy'],
               energy_error_per_site=(energy-truth['energy'])/ham.n_sites,
               fidelity=float((amplitude @ truth['amplitude'])**2),
               ground_probability_tv=float(abs(probability-truth['probability']).sum()/2))
    if data is not None:
        ids = ((1-data.astype(np.int64))//2) @ (1 << np.arange(ham.n_sites))
        empirical = np.bincount(ids, minlength=len(probability))/len(ids)
        keep = empirical > 0
        logp = 2*logs-np.logaddexp.reduce(2*logs)
        row['empirical_target_kl'] = float((empirical[keep]*(np.log(empirical[keep])-logp[keep])).sum())
    return row


def run(args):
    if (args.L < 2 or args.stints < 1 or args.walkers < 8 or args.updates < 1
            or args.batch < 2 or args.gibbs < 1 or args.sample_walkers < 1
            or args.sample_walkers > args.walkers or args.initial_sweeps < 1
            or not np.isfinite([args.gamma, args.equil, args.collect, args.delta, args.lr]).all()
            or min(args.gamma, args.equil, args.collect, args.delta, args.lr) <= 0):
        raise ValueError('invalid self-learning pilot settings')
    for duration in [args.equil, args.collect]:
        if not np.isclose(duration/args.delta, round(duration/args.delta), atol=1e-10, rtol=0):
            raise ValueError('stint times must be integer multiples of delta')
    args.output.mkdir(parents=True, exist_ok=True)
    sources = ['self_learning_rbm.py', 'simple_guides.py', 'projector.py', 'physics.py']
    signature = dict(format=1, settings={k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items() if k != 'output'},
        initial_sha256=sha(args.initial) if args.initial else None,
        sources={s: sha(Path(__file__).with_name(s)) for s in sources},
        numerics_sha256=sha(Path(__file__).parent.parent/'qmc_numerics.py'),
        target='normalized mixed walker f_s ~ psi_s*psi_0; fit normalized psi_theta**2',
        update='psi_(s+1) ~ sqrt(f_s), ideal log-amplitude error halves per stint',
        negative_phase='persistent two-hidden-copy Gibbs; finite-chain approximation',
        population='one single-replica training population; new population each stint',
        measurements='adaptation traces only; independent final-guide production required',
        ground_state_qualified=False)
    manifest = args.output/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError('self-learning manifest mismatch')
    else:
        write(manifest, signature)
    ham = make_af_sample(args.L, args.sample_id).hamiltonian(.15)
    model = RBM(ham.n_sites, 2, args.seed).to(args.device)
    if args.initial:
        initial = torch.load(args.initial, map_location=args.device, weights_only=False)
        model.load_state_dict(initial.get('model', initial))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(np.random.SeedSequence([20261103, args.L, args.sample_id, args.seed]))
    generator = torch.Generator(device=args.device).manual_seed(args.seed+20261103)
    cp = args.output/'checkpoint.pt'
    state = dict(signature=signature, stint=0, phase='start', rows=[], step=0, update=0)
    if cp.exists():
        state = torch.load(cp, map_location='cpu', weights_only=False)
        if state['signature'] != signature:
            raise ValueError('self-learning checkpoint mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng']; generator.set_state(state['torch_rng'])
    def save():
        state.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     rng=rng.bit_generator.state, torch_rng=generator.get_state())
        save_checkpoint(cp, state)
        write(args.output/'status.json', dict(state=state['phase'], stint=state['stint'],
            planned_stints=args.stints, projection_step=state['step'], fit_update=state['update'],
            completed_stints=len(state['rows']), ground_state_qualified=False))
    while state['stint'] < args.stints:
        stint = state['stint']; directory = args.output/f'stint_{stint:03d}'
        directory.mkdir(exist_ok=True)
        if state['phase'] == 'start':
            save_checkpoint(directory/'input_model.pt', dict(model=model.state_dict()))
            frozen = RBM(ham.n_sites, 2); frozen.load_state_dict(model.state_dict())
            guide = SimpleGuide(frozen, sweeps=args.initial_sweeps)
            states = guide.sample(args.walkers, int(rng.integers(2**32)))[:, None, :]
            state.update(phase='project', step=0, update=0, states=states,
                shift=float(rates_and_local(ham, args.gamma, guide, states)[1].mean()),
                samples=[], energies=[], weight_ess=[], lineages=[], losses=[],
                ancestors=np.arange(args.walkers))
            save()
        if state['phase'] == 'project':
            frozen = RBM(ham.n_sites, 2); frozen.load_state_dict(model.state_dict())
            guide = SimpleGuide(frozen)
            equil = round(args.equil/args.delta); total = equil+round(args.collect/args.delta)
            while state['step'] < total:
                states, children, shift, energy, growth, ess = propagate(
                    ham, args.gamma, guide, state['states'], state['shift'], args.delta, rng)
                state['step'] += 1
                state.update(states=states, shift=shift)
                state['ancestors'] = state['ancestors'][children]
                counts = np.bincount(state['ancestors'], minlength=args.walkers)
                state['energies'].append([energy, growth]); state['weight_ess'].append(ess)
                state['lineages'].append([args.walkers**2/float(counts @ counts), counts.max()/args.walkers])
                if state['step'] == equil:
                    state['ancestors'] = np.arange(args.walkers)
                if state['step'] > equil:
                    take = rng.choice(args.walkers, args.sample_walkers, replace=False)
                    state['samples'].append(states[take, 0].copy())
                if state['step'] % 25 == 0 or state['step'] == total:
                    save()
            data = np.concatenate(state['samples'])
            np.savez_compressed(directory/'walkers.npz', samples=data,
                energies=state['energies'], weight_ess=state['weight_ess'], lineages=state['lineages'])
            state['before'] = exact_evaluation(model, ham, args.gamma, data) if args.L == 2 else None
            state.update(phase='fit', negative=(2*torch.randint(2, (args.batch, ham.n_sites),
                device=args.device, generator=generator)-1).double().cpu())
            save()
        if state['phase'] == 'fit':
            data = torch.tensor(np.concatenate(state['samples']), device=args.device, dtype=torch.float64)
            negative = state['negative'].to(args.device)
            while state['update'] < args.updates:
                for _ in range(args.gibbs):
                    negative = model.sweep(negative, generator)
                indices = torch.randint(len(data), (args.batch,), device=args.device, generator=generator)
                optimizer.zero_grad()
                loss = likelihood_loss(model, data[indices], negative)/ham.n_sites
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('nonfinite likelihood gradient')
                optimizer.step(); state['update'] += 1
                if state['update'] % 25 == 0 or state['update'] == args.updates:
                    state['losses'].append([state['update'], float(loss.detach()), float(norm)])
                    state['negative'] = negative.cpu(); save()
            save_checkpoint(directory/'model.pt', dict(model=model.state_dict(), signature=signature))
            np.savez_compressed(directory/'fit.npz', gradient_trace=state['losses'],
                                negative_samples=negative.cpu().numpy().astype(np.int8))
            lineage = np.asarray(state['lineages'])[round(args.equil/args.delta):]
            row = dict(stint=stint, model_sha256=sha(directory/'model.pt'),
                input_model_sha256=sha(directory/'input_model.pt'), raw_sha256=sha(directory/'walkers.npz'),
                fit_sha256=sha(directory/'fit.npz'), training_samples=len(data), before=state['before'],
                after=exact_evaluation(model, ham, args.gamma, data.cpu().numpy()) if args.L == 2 else None,
                minimum_training_lineage_ess=float(lineage[:, 0].min()),
                largest_training_lineage=float(lineage[:, 1].max()),
                training_ancestry_alarm=bool(lineage[:, 0].min() < 8 or lineage[:, 1].max() > .5),
                ground_state_qualified=False)
            write(directory/'summary.json', row); state['rows'].append(row)
            write(args.output/'summary.json', dict(rows=state['rows'], ground_state_qualified=False))
            print(json.dumps(row), flush=True)
            state.update(stint=stint+1, phase='start', step=0, update=0); save()
    state['phase'] = 'completed_unqualified'; save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial', type=Path)
    p.add_argument('--L', type=int, default=2); p.add_argument('--gamma', type=float, default=.6)
    p.add_argument('--sample-id', type=int, default=60000); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stints', type=int, default=6); p.add_argument('--walkers', type=int, default=1024)
    p.add_argument('--equil', type=float, default=8.); p.add_argument('--collect', type=float, default=12.)
    p.add_argument('--delta', type=float, default=.1); p.add_argument('--sample-walkers', type=int, default=64)
    p.add_argument('--updates', type=int, default=1000); p.add_argument('--batch', type=int, default=256)
    p.add_argument('--gibbs', type=int, default=10); p.add_argument('--initial-sweeps', type=int, default=400)
    p.add_argument('--lr', type=float, default=.001); p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    try:
        run(args)
    except Exception as error:
        write(args.output/'failure.json', dict(error=repr(error), ground_state_qualified=False))
        raise


if __name__ == '__main__':
    main()
