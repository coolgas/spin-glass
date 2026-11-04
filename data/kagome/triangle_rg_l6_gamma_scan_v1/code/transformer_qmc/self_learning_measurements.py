"""Independent L2 validation after all finite self-learning stints finish.

Exact states are evaluation only. Frozen Gibbs audits use new chains and RNGs;
physical observables use the established independent four-population projector.
Two repeat SEs are descriptive, not an uncertainty or convergence certificate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import write
from .correlation_lengths import sha
from .physics import basis, exact_reference, make_af_sample
from .production_protocol import exact_spectra, geometry, identity
from .projected_overlap_histograms import run_one
from .self_learning_rbm import exact_evaluation
from .simple_guides import RBM
from .simple_overlap_scan import checked_pq


def observables(probability, states, lattice):
    q, mass = checked_pq(probability)
    spectrum = exact_spectra(probability, states, lattice)['connected_spectrum']
    return dict(q_grid=q.tolist(), mass=mass.tolist(), q=float(q@mass),
                q2=float(q**2@mass),
                spectrum=spectrum[[0]+lattice['kmin_indices']].tolist())


def compare(row, truth):
    difference = np.abs(np.asarray(row['spectrum'])-truth['spectrum'])
    tolerance = np.maximum(.1*np.abs(truth['spectrum']), .001)
    return dict(q2_error=float(row['q2']-truth['q2']),
        histogram_tv=float(np.abs(np.asarray(row['mass'])-truth['mass']).sum()/2),
        spectrum_max_tolerance_ratio=float(np.max(difference/tolerance)),
        spectrum_error=difference.tolist())


@torch.no_grad()
def frozen_sampler(model, probability, states, seed, output):
    """Multistart finite-chain check against the exact frozen model density.

    Chain-wise standard errors retain time correlation within each chain.
    Their validity still requires equilibration; they are not bias bounds.
    """
    dest = output/f'frozen_seed{seed}.json'
    if dest.exists():
        row = json.loads(dest.read_text())
        assert sha(dest.with_suffix('.npz')) == row['raw_sha256']
        return row
    device = next(model.parameters()).device
    x = torch.tensor(states, dtype=torch.float64, device=device)
    p = torch.tensor(probability, dtype=torch.float64, device=device)
    hidden = torch.tanh(x@model.w+model.b)
    target = torch.cat((2*(p@x), 2*(p@hidden),
                        (2*x.T@(p[:, None]*hidden)).flatten())).cpu().numpy()
    rows = []; arrays = {}
    for start_index, start in enumerate(['random', 'up', 'down']):
        generator = torch.Generator(device=device).manual_seed(20261104+100*seed+start_index)
        spins = (2*torch.randint(2, (256, 12), device=device, generator=generator)-1).double()
        if start != 'random':
            spins.fill_(1 if start == 'up' else -1)
        samples = []; statistics = torch.zeros((256, len(target)), dtype=torch.float64, device=device)
        for step in range(1500):
            spins = model.sweep(spins, generator)
            if step >= 500 and (step-500) % 10 == 0:
                samples.append(spins.cpu().numpy().astype(np.int8))
                h = torch.tanh(spins@model.w+model.b)
                statistics += torch.cat((2*spins, 2*h,
                    (2*spins[:, :, None]*h[:, None, :]).flatten(1)), dim=1)
        raw = np.asarray(samples); arrays[start] = raw
        per_chain = (statistics/len(raw)).cpu().numpy()
        mean = per_chain.mean(0); se = per_chain.std(0, ddof=1)/np.sqrt(len(per_chain))
        error = mean-target
        ids = ((1-raw.astype(np.int64))//2)@(1 << np.arange(12))
        empirical = np.bincount(ids.ravel(), minlength=4096)/ids.size
        arrays[start+'_gradient_mean'] = mean; arrays[start+'_gradient_se'] = se
        rows.append(dict(start=start, draws_per_chain=len(raw), chains=256,
            density_tv=float(np.abs(empirical-probability).sum()/2),
            gradient_rms_error=float(np.sqrt(np.mean(error**2))),
            gradient_max_absolute_error=float(np.abs(error).max()),
            gradient_max_standardized_error=float(np.max(np.abs(error)/np.maximum(se, 1e-15))),
            gradient_fraction_outside_3se=float(np.mean(np.abs(error)>3*se))))
    arrays['exact_gradient'] = target
    np.savez_compressed(dest.with_suffix('.npz'), **arrays)
    row = dict(seed=seed, starts=rows, raw_sha256=sha(dest.with_suffix('.npz')),
        burn_sweeps=500, retained_draws=100, thin_sweeps=10,
        caveat='Finite frozen-chain audit, not proof of adaptive negative-phase mixing.',
        ground_state_qualified=False)
    write(dest, row)
    return row


def run(pilot, output):
    output.mkdir(parents=True, exist_ok=True)
    models = []
    for seed in [0, 1]:
        folder = pilot/f'seed{seed}'
        manifest = json.loads((folder/'manifest.json').read_text())
        status = json.loads((folder/'status.json').read_text())
        assert manifest['settings']['L'] == 2
        assert manifest['settings']['gamma'] == .6 and manifest['settings']['sample_id'] == 60000
        assert status['state'] == 'completed_unqualified' and status['completed_stints'] == 6
        models.append(folder/'stint_005/model.pt')
    specs = [dict(model=str(models[seed]), task=dict(L=2, gamma=.6, sample_id=60000,
        kind='rbm_2', training_seed=seed, audit='self_learning_final_l2',
        walkers=512, equil=16., forward=16., start='guide', delta=.1, cycles=8, repeat=repeat))
        for seed in [0, 1] for repeat in [0, 1]]
    plan = dict(specs=specs, model_hashes=[sha(p) for p in models],
        source_sha256=sha(Path(__file__)), planned_runs=4, deadline=None,
        independent_from_training=True, ground_state_qualified=False)
    plan_path = output/'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        write(plan_path, plan)
    ham = make_af_sample(2, 60000).hamiltonian(.15)
    states = basis(12); lattice = geometry(2)
    exact = exact_reference(ham, .6)
    truth = dict(energy=exact['energy'], residual=exact['residual'],
                 **observables(exact['probability'], states, lattice))
    write(output/'exact.json', truth)
    np.savez_compressed(output/'exact.npz', states=states, probability=exact['probability'],
                        amplitude=exact['amplitude'])
    guides = []
    for seed, path in enumerate(models):
        model = RBM(12, 2).to('cuda')
        model.load_state_dict(torch.load(path, map_location='cuda', weights_only=False)['model'])
        with torch.no_grad():
            p = torch.softmax(2*model(torch.tensor(states, dtype=torch.float64, device='cuda')), 0).cpu().numpy()
        guide = dict(seed=seed, model_sha256=sha(path),
            **exact_evaluation(model, ham, .6), **observables(p, states, lattice))
        guide['errors'] = compare(guide, truth)
        guide['frozen_sampler'] = frozen_sampler(model, p, states, seed, output)
        guides.append(guide)
        write(output/'guides.json', dict(rows=guides))
    rows = []
    for spec in specs:
        write(output/'status.json', dict(state='projecting', completed=len(rows), planned=4,
                                        current=spec['task'], ground_state_qualified=False))
        record = run_one(spec, output/'projector')
        record['errors'] = dict(energy_error_per_site=(record['energy']-truth['energy'])/12,
                                **compare(record, truth))
        rows.append(record)
        write(output/'measurements.json', dict(rows=rows, ground_state_qualified=False))
        print(json.dumps(dict(seed=spec['task']['training_seed'], repeat=spec['task']['repeat'],
            errors=record['errors'], ancestry_alarm=record['ancestry_alarm'])), flush=True)
    summary = []
    for seed in [0, 1]:
        group = [r for r in rows if r['task']['training_seed'] == seed]
        mean = {}; se = {}
        for key in ['q', 'q2', 'mass', 'spectrum', 'energy']:
            values = np.asarray([r[key] for r in group])
            mean[key] = values.mean(0).tolist(); se[key] = (values.std(0, ddof=1)/np.sqrt(2)).tolist()
        summary.append(dict(seed=seed, mean=mean, descriptive_repeat_se=se,
            errors=dict(energy_error_per_site=(mean['energy']-truth['energy'])/12, **compare(mean, truth)),
            ancestry_alarms=sum(r['ancestry_alarm'] for r in group),
            minimum_descendant_ess=min(r['minimum_descendant_ess'] for r in group)))
    write(output/'summary.json', dict(rows=summary, exact=truth,
        uncertainty='Two-repeat SE is descriptive only; population, time and start studies remain required.',
        ground_state_qualified=False))
    write(output/'status.json', dict(state='completed_unqualified', completed=4, planned=4,
                                    ground_state_qualified=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    try:
        with threadpool_limits(limits=1):
            run(args.pilot, args.output)
    except Exception as error:
        write(args.output/'failure.json', dict(error=repr(error), ground_state_qualified=False))
        raise


if __name__ == '__main__':
    main()
