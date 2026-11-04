"""Pure overlap histograms on AutoDL, with four independent GFMC populations.

L4 replays frozen production trajectories and verifies their original moments.
Production source and checkpoints remain separate from this measurement run.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import checkpoint, label
from .disorder_scaling import observables
from .physics import make_af_sample
from .preliminary_overlap_notebook import digest, dump
from .production_protocol import identity
from .projector import propagate, rates_and_local
from .sampled_simple_guides import TemperedChains
from .simple_guides import Jastrow, RBM, SimpleGuide


def histogram(tags, counts):
    """Pool equal cycles, then cross measures from distinct populations.

    Descendant weights are paired only across independent populations. This
    integrates to products of cycle-averaged moments, as in production.
    """
    cycles, populations, walkers, n = tags.shape
    measures = []
    for population in range(populations):
        spins = tags[:, population].reshape(-1, n)
        weights = counts[:, population].reshape(-1) / (cycles * walkers)
        keep = weights > 0
        spins, inverse = np.unique(spins[keep], axis=0, return_inverse=True)
        mass = np.bincount(inverse, weights=weights[keep], minlength=len(spins))
        measures.append((spins.astype(np.int16), mass))
    result = np.zeros(n + 1)
    pairs = 0
    for a in range(populations):
        for b in range(a + 1, populations):
            sa, wa = measures[a]; sb, wb = measures[b]
            for start in range(0, len(sa), 128):
                indices = (sa[start:start+128] @ sb.T + n) // 2
                weights = wa[start:start+128, None] * wb[None, :]
                result += np.bincount(indices.ravel(), weights=weights.ravel(), minlength=n+1)
            pairs += 1
    result /= pairs
    np.testing.assert_allclose(result.sum(), 1., atol=1e-12)
    return np.linspace(-1, 1, n+1), result


def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_one(spec, output):
    with threadpool_limits(limits=1):
        return _run_one(spec, Path(output))


def _run_one(spec, root):
    task = spec['task']; key = identity(task)[:24]
    dest = root/'runs'/(key+'.json'); cp = root/'checkpoints'/(key+'.npz')
    provenance = dict(model_sha256=digest(Path(spec['model'])), source_sha256=digest(Path(__file__)),
        numerics='log_weight_v1',numerics_sha256=digest(Path(__file__).parent.parent/'qmc_numerics.py'),
        sources={name:digest(Path(__file__).with_name(name)) for name in
                 ['projector.py','disorder_scaling.py','physics.py','simple_guides.py',
                  'independent_replica_projector.py']})
    projection_device = spec.get('projection_device', 'cpu')
    if projection_device != 'cpu':
        provenance['projection_device'] = projection_device
    if dest.exists():
        record = json.loads(dest.read_text())
        assert record['provenance'] == provenance
        assert digest(dest.with_suffix('.npz')) == record['raw_sha256']
        return record
    n = 3*task['L']**2; w = task['walkers']; dt = task['delta']
    ham = make_af_sample(task['L'], task['sample_id']).hamiltonian(.15)
    model = Jastrow(n) if task['kind'] == 'jastrow' else RBM(n, 2)
    model = model.to(projection_device)
    state = torch.load(spec['model'], map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    guide = SimpleGuide(model, sweeps=200)
    rng = np.random.default_rng(int(identity(task)[:16], 16))
    equil = round(task['equil']/dt); forward = round(task['forward']/dt)
    started = time.monotonic(); elapsed = 0.
    if cp.exists():
        with np.load(cp) as data:
            meta = json.loads(str(data['metadata']))
            assert meta['task'] == task and meta['provenance'] == provenance
            rng.bit_generator.state = meta['rng']; step = meta['step']; elapsed = meta['seconds']
            states = data['states'].copy(); shifts = data['shifts'].copy()
            tagged = data['tagged'].copy(); ancestors = data['ancestors'].copy()
            all_tags = list(data['tags']); all_counts = list(data['counts'])
            weight_ess = list(data['weight_ess'])
            energies=list(data['energy']);growths=list(data['growth'])
            log_weight_ranges=list(data['log_weight_ranges'])
    else:
        states = (guide.sample(4*w, int(rng.integers(2**32))).reshape(4, w, n)
                  if task['start'] == 'guide'
                  else (1-2*rng.integers(2, size=(4, w, n))).astype(np.int8))
        shifts = np.array([rates_and_local(ham, task['gamma'], guide, s[:, None, :])[1].mean() for s in states])
        tagged = states.astype(float); ancestors = np.tile(np.arange(w), (4, 1))
        all_tags = []; all_counts = []; weight_ess = []; step = 0
        energies=[];growths=[];log_weight_ranges=[]

    def save():
        cp.parent.mkdir(parents=True, exist_ok=True)
        checkpoint(cp, dict(states=states, shifts=shifts, tagged=tagged, ancestors=ancestors,
                            tags=np.asarray(all_tags, dtype=np.int8), counts=np.asarray(all_counts),
                            weight_ess=np.asarray(weight_ess),energy=np.asarray(energies),
                            growth=np.asarray(growths),log_weight_ranges=np.asarray(log_weight_ranges)),
                   dict(task=task, provenance=provenance, rng=rng.bit_generator.state,
                        step=step, seconds=elapsed+time.monotonic()-started))
        dump(root/'progress'/(key+'.json'),dict(task=task,step=step,
             total=equil+task['cycles']*forward,cycles=len(all_tags),numerics='log_weight_v1'))

    while step < equil+task['cycles']*forward:
        weights = [];energy=[];growth=[];ranges=[]
        for r in range(4):
            diagnostic={}
            s, children, shifts[r], mixed, eg, we = propagate(
                ham, task['gamma'], guide, states[r, :, None, :], shifts[r], dt, rng,diagnostic)
            states[r] = s[:, 0]; weights.append(we)
            energy.append(mixed);growth.append(eg)
            ranges.append([diagnostic['min_log_weight'],diagnostic['max_log_weight']])
            if step >= equil:
                ancestors[r] = ancestors[r, children]
        step += 1; weight_ess.append(weights)
        energies.append(energy);growths.append(growth);log_weight_ranges.append(ranges)
        boundary = step == equil or (step > equil and (step-equil) % forward == 0)
        if step > equil and boundary:
            all_tags.append(tagged.astype(np.int8))
            all_counts.append(np.array([np.bincount(a, minlength=w) for a in ancestors]))
        if boundary:
            tagged = states.astype(float); ancestors = np.tile(np.arange(w), (4, 1))
        if boundary or step % 100 == 0:
            save()
    tags = np.asarray(all_tags, dtype=np.int8); counts = np.asarray(all_counts)
    grid, mass = histogram(tags, counts)
    ms = np.einsum('crw,crwn->crn', counts/w, tags)
    ps = np.einsum('crw,crwi,crwj->crij', counts/w, tags, tags)
    moments = observables(ms.mean(0), ps.mean(0), task['L'])
    np.testing.assert_allclose([grid@mass, grid**2@mass], [moments['q'], moments['q2']], atol=1e-12)
    if spec.get('original'):
        original = json.loads(Path(spec['original']).read_text())
        np.testing.assert_allclose([moments['q'], moments['q2']], [original['q'], original['q2']], atol=1e-12, rtol=0)
        np.testing.assert_allclose(moments['spectrum'], original['spectrum'], atol=1e-12, rtol=0)
    ess = w*w/(counts.astype(float)**2).sum(2); lineage = counts.max(2)/w
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest.with_suffix('.npz'), tags=tags, counts=counts,
                        descendant_ess=ess, largest_lineage=lineage, weight_ess=weight_ess,
                        energy=energies,growth=growths,log_weight_ranges=log_weight_ranges)
    record = dict(task=task, provenance=provenance, q_grid=grid.tolist(), mass=mass.tolist(),
                  **moments, minimum_descendant_ess=float(ess.min()),
                  largest_lineage=float(lineage.max()),energy=float(np.mean(energies[equil:])),
                  growth_energy=float(np.mean(growths[equil:])),
                  ancestry_alarm=bool(ess.min()<8 or lineage.max()>.5),
                  replay_verified=bool(spec.get('original')), ground_state_qualified=False,
                  raw_file=str(dest.with_suffix('.npz').relative_to(root.parent)),
                  raw_sha256=digest(dest.with_suffix('.npz')), seconds=elapsed+time.monotonic()-started)
    dump(dest, record)
    return record


@torch.no_grad()
def sample_l2(spec, output):
    """Finite MCMC, using independent chain columns for overlap replicas."""
    t = spec['task']; root = Path(output); key = identity(t)[:24]
    dest = root/'guide_samples_l2'/(key+'.json')
    if dest.exists():
        r = json.loads(dest.read_text())
        assert digest(dest.with_suffix('.npz')) == r['raw_sha256']
        return r
    model = Jastrow(12) if t['kind']=='jastrow' else RBM(12, 2)
    model.load_state_dict(torch.load(spec['model'], map_location='cpu', weights_only=False))
    arrays = {}; starts = []
    for start in ['random', 'up', 'down']:
        sampler = TemperedChains(model, 256, 8, int(identity(dict(t, start=start))[:8], 16), start)
        samples = []
        for step in range(1000):
            spins = sampler.sweep()
            if step >= 500 and (step-500) % 5 == 0:
                samples.append((spins[:128] != spins[128:]).sum(1).numpy())
        values = np.asarray(samples, dtype=np.int8); arrays[start+'_mismatches'] = values
        mass = np.bincount(values.ravel(), minlength=13)[::-1]/values.size
        starts.append(dict(start=start, mass=mass.tolist(), **sampler.diagnostics()))
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest.with_suffix('.npz'), **arrays)
    result = dict(task=t, q_grid=np.linspace(-1, 1, 13).tolist(), starts=starts,
                  model_sha256=digest(Path(spec['model'])),
                  raw_file=str(dest.with_suffix('.npz').relative_to(root)), raw_sha256=digest(dest.with_suffix('.npz')))
    dump(dest, result)
    return result


def validate_histogram():
    from .simple_overlap_scan import checked_pq
    from .physics import basis
    from itertools import combinations
    rng = np.random.default_rng(418)
    tags = (1-2*rng.integers(2, size=(3, 4, 19, 6))).astype(np.int8)
    counts = np.array([[rng.multinomial(19, np.ones(19)/19) for _ in range(4)] for _ in range(3)])
    q, p = histogram(tags, counts)
    m = np.einsum('crw,crwn->rn', counts/57, tags)
    pair = np.einsum('crw,crwi,crwj->rij', counts/57, tags, tags)
    np.testing.assert_allclose(q@p, np.mean([np.mean(m[a]*m[b]) for a,b in combinations(range(4),2)]), atol=1e-12)
    np.testing.assert_allclose(q**2@p, np.mean([np.mean(pair[a]*pair[b]) for a,b in combinations(range(4),2)]), atol=1e-12)
    states = basis(6); weights = rng.integers(1, 8, size=64)
    exact_q, exact_p = checked_pq(weights/weights.sum())
    expanded = np.repeat(states, weights, axis=0)
    tags = np.tile(expanded[None,None,:,:], (1,4,1,1))
    q, p = histogram(tags, np.ones(tags.shape[:-1], dtype=int))
    np.testing.assert_allclose(q, exact_q, atol=1e-12)
    np.testing.assert_allclose(p, exact_p, atol=1e-12)
    print('Validated weighted pure histogram against exact convolution and independent moments.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--exact-scan', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    a = parser.parse_args(); validate_histogram(); specs = []
    for record in json.loads((a.snapshot/'run_records.json').read_text())['rows']:
        t = record['task']
        if t['L']==4 and t['gamma'] in [.6,.65] and t['audit']=='main':
            name = 'L{L}_s{sample_id}_g{gamma:g}_{kind}_anneal_seed{training_seed}'.format(**t).replace('.', 'p')
            specs.append(dict(task=t, model=str(a.production/'training'/(name+'.pt')),
                              original=str(a.snapshot/'projector/runs'/(identity(t)[:24]+'.json'))))
    assert len(specs)==64
    l2 = []
    for row in json.loads((a.snapshot/'exact_l2_reference.json').read_text())['rows']:
        if row['sample_id']==50043:
            continue
        for fit in row['fits']:
            if not fit['selected']:
                continue
            model = a.exact_scan/(label(row['gamma'],row['sample_id'])+f"_{fit['kind']}_seed{fit['seed']}.pt")
            for repeat in [0,1]:
                t = dict(L=2, gamma=row['gamma'], sample_id=row['sample_id'], kind=fit['kind'],
                         training_seed=fit['seed'], audit='main', walkers=512, equil=16., forward=16.,
                         start='guide', delta=.1, cycles=8, repeat=repeat)
                item = dict(task=t, model=str(model)); specs.append(item)
                if repeat==0:
                    l2.append(item)
    assert len(l2)==32
    dump(a.snapshot/'histogram_protocol.json', dict(source_sha256=digest(Path(__file__)), specs=specs, guide_specs=l2,
        estimator='Four independently branching populations; pool equal forward cycles, histogram all cross-population weighted pairs; average six population pairs, two repeats, then eight disorders equally.',
        L4='Deterministic replay; q, q2 and spectra must agree with frozen production to 1e-12.',
        L2='Selected full-basis energy-trained guides; before: actual parallel-tempering MCMC; after: CT GFMC, 512 walkers/population, equil=forward=16, eight cycles, two repeats.',
        uncertainty='Capped one-standard-error bars across eight disorder means; exclude mixing and projection-bias bounds.'))
    projected = []; sampled = []; failures = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=configure) as pool:
        futures = {pool.submit(run_one, s, str(a.snapshot/'projected_histograms')): ('projected', s) for s in specs}
        futures.update({pool.submit(sample_l2, s, str(a.snapshot)): ('sampled', s) for s in l2})
        for future in as_completed(futures):
            kind, spec = futures[future]
            try:
                r = future.result()
            except Exception as error:
                failures.append(dict(kind=kind, spec=spec, error=repr(error)))
                dump(a.snapshot/'histogram_failures.json', dict(rows=failures)); raise
            (projected if kind=='projected' else sampled).append(r)
            dump(a.snapshot/'projected_distributions.json', dict(rows=projected))
            dump(a.snapshot/'guide_distributions_l2.json', dict(rows=sampled))
            print(kind, len(projected), len(sampled), spec['task'], flush=True)
    dump(a.snapshot/'histogram_complete.json', dict(projected=len(projected), sampled=len(sampled), failures=failures))


if __name__=='__main__':
    main()
