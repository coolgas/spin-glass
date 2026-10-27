import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, time, inspect, traceback, hashlib
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import kagome_qea.pqmc as pq
import kagome_qea.trial_overlap as trial
from kagome_qea.sample import make_sample
BASE = Path('/root/autodl-tmp/issue9-convergence-20261025-bc375b3')
CONFIGS = [
    dict(name='L6_eq40_t8', L=6, gamma=.05, equil=40, tau=8., populations=8, seed=20261025),
    dict(name='L6_eq40_t16', L=6, gamma=.05, equil=40, tau=16., populations=8, seed=20261025),
    dict(name='L6_eq80_t8', L=6, gamma=.05, equil=80, tau=8., populations=8, seed=20261025),
    dict(name='L6_eq160_t8', L=6, gamma=.05, equil=160, tau=8., populations=8, seed=20261025),
    dict(name='L2_completed_control', L=2, gamma=.2, equil=40, tau=8., populations=2, seed=0),
]

def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False))

def run(task):
    config, child = task
    r = pq._project_once(make_sample(config['L'], True, 0).hamiltonian(0.), config['gamma'],
        np.random.default_rng(np.random.SeedSequence(config['seed']).spawn(32)[child]),
        n_replicas=2, n_walkers=5000, tau_block=.2, n_equilibrate=config['equil'],
        n_measure=40, tau_forward=config['tau'], n_forward_cycles=8)
    np.savez_compressed(ROOT / 'runs' / (config['name'] + '_p' + str(child) + '.npz'), **r)
    assert pq.qualify_production(r)['qualified'] is False
    return config['name'], child

def load(name, child):
    with np.load(ROOT / 'runs' / (name + '_p' + str(child) + '.npz')) as data:
        return {key: data[key].item() if data[key].shape == () else data[key].copy() for key in data.files}

def se(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))

def trace_rollback():
    source, first = inspect.getsourcelines(pq._project_once)
    end_line = first + next(i for i, s in enumerate(source) if s.strip().startswith(
        'tags = np.fromiter((walker.tag for walker in saved)'))
    cycle_line = first + next(i for i, s in enumerate(source) if s.strip() ==
        'for identity, walker in enumerate(walkers):')
    rows = []
    pending = []
    def state(walkers):
        return [(w.spins.copy(), w.diagonal, w.weight, w.tag, w.ancestor) for w in walkers]
    def equal(a, b):
        return all(np.array_equal(x[0], y[0]) and x[1:] == y[1:] for x, y in zip(a, b))
    def local_trace(frame, event, arg):
        if event != 'line':
            return local_trace
        f = frame.f_locals
        if frame.f_lineno == cycle_line and pending:
            expected, shift = pending.pop()
            assert equal(state(f['walkers']), expected) and f['shift'] == shift
            rows[-1]['next_cycle_equal_backup'] = True
        if frame.f_lineno == end_line and f['rejected'] is not None:
            snapshot = state(f['backup'])
            assert equal(state(f['saved']), snapshot)
            assert equal(state(f['walkers']), snapshot)
            assert f['shift'] == f['shift_backup']
            weights = np.array([w.weight for w in f['saved']])
            assert np.all(weights == 1.)
            rows.append(dict(cycle=len(rows), achieved=float(f['elapsed']),
                rejected_dt=float(f['rejected'][0]), state_equal_backup=True,
                shift_equal_backup=True, unit_weights=True))
            pending.append((snapshot, f['shift_backup']))
        return local_trace
    def global_trace(frame, event, arg):
        return local_trace if frame.f_code is pq._project_once.__code__ else None
    sys.settrace(global_trace)
    try:
        result = pq._project_once(make_sample(6, True, 0).hamiltonian(0.), .05,
            np.random.default_rng(np.random.SeedSequence(20261025).spawn(32)[0]),
            n_walkers=5000, tau_block=.2, n_equilibrate=40, n_measure=40,
            tau_forward=8., n_forward_cycles=8)
    finally:
        sys.settrace(None)
    previous = load('L6_eq40_t8', 0)
    for key in ('chi_sg', 'energy', 'cycle_q2', 'forward_times'):
        assert np.array_equal(result[key], previous[key])
    assert len(rows) == 8 and sum(x.get('next_cycle_equal_backup', False) for x in rows) == 7
    dump('rollback_trace.json', rows)

def archive_checks(results):
    accepted = {}
    for config in CONFIGS:
        parts = results[config['name']]
        combined = pq._combine_populations(parts)
        sample = make_sample(config['L'], True, 0)
        name = config['name']
        target = ROOT/'archives'/name/'gaussian_nn'/'pbc'/('L'+str(config['L']))/(
            'pqmc_k'+str(config['populations']))/'h0'/'sigma1'/(
            'gamma'+trial.token(config['gamma']))/'sample0_seed0.npz'
        with patch.multiple(trial, N_POPULATIONS=config['populations'],
            N_EQUILIBRATE=config['equil'], TAU_FORWARD_AT_SIGMA_1=config['tau']):
            trial.save_run(target, sample, combined, 1., config['gamma'], .2, config['tau'], 0.)
            assert trial.result_ready(target)
            accepted[name] = True
            if name == 'L6_eq40_t8':
                with np.load(target) as data:
                    base = {k:data[k].copy() for k in data.files}
                shape = base['forward_times'].shape
                cases = {
                    'negative_rejected_ess': {'cycle_rejected_weight_ess':np.full(shape,-1.)},
                    'large_weight_ess': {'cycle_weight_ess':np.full(shape,50000.)},
                    'small_fraction': {'cycle_max_fraction':np.full(shape,1e-6)},
                    'fractional_floor': {'floor_unique':np.float64(2.9)},
                    'long_rejected_step': {'cycle_rejected_dt':np.full(shape,100.)},
                    'endpoint_mismatch': {'cycle_unique':base['cycle_unique']+1,
                        'min_unique_ancestors':base['min_unique_ancestors']+1},
                }
                points=base['cycle_checkpoints'].copy()
                mask=np.isfinite(points).all(axis=-1);points[mask,1:]=[-10.,-20.,-30.,-40.]
                cases['negative_checkpoints']={'cycle_checkpoints':points}
                points=base['cycle_checkpoints'].copy()
                for pop in points:
                    for cycle in pop:
                        indices=np.where(np.isfinite(cycle).all(axis=1))[0]
                        cycle[indices]=cycle[indices[::-1]].copy()
                cases['reverse_checkpoints']={'cycle_checkpoints':points}
                rejected={}
                for case, changes in cases.items():
                    data=dict(base);data.update(changes)
                    path=ROOT/'archives'/(case+'.npz');np.savez_compressed(path,**data)
                    ok=trial.result_ready(path,expected=dict(L=6,sample_id=0,sigma=1.,gamma=.05))
                    rejected[case]=not ok
                    assert not ok,case
                dump('archive_corruption_checks.json', rejected)
    dump('archive_roundtrips.json', accepted)

def main():
    started=time.monotonic()
    (ROOT/'runs').mkdir(exist_ok=True)
    source=json.loads((ROOT/'source_manifest.json').read_text())
    for name,digest in source['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    dump('campaign.json',dict(started=datetime.now(timezone.utc).isoformat(),configs=CONFIGS,
        walkers=5000,measure=40,cycles=8,block=.2,sample=0,sigma=1.,h=0.,periodic=True,
        guide='constant',floors=dict(effective=8,unique=2),workers=8,
        rng='SeedSequence(seed).spawn(32)[child]; matched across settings',
        scope='Single L6 disorder sample plus two completed L2 regression controls',source=source))
    tasks=[(c,k) for c in CONFIGS for k in range(c['populations'])]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name,child in pool.map(run,tasks):
            print('DONE',name,child,flush=True)
    results={c['name']:[load(c['name'],k) for k in range(c['populations'])] for c in CONFIGS}
    controls=[]
    for k,r in enumerate(results['L2_completed_control']):
        old=json.loads((BASE/'populations'/('reproduce_s0_g02_t8_p'+str(k)+'.json')).read_text())
        for new,prior in [('chi_sg','chi'),('q2','q2'),('q4','q4'),('energy','energy'),
            ('energy_growth','energy_growth'),('cycle_q2','cycles_q2'),('forward_times','achieved')]:
            assert np.array_equal(r[new],old[prior]),(new,k)
        assert r['projection_complete']
        controls.append(dict(child=k,unchanged=True,chi=r['chi_sg']))
    summaries=[]
    for c in CONFIGS:
        rows=results[c['name']];values=[r['chi_sg'] for r in rows];times=np.stack([r['forward_times'] for r in rows])
        fixed=[float(np.mean(r['energy_measure_trace'])) for r in rows]
        summaries.append(dict(name=c['name'],populations=len(rows),chi=float(np.mean(values)),se=se(values),
            mixed_energy=float(np.mean([r['energy'] for r in rows])),fixed_energy=float(np.mean(fixed)),
            fixed_energy_se=se(fixed),attained_min=float(times.min()),attained_max=float(times.max()),
            truncated=sum(r['n_forward_truncated'] for r in rows),cycles=int(times.size),
            fixed_first_half=float(np.mean([r['energy_measure_trace'][:20].mean() for r in rows])),
            fixed_second_half=float(np.mean([r['energy_measure_trace'][20:].mean() for r in rows]))))
    pairs=[]
    for second in ['L6_eq40_t16','L6_eq80_t8','L6_eq160_t8']:
        a=results['L6_eq40_t8'];b=results[second]
        delta=np.array([y['chi_sg']-x['chi_sg'] for x,y in zip(a,b)])
        energy=np.array([y['energy_measure_trace'].mean()-x['energy_measure_trace'].mean() for x,y in zip(a,b)])
        pairs.append(dict(first='L6_eq40_t8',second=second,chi_delta=float(delta.mean()),chi_delta_se=se(delta),
            fixed_energy_delta=float(energy.mean()),fixed_energy_delta_se=se(energy),
            identical_cycles=all(np.array_equal(x['cycle_q2'],y['cycle_q2']) for x,y in zip(a,b)),
            identical_attained=all(np.array_equal(x['forward_times'],y['forward_times']) for x,y in zip(a,b))))
    differences=[]
    for k,r in enumerate(results['L6_eq40_t8']):
        old=json.loads((BASE/'populations'/('fresh_L6_g005_t8_p'+str(k)+'.json')).read_text())
        differences.append(dict(child=k,old_chi=old['chi'],new_chi=r['chi_sg'],
            chi_delta=r['chi_sg']-old['chi'],first_cycle_unchanged=r['cycle_q2'][0]==old['cycles_q2'][0],
            later_cycles_changed=not np.array_equal(r['cycle_q2'][1:],old['cycles_q2'][1:])))
    dump('preliminary.json',dict(summaries=summaries,pairs=pairs,controls=controls,old_new=differences))
    archive_checks(results)
    trace_rollback()
    final=dict(summaries=summaries,pairs=pairs,controls=controls,old_new=differences,
        elapsed=time.monotonic()-started,finished=datetime.now(timezone.utc).isoformat(),failures=[])
    dump('final.json',final)
    print('FINAL',json.dumps(final),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc())
        raise
