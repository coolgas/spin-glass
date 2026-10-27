import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, time, hashlib, traceback, importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import kagome_qea.pqmc as pq
import kagome_qea.trial_overlap as trial
from kagome_qea.sample import make_sample
from ising import chain
REVISION = '6f130ae809215bdcfc579239b0f6d2fdad540ac9'
BASE = Path('/root/autodl-tmp/issue9-convergence-20261025-bc375b3')

def dump(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, allow_nan=False))

def same(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))

def replay(task):
    c, child = task
    start = time.monotonic()
    r = pq._project_once(make_sample(c['L'], True, c['sample']).hamiltonian(0.),
        c['gamma'], np.random.default_rng(np.random.SeedSequence(c['seed']).spawn(32)[child]),
        n_replicas=2, n_walkers=c['walkers'], tau_block=c['block'],
        n_equilibrate=c['equil'], n_measure=c['measure'], tau_forward=c['tau'], n_forward_cycles=c['cycles'])
    name = c['name'] + '_p' + str(child)
    np.savez_compressed(ROOT / 'runs' / (name + '.npz'), **r)
    old = json.loads((BASE / 'populations' / (name + '.json')).read_text())
    fields = {'chi':'chi_sg', 'q2':'q2', 'q4':'q4', 'energy':'energy',
        'energy_growth':'energy_growth', 'achieved':'forward_times', 'truncated':'n_forward_truncated',
        'complete':'projection_complete', 'unique':'min_unique_ancestors',
        'effective':'min_effective_ancestors', 'cycles_q2':'q2_series'}
    differences = [key for key,new in fields.items() if not same(old[key],r[new])]
    assert not differences, (name,differences)
    checks = []
    for i, (achieved, rows) in enumerate(zip(r['forward_times'], r['cycle_checkpoints'])):
        rows = rows[np.isfinite(rows).all(axis=1)]
        final = rows[-1]
        expected = [achieved, r['cycle_unique'][i], r['cycle_effective'][i],
                    r['cycle_weight_ess'][i], r['cycle_max_fraction'][i]]
        assert np.array_equal(final, expected), (name, i, 'checkpoint mismatch')
        assert np.all(np.diff(rows[:,0]) >= 0)
        stop = int(r['cycle_stop'][i])
        if stop:
            hit_k = r['cycle_rejected_effective'][i] < r['floor_effective']
            hit_u = r['cycle_rejected_unique'][i] < r['floor_unique']
            assert stop == int(hit_k) + 2 * int(hit_u)
        else:
            assert np.isnan(r['cycle_rejected_dt'][i])
        checks.append(stop)
    assert np.allclose(r['energy_equilibration_time'], c['block']*np.arange(1,c['equil']+1))
    assert np.allclose(r['energy_measure_time'], c['block']*np.arange(c['equil']+1,c['equil']+c['measure']+1))
    return dict(name=name, baseline_identical=True, chi=float(r['chi_sg']), energy=float(r['energy']),
        achieved=np.asarray(r['forward_times']).tolist(), stops=checks,
        fixed_window_energy=float(np.mean(r['energy_measure_trace'])),
        min_kish=float(r['min_effective_ancestors']), min_unique=int(r['min_unique_ancestors']),
        seconds=time.monotonic()-start)

def rollback_probe(module, label):
    rows=[]
    source=Path(module.__file__).read_text().splitlines()
    target=next(i for i,s in enumerate(source,1) if s.strip().startswith('tags = np.fromiter((walker.tag for walker in saved)'))
    def trace(frame,event,arg):
        if event=='line' and frame.f_code is module._project_once.__code__ and frame.f_lineno==target:
            f=frame.f_locals
            if f['elapsed'] + 1e-8 < 30.:
                saved, backup = f['saved'], f['backup']
                rows.append(dict(elapsed=float(f['elapsed']),
                    saved_is_backup=saved is backup, walkers_is_backup=f['walkers'] is backup,
                    changed_spins=sum(not np.array_equal(a.spins,b.spins) for a,b in zip(saved,backup)),
                    changed_diagonals=sum(a.diagonal!=b.diagonal for a,b in zip(saved,backup)),
                    changed_weights=sum(a.weight!=b.weight for a,b in zip(saved,backup)),
                    saved_weight_min=float(min(w.weight for w in saved)),
                    saved_weight_max=float(max(w.weight for w in saved)),
                    backup_weight_min=float(min(w.weight for w in backup)),
                    backup_weight_max=float(max(w.weight for w in backup))))
        return trace
    sys.settrace(trace)
    try:
        r=module._project_once(chain(4,J=1.),1.,np.random.default_rng(17),n_walkers=30,
            tau_block=.2,n_equilibrate=2,n_measure=2,tau_forward=30.,n_forward_cycles=3)
    finally:
        sys.settrace(None)
    return dict(label=label, rows=rows, chi=float(r['chi_sg']), times=r['forward_times'].tolist())

def probes():
    findings={}
    findings['rollback_current']=rollback_probe(pq,'6f130ae')
    spec=importlib.util.spec_from_file_location('old_pqmc', BASE/'code/kagome_qea/pqmc.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    findings['rollback_baseline']=rollback_probe(old,'bc375b3')
    floor_cases=[]
    for effective,unique in [(8.,2.9),(31.,2),(8.,31),(8.,np.bool_(True)),(8.,np.nan),(8.,np.inf)]:
        try:
            value=pq._resolve_floors(30,effective,unique)
            floor_cases.append(dict(input=[str(effective),str(unique)],accepted=True,output=value))
        except Exception as e:
            floor_cases.append(dict(input=[str(effective),str(unique)],accepted=False,error=type(e).__name__))
    findings['floor_cases']=floor_cases
    ess=[]
    for weights in ([1.,1.],[1e200,1e200],[1e-200,1e-200]):
        walkers=[pq._Walker(np.ones((2,2),dtype=np.int8),0.) for _ in weights]
        for w,value in zip(walkers,weights):w.weight=value
        with np.errstate(all='ignore'):value=pq._weight_ess(walkers)
        ess.append(dict(weights=[str(v) for v in weights],ess=str(value)))
    findings['weight_scale_invariance']=ess
    sample=make_sample(2,True,0)
    r=pq.project(sample.hamiltonian(0.),1.,n_walkers=30,n_equilibrate=2,n_measure=3,
        tau_block=.05,tau_forward=30.,n_forward_cycles=2,n_populations=2,seed=42)
    archive=ROOT/'archive'/'gaussian_nn'/'pbc'/'L2'/'pqmc_k2'/'h0'/'sigma1'/'gamma1'/'sample0_seed0.npz'
    settings=dict(N_POPULATIONS=2,N_WALKERS=30,N_EQUILIBRATE=2,N_MEASURE=3,
        N_FORWARD_CYCLES=2,TAU_BLOCK_AT_SIGMA_1=.05,TAU_FORWARD_AT_SIGMA_1=30.)
    with patch.multiple(trial,**settings):
        trial.save_run(archive,sample,r,1.,1.,.05,30.,0.)
        with np.load(archive) as data:payload={key:data[key].copy() for key in data.files}
        expected=dict(L=2,sample_id=0,sigma=1.,gamma=1.)
        findings['real_archive_roundtrip']=trial.result_ready(archive,expected=expected)
        tests={
            'negative_rejected_weight_ess':{'cycle_rejected_weight_ess':np.full((2,2),-1.)},
            'above_population_weight_ess':{'cycle_weight_ess':np.full((2,2),3000.)},
            'impossible_descendant_fraction':{'cycle_max_fraction':np.full((2,2),1e-6)},
            'fractional_unique_floor':{'floor_unique':np.float64(2.9)},
            'rejected_step_past_requested':{'cycle_rejected_dt':np.full((2,2),100.)},
        }
        points=payload['cycle_checkpoints'].copy()
        finite=np.isfinite(points).all(axis=-1)
        points[finite,1:]=[-10.,-20.,-30.,-40.]
        tests['negative_checkpoint_diagnostics']={'cycle_checkpoints':points}
        points=payload['cycle_checkpoints'].copy()
        for pop in points:
            for cycle in pop:
                index=np.where(np.isfinite(cycle).all(axis=1))[0]
                cycle[index]=cycle[index[::-1]].copy()
        tests['reverse_checkpoint_time']={'cycle_checkpoints':points}
        tests['inconsistent_final_checkpoint']={'cycle_unique':payload['cycle_unique']+1,
            'min_unique_ancestors':payload['min_unique_ancestors']+1}
        outcomes={}
        for name,changes in tests.items():
            mutated=dict(payload);mutated.update(changes)
            path=ROOT/'archive'/(name+'.npz');np.savez_compressed(path,**mutated)
            outcomes[name]=trial.result_ready(path,expected=expected)
        findings['corrupt_archive_accepted']=outcomes
    dump('probes.json',findings)
    print('PROBES',json.dumps(findings),flush=True)

def main():
    start=time.monotonic()
    (ROOT/'runs').mkdir(exist_ok=True)
    probes()
    manifest=json.loads((BASE/'manifest.json').read_text())
    counts={'reproduce_s0_g02_t8':16,'fresh_s0_g02_t8':2,'fresh_s0_g02_t16':2,
        'fresh_s0_g1_t8':2,'fresh_s0_g1_t16':2,'fresh_L6_g005_t8':2,
        'fresh_L6_g005_t16':2,'fresh_L6_g005_eq80_t8':2}
    tasks=[(c,k) for c in manifest['configs'] if c['name'] in counts for k in range(counts[c['name']])]
    results=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row in pool.map(replay,tasks):
            results.append(row);print('REPLAY',json.dumps(row),flush=True)
    out=dict(revision=REVISION,replays=results,elapsed=time.monotonic()-start)
    dump('final.json',out)
    print('FINISHED',len(results),out['elapsed'],flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc())
        raise
