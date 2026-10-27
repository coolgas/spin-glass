import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, time, traceback, hashlib, platform, concurrent.futures
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from kagome_qea.sample import make_sample
from kagome_qea.pqmc import _project_once
from kagome_qea.exact import ground_state, operator_matrix

REVISION = 'bc375b3d79cd51a5d56bca04f656e0c17f6e803d'
WORKERS = 16
CONFIGS = []
def add(name, L=2, sample=0, gamma=.2, walkers=5000, equil=40, tau=8., populations=16, seed=20261025):
    CONFIGS.append(dict(name=name, L=L, sample=sample, gamma=gamma, walkers=walkers,
                        equil=equil, tau=tau, populations=populations, measure=40, cycles=8, block=.2, seed=seed))
add('reproduce_s0_g02_t8', seed=0)
add('reproduce_s0_g02_t16', tau=16., populations=8, seed=0)
for tau in (4.,8.,16.):
    add('fresh_s0_g02_t'+str(int(tau)), tau=tau, populations=32 if tau==8 else 16)
add('fresh_s0_g02_w10000_t8', walkers=10000)
add('fresh_s0_g02_eq80_t8', equil=80)
for sample in (1,2):
    for tau in (8.,16.):
        add('fresh_s'+str(sample)+'_g02_t'+str(int(tau)), sample=sample, tau=tau)
for tau in (8.,16.):
    add('fresh_s0_g1_t'+str(int(tau)), gamma=1., tau=tau)
add('fresh_s0_g005_t8', gamma=.05, populations=8)
add('fresh_s0_g3_t4', gamma=3., tau=4., populations=16)
for tau in (8.,16.):
    add('fresh_L6_g005_t'+str(int(tau)), L=6, gamma=.05, tau=tau, populations=8)
add('fresh_L6_g005_w10000_t8', L=6, gamma=.05, walkers=10000, populations=8)
add('fresh_L6_g005_eq80_t8', L=6, gamma=.05, equil=80, populations=8)

def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)

def exact_reference(sample_id, gamma):
    ham = make_sample(2, True, sample_id).hamiltonian(0.)
    exact = ground_state(ham, gamma)
    H = operator_matrix(ham, gamma)
    half = H.shape[0] // 2
    even = H[:half,:half] + H[:half,half:][:,::-1]
    vals, vecs = eigsh(even, k=1, which='SA', tol=1e-13, v0=np.ones(half))
    E = float(vals[0])
    v = vecs[:,0]
    if v.sum() < 0: v = -v
    psi = np.concatenate((v, v[::-1])) / np.sqrt(2.)
    if np.linalg.norm(H @ psi - E * psi) > 1e-10:
        raise RuntimeError('Even-sector reference residual too large')
    index = np.arange(H.shape[0])
    spins = 1. - 2. * ((index[:, None] >> np.arange(ham.n_sites)) & 1)
    def susceptibility(prob):
        prob = prob / prob.sum()
        C = spins.T @ (spins * prob[:, None])
        return float(np.sum(C * C) / ham.n_sites)
    check_chi = susceptibility(psi * psi)
    if abs(check_chi - exact['chi_sg']) > 1e-7:
        raise RuntimeError('Independent exact reference disagreement')
    shifted = H - E * sparse.eye(H.shape[0], format='csr')
    ones = np.ones(H.shape[0])
    def prop(vector, tau):
        out = expm_multiply(-tau * shifted, vector)
        return out / np.linalg.norm(out)
    finite = []
    for tau in (2., 4., 8., 16., 32.):
        left = prop(ones, tau)
        finite.append(dict(forward=tau, exact_right_chi=susceptibility(left * psi),
                           right16_chi=susceptibility(left * prop(ones, 16.))))
    return dict(sample=sample_id, gamma=gamma, chi=float(exact['chi_sg']),
                energy=float(exact['energy']), residual=float(exact['residual']),
                independent_chi=check_chi, independent_residual=float(np.linalg.norm(H @ psi - E * psi)),
                gap=float(exact['gap']), mixed=bool(exact['mixed']), finite_projection=finite)

def work(task):
    c, child = task
    started = time.monotonic()
    ham = make_sample(c['L'], True, c['sample']).hamiltonian(0.)
    rng = np.random.default_rng(np.random.SeedSequence(c['seed']).spawn(32)[child])
    r = _project_once(ham, c['gamma'], rng, n_replicas=2, n_walkers=c['walkers'],
                      tau_block=c['block'], n_equilibrate=c['equil'], n_measure=c['measure'],
                      tau_forward=c['tau'], n_forward_cycles=c['cycles'])
    times = np.asarray(r['forward_times'])
    row = dict(config=c['name'], child=child, chi=float(r['chi_sg']), q2=float(r['q2']),
               q4=float(r['q4']), energy=float(r['energy']), energy_growth=float(r['energy_growth']),
               achieved=times.tolist(), truncated=int(r['n_forward_truncated']),
               complete=bool(r['projection_complete']), unique=int(r['min_unique_ancestors']),
               effective=float(r['min_effective_ancestors']),
               cycles_q2=np.asarray(r['q2_series']).tolist(), seconds=time.monotonic()-started)
    dump(ROOT / 'populations' / (c['name'] + '_p' + str(child) + '.json'), row)
    return row

def stderr(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / np.sqrt(values.size))

def summarize(config, rows, refs):
    rows = sorted(rows, key=lambda x: x['child'])
    values = np.array([r['chi'] for r in rows])
    times = np.array([r['achieved'] for r in rows])
    ref = refs.get(str(config['sample']) + ':' + str(config['gamma'])) if config['L'] == 2 else None
    result = dict(config=config, count=len(rows), chi=float(values.mean()), se=stderr(values),
                  energy=float(np.mean([r['energy'] for r in rows])),
                  energy_se=stderr([r['energy'] for r in rows]),
                  achieved_min=float(times.min()), achieved_median=float(np.median(times)),
                  achieved_max=float(times.max()),
                  complete_populations=sum(r['complete'] for r in rows),
                  truncated_cycles=sum(r['truncated'] for r in rows), cycles=int(times.size),
                  min_unique=min(r['unique'] for r in rows), min_effective=min(r['effective'] for r in rows),
                  cpu_seconds=sum(r['seconds'] for r in rows), chi_populations=values.tolist())
    if ref:
        result.update(exact=ref['chi'], difference=float(values.mean()-ref['chi']),
                      difference_se=float((values.mean()-ref['chi'])/stderr(values)))
    return result

def main():
    start = time.monotonic()
    (ROOT / 'populations').mkdir(exist_ok=True)
    (ROOT / 'summaries').mkdir(exist_ok=True)
    metadata = dict(revision=REVISION, started=datetime.now(timezone.utc).isoformat(),
                    workers=WORKERS, platform=platform.platform(), python=sys.version, numpy=np.__version__,
                    rng='Per-config SeedSequence(seed).spawn(32)[child]; seed0 reproduction and seed20261025 holdout',
                    configs=CONFIGS, tasks=sum(c['populations'] for c in CONFIGS))
    dump(ROOT / 'manifest.json', metadata)
    print('START', json.dumps(metadata), flush=True)
    refs = {}
    for sample, gamma in sorted({(c['sample'],c['gamma']) for c in CONFIGS if c['L']==2}):
        ref = exact_reference(sample, gamma)
        refs[str(sample)+':'+str(gamma)] = ref
        print('EXACT', json.dumps(ref), flush=True)
    dump(ROOT / 'exact.json', refs)
    results = {c['name']:[] for c in CONFIGS}
    failures = []
    tasks = [(c,k) for k in range(32) for c in CONFIGS if k<c['populations']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(work, task):task for task in tasks}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            config, child = futures[future]
            try:
                row = future.result()
                results[config['name']].append(row)
                if len(results[config['name']]) == config['populations']:
                    summary = summarize(config, results[config['name']], refs)
                    dump(ROOT / 'summaries' / (config['name']+'.json'), summary)
                    print('SUMMARY', json.dumps(summary), flush=True)
            except Exception:
                failure = dict(config=config['name'], child=child, traceback=traceback.format_exc())
                failures.append(failure)
                print('FAILURE', json.dumps(failure), flush=True)
            done += 1
            if done % 16 == 0:
                dump(ROOT / 'progress.json', dict(done=done, total=len(tasks), failures=len(failures), seconds=time.monotonic()-start))
                print('PROGRESS', done, '/', len(tasks), round(time.monotonic()-start,1), flush=True)
    summaries = [summarize(c, results[c['name']], refs) for c in CONFIGS if len(results[c['name']])>=2]
    paired = []
    comparisons = [('reproduce_s0_g02_t8', 'reproduce_s0_g02_t16'), ('fresh_s0_g02_t4', 'fresh_s0_g02_t8'), ('fresh_s0_g02_t8', 'fresh_s0_g02_t16'), ('fresh_s0_g02_t8', 'fresh_s0_g02_w10000_t8'), ('fresh_s0_g02_t8', 'fresh_s0_g02_eq80_t8'), ('fresh_s1_g02_t8', 'fresh_s1_g02_t16'), ('fresh_s2_g02_t8', 'fresh_s2_g02_t16'), ('fresh_s0_g1_t8', 'fresh_s0_g1_t16'), ('fresh_L6_g005_t8', 'fresh_L6_g005_t16'), ('fresh_L6_g005_t8', 'fresh_L6_g005_w10000_t8'), ('fresh_L6_g005_t8', 'fresh_L6_g005_eq80_t8')]
    for a,b in comparisons:
        left={r['child']:r for r in results[a]}; right={r['child']:r for r in results[b]}
        keys=sorted(left.keys() & right.keys())
        if len(keys)<2: continue
        delta=np.array([right[k]['chi']-left[k]['chi'] for k in keys])
        paired.append(dict(first=a,second=b,populations=len(keys),delta=float(delta.mean()),se=stderr(delta),
                           identical_q2_cycles=all(left[k]['cycles_q2']==right[k]['cycles_q2'] for k in keys),
                           identical_achieved_times=all(left[k]['achieved']==right[k]['achieved'] for k in keys)))
    final=dict(metadata=metadata,refs=refs,summaries=summaries,paired=paired,failures=failures,
               elapsed=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat())
    dump(ROOT/'final.json',final)
    print('PAIRED',json.dumps(paired),flush=True)
    print('FINISHED',json.dumps(dict(seconds=final['elapsed'],failures=len(failures))),flush=True)
    if failures: raise SystemExit(1)

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (ROOT/'failure.txt').write_text(traceback.format_exc())
        raise
