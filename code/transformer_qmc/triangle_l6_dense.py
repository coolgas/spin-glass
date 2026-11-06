"""Finite, restartable L6 m6/chi3 scan; raw artifacts remain remote."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

GAMMAS = tuple(i / 1000 for i in range(100, 1501, 25))
IDS = tuple(range(60000, 60008))
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(gamma, sid):
    return f'g{round(gamma * 1000):04d}_s{sid}'


def qualify(path, output):
    import numpy as np
    from .triangle_effective_rg import physical_correlations
    from .triangle_correlation_observables import overlap_observables
    from .triangle_overlap_report import difference, within
    row = json.loads(path.read_text())
    assert (row['L'], row['m'], row['chi'], row['h'], row['lambda_rg']) == (6, 6, 3, .15, 1.)
    assert row['sample_id'] in IDS and row['gamma'] in GAMMAS
    raw = path.with_suffix('.npz')
    assert digest(raw) == row['raw_sha256']
    with np.load(raw, allow_pickle=False) as a:
        assert list(a['disorder_entropy']) == [20261027, 6, row['sample_id'], 1]
        assert np.all((-a['J'] >= .5) & (-a['J'] <= 1.5))
        np.testing.assert_array_equal(a['J'], a['original_J'])
        values = overlap_observables(a['magnetization'], a['zz'])
        np.testing.assert_allclose([values['q_ea_finite'], values['q2_raw']],
                                   [row['metrics']['q'], row['metrics']['q2']], atol=1e-10, rtol=1e-10)
        digits = np.array(np.unravel_index(np.arange(4096), (8,) * 4)).T
        spins = (1 - 2 * ((digits[:, :, None] >> np.arange(3)) & 1)).reshape(4096, 12)
        model = dict(decoders=[a[f'decoder_{b}'] for b in range(9)], sites=a['sites'], spins=spins)
        starts = [overlap_observables(*physical_correlations(model, a[f'state_start_{seed}'])) for seed in (0, 1)]
        agreement = difference(*starts)
        selected = int(np.argmin(a['start_energies']))
        for name, value in values.items():
            np.testing.assert_allclose(value, starts[selected][name], atol=1e-9, rtol=1e-9)
        residuals = a['start_residuals'].tolist()
        bond_hash = hashlib.sha256(a['bonds'].tobytes() + a['J'].tobytes()).hexdigest()
    checks = dict(residuals=max(residuals) <= 1e-7,
                  start_energy=row['start_energy_difference'] <= 1e-8,
                  start_fidelity=row['start_fidelity'] >= 1 - 1e-6,
                  start_alarm=not row['start_alarm'],
                  physical_energy=row['physical_energy_check_error'] < 1e-8,
                  cutoff=not row['unresolved_cutoff'],
                  sampling=not any(c['alarm'] for c in row['sampling_checks']),
                  observables=bool(within(agreement, .001, .01, .01)))
    result = dict(L=6, m=6, chi=3, gamma=row['gamma'], sample_id=row['sample_id'],
                  qualified=all(checks.values()), checks=checks, observables=values,
                  start_observables=starts, start_difference=agreement, residuals=residuals,
                  result_path=str(path), result_sha256=digest(path), raw_sha256=row['raw_sha256'],
                  bonds_sha256=bond_hash, seconds=row['seconds'],
                  raw_bytes=raw.stat().st_size, physical_accuracy_qualified=False)
    write(output, result)
    return result


def worker(root, gamma, sid):
    from .triangle_effective_l6_adaptive import run_case
    from .triangle_effective_l6_uniform import SAMPLES
    assert tuple(SAMPLES) == IDS
    assert gamma in GAMMAS and sid in IDS
    for name in ('cases', 'audits', 'failures'):
        (root / name).mkdir(parents=True, exist_ok=True)
    tag = key(gamma, sid)
    try:
        row = run_case(root, 6, gamma, 1., 6, 3, sid)
        result = qualify(root / 'cases' / (row['key'] + '.json'), root / 'audits' / (tag + '.json'))
        result['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['threads'] = int(os.environ.get('OPENBLAS_NUM_THREADS', '1'))
        write(root / 'audits' / (tag + '.json'), result)
        print(json.dumps(dict(case=tag, qualified=result['qualified'], seconds=result['seconds'], peak_rss_kib=result['peak_rss_kib'])), flush=True)
    except Exception:
        write(root / 'failures' / (tag + '.json'), dict(gamma=gamma, sample_id=sid, traceback=traceback.format_exc()))
        raise


def controller(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('cases', 'audits', 'logs', 'failures'):
        (root / name).mkdir(exist_ok=True)
    # Refuse to duplicate orphaned workers after a controller interruption.
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            cmd = p.read_bytes().decode().split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if str(root) in cmd and '--case' in cmd and 'transformer_qmc.triangle_l6_dense' in cmd:
            raise RuntimeError(f'Existing worker {p.parent.name}; preserve it and restart only after it exits')
    import numpy as np
    previous = json.loads(args.previous_cases.read_text())
    reused = []
    for r in previous:
        if r['kind'] != 'archived_exact_retained_space' or (r['L'], r['m'], r['chi']) != (6, 6, 3):
            continue
        if r['gamma'] not in GAMMAS or r['sample_id'] not in IDS:
            continue
        tag = key(r['gamma'], r['sample_id'])
        if not (root / 'audits' / (tag + '.json')).exists():
            qualify(Path(r['result_path']), root / 'audits' / (tag + '.json'))
        reused.append(tag)
    manifest = dict(L=6, m=6, chi=3, h=.15, lambda_rg=1., gamma=list(GAMMAS), disorders=list(IDS),
                    cases=456, reused=sorted(set(reused)), solver='original retained-space Lanczos, two starts 0/1',
                    eigen_tolerance=1e-11, eigen_maxiter=20000, residual_limit=1e-7,
                    observable_q_absolute=.001, observable_relative=.01,
                    physical_accuracy_qualified=False, worker_limit=args.workers, threads=1,
                    max_worker_memory_gib=48, min_available_memory_gib=16, min_disk_free_gib=3,
                    previous_cases_sha256=digest(args.previous_cases),
                    sources={str(p.relative_to(Path(__file__).parents[1])):digest(p) for p in Path(__file__).parents[1].rglob('*.py')})
    write(root / 'manifest.json', manifest)
    running = {}
    def update(state):
        audits = [json.loads(p.read_text()) for p in (root / 'audits').glob('*.json')]
        signatures = {}
        for row in audits:
            old = signatures.setdefault(row['sample_id'], row['bonds_sha256'])
            assert old == row['bonds_sha256'], 'Disorder identity changed across fields'
        write(root / 'status.json', dict(utc=time.time(), state=state, total=456, completed=len(audits),
              qualified=sum(r['qualified'] for r in audits), excluded=sum(not r['qualified'] for r in audits),
              failed=[p.stem for p in (root / 'failures').glob('*.json')], reused=len(set(reused)),
              running={k:dict(pid=v[0].pid, gamma=v[2], sample_id=v[3], threads=1) for k,v in running.items()}))
    update('running')
    # Prioritize eight-disorder high-field coverage, then fill the full grid.
    priority = [1.25, 1.5, 1.125, 1.375, .125]
    fields = priority + [g for g in GAMMAS if g not in priority]
    queue = [(g,s) for g in fields for s in IDS
             if not (root/'audits'/(key(g,s)+'.json')).exists()
             and not (root/'failures'/(key(g,s)+'.json')).exists()]
    paused = None
    while queue or running:
        for tag, (proc, stream, g, sid) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                stream.close(); del running[tag]
                if rc != 0 and not (root/'failures'/(tag+'.json')).exists():
                    write(root/'failures'/(tag+'.json'), dict(gamma=g, sample_id=sid, returncode=rc))
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        cap = int(Path('/sys/fs/cgroup/memory.max').read_text())
        stat = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        available = cap - current + int(stat.get('inactive_file', 0))
        free = shutil.disk_usage(root).free
        paused = 'disk' if free < (3 + len(queue)*.003) * 2**30 else ('memory' if available < (16+2*len(running)) * 2**30 else None)
        while queue and len(running) < args.workers and not paused:
            g, sid = queue.pop(0); tag = key(g, sid)
            env = dict(os.environ, **{name:'1' for name in THREAD_VARS})
            log = (root/'logs'/(tag+'.log')).open('ab')
            proc = subprocess.Popen([sys.executable,'-u','-m','transformer_qmc.triangle_l6_dense',
                       '--output',str(root),'--case','--gamma',str(g),'--sample-id',str(sid)],
                       env=env, stdout=log, stderr=subprocess.STDOUT)
            running[tag] = (proc, log, g, sid)
        update('resource_deferred_'+paused if paused else 'running')
        if paused and not running:
            return
        if queue or running:
            time.sleep(2)
    failures = list((root/'failures').glob('*.json'))
    audits = [json.loads(p.read_text()) for p in (root/'audits').glob('*.json')]
    update('complete' if not failures and all(r['qualified'] for r in audits) else 'complete_with_exclusions')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='store_true')
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--sample-id', type=int)
    parser.add_argument('--previous-cases', type=Path)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.case:
        worker(args.output, args.gamma, args.sample_id)
    else:
        assert args.previous_cases and 1 <= args.workers <= 16
        controller(args)


if __name__ == '__main__':
    main()
