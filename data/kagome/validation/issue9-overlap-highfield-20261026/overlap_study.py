"""Pure overlap forward-time comparison, using the frozen flat-guide solver."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'issue9-energy2048-20261026'/'source'
sys.path.insert(0, str(SOURCE/'code'))
from kagome_qea import pqmc
from kagome_qea.sample import make_sample

SETTINGS = dict(L=6, sample_id=0, sigma=1.0, gamma=5.0, h=0.0,
                n_walkers=10000, n_replicas=2, tau_block=0.2,
                n_equilibrate=32, n_measure=32, n_forward_cycles=8,
                forward_times=[0.1, 0.2, 0.4, 0.8], populations=8,
                seed=2026102603, spawned_children=32, workers=16)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def project(duration, child, walkers=10000, blocks=32, cycles=8):
    ham = make_sample(6, True, 0).hamiltonian(0.0)
    rng = np.random.default_rng(np.random.SeedSequence(SETTINGS['seed']).spawn(32)[child])
    return pqmc._project_once(ham, 5.0, rng, n_walkers=walkers, n_replicas=2,
                              tau_block=0.2, n_equilibrate=blocks, n_measure=blocks,
                              tau_forward=duration, n_forward_cycles=cycles)


def smoke():
    start = time.monotonic()
    a = project(.2, 0, walkers=128, blocks=2, cycles=2)
    b = project(.2, 0, walkers=128, blocks=2, cycles=2)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_equal(a[key], b[key], err_msg=key)
    np.testing.assert_allclose(a['chi_sg'], 108*a['q2'])
    row = dict(exact_repeatability=True, result_fields=len(a),
               benchmark_seconds=(time.monotonic()-start)/2,
               benchmark_walkers=128, benchmark_initial_blocks=4,
               benchmark_cycles=2, benchmark_forward_time=.2,
               projection_complete=bool(a['projection_complete']))
    write_json(ROOT/'smoke.json', row)
    print('SMOKE', json.dumps(row), flush=True)


def run_population(index, duration, child):
    path = ROOT/'runs'/f't{index}_p{child}.npz'
    assert not path.exists()
    start = time.monotonic()
    result = project(duration, child)
    assert len(result['q2_series']) == 8
    np.testing.assert_allclose(result['chi_sg'], 108*result['q2'])
    np.testing.assert_allclose(result['energy_measure_time'], np.arange(33, 65)*.2)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **result)
    temporary.replace(path)
    row = dict(child=child, duration=duration, index=index,
               seconds=time.monotonic()-start,
               finished=datetime.now(timezone.utc).isoformat())
    write_json(path.with_suffix('.json'), row)
    return row


def estimate(values):
    values = np.asarray(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    half = float(t.ppf(.975, len(values)-1)*se)
    return dict(mean=mean, se=se, ci95=[mean-half, mean+half], values=values.tolist())


def analyze():
    summaries, contrasts, chi_values = {}, {}, []
    for i, duration in enumerate(SETTINGS['forward_times']):
        parts = []
        for child in range(8*i, 8*(i+1)):
            with np.load(ROOT/'runs'/f't{i}_p{child}.npz', allow_pickle=False) as archive:
                parts.append({key: archive[key] for key in archive.files})
        q2 = np.array([p['q2_series'].mean() for p in parts])
        q4 = np.array([p['q4_series'].mean() for p in parts])
        chi = 108*q2
        chi_values.append(chi)
        second = estimate(q2)
        row = dict(q2=second, q4=estimate(q4), chi_sg=estimate(chi),
                   q_ea_overlap_rms=float(np.sqrt(q2.mean())),
                   q_ea_overlap_rms_ci95=np.sqrt(np.maximum(0, second['ci95'])).tolist(),
                   all_projection_complete=all(bool(p['projection_complete']) for p in parts),
                   truncated_cycles=sum(int(p['n_forward_truncated']) for p in parts),
                   achieved_forward_times=[p['forward_times'].tolist() for p in parts],
                   min_unique_ancestors=min(int(p['min_unique_ancestors']) for p in parts),
                   min_effective_ancestors=min(float(p['min_effective_ancestors']) for p in parts),
                   max_descendant_fraction=max(float(p['cycle_max_fraction'].max()) for p in parts))
        summaries[str(duration)] = row
    for i in range(1, len(chi_values)):
        a, b = chi_values[i-1], chi_values[i]
        va, vb = a.var(ddof=1)/8, b.var(ddof=1)/8
        df = (va+vb)**2/(va*va/7+vb*vb/7)
        se = np.sqrt(va+vb)
        mean = b.mean()-a.mean()
        half = t.ppf(.975, df)*se
        key = f"{SETTINGS['forward_times'][i]}_minus_{SETTINGS['forward_times'][i-1]}"
        contrasts[key] = dict(mean=float(mean), se=float(se), df=float(df),
                              ci95=[float(mean-half), float(mean+half)])
    output = dict(settings=SETTINGS, summaries=summaries, chi_forward_contrasts=contrasts,
                  q_ea_definition='sqrt(disorder mean of pure q2); here single-sample RMS overlap',
                  disorder_samples=1, phase_established=False,
                  finished=datetime.now(timezone.utc).isoformat())
    write_json(ROOT/'final.json', output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = SETTINGS['forward_times']
    y = np.array([summaries[str(v)]['chi_sg']['mean'] for v in x])
    e = np.array([summaries[str(v)]['chi_sg']['se'] for v in x])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.errorbar(x, y, yerr=e, marker='o', capsize=4, label='Mean ± SE across populations')
    ax.axhline(1, color='gray', ls='--', label='Uncoupled limit')
    ax.set(xlabel='Requested forward time', ylabel='Spin-glass susceptibility',
           title='L6 sample0, Gamma/sigma=5, flat guide, 10,000 walkers')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(ROOT/'forward_comparison.png', dpi=150)
    plt.close(fig)
    print('FINAL', json.dumps(output), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    campaign = json.loads((ROOT/'campaign.json').read_text())
    assert campaign['settings'] == SETTINGS
    for name, digest in campaign['source_sha256'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest, name
    for name, digest in campaign['study_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    (ROOT/'runs').mkdir(exist_ok=True)
    if args.analyze_only:
        analyze()
        return
    smoke()
    if args.smoke_only:
        return
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(run_population, i, duration, child)
                   for i, duration in enumerate(SETTINGS['forward_times'])
                   for child in range(8*i, 8*(i+1))]
        for future in as_completed(futures):
            print('DONE', json.dumps(future.result()), flush=True)
    analyze()
    write_json(ROOT/'exit.json', dict(exit_code=0, elapsed_seconds=time.monotonic()-start,
                                    finished=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write_json(ROOT/'exit.json', dict(exit_code=1, finished=datetime.now(timezone.utc).isoformat()))
        (ROOT/'failure.txt').write_text(traceback.format_exc())
        raise
