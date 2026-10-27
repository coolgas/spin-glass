"""Bounded Gaussian-bond size/field screen using the frozen flat-guide PQMC."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'issue9-energy2048-20261026'/'source'
sys.path.insert(0, str(SOURCE/'code'))
from kagome_qea import pqmc
from kagome_qea.sample import make_sample

SEED = 2026102604
BASE = dict(n_walkers=10000, n_replicas=2, tau_block=.2,
            n_equilibrate=32, n_measure=32, tau_forward=.8,
            n_forward_cycles=8, min_effective_ancestors=8,
            min_unique_ancestors=2)
VARIANTS = dict(base={}, projection=dict(n_equilibrate=64, n_measure=64),
                forward=dict(tau_forward=1.6), walkers=dict(n_walkers=20000))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def tasks():
    rows = []
    for L in (4, 6, 8):
        for gamma in (.5, 1., 2., 3., 5.):
            for sample_id in (0, 1):
                variants = VARIANTS if sample_id == 0 and gamma in (1., 3.) else {'base': {}}
                for variant in variants:
                    for population in range(4):
                        identity = len(rows)
                        rows.append(dict(id=identity, L=L, gamma=gamma,
                            sample_id=sample_id, variant=variant, population=population,
                            seed_entropy=[SEED, identity],
                            settings={**BASE, **VARIANTS[variant]},
                            archive=f'runs/p{identity:03d}.npz'))
    return rows


def verify_protocol():
    protocol = json.loads((ROOT/'campaign.json').read_text())
    assert protocol['tasks'] == tasks()
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest, name
    for name, digest in protocol['study_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    return protocol


def project(task, smoke=False):
    settings = dict(task['settings'])
    if smoke:
        settings.update(n_walkers=64, n_equilibrate=2, n_measure=2,
                        n_forward_cycles=2, tau_forward=.05)
    sample = make_sample(task['L'], True, task['sample_id'])
    rng = np.random.default_rng(np.random.SeedSequence(task['seed_entropy']))
    result = pqmc._project_once(sample.hamiltonian(0.), task['gamma'], rng, **settings)
    # Store the actual physical disorder, not just a recipe for recreating it.
    result.update(sample_L=sample.L, sample_id=sample.sample_id,
                  sample_periodic=sample.periodic, sample_seed_entropy=sample.seed_entropy,
                  sample_bonds=sample.bonds, sample_J=sample.J,
                  gamma=task['gamma'], sigma=1., h=0.,
                  solver_seed_entropy=np.asarray(task['seed_entropy']),
                  task_id=task['id'], variant=task['variant'])
    return result


def smoke():
    start = time.monotonic()
    for task in (tasks()[0], tasks()[-1]):
        a, b = project(task, True), project(task, True)
        for key in a:
            np.testing.assert_equal(a[key], b[key], err_msg=key)
        np.testing.assert_allclose(a['chi_sg'], 3*task['L']**2*a['q2'])
    row = dict(exact_repeatability=True, sizes_checked=[4, 8],
               seconds=time.monotonic()-start, tasks=len(tasks()),
               full_settings_validated_by_smoke=False)
    write_json(ROOT/'smoke.json', row)
    print('SMOKE', json.dumps(row), flush=True)


def run_population(task):
    path = ROOT/task['archive']
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    start = time.monotonic()
    result = project(task)
    assert len(result['q2_series']) == task['settings']['n_forward_cycles']
    np.testing.assert_allclose(result['chi_sg'], 3*task['L']**2*result['q2'])
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **result)
    temporary.replace(path)
    row = dict(task=task, seconds=time.monotonic()-start,
               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               projection_complete=bool(result['projection_complete']),
               truncated_cycles=int(result['n_forward_truncated']),
               finished=datetime.now(timezone.utc).isoformat())
    write_json(path.with_suffix('.json'), row)
    return row


def estimate(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    half = float(t.ppf(.975, len(values)-1)*se)
    return dict(mean=mean, se=se, ci95=[mean-half, mean+half], values=values.tolist())


def contrast(a, b):
    """Independent-population Welch interval for b minus a."""
    a, b = np.asarray(a), np.asarray(b)
    variances = np.array([a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)])
    se = float(np.sqrt(variances.sum()))
    mean = float(b.mean()-a.mean())
    if se == 0:
        return dict(mean=mean, se=0., df=None, ci95=[mean, mean])
    df = float(variances.sum()**2/np.sum(variances**2/[len(a)-1, len(b)-1]))
    half = float(t.ppf(.975, df)*se)
    return dict(mean=mean, se=se, df=df, ci95=[mean-half, mean+half])


def group_key(task):
    return f"L{task['L']}_g{task['gamma']:g}_s{task['sample_id']}_{task['variant']}"


def analyze():
    protocol = verify_protocol()
    grouped = {}
    for task in tasks():
        with np.load(ROOT/task['archive'], allow_pickle=False) as archive:
            part = {key: archive[key] for key in archive.files}
        grouped.setdefault(group_key(task), []).append((task, part))
    groups = {}
    for key, parts in grouped.items():
        task = parts[0][0]
        n_sites = 3*task['L']**2
        matrix = np.array([[p[name+'_series'].mean() for name in ('q', 'q2', 'q3', 'q4')]
                           for _, p in parts])
        energies = [p['energy_measure_trace'].mean()/n_sites for _, p in parts]
        drifts = [np.diff(p['energy_measure_trace'].reshape(2, -1).mean(axis=1))[0]/n_sites
                  for _, p in parts]
        means = matrix.mean(axis=0)
        groups[key] = dict(L=task['L'], gamma=task['gamma'], sample_id=task['sample_id'],
            variant=task['variant'], archives=[a['archive'] for a, _ in parts],
            population_moments=matrix.tolist(),
            q2=estimate(matrix[:, 1]), q4=estimate(matrix[:, 3]),
            moment_mean_covariance=(np.cov(matrix, rowvar=False)/len(parts)).tolist(),
            chi_sg=estimate(n_sites*matrix[:, 1]), q_ea=float(np.sqrt(means[1])),
            binder=float(.5*(3-means[3]/means[1]**2)),
            energy_per_spin=estimate(energies), energy_late_minus_early_per_spin=estimate(drifts),
            all_projection_complete=all(bool(p['projection_complete']) for _, p in parts),
            truncated_cycles=sum(int(p['n_forward_truncated']) for _, p in parts),
            achieved_forward_times=[p['forward_times'].tolist() for _, p in parts],
            min_effective_ancestors=min(float(p['min_effective_ancestors']) for _, p in parts),
            min_unique_ancestors=min(int(p['min_unique_ancestors']) for _, p in parts),
            max_descendant_fraction=max(float(p['cycle_max_fraction'].max()) for _, p in parts))
    comparisons = {}
    for key, row in groups.items():
        if row['variant'] == 'base':
            continue
        base = groups[key.rsplit('_', 1)[0]+'_base']
        comparisons[key] = {name: contrast(base[name]['values'], row[name]['values'])
                            for name in ('chi_sg', 'q2', 'q4', 'energy_per_spin')}
    curves = []
    bootstrap = {}
    for L in (4, 6, 8):
        # The same disorder resampling is used at every field for this size.
        ids = np.random.default_rng([SEED, L, 999]).integers(0, 2, size=(10000, 2))
        bootstrap[str(L)] = dict(seed_entropy=[SEED, L, 999], draws=10000,
            independent_disorder_samples=2, interval_reliability='exploratory; only two samples')
        for gamma in (.5, 1., 2., 3., 5.):
            rows = [groups[f'L{L}_g{gamma:g}_s{s}_base'] for s in (0, 1)]
            second = np.array([r['q2']['mean'] for r in rows])
            fourth = np.array([r['q4']['mean'] for r in rows])
            center = float(second.mean())
            projector_var = float(sum(r['q2']['se']**2 for r in rows)/4)
            scatter_se = float(second.std(ddof=1)/np.sqrt(2))
            bs2, bs4 = second[ids].mean(axis=1), fourth[ids].mean(axis=1)
            interval = lambda values: np.quantile(values, [.025, .975]).tolist()
            curves.append(dict(L=L, n_sites=3*L*L, gamma_over_sigma=gamma,
                disorder_samples=2, q_ea_squared=center, q_ea=float(np.sqrt(center)),
                chi_sg=3*L*L*center, q4=float(fourth.mean()),
                binder_ratio_of_disorder_moments=float(.5*(3-fourth.mean()/center**2)),
                sample_binders=[r['binder'] for r in rows],
                q2_projector_se=float(np.sqrt(projector_var)),
                q2_sample_scatter_se=scatter_se,
                q2_disorder_se_estimate=float(np.sqrt(max(0., second.var(ddof=1)
                    -np.mean([r['q2']['se']**2 for r in rows]))/2)),
                chi_sg_disorder_bootstrap_ci95=interval(3*L*L*bs2),
                q_ea_disorder_bootstrap_ci95=interval(np.sqrt(bs2)),
                binder_disorder_bootstrap_ci95=interval(.5*(3-bs4/bs2**2)),
                all_projection_complete=all(r['all_projection_complete'] for r in rows)))
    output = dict(campaign=protocol['name'], groups=groups, convergence_contrasts=comparisons,
        curves=curves, disorder_bootstrap=bootstrap, phase_established=False,
        convergence_qualified=False, disorder_samples_per_size=2,
        uncertainty='Population means give projector SE; disorder bootstrap resamples whole samples across fields. Systematic bias is separate.',
        binder_definition='0.5*(3 - [<q^4>]/[<q^2>]^2); sample ratios retained separately',
        variance_scaling=dict(sigmas=[.05, .1, .2, .5, 1., 2.],
            rule='Gamma = sigma * gamma_over_sigma; dimensionless observables unchanged',
            independent_simulations=False),
        finished=datetime.now(timezone.utc).isoformat())
    write_json(ROOT/'final.json', output)
    plot(output)
    print('ANALYZED', len(groups), 'groups', flush=True)


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    labels = {
        'chi_sg': r'$\chi_{\mathrm{SG}}$',
        'q_ea': r'$q_{\mathrm{EA}}$',
        'binder_ratio_of_disorder_moments': r'$U_4$',
    }
    for L in (4, 6, 8):
        rows = [r for r in output['curves'] if r['L'] == L]
        x = [r['gamma_over_sigma'] for r in rows]
        for ax, name in zip(axes, ('chi_sg', 'q_ea', 'binder_ratio_of_disorder_moments')):
            ax.plot(x, [r[name] for r in rows], 'o-', label=rf'$L={L}$')
            ax.set(xlabel=r'$\Gamma/\sigma$', ylabel=labels[name])
            ax.grid(alpha=.2)
        # Show disorder scatter for chi and qEA; only two samples in this screen.
        for ax, name, factor in ((axes[0], 'chi_sg', 3*L*L), (axes[1], 'q_ea', None)):
            errors = [r['q2_sample_scatter_se']*(factor if factor else .5/r['q_ea']) for r in rows]
            ax.errorbar(x, [r[name] for r in rows], yerr=errors, fmt='none', capsize=3)
    axes[0].axhline(1, color='gray', ls='--')
    axes[0].legend()
    fig.suptitle(r'Gaussian bonds: exploratory screen ($n_{\mathrm{dis}}=2$; convergence unqualified)')
    fig.savefig(ROOT/'size_field_screen.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    rows = [r for r in output['curves'] if r['L'] == 6]
    for sigma in output['variance_scaling']['sigmas']:
        ax.plot([sigma*r['gamma_over_sigma'] for r in rows], [r['chi_sg'] for r in rows],
                'o-', label=rf'$\mathrm{{Var}}(J)=\sigma^2={sigma**2:g}$')
    ax.set(xlabel=r'Transverse field, $\Gamma$',
           ylabel=r'Spin-glass susceptibility, $\chi_{\mathrm{SG}}$',
           title=r'$L=6$: exact Gaussian-bond variance rescaling')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(ROOT/'variance_rescaling.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    protocol = verify_protocol()
    (ROOT/'runs').mkdir(exist_ok=True)
    if args.analyze_only:
        analyze()
        return
    smoke()
    if args.smoke_only:
        return
    start = time.monotonic()
    # A new campaign never silently resumes partially written populations.
    assert not list((ROOT/'runs').iterdir()), 'runs must be empty before launch'
    with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
        futures = [pool.submit(run_population, task) for task in tasks()]
        for count, future in enumerate(as_completed(futures), 1):
            row = future.result()
            print('DONE', count, '/', len(futures), json.dumps(row), flush=True)
            write_json(ROOT/'progress.json', dict(completed=count, total=len(futures), last=row))
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
