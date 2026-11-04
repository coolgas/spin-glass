"""Finite L4/L6 diagnostic with L2-selected lambda and original-model controls."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from ising.hamiltonian import Hamiltonian
from .physics import make_af_sample
from .triangle_effective_benchmark import context, digest, write_json
from .triangle_effective_rg import hierarchy, joint_samples


SCHEDULES = ((4, 2), (6, 2), (6, 3))
PAIRS = 4096


def choose(calibration):
    data = json.loads(calibration.read_text())
    assert data['reference_lambda'] == 1
    selected = {}
    for gamma in (1., 2.):
        rows = [r for r in data['measurements'] if r['gamma'] == gamma and r['m'] == 6
                and r['target_tv'] < .05 and r['target_q2_error'] < .01]
        selected[gamma] = min(rows, key=lambda r: r['lambda_rg'])
    return selected


def run_case(root, L, gamma, lam, m, chi, sample_id=60000):
    key = f'l{L}_g{gamma:g}_l{lam:g}_m{m}_c{chi}'
    if sample_id != 60000:
        key = f's{sample_id}_'+key
    path = root/'cases'/(key+'.json')
    if path.exists():
        row = json.loads(path.read_text())
        assert digest(path.with_suffix('.npz')) == row['raw_sha256']
        return row
    start = time.monotonic()
    sample = make_af_sample(L, sample_id)
    original = Hamiltonian(3*L*L, sample.bonds, sample.J, h=.15)
    model = hierarchy(original, gamma, m, chi, lam=lam)
    owner = model['first']['owner']
    mask = np.array([owner[int(i)][0] != owner[int(j)][0] for i,j in sample.bonds])
    physical = Hamiltonian(original.n_sites, sample.bonds,
                           sample.J*np.where(mask, lam, 1.), h=.15)
    assert len(model['sites']) == L*L//4
    assert sorted(i for block in model['sites'] for i in block) == list(range(3*L*L))
    for D in model['decoders']:
        np.testing.assert_allclose(D.T@D, np.eye(chi), atol=1e-11)
    built = time.monotonic()
    solutions = [model['operator'].solve(seed) for seed in (0, 1)]
    energy, state, gap, residual = min(solutions, key=lambda s:s[0])
    start_energy_difference = abs(solutions[0][0]-solutions[1][0])
    fidelity = float(abs(solutions[0][1]@solutions[1][1])**2)
    start_alarm = start_energy_difference > 1e-8 or fidelity < 1-1e-6
    solved = time.monotonic()
    magnetization, zz, metrics = context(physical, model, state, sample)
    # Independent physical X expectation, combined with saved Z and ZZ products.
    xmag = np.zeros(physical.n_sites)
    chi_state = state.reshape((chi,)*len(model['sites']))
    digits = np.arange(4096)
    for b, (D, sites) in enumerate(zip(model['decoders'], model['sites'])):
        moved = np.moveaxis(chi_state, b, 0).reshape(chi, -1)
        rho = moved@moved.T
        for a, site in enumerate(sites):
            t, local = divmod(a, 3)
            flipped = digits ^ (1 << (3*(3-t)+local))
            xmag[site] = np.sum(rho*(D.T@D[flipped]).T)
    field_energy = -.15*magnetization.sum()-gamma*xmag.sum()
    physical_energy = -sum(j*zz[i,k] for (i,k),j in zip(physical.bonds, physical.J))+field_energy
    assert abs(physical_energy-energy) < 1e-8
    target_energy = -sum(j*zz[i,k] for (i,k),j in zip(original.bonds, original.J))+field_energy
    seeds = [int(np.random.SeedSequence([20261104, 108, L, int(gamma*100),
              int(lam*1000), m, chi, b, sample_id]).generate_state(1)[0]) for b in range(2)]
    spins = np.concatenate([joint_samples(model, state, PAIRS, seed) for seed in seeds])
    equal = np.sum(spins[::2] == spins[1::2], axis=1)
    counts = np.bincount(equal, minlength=physical.n_sites+1)
    mass = counts/PAIRS
    q_samples = 2*equal/physical.n_sites-1
    sampling_checks = []
    for power, expected in ((1, metrics['q']), (2, metrics['q2'])):
        values = q_samples**power
        se = float(values.std(ddof=1)/np.sqrt(PAIRS))
        sampling_checks.append(dict(power=power, measured=float(values.mean()),
            expected=expected, se=se, alarm=bool(abs(values.mean()-expected) > 6*se+1e-10)))
    batches = np.array([np.bincount(e, minlength=physical.n_sites+1)/(PAIRS//2)
                        for e in np.split(equal, 2)])
    cutoffs = [float(s[chi]-s[chi-1]) for s in model['super_spectra']]
    arrays = dict(bonds=sample.bonds, original_J=sample.J, J=physical.J, mask=mask,
        points=sample.points, disorder_entropy=sample.seed_entropy, state=state,
        state_start_0=solutions[0][1], state_start_1=solutions[1][1],
        start_energies=np.array([s[0] for s in solutions]),
        start_residuals=np.array([s[3] for s in solutions]),
        spins=spins, counts=counts, batches=batches, magnetization=magnetization,
        zz=zz, xmag=xmag, sites=np.array(model['sites']),
        triangle_sites=np.array(model['first']['sites']),
        triangle_bases=np.array(model['first']['bases']),
        triangle_spectra=np.array(model['first']['spectra']))
    for b in range(len(model['sites'])):
        arrays[f'W_{b}'] = model['retained'][b]
        arrays[f'decoder_{b}'] = model['decoders'][b]
        arrays[f'spectrum_{b}'] = model['super_spectra'][b]
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    row = dict(key=key, L=L, n=physical.n_sites, gamma=gamma, lambda_rg=lam,
        m=m, chi=chi, dimension=model['operator'].dimension, sample_id=sample_id,
        h=.15, orientation=0, origin=[0,0], pairs=PAIRS, seeds=seeds,
        energy=energy, energy_per_spin=energy/physical.n_sites,
        target_energy_expectation_per_spin=float(target_energy/physical.n_sites),
        physical_energy_check_error=float(abs(physical_energy-energy)),
        residual=residual, gap=gap, solver_seeds=[0,1], start_fidelity=fidelity,
        start_energy_difference=start_energy_difference, start_alarm=start_alarm,
        metrics=metrics, mass=mass.tolist(), cutoff_gaps=cutoffs,
        unresolved_cutoff=min(cutoffs)<1e-9, sampling_checks=sampling_checks,
        raw_sha256=digest(path.with_suffix('.npz')), seconds=time.monotonic()-start,
        build_seconds=built-start, solve_seconds=solved-built,
        physical_accuracy_qualified=False)
    write_json(path, row)
    print(json.dumps({k:row[k] for k in ('key','seconds','metrics','start_alarm')}), flush=True)
    return row


def report(root, selected):
    rows = [json.loads(p.read_text()) for p in (root/'cases').glob('*.json')]
    lookup = {(r['L'],r['gamma'],r['lambda_rg'],r['m'],r['chi']):r for r in rows}
    cells = [nbformat.v4.new_markdown_cell(r'''# Adaptive-lambda pilot: L4 and L6

This is a first L6 diagnostic, with 108 physical spins → 36 triangle sites →
9 superblocks. L4 has 48 → 16 → 4. All bonds start from K~Uniform(0.5,1.5),
one disorder60000 draw per size; internal bonds remain fixed and only connecting
bonds receive lambda. The target is always the original lambda=1 model, at h=0.15.
Different sizes have different disorder draws.

Choose the smallest sampled lambda whose **L2 m6** distribution TV versus the
original model is <0.05 and whose q² error is <0.01. This gives lambda=0.85 at
Gamma=1 and lambda=0.3 at Gamma=2. These are observable-specific choices from
one L2 disorder; energy errors remain and L6 target accuracy is unverified.

Each choice is run beside a lambda=1 control, at (m,chi)=(4,2),(6,2),(6,3).
L6 solved dimensions are 2^9=512 and 3^9=19683. Such small chi is deliberately
diagnostic: retention stability has not been established. Lanczos uses two
starts; physical energies independently check the composed projection.
Each P(q) uses 4096 independent physical replica pairs. Bars are ±1 sampling
SE and exclude truncation and disorder uncertainty. Exact physical q and q²
moments are also evaluated directly in each retained state.

Adaptive lambda changes with Gamma, so its curves form a path through different
Hamiltonians. Compare with the lambda=1 controls at each field; do not interpret
the adaptive path alone as the original model's field response or a phase boundary.
'''+f'\nSaved models: **{len(rows)}/24**. Figures use completed cases only.\n')]
    for kind in ('controls','retention'):
        fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
        for i,L in enumerate((4,6)):
            for j,gamma in enumerate((1.,2.)):
                ax=axes[i,j];lam=selected[gamma]['lambda_rg']
                settings = [(lam,6,3,'#0072B2',f'adaptive lambda={lam:g}'),
                            (1.,6,3,'#D55E00','original lambda=1')]
                if kind=='retention':
                    settings=[(lam,m,c,color,f'm={m}, chi={c}') for (m,c),color in
                              zip(SCHEDULES,('#0072B2','#D55E00','#009E73'))]
                for l,m,c,color,label in settings:
                    r=lookup.get((L,gamma,l,m,c))
                    if r is None: continue
                    p=np.array(r['mass']);q=np.linspace(-1,1,r['n']+1)
                    ax.errorbar(q,p,yerr=np.sqrt(p*(1-p)/r['pairs']),color=color,
                        marker='.',ms=3,lw=1,capsize=1,label=label)
                ax.set(title=f'L={L}, Gamma={gamma:g}'+(f', lambda={lam:g}' if kind=='retention' else ''),
                       xlabel='Physical overlap q',ylabel='Probability mass per bin',xlim=(-1,1),ylim=(0,None))
                if ax.lines: ax.legend(fontsize=9)
                ax.grid(alpha=.2)
        name=f'l6_adaptive_{kind}.png';fig.savefig(root/name,dpi=150)
        fig.savefig(root/name.replace('.png','.pdf'));plt.close(fig)
        title='Adaptive choices versus original-model RG controls' if kind=='controls' else 'Retention sensitivity of the adaptive choices'
        cells.append(nbformat.v4.new_markdown_cell('## '+title+f'\n\n![{title}](attachment:{name})',
            attachments={name:{'image/png':base64.b64encode((root/name).read_bytes()).decode()}}))
    table=['## Direct physical moments and diagnostics','','| L | Gamma | lambda | m | chi | mean q | q² | target energy expectation/spin | alarms |',
           '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |']
    for r in sorted(rows,key=lambda r:(r['L'],r['gamma'],r['lambda_rg'],r['m'],r['chi'])):
        alarm=r['start_alarm'] or r['unresolved_cutoff'] or any(t['alarm'] for t in r['sampling_checks'])
        table.append(f"| {r['L']} | {r['gamma']:g} | {r['lambda_rg']:g} | {r['m']} | {r['chi']} | {r['metrics']['q']:.5f} | {r['metrics']['q2']:.5f} | {r['target_energy_expectation_per_spin']:.6f} | {'yes' if alarm else 'no'} |")
    table+=['','The target energy column evaluates each reconstructed state under the original '
        'lambda=1 Hamiltonian. It is an expectation, not an exact L4/L6 energy error. '
        'Both control and adaptive curves are RG approximations. Histogram differences '
        'include sampling noise, with different bin spacings at L4 and L6.', '',
        'A size trend here can contain disorder and truncation effects. Increase chi, '
        'check more disorder samples and test lambda transfer before using these results '
        'for q_EA or correlation-length scaling. Raw states, isometries, spectra, physical '
        'samples, seeds, failures and source hashes accompany the report.']
    cells.append(nbformat.v4.new_markdown_cell('\n'.join(table)))
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(execution_location='AutoDL',
        preliminary=True,kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbformat.validate(nb);nbformat.write(nb,root/'triangle_effective_l6_adaptive.ipynb')
    write_json(root/'report_provenance.json',dict(files={p.name:digest(p) for p in root.iterdir()
        if p.is_file() and p.suffix in ('.png','.pdf','.ipynb')},cases=len(rows)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    args=parser.parse_args();root=args.output
    for name in ('cases','failures'): (root/name).mkdir(parents=True,exist_ok=True)
    selected=choose(args.calibration)
    shutil.copy2(args.calibration,root/'l2_calibration.json')
    manifest=dict(selection_rule='Minimum sampled lambda with L2 m6 target TV<.05 and q2 error<.01',
        selected={str(g):r for g,r in selected.items()},sizes=[4,6],schedules=SCHEDULES,
        pairs=PAIRS,worker_threads=1,total=24,
        calibration_sha256=digest(args.calibration),
        sources={str(p.relative_to(Path(__file__).parents[1])):digest(p)
                 for p in Path(__file__).parents[1].rglob('*.py')})
    if (root/'launch.json').exists():
        assert json.loads((root/'launch.json').read_text())==json.loads(json.dumps(manifest))
    else: write_json(root/'launch.json',manifest)
    report(root,selected)
    completed=0;failures=[]
    for L in (4,6):
        for gamma in (1.,2.):
            for lam in (selected[gamma]['lambda_rg'],1.):
                for m,chi in SCHEDULES:
                    key=f'l{L}_g{gamma:g}_l{lam:g}_m{m}_c{chi}'
                    write_json(root/'status.json',dict(state='running',completed=completed,total=24,
                        current=key,failures=failures))
                    try:
                        run_case(root,L,gamma,lam,m,chi);completed+=1
                    except Exception:
                        (root/'failures'/(key+'.txt')).write_text(traceback.format_exc())
                        failures.append(key);traceback.print_exc()
                    report(root,selected)
    report(root,selected)
    write_json(root/'status.json',dict(state='complete' if not failures else 'complete_with_failures',
        completed=completed,total=24,failures=failures))


if __name__=='__main__': main()
