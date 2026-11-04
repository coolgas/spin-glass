"""Finite weak-intertriangle-coupling screen; scientific execution on AutoDL."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from scipy.sparse.linalg import eigsh

from ising.hamiltonian import Hamiltonian
from kagome_qea.exact import operator_matrix
from .physics import make_af_sample, basis, correlations
from .triangle_effective_rg import triangles, hierarchy, joint_samples
from .triangle_effective_benchmark import write_json, digest, lift, xor_mass, context

GAMMAS = (.25, .5, 1., 2.)
LAMBDAS = (0., .1, .3, 1.)
SCHEDULES = ((4, 8), (6, 8), (6, 16))
SID = 60000
H = .15


def physical_hamiltonian(L, lam):
    sample = make_af_sample(L, SID)
    original = Hamiltonian(3*L*L, sample.bonds, sample.J, h=H)
    first = triangles(original, 1., 8)
    owner = first['owner']
    mask = np.array([owner[int(i)][0] != owner[int(j)][0] for i, j in sample.bonds])
    J = sample.J*np.where(mask, lam, 1.)
    return sample, original, Hamiltonian(3*L*L, sample.bonds, J, h=H), mask


def exact_small(root, gamma, lam):
    key = f'l2_g{gamma:g}_l{lam:g}'
    path = root/'exact'/(key+'.json')
    if path.exists():
        return json.loads(path.read_text())
    sample, original, ham, mask = physical_hamiltonian(2, lam)
    matrix = operator_matrix(ham, gamma)
    energies, vectors = eigsh(matrix, k=2, which='SA', tol=1e-12,
                              v0=np.random.default_rng(710).normal(size=4096))
    order = np.argsort(energies)
    energy, state = energies[order[0]], vectors[:, order[0]]
    residual = np.linalg.norm(matrix@state-energy*state)
    assert residual < 1e-8
    mass = xor_mass(state**2, 12)
    exact_metrics = correlations(state**2, basis(12))
    arrays = dict(bonds=sample.bonds, original_J=sample.J, J=ham.J,
                  mask=mask, points=sample.points, exact_state=state, mass=mass)
    checks = []
    for m in (4, 6, 8):
        first = triangles(original, gamma, m, lam=lam)
        e, v, gap, res = first['operator'].solve()
        psi = lift(first, v, m)
        # Independent physical action tests the lambda weighting and tensor order.
        rng = np.random.default_rng(721+m)
        test = rng.normal(size=m**4); test /= np.linalg.norm(test)
        lifted = lift(first, test, m)
        action_error = abs(test@first['operator'].action(test)-lifted@(matrix@lifted))
        assert action_error < 1e-9
        pq = xor_mass(psi**2, 12)
        metrics = correlations(psi**2, basis(12))
        tv = float(abs(pq-mass).sum()/2)
        if m == 8:
            assert tv < 1e-7 and abs(e-energy) < 1e-8
        checks.append(dict(m=m, energy_error_per_site=float((e-energy)/12),
                           tv=tv, q2_error=float(abs(metrics['q2']-exact_metrics['q2'])),
                           residual=res, action_error=float(action_error)))
        arrays[f'mass_m{m}'] = pq
        arrays[f'bases_m{m}'] = np.asarray(first['bases'])
        arrays[f'state_m{m}'] = v
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    result = dict(key=key, gamma=gamma, lam=lam, residual=float(residual),
                  mass=mass.tolist(), checks=checks, raw_sha256=digest(path.with_suffix('.npz')))
    write_json(path, result)
    return result


def endpoint_mass(first):
    """Exact full-spin decoupled P(q), convolving independent triangle overlaps."""
    mass = np.ones(1)
    for U in first['bases']:
        mass = np.convolve(mass, xor_mass(U[:, 0]**2, 3))
    return mass


def run_case(root, gamma, lam, m, chi, pairs):
    key = f'l4_s{SID}_g{gamma:g}_l{lam:g}_m{m}_c{chi}'
    path = root/'cases'/(key+'.json')
    if path.exists():
        row = json.loads(path.read_text())
        assert digest(path.with_suffix('.npz')) == row['raw_sha256']
        return row
    start = time.monotonic()
    sample, original, physical, mask = physical_hamiltonian(4, lam)
    model = hierarchy(original, gamma, m, chi, lam=lam)
    built = time.monotonic()
    solutions = [model['operator'].solve(seed) for seed in (0, 1)]
    energy, state, gap, residual = min(solutions, key=lambda x: x[0])
    fidelity = float(abs(solutions[0][1]@solutions[1][1])**2)
    assert abs(solutions[0][0]-solutions[1][0]) < 1e-8 and fidelity > 1-1e-6
    magnetization, zz, metrics = context(physical, model, state, sample)
    # Different seeds per context, schedule and independent batch.
    seeds = [int(np.random.SeedSequence([20261104, 91, SID, GAMMAS.index(gamma),
             LAMBDAS.index(lam), m, chi, b]).generate_state(1)[0]) for b in range(2)]
    spins = np.concatenate([joint_samples(model, state, pairs, seed) for seed in seeds])
    q_samples = np.mean(spins[::2]*spins[1::2], axis=1)
    counts = np.bincount(np.sum(spins[::2] == spins[1::2], axis=1), minlength=49)
    mass = counts/counts.sum()
    batches = np.array([np.bincount(np.sum(s[::2] == s[1::2], axis=1), minlength=49)/(pairs//2)
                        for s in np.split(spins, 2)])
    for power, expected in ((1, metrics['q']), (2, metrics['q2'])):
        values = q_samples**power
        assert abs(values.mean()-expected) < 6*values.std(ddof=1)/np.sqrt(pairs)+1e-10
    cutoffs = [float(s[chi]-s[chi-1]) for s in model['super_spectra']]
    arrays = dict(bonds=sample.bonds, original_J=sample.J, J=physical.J,
                  intertriangle_mask=mask, points=sample.points, state=state,
                  spins=spins, counts=counts, batches=batches, magnetization=magnetization,
                  zz=zz, sites=np.array(model['sites']),
                  triangle_sites=np.array(model['first']['sites']),
                  triangle_bases=np.array(model['first']['bases']),
                  triangle_spectra=np.array(model['first']['spectra']))
    for b in range(4):
        arrays[f'W_{b}'] = model['retained'][b]
        arrays[f'decoder_{b}'] = model['decoders'][b]
        arrays[f'spectrum_{b}'] = model['super_spectra'][b]
    endpoint_tv = None
    if lam == 0:
        exact = endpoint_mass(model['first'])
        endpoint_tv = float(abs(exact-mass).sum()/2)
        arrays['exact_endpoint_mass'] = exact
        expected_energy = sum(s[0] for s in model['first']['spectra'])
        assert abs(energy-expected_energy) < 1e-8
        assert abs(exact@np.linspace(-1, 1, 49)**2-metrics['q2']) < 1e-9
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    result = dict(key=key, gamma=gamma, lam=lam, m=m, chi=chi, sample_id=SID,
                  h=H, orientation=0, origin=[0, 0], pairs=pairs, seeds=seeds,
                  energy=float(energy), energy_per_site=float(energy/48), gap=gap,
                  residual=residual, fidelity=fidelity, metrics=metrics,
                  mass=mass.tolist(), batches=batches.tolist(),
                  cutoff_gaps=cutoffs, unresolved_cutoff=bool(min(cutoffs) < 1e-9),
                  endpoint_tv=endpoint_tv, raw_sha256=digest(path.with_suffix('.npz')),
                  seconds=time.monotonic()-start, build_seconds=built-start,
                  physical_accuracy_qualified=False)
    write_json(path, result)
    print(json.dumps({k:result[k] for k in ['key','seconds','residual','metrics']}), flush=True)
    return result


def report(root):
    rows = [json.loads(p.read_text()) for p in (root/'cases').glob('*.json')]
    small = [json.loads(p.read_text()) for p in (root/'exact').glob('*.json')]
    assert len(rows) == 48 and len(small) == 16
    for row in rows+small:
        folder = 'cases' if row['key'].startswith('l4') else 'exact'
        assert digest(root/folder/(row['key']+'.npz')) == row['raw_sha256']
    q = np.linspace(-1, 1, 49)
    fig, axes = plt.subplots(4, 4, figsize=(17, 14), sharex=True, layout='constrained')
    fig2, axes2 = plt.subplots(4, 4, figsize=(17, 14), sharex=True, layout='constrained')
    panel_rows = []
    for i, g in enumerate(GAMMAS):
        for j, lam in enumerate(LAMBDAS):
            selected = [next(r for r in rows if (r['gamma'],r['lam'],r['m'],r['chi']) ==
                             (g,lam,m,c)) for m,c in SCHEDULES]
            ax, bx = axes[i,j], axes2[i,j]
            for r, color in zip(selected, ['#0072B2','#D55E00','#009E73']):
                p = np.array(r['mass'])
                ax.plot(q, p, color=color, marker='.', ms=2,
                            lw=1, label=f"RG m={r['m']}, chi={r['chi']}")
            if lam == 0:
                with np.load(root/'cases'/(selected[0]['key']+'.npz')) as raw:
                    ax.plot(q, raw['exact_endpoint_mass'], color='black', ls='--',
                            lw=1.3, label='Exact decoupled endpoint')
            s = next(r for r in small if (r['gamma'],r['lam']) == (g,lam))
            with np.load(root/'exact'/(s['key']+'.npz')) as raw:
                for name, color, label in [('mass','black','Exact L2'),
                        ('mass_m4','#0072B2','L2 m=4'), ('mass_m6','#D55E00','L2 m=6')]:
                    bx.plot(np.linspace(-1,1,13), raw[name], '.-', color=color, label=label)
            tv_m = float(abs(np.array(selected[0]['mass'])-selected[1]['mass']).sum()/2)
            tv_chi = float(abs(np.array(selected[1]['mass'])-selected[2]['mass']).sum()/2)
            delta_e = abs(selected[1]['energy_per_site']-selected[2]['energy_per_site'])
            delta_q2 = abs(selected[1]['metrics']['q2']-selected[2]['metrics']['q2'])
            l2check = next(c for c in s['checks'] if c['m'] == 6)
            candidate = (lam > 0 and tv_m < .05 and tv_chi < .05 and delta_e < .001
                         and delta_q2 < .01 and l2check['tv'] < .05
                         and l2check['energy_error_per_site'] < .001
                         and l2check['q2_error'] < .01
                         and not any(r['unresolved_cutoff'] for r in selected))
            panel_rows.append(dict(gamma=g,lam=lam,tv_m=tv_m,tv_chi=tv_chi,
                delta_energy_chi=delta_e,delta_q2_chi=delta_q2,
                l2_m6=l2check,candidate_only=candidate))
            ax.set_title(f'Gamma={g:g}, lambda={lam:g}\nTV(m)={tv_m:.3f}; TV(chi)={tv_chi:.3f}', fontsize=10)
            bx.set_title(f'Gamma={g:g}, lambda={lam:g}')
            for a in (ax,bx):
                a.set(xlim=(-1.02,1.02), ylim=(0,None), xticks=[-1,-.5,0,.5,1])
                a.grid(alpha=.2)
                if i==3: a.set_xlabel('Physical overlap q')
                if j==0: a.set_ylabel('Probability mass')
    figures = []
    for figure, axs, name, title in [
        (fig,axes,'l4_interaction_grid','L4, N=48 | fixed disorder60000 | h=.15 | intertriangle coupling scan'),
        (fig2,axes2,'l2_exact_interaction_grid','L2, N=12 | fixed disorder60000 | h=.15 | exact companion')]:
        handles, labels = axs[0,0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside upper center', ncol=4, title=title)
        for ext in ('png','pdf'): figure.savefig(root/f'{name}.{ext}', dpi=160)
        plt.close(figure); figures.append(name+'.png')
    # Overlay fields at fixed lambda to expose motion hidden by panel separation.
    fig, axs = plt.subplots(1,4,figsize=(17,4.5),sharex=True,layout='constrained')
    shifts=[]
    for ax, lam in zip(axs,LAMBDAS):
        group=[]
        for g in GAMMAS:
            r=next(r for r in rows if (r['gamma'],r['lam'],r['m'],r['chi'])==(g,lam,6,16))
            ax.plot(q,r['mass'],'.-',ms=2,label=f'Gamma={g:g}')
            group.append(r)
        for a,b in zip(group,group[1:]):
            delta=b['metrics']['q']-a['metrics']['q']
            va=np.array(a['mass'])@q**2-a['metrics']['q']**2
            vb=np.array(b['mass'])@q**2-b['metrics']['q']**2
            shifts.append(dict(lam=lam,gamma_from=a['gamma'],gamma_to=b['gamma'],
                delta_mean_q=delta,measurement_se=float(np.sqrt(max(0,va)/a['pairs']+max(0,vb)/b['pairs'])),
                tv=float(abs(np.array(a['mass'])-b['mass']).sum()/2)))
        ax.set(title=f'lambda={lam:g}',xlabel='Physical overlap q',xlim=(-1.02,1.02))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[0].set_ylabel('Probability mass')
    fig.suptitle('L4 field-driven P(q) shifts | RG m=6, chi=16 | h=.15 | one disorder')
    for ext in ('png','pdf'): fig.savefig(root/f'l4_field_shifts.{ext}',dpi=160)
    plt.close(fig);figures.append('l4_field_shifts.png')
    assessment=dict(panels=panel_rows,shifts=shifts,physical_accuracy_qualified=False,
        scope='One disorder; L2 exact tests plus L4 retention sensitivity; no matched full-spin L4 reference at lambda>0',
        candidate_rule='TV m,chi<.05; chi energy/spin<.001; chi q2<.01; L2 m6 same screens; resolved cutoffs; lambda>0')
    write_json(root/'assessment.json',assessment)
    text=r'''# Weak-intertriangle-coupling screen for spatial RG

The eventual goal is finite-size scaling of $q_{\rm EA}$ and the physical
correlation length in a validated RG regime. This first screen asks whether
physical $P(q)$ still moves with field where retained-space sensitivity is small.

The 4 × 4 layout follows the selected comparison in
`exact_guide_overlap_distributions.ipynb`. Here rows are Gamma=.25,.5,1,2;
columns are lambda=0,.1,.3,1; h=.15. L4 means 48 physical kagome spins.
Original bonds are fixed (disorder60000); only bonds between orientation0
triangles are multiplied by lambda. Internal bonds and fields are unchanged.
The same physical interpolation is used by every retained schedule.

![L4 grid](attachment:l4_interaction_grid.png)

L4 curves compare (m,chi)=(4,8),(6,8),(6,16), with origin(0,0).
Each effective Hamiltonian is solved by Lanczos from two independent starts.
Each curve uses 8192 independent physical replica pairs in two measurement
batches. No error bars are shown for one disorder; production bars represent disorder SE only.
The black curve at lambda=0 is an exact full-spin product-triangle reference.
At lambda>0 no exact full-space N48 solution or matched GFMC curve is supplied.
Agreement between retained spaces is a sensitivity screen, not an accuracy bound.

![L2 companion](attachment:l2_exact_interaction_grid.png)

The companion uses exact full-space L2 references at identical model parameters
and checks the first projection at m=4,6,8. L2 disorder60000 is a different bond
realization from L4 disorder60000 under the size-dependent disorder generator.
L2 agreement does not certify the L4 superblock truncation. All m=8 projections
must recover their exact reference. Independent physical action tests verify
the bond weighting and projection.

![Field shifts](attachment:l4_field_shifts.png)

Field shifts are visible directly in the overlay; `assessment.json` stores
paired field changes, their sampling SE, per-panel TV sensitivity, energy and
q2 changes, and exact L2 errors. Its candidate flag is provisional: unresolved
cutoffs or failed screens exclude a panel, and nonpassing points are retained.
The decoupled endpoint has field-driven local changes but no collective
intertriangle transition; motion of P(q) alone does not establish glass order.

Next validate candidate intervals with higher retention, shifted superblocks,
held-out disorders and matched full-spin L4 benchmarks. Refine Gamma/lambda
near the surviving interval before a disorder-averaged size study. At h=.15,
define q_EA with its longitudinal-field background explicitly; a finite-size
raw mean overlap or peak is not automatically a thermodynamic glass order
parameter. Define correlation length from physical connected correlations,
using one fixed Fourier/sublattice convention. Scaling uses physical L and
identical blocking rules, with separate disorder and truncation uncertainties.

All scientific calculations and figure generation ran on AutoDL with one CPU
worker. Raw bonds, modified couplings, states, isometries, spectra, physical
samples, seeds, residuals, failures and source hashes accompany the figures.
'''
    table = ['\n## Measured retention screen\n',
             '| Gamma | lambda | TV(m) | TV(chi) | chi energy change/spin | chi q2 change | Candidate only |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | :---: |']
    for r in panel_rows:
        table.append(f"| {r['gamma']:g} | {r['lam']:g} | {r['tv_m']:.4f} | {r['tv_chi']:.4f} | {r['delta_energy_chi']:.6f} | {r['delta_q2_chi']:.5f} | {'yes' if r['candidate_only'] else 'no'} |")
    table.extend(['\n## Measured field shifts at m=6, chi=16\n',
        '| lambda | Gamma change | Change in mean q | Sampling SE | TV between fields |',
        '| ---: | :--- | ---: | ---: | ---: |'])
    for r in shifts:
        table.append(f"| {r['lam']:g} | {r['gamma_from']:g} to {r['gamma_to']:g} | {r['delta_mean_q']:+.5f} | {r['measurement_se']:.5f} | {r['tv']:.4f} |")
    text += '\n'.join(table)+'\n'
    cells=[nbformat.v4.new_markdown_cell(text,attachments={name:{'image/png':
        base64.b64encode((root/name).read_bytes()).decode()} for name in figures})]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'execution_location':'AutoDL'})
    nbformat.validate(nb);nbformat.write(nb,root/'triangle_effective_interaction_scan.ipynb')
    (root/'REPORT.md').write_text(text.replace('attachment:',''))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.output
    for name in ('cases','exact'): (root/name).mkdir(parents=True,exist_ok=True)
    manifest=dict(gammas=GAMMAS,lambdas=LAMBDAS,schedules=SCHEDULES,sample_id=SID,h=H,
        pairs=8192,threads=1,scope='16 exact L2 contexts and 48 L4 retained models; no L6',
        sources={str(f.relative_to(Path(__file__).parents[1])):digest(f)
                 for f in Path(__file__).parents[1].rglob('*.py')})
    if (root/'launch.json').exists():
        assert json.loads((root/'launch.json').read_text()) == json.loads(json.dumps(manifest))
    else: write_json(root/'launch.json',manifest)
    try:
        for g in GAMMAS:
            for lam in LAMBDAS: exact_small(root,g,lam)
        done=0
        for g in GAMMAS:
            for lam in LAMBDAS:
                for m,c in SCHEDULES:
                    run_case(root,g,lam,m,c,8192);done+=1
                    write_json(root/'status.json',dict(state='running',completed=done,total=48))
        report(root)
        write_json(root/'status.json',dict(state='complete',completed=48,total=48))
    except Exception:
        (root/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__': main()
