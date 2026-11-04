"""All-bond endpoint screen; run calculations and reports on AutoDL."""
import argparse
import base64
import itertools
import json
from pathlib import Path
import platform
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

from ising.hamiltonian import Hamiltonian
from kagome_qea.exact import operator_matrix
from .physics import make_af_sample, basis, correlations
from .triangle_effective_rg import triangles, hierarchy, joint_samples, apply_axis
from .triangle_effective_benchmark import write_json, digest, lift, xor_mass, context

SCALES = (1., .5, .2, .1)
GAMMAS = (.05, .10, .25, .50)
FIELDS = (0., .025, .075, .15)
SCHEDULES = ((4, 8), (6, 8), (6, 16))
SID = 60000
PAIRS = 8192


def even_solve(action, dimension, parity, seed=0):
    """Restrict Krylov action to global even spin-flip parity at h=0."""
    def project(v):
        return (v+parity(v))/2
    def matvec(v):
        return project(action(project(v)))
    linear = LinearOperator((dimension,dimension),matvec=matvec,dtype=np.float64)
    values,vectors = eigsh(linear,k=2,which='SA',tol=1e-11,maxiter=20000,
        v0=project(np.random.default_rng(seed).normal(size=dimension)))
    order = np.argsort(values)
    state = project(vectors[:,order[0]])
    state /= np.linalg.norm(state)
    energy = float(state@action(state))
    residual = float(np.linalg.norm(action(state)-energy*state))
    assert residual < 1e-7
    return energy,state,float(values[order[1]]-values[order[0]]),residual


def retained_solve(operator, bases, h, seed=0):
    if h != 0:
        return operator.solve(seed)
    flips = [b.T@b[::-1] for b in bases]
    for flip in flips:
        assert np.max(abs(flip@flip-np.eye(len(flip)))) < 1e-7
    def parity(v):
        t = v.reshape(operator.dims)
        for axis,flip in enumerate(flips):
            t = apply_axis(t,flip,axis)
        return t.ravel()
    return even_solve(operator.action,operator.dimension,parity,seed)


def physical_hamiltonian(L, scale, h):
    sample = make_af_sample(L, SID)
    return sample, Hamiltonian(3*L*L, sample.bonds, sample.J*scale, h=h)


def cached(path):
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    assert digest(path.with_suffix('.npz')) == row['raw_sha256']
    return row


def exact_small(root, scale, gamma, h):
    key = f'l2_s{SID}_k{scale:g}_g{gamma:g}_h{h:g}'
    path = root/'exact'/(key+'.json')
    old = cached(path)
    if old is not None:
        return old
    sample, ham = physical_hamiltonian(2, scale, h)
    matrix = operator_matrix(ham, gamma)
    energies, vectors = eigsh(matrix, k=2, which='SA', tol=1e-12,
                              maxiter=20000,
                              v0=np.random.default_rng(710).normal(size=4096))
    order = np.argsort(energies)
    energy, state = energies[order[0]], vectors[:, order[0]]
    physical_gap = float(np.diff(energies[order])[0])
    if h == 0:
        energy,state,even_gap,_ = even_solve(lambda v: matrix@v,4096,lambda v:v[::-1],710)
    else:
        even_gap = None
    residual = float(np.linalg.norm(matrix@state-energy*state))
    assert residual < 1e-8
    mass = xor_mass(state**2, 12)
    metrics = {k: np.asarray(v).tolist() for k,v in correlations(state**2, basis(12)).items()}
    arrays = dict(bonds=sample.bonds, original_J=sample.J, J=ham.J,
                  points=sample.points, exact_state=state,
                  exact_energies=energies[order], mass=mass)
    checks = []
    for m in (4, 6, 8):
        first = triangles(ham, gamma, m)
        e, v, gap, res = retained_solve(first['operator'],first['bases'],h)
        psi = lift(first, v, m)
        test = np.random.default_rng(721+m).normal(size=m**4)
        test /= np.linalg.norm(test)
        lifted = lift(first, test, m)
        action_error = float(abs(test@first['operator'].action(test)-lifted@(matrix@lifted)))
        assert action_error < 1e-9
        pq = xor_mass(psi**2, 12)
        obs = correlations(psi**2, basis(12))
        tv = float(abs(pq-mass).sum()/2)
        # Near-degenerate states can differ despite equal energies; retain that alarm.
        assert m != 8 or abs(e-energy) < 1e-8
        cutoff = min(float(s[m]-s[m-1]) for s in first['spectra']) if m < 8 else None
        checks.append(dict(m=m, energy_error_per_site=float((e-energy)/12),
                           tv=tv, q2_error=float(abs(obs['q2']-metrics['q2'])),
                           residual=res, action_error=action_error, gap=gap,
                           cutoff_gap=cutoff, full_recovery_alarm=bool(m == 8 and tv > 1e-7)))
        arrays[f'mass_m{m}'] = pq
        arrays[f'bases_m{m}'] = np.asarray(first['bases'])
        arrays[f'spectra_m{m}'] = np.asarray(first['spectra'])
        arrays[f'state_m{m}'] = v
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    row = dict(key=key, scale=scale, gamma=gamma, h=h, sample_id=SID,
               exact_energy=float(energy), exact_gap=physical_gap,even_sector_gap=even_gap,
               residual=residual, metrics=metrics, mass=mass.tolist(), checks=checks,
               raw_sha256=digest(path.with_suffix('.npz')))
    write_json(path, row)
    print(json.dumps(dict(key=key, exact_gap=row['exact_gap'], checks=checks)), flush=True)
    return row


def run_case(root, scale, gamma, h, m, chi):
    key = f'l4_s{SID}_k{scale:g}_g{gamma:g}_h{h:g}_m{m}_c{chi}'
    path = root/'cases'/(key+'.json')
    old = cached(path)
    if old is not None:
        return old
    start = time.monotonic()
    sample, ham = physical_hamiltonian(4, scale, h)
    model = hierarchy(ham, gamma, m, chi)
    built = time.monotonic()
    solutions = [retained_solve(model['operator'],model['decoders'],h,seed) for seed in (0, 1)]
    energy, state, gap, residual = min(solutions, key=lambda x: x[0])
    fidelity = float(abs(solutions[0][1]@solutions[1][1])**2)
    start_energy_change = float(abs(solutions[0][0]-solutions[1][0]))
    magnetization, zz, metrics = context(ham, model, state, sample)
    seeds = [int(np.random.SeedSequence([20261104, 92, SID, SCALES.index(scale),
             GAMMAS.index(gamma), FIELDS.index(h), m, chi, b]).generate_state(1)[0])
             for b in range(2)]
    spins = np.concatenate([joint_samples(model, state, PAIRS, seed) for seed in seeds])
    q_samples = np.mean(spins[::2]*spins[1::2], axis=1)
    counts = np.bincount(np.sum(spins[::2] == spins[1::2], axis=1), minlength=49)
    mass = counts/counts.sum()
    batches = np.array([np.bincount(np.sum(s[::2] == s[1::2], axis=1), minlength=49)/(PAIRS//2)
                        for s in np.split(spins, 2)])
    sampling_checks = []
    for power, expected in ((1, metrics['q']), (2, metrics['q2'])):
        values = q_samples**power
        se = float(values.std(ddof=1)/np.sqrt(PAIRS))
        delta = float(values.mean()-expected)
        sampling_checks.append(dict(power=power, delta=delta, se=se,
                                    alarm=bool(abs(delta) > 6*se+1e-10)))
    cutoffs = [float(s[chi]-s[chi-1]) for s in model['super_spectra']]
    triangle_cutoffs = [float(s[m]-s[m-1]) for s in model['first']['spectra']]
    arrays = dict(bonds=sample.bonds, original_J=sample.J, J=ham.J,
                  points=sample.points, state=state, start_states=np.array([s[1] for s in solutions]),
                  start_energies=np.array([s[0] for s in solutions]),
                  spins=spins, counts=counts, batches=batches, magnetization=magnetization,
                  zz=zz, sites=np.array(model['sites']), groups=np.array(model['groups']),
                  triangle_sites=np.array(model['first']['sites']),
                  triangle_bases=np.array(model['first']['bases']),
                  triangle_spectra=np.array(model['first']['spectra']))
    for b in range(4):
        arrays[f'W_{b}'] = model['retained'][b]
        arrays[f'decoder_{b}'] = model['decoders'][b]
        arrays[f'spectrum_{b}'] = model['super_spectra'][b]
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    row = dict(key=key, scale=scale, K_range=[.5*scale,1.5*scale], gamma=gamma,
               h=h, m=m, chi=chi, sample_id=SID, orientation=0, origin=[0,0],
               intertriangle_multiplier=1., pairs=PAIRS, seeds=seeds,
               energy=float(energy), energy_per_site=float(energy/48), gap=gap,
               gap_sector='even' if h == 0 else 'full',
               residual=residual, fidelity=fidelity, start_energy_change=start_energy_change,
               start_alarm=bool(start_energy_change > 1e-8 or fidelity < 1-1e-6),
               sampling_checks=sampling_checks, metrics=metrics,
               mass=mass.tolist(), batches=batches.tolist(),
               cutoff_gaps=cutoffs, triangle_cutoff_gaps=triangle_cutoffs,
               unresolved_cutoff=bool(min(cutoffs+triangle_cutoffs) < 1e-9),
               raw_sha256=digest(path.with_suffix('.npz')),
               seconds=time.monotonic()-start, build_seconds=built-start,
               physical_accuracy_qualified=False)
    write_json(path, row)
    print(json.dumps({k:row[k] for k in ('key','seconds','residual','metrics','start_alarm','unresolved_cutoff')}), flush=True)
    return row


def verify_raw(root, rows, small):
    for r in rows+small:
        folder = 'cases' if r['key'].startswith('l4') else 'exact'
        path = root/folder/(r['key']+'.npz')
        assert digest(path) == r['raw_sha256'], path
        with np.load(path) as a:
            assert np.array_equal(a['J'], a['original_J']*r['scale'])
            if folder == 'cases':
                counts = np.bincount((a['spins'][::2] == a['spins'][1::2]).sum(1), minlength=49)
                assert np.array_equal(counts, a['counts'])
                assert np.array_equal(counts/counts.sum(), r['mass'])
            else:
                assert np.allclose(xor_mass(a['exact_state']**2,12), r['mass'], atol=1e-14)


def report(root):
    rows = [json.loads(p.read_text()) for p in (root/'cases').glob('*.json')]
    small = [json.loads(p.read_text()) for p in (root/'exact').glob('*.json')]
    assert len(rows) == 192 and len(small) == 64
    verify_raw(root, rows, small)
    lookup = {(r['scale'],r['gamma'],r['h'],r['m'],r['chi']):r for r in rows}
    reference = {(r['scale'],r['gamma'],r['h']):r for r in small}
    q = np.linspace(-1,1,49)
    panels, figures = [], []
    for scale in SCALES:
        fig, axs = plt.subplots(4,4,figsize=(17,14),sharex=True,layout='constrained')
        fig2, axs2 = plt.subplots(4,4,figsize=(17,14),sharex=True,layout='constrained')
        for i,g in enumerate(GAMMAS):
            for j,h in enumerate(FIELDS):
                selected = [lookup[scale,g,h,m,c] for m,c in SCHEDULES]
                exact = reference[scale,g,h]
                ax,bx = axs[i,j],axs2[i,j]
                for r,color in zip(selected,('#0072B2','#D55E00','#009E73')):
                    p = np.array(r['mass'])
                    ax.plot(q,p,color=color,ls='-',
                                marker='.',ms=2,lw=1,
                                label=f"RG m={r['m']}, chi={r['chi']}")
                with np.load(root/'exact'/(exact['key']+'.npz')) as raw:
                    for name,color,label in [('mass','black','Exact L2'),
                            ('mass_m4','#0072B2','L2 m=4'),('mass_m6','#D55E00','L2 m=6')]:
                        bx.plot(np.linspace(-1,1,13),raw[name],'.-',color=color,label=label)
                a,b,c = selected
                tv_m = float(abs(np.array(a['mass'])-b['mass']).sum()/2)
                tv_chi = float(abs(np.array(b['mass'])-c['mass']).sum()/2)
                de = abs(b['energy_per_site']-c['energy_per_site'])
                dq2 = abs(b['metrics']['q2']-c['metrics']['q2'])
                check = next(t for t in exact['checks'] if t['m']==6)
                alarm = any(r['start_alarm'] or r['unresolved_cutoff'] or
                            any(t['alarm'] for t in r['sampling_checks']) for r in selected)
                alarm |= any(t['full_recovery_alarm'] for t in exact['checks'])
                # Compare energies in units of the bond scale as well as absolute units.
                candidate = (tv_m < .05 and tv_chi < .05 and de/scale < .001
                             and dq2 < .01 and check['tv'] < .05
                             and check['energy_error_per_site']/scale < .001
                             and check['q2_error'] < .01 and not alarm)
                panels.append(dict(scale=scale,K_range=[.5*scale,1.5*scale],gamma=g,h=h,
                    tv_m=tv_m,tv_chi=tv_chi,delta_energy_chi_per_site=de,
                    delta_energy_chi_per_site_over_scale=de/scale,delta_q2_chi=dq2,
                    l2_m6=check,alarm=bool(alarm),candidate_only=bool(candidate),
                    mean_q=c['metrics']['q'],q2=c['metrics']['q2']))
                ax.set_title(f'Gamma={g:g}, h={h:g}\nTV(m)={tv_m:.3f}; TV(chi)={tv_chi:.3f}',fontsize=10)
                bx.set_title(f'Gamma={g:g}, h={h:g}')
                for axis in (ax,bx):
                    axis.set(xlim=(-1.02,1.02),ylim=(0,None),xticks=[-1,-.5,0,.5,1])
                    axis.grid(alpha=.2)
                    if i == 3: axis.set_xlabel('Physical overlap q')
                    if j == 0: axis.set_ylabel('Probability mass')
        for figure,axes,size,name in [(fig,axs,'L4, N=48',f'l4_k{scale:g}_field_grid'),
                                      (fig2,axs2,'L2, N=12',f'l2_k{scale:g}_exact_grid')]:
            handles,labels = axes[0,0].get_legend_handles_labels()
            figure.legend(handles,labels,loc='outside upper center',ncol=3,
                          title=f'{size} | K~U({.5*scale:g},{1.5*scale:g}) | disorder60000')
            for ext in ('png','pdf'): figure.savefig(root/f'{name}.{ext}',dpi=160)
            plt.close(figure)
            figures.append(name+'.png')
    shifts = []
    by_context = {(r['scale'],r['gamma'],r['h']):r for r in panels}
    for scale in SCALES:
        for variable,fixed_values,varying in [('gamma',FIELDS,GAMMAS),('h',GAMMAS,FIELDS)]:
            for fixed in fixed_values:
                group = [lookup[(scale,v,fixed,6,16) if variable=='gamma'
                                else (scale,fixed,v,6,16)] for v in varying]
                for a,b in zip(group,group[1:]):
                    variance = [max(0,r['metrics']['q2']-r['metrics']['q']**2)/r['pairs'] for r in (a,b)]
                    pa,pb = (by_context[scale,r['gamma'],r['h']] for r in (a,b))
                    shifts.append(dict(scale=scale,variable=variable,
                        gamma_from=a['gamma'],gamma_to=b['gamma'],h_from=a['h'],h_to=b['h'],
                        delta_mean_q=b['metrics']['q']-a['metrics']['q'],
                        delta_q2=b['metrics']['q2']-a['metrics']['q2'],
                        mean_q_sampling_se=float(np.sqrt(sum(variance))),
                        tv=float(abs(np.array(a['mass'])-b['mass']).sum()/2),
                        both_candidates=bool(pa['candidate_only'] and pb['candidate_only'])))
    assessment = dict(panels=panels,shifts=shifts,physical_accuracy_qualified=False,
        candidate_rule='TV(m),TV(chi)<.05; chi and L2 m6 energy error/spin/scale<.001; q2<.01; L2 TV<.05; no alarms',
        energy_units='scale is the mean K; normalization avoids trivially passing by lowering the energy unit',
        scope='One disorder, fixed partition. Exact L2 and L4 retention sensitivity; no full-spin L4 reference.')
    write_json(root/'assessment.json',assessment)
    text = r'''# Lowering both endpoints of every bond distribution

The goal is to locate a candidate field region where RG is stable and physical
$P(q)$ changes, before finite-size scaling of $q_{\rm EA}$ and correlation length.
All bonds use $K=sK^{(0)}$, with $K^{(0)}\sim U(0.5,1.5)$ and
$s=1,0.5,0.2,0.1$. Internal and boundary bonds receive the same factor.
For each size, all contexts share one archived base disorder realization.

Rows: $\Gamma=0.05,0.10,0.25,0.50$. Columns: $h=0,0.025,0.075,0.15$.
L4 contains 48 physical spins. Curves compare $(m,\chi)=(4,8),(6,8),(6,16)$;
8192 independent physical replica pairs estimate each histogram. No error
bars are plotted because this study has only one disorder. Production error
bars are reserved for standard errors across independent disorders. Sampling
diagnostics are preserved in the raw records. Exact L2 probabilities are deterministic.
The L2 companion compares the first projection with exact full-spin results;
its disorder differs from L4 because the seed includes size. No full-spin L4
reference is supplied. Retention agreement is a sensitivity screen.

Lowering $K$ at fixed fields changes $\Gamma/K,h/K$. Rescaling all three
together preserves the ground state and truncation difficulty. Energy screens
below divide by the mean bond scale to avoid changing the tolerance merely
by changing the energy unit. At $h=0$, symmetry enforces zero mean overlap for
a nondegenerate exact finite-system ground state, while $P(q)$ and $q^2$ can
still change. Cutoff degeneracies, solver-start disagreement and sampling
alarms are saved and exclude provisional candidates.
'''
    for scale in SCALES:
        text += f'\n## K sampled from U({.5*scale:g}, {1.5*scale:g})\n\n'
        text += f'![L4 field grid](attachment:l4_k{scale:g}_field_grid.png)\n\n'
        text += f'![Exact L2 companion](attachment:l2_k{scale:g}_exact_grid.png)\n\n'
        text += '| Gamma | h | TV(m) | TV(chi) | chi energy/spin/scale | chi q2 change | Candidate |\n'
        text += '| ---: | ---: | ---: | ---: | ---: | ---: | :---: |\n'
        for r in panels:
            if r['scale'] == scale:
                text += f"| {r['gamma']:g} | {r['h']:g} | {r['tv_m']:.4f} | {r['tv_chi']:.4f} | {r['delta_energy_chi_per_site_over_scale']:.6f} | {r['delta_q2_chi']:.5f} | {'yes' if r['candidate_only'] else 'no'} |\n"
    text += '\n## Shifts between neighboring passing panels\n\n'
    text += '| Mean K | Variable | Gamma from/to | h from/to | Change in mean q | Sampling SE | Change in q2 | TV |\n'
    text += '| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: |\n'
    for r in shifts:
        if r['both_candidates']:
            text += f"| {r['scale']:g} | {r['variable']} | {r['gamma_from']:g}/{r['gamma_to']:g} | {r['h_from']:g}/{r['h_to']:g} | {r['delta_mean_q']:+.5f} | {r['mean_q_sampling_se']:.5f} | {r['delta_q2']:+.5f} | {r['tv']:.4f} |\n"
    text += r'''

Candidate flags do not certify physical accuracy or a transition. One disorder
and one partition cannot establish glass order. Visible motion may be local
polarization; the field-induced background in $q_{\rm EA}$ must be defined
explicitly. Next validate candidates with higher retention, alternate partitions,
held-out disorder and matched full-spin L4 references before size scaling.
Scaling uses physical L, connected physical correlations and fixed blocking.

Scientific calculations and figure generation ran on AutoDL with one CPU worker.
Raw couplings, seeds, both solver-start states, isometries, spectra, physical
replicas, observables, failures and immutable launch sources accompany the report.
`assessment.json` stores every panel and adjacent field change, including failures
of the retention screen. Energy cutoffs are diagnostic and are not error bounds.
'''
    attachments = {name:{'image/png':base64.b64encode((root/name).read_bytes()).decode()}
                   for name in figures}
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text,attachments=attachments)],
                                  metadata={'execution_location':'AutoDL'})
    nbformat.validate(nb)
    nbformat.write(nb,root/'triangle_effective_bond_scan.ipynb')
    (root/'REPORT.md').write_text(text.replace('attachment:',''))
    write_json(root/'verification.json',dict(raw_hashes_verified=len(rows)+len(small),
        l4_histograms_reconstructed=len(rows),l2_histograms_reconstructed=len(small),
        all_bonds_scaled=True,report_sources_sha256=digest(Path(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report-only',action='store_true')
    args = p.parse_args()
    root = args.output
    for name in ('cases','exact','failures'): (root/name).mkdir(parents=True,exist_ok=True)
    if args.report_only:
        report(root)
        return
    manifest = dict(scales=SCALES,gammas=GAMMAS,fields=FIELDS,schedules=SCHEDULES,
        sample_id=SID,pairs=PAIRS,threads=1,intertriangle_multiplier=1.,
        scope='64 exact L2 contexts and 192 L4 retained models; one disorder; no L6',
        sources={str(f.relative_to(Path(__file__).parents[1])):digest(f)
                 for f in Path(__file__).parents[1].rglob('*.py')})
    manifest['zero_field_solver'] = 'even global spin-flip parity; unrestricted pilot archived separately'
    if (root/'launch.json').exists():
        assert json.loads((root/'launch.json').read_text()) == json.loads(json.dumps(manifest))
    else:
        write_json(root/'launch.json',manifest)
        write_json(root/'environment.json',dict(python=platform.python_version(),numpy=np.__version__))
    failures,completed = [],0
    for scale,g,h in itertools.product(SCALES,GAMMAS,FIELDS):
        tasks = [('exact',()), *[('case',schedule) for schedule in SCHEDULES]]
        for kind,schedule in tasks:
            label = f'{kind}_k{scale:g}_g{g:g}_h{h:g}_'+'_'.join(map(str,schedule))
            try:
                if kind == 'exact': exact_small(root,scale,g,h)
                else: run_case(root,scale,g,h,*schedule)
                completed += 1
            except Exception:
                failures.append(label)
                (root/'failures'/(label+'.txt')).write_text(traceback.format_exc())
                print(traceback.format_exc(),flush=True)
            write_json(root/'status.json',dict(state='running',completed=completed,
                attempted=completed+len(failures),total=256,failures=failures))
    if not failures:
        try:
            report(root)
        except Exception:
            (root/'failures'/'report.txt').write_text(traceback.format_exc())
            failures.append('report')
            print(traceback.format_exc(),flush=True)
    write_json(root/'status.json',dict(state='complete' if not failures else 'complete_with_failures',
        completed=completed,total=256,failures=failures))


if __name__ == '__main__':
    main()
