"""Append auditable strong-field repair controls to the main overlap notebook."""
from pathlib import Path


RELATIVE = Path('data/kagome/ltfim_disorder_scaling_v1/rbm_repair/20261103_v1')
MARKER = 'rbm-strong-field-repair-controls'


def notebook_cells():
    import nbformat as nb
    return [nb.v4.new_markdown_cell(r'''## Strong-field repair controls: L=2 and L=6, periodic boundaries
<!-- rbm-strong-field-repair-controls -->

**Hamiltonian and scope:** $H=\sum_{\langle ij\rangle}K_{ij}Z_iZ_j
-0.15\sum_iZ_i-\Gamma\sum_iX_i$, periodic kagome, $N=3L^2$.
$K_{ij}\sim U(0.5,1.5)$ independently, stored as $J_{ij}=-K_{ij}$, using
`default_rng(SeedSequence([20261027,L,sample_id,1]))`. These controls use
the prespecified disorder **60000** and training seed **0**; they are not
a disorder average. The weak-field investigation remains at **$h=0.15$**.

**Two corrections:**
1. The $N=108$ RBM Adam learning rate is reduced from $0.01$ to $0.001$.
   A matched 200-update ablation keeps the model initialization, chains,
   temperature ladder, sweeps, and disorder identical. Norm clipping did not
   prevent instability at the old Adam step. No sampler equilibrium formula
   or guide-ratio clipping is changed. Failed frozen transport/start checks
   now block guide selection; a completed optimizer is not a converged fit.
2. GFMC now accumulates branching weights and weighted-time integrals in
   log space. A preserved failing uniform-start run reproduces all-weight
   underflow with the old code. The numerical repair does not restore the
   ancestral diversity lost by a poor guide.

The interacting strong-field check uses **$\Gamma=4$**, with exact
diagonalization at **$L=2,N=12$**, and matched RBM/Jastrow projection at
**$L=6,N=108$**. Separate analytic $K=0$ independent-spin tests at $N=108$
are numerical controls, not kagome-disorder measurements. At $\Gamma=4$
both $L=6$ benchmark guides use an explicit Adam rate $0.001$; the Jastrow
comparison does not change its production default. Two benchmark-launch
failures (worker thread reinitialization and a shared status-file collision)
are retained alongside the successfully recovered runs.

The projection checks compare guide/uniform starts, doubled walkers, and
doubled equilibration/forward time. Each has four independently branching
physical populations and two repeats; overlap uses cross-population products.
Capped bars show repeat SE for this one disorder, not disorder uncertainty.
All raw tags, descendant counts, energies, checkpoints, seeds, failures,
and numerical/ancestry diagnostics remain available in the source bundle.
'''), nb.v4.new_code_cell(r'''repair_folder=root/'data/kagome/ltfim_disorder_scaling_v1/rbm_repair/20261103_v1'
repair_manifest=json.loads((repair_folder/'sha256.json').read_text())
for relative,digest in repair_manifest.items():
    assert hashlib.sha256((repair_folder/relative).read_bytes()).hexdigest()==digest,relative
print('Repair artifact hashes verified:',len(repair_manifest))
repair_optimizer=[]
for path in sorted((repair_folder/'optimizer').glob('g*.json')):
    r=json.loads(path.read_text())
    if 'diagnostics' not in r:continue
    repair_optimizer.append(r)
display(Markdown('### Matched L=6, N=108 optimizer ablation'))
table=['| Gamma | Adam rate | Mean frozen energy | Min round trips | Min exchange |',
       '|---|---|---|---|---|']
for r in repair_optimizer:
    ds=r['diagnostics']
    table.append(f"| {r['gamma']:.2f} | {r['lr']:.3g} | {np.mean([d['energy'] for d in ds]):.5f} | "
                 f"{min(d['round_trips'] for d in ds)} | {min(min(d['exchange_fraction']) for d in ds):.3f} |")
display(Markdown('\n'.join(table)))
repair_failure=json.loads((repair_folder/'failed_start_reproduction.json').read_text())
display(Markdown('**Preserved failure replay:** `'+repair_failure['original']['error']+'`. '
    'The corrected first block completes with weight ESS '
    f"{repair_failure['log_weight']['ess']:.2f}/512; this old guide remains unqualified."))
repair_fits=[]
for path in sorted((repair_folder/'validation').glob('**/training/*.json')):
    r=json.loads(path.read_text())
    if 'task' in r and 'diagnostics' in r:repair_fits.append(r)
display(Markdown('### Frozen retraining diagnostics'))
table=['| L / N | Gamma | Guide | Energy range across starts | Min round trips | Sampling alarms |',
       '|---|---|---|---|---|---|']
for r in repair_fits:
    t=r['task'];ds=r['diagnostics'];energies=[d['energy'] for d in ds]
    table.append(f"| {t['L']} / {3*t['L']**2} | {t['gamma']:.2f} | {t['kind']} | "
                 f"{min(energies):.5f} to {max(energies):.5f} | {min(d['round_trips'] for d in ds)} | "
                 f"{'; '.join(r.get('sampling_alarms',[])) or 'none detected'} |")
    if 'exact_evaluation' in r:
        ex=r['exact_evaluation']
        display(Markdown(f"L=2 frozen RBM: exact variational energy error **{ex['energy_error']:.6g}**, "
                         f"fidelity **{ex['fidelity']:.8f}**. These evaluate the trained guide before GFMC."))
display(Markdown('\n'.join(table)))
repair_projection=json.loads((repair_folder/'validation/projection_summary.json').read_text())['results']
repair_errors=[r for r in repair_projection if 'error' in r]
if repair_errors:display(Markdown('**Retained projection failures:** `'+str(repair_errors)+'`'))
repair_groups={}
for r in repair_projection:
    if 'error' in r:continue
    c=r['case'];repair_groups.setdefault((c['L'],c['kind'],c['audit']),[]).append(r)
table=['| L / N | Guide | Audit | Energy ± repeat SE | Mean q ± repeat SE | Min descendant ESS | Max lineage | Alarms |',
       '|---|---|---|---|---|---|---|---|']
for (L,kind,audit),rs in sorted(repair_groups.items()):
    assert sorted(r['case']['repeat'] for r in rs)==[0,1]
    def mean_se(key):
        x=np.array([r[key] for r in rs]);return x.mean(),x.std(ddof=1)/np.sqrt(len(x))
    e,ee=mean_se('energy');q,qe=mean_se('mean_q')
    table.append(f"| {L} / {3*L**2} | {kind} | {audit} | {e:.5f} ± {ee:.5f} | {q:.6f} ± {qe:.6f} | "
        f"{min(r['minimum_descendant_ess'] for r in rs):.1f} | {max(r['largest_lineage'] for r in rs):.3f} | {sum(r['ancestry_alarm'] for r in rs)}/2 |")
display(Markdown('### Gamma=4 projection sensitivity checks\n'+'\n'.join(table)))
exact_energy=next(r['exact']['energy'] for r in repair_projection if 'exact' in r)
l2_start=np.array([r['energy'] for r in repair_groups[(2,'rbm_2','start')]])
l6_rbm_means=[np.mean([r['energy'] for r in repair_groups[(6,'rbm_2',audit)]])
              for audit in ['main','start','population','time']]
display(Markdown('**Residual sensitivity:** the L=2 uniform-start energy differs from exact by '
    f"{l2_start.mean()-exact_energy:+.6f}, versus repeat SE {l2_start.std(ddof=1)/np.sqrt(2):.6f}. "
    f"The L=6 RBM audit means span {np.ptp(l6_rbm_means):.6f} in total energy. "
    'Numerical completion and passing ancestry thresholds therefore do not establish fully converged accuracy.'))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
audits=['main','start','population','time']
for L,kind,color,offset in [(2,'rbm_2','tab:green',-.18),(6,'rbm_2','tab:blue',0),(6,'jastrow','tab:orange',.18)]:
    for ax,key in zip(axes,['energy','mean_q']):
        means=[];errors=[]
        for audit in audits:
            values=np.array([r[key] for r in repair_groups.get((L,kind,audit),[])])
            assert len(values)==2
            if key=='energy':values=values/(3*L**2)
            means.append(values.mean());errors.append(values.std(ddof=1)/np.sqrt(2))
        ax.errorbar(np.arange(4)+offset,means,yerr=errors,fmt='o-',capsize=3,color=color,
                    label=f'L={L}, N={3*L**2}, {kind}')
truth=next(r['exact'] for r in repair_projection if 'exact' in r)
axes[0].axhline(truth['energy']/12,color='tab:green',ls='--',label='L=2 exact')
axes[1].axhline(truth['q_mean'],color='tab:green',ls='--',label='L=2 exact')
for ax in axes:
    ax.set_xticks(range(4),audits);ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0].set_ylabel('Energy per spin');axes[1].set_ylabel('Mean overlap q')
fig.suptitle('Strong-field controls | Gamma=4 | h=0.15 | periodic | disorder 60000 | ±1 repeat SE')
finish(fig,'strong_field_repair_controls')
print('Ground-state qualification remains false: one-disorder controls do not validate the weak-field ensemble.')
'''), nb.v4.new_markdown_cell(r'''**Interpretation:** comparison with exact $L=2$ results is a direct accuracy
check. The uniform-start audit retains a discrepancy larger than its two-repeat
SE. At $L=6$, comparisons across guides, starts and projection settings provide
finite-budget consistency checks without an exact interacting reference; the
remaining audit spread must be resolved before claiming precision convergence.
Temperature round trips alone do not prove valley equilibration. The original
$L=6$ near-unit-overlap curves are not retroactively corrected by these tests.
The $L=4$ weak-field fine grid and original higher-field $L=6$ coverage remain
part of the campaign; corrected weak-field fits and full disorder ensembles
require their own convergence checks.

A separately logged corrected $L=6$ training pilot prioritizes
$\Gamma=0.55$ then $0.60$ (disorder 60000, seeds 0 and 1, fixed $h=0.15$).
Its pending fits are not included as physical measurements here.
''')]
