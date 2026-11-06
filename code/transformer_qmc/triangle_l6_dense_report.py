"""Render qualified L6 results remotely; deliver local notebooks to notebook/."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from .triangle_l6_dense import GAMMAS, IDS, digest, write
from .triangle_overlap_report import figure_cell

OBS = ('q_ea_finite', 'chi_sg_connected_equal_time', 'q2_raw', 'chi_sg_raw_equal_time')
LABELS = (r'$q_{\rm EA}^{(N)}$', r'$\chi_{\rm SG,conn}^{\rm eq}$',
          r'$\langle q^2\rangle_{\rm raw}$', r'$\chi_{\rm SG,raw}^{\rm eq}$')


def report(root, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in sorted((root/'audits').glob('*.json'))]
    lookup = {(r['gamma'], r['sample_id']):r for r in rows}
    assert len(lookup) == len(rows)
    signatures = {}
    for row in rows:
        assert signatures.setdefault(row['sample_id'], row['bonds_sha256']) == row['bonds_sha256']
        path = Path(row['result_path'])
        assert digest(path) == row['result_sha256']
        assert digest(path.with_suffix('.npz')) == row['raw_sha256']
        assert abs(row['observables']['chi_sg_raw_equal_time'] - 108*row['observables']['q2_raw']) < 1e-10
    qualified = {key:r for key,r in lookup.items() if r['qualified']}
    indices = np.random.default_rng(20261105).integers(len(IDS), size=(5000, len(IDS)))
    np.save(out/'bootstrap_indices.npy', indices)
    means = []
    for g in GAMMAS:
        group = [qualified.get((g, sid)) for sid in IDS]
        if not all(r is not None for r in group):
            continue
        item = dict(gamma=g, disorders=list(IDS), observables={})
        for obs in (*OBS, 'chi_sg_connected_diagonal', 'chi_sg_connected_off_diagonal', 'magnetization'):
            values = np.array([r['observables'][obs] for r in group])
            item['observables'][obs] = dict(mean=float(values.mean()),
                interval95=np.quantile(values[indices].mean(axis=1), [.025,.975]).tolist())
        means.append(item)
    mean_lookup = {r['gamma']:r for r in means}
    failures = [dict(path=str(p), **json.loads(p.read_text())) for p in sorted((root/'failures').glob('*.json'))]
    now = datetime.now(timezone.utc).isoformat()
    counts = dict(total=456, completed=len(rows), qualified=len(qualified),
                  excluded=sum(not r['qualified'] for r in rows), failed=len(failures),
                  complete_eight_disorder_fields=len(means), fields=57,
                  reused=sum('/triangle-L6-m6-c3-dense-20261105-v1/results/' not in r['result_path'] for r in rows))
    intro = r'''# Dense L6 field scan: m=6, chi=3

**108 physical spins; K uniformly distributed on (0.5,1.5); h=0.15; lambda=1.**
Gamma=0.100:0.025:1.500 (57 fields), eight disorders 60000–60007, 456 cases.
Identical bonds are used across fields for each disorder; hashes are verified.
The hierarchy is 36 triangles → 9 superblocks, with final dimension 3^9=19,683.
Forty archived points are reused by hash-verified reference.

The unchanged retained-space Lanczos solver uses starts 0/1, tolerance 1e-11,
maximum 20,000 iterations and residual <=1e-7. Original energy, fidelity,
projection-energy, cutoff and sampling diagnostics are retained; failed
cases are excluded. Physical moments are reconstructed independently for
both starts: qEA agrees within 0.001, connected susceptibility and raw q2
within 1% (larger absolute endpoint denominator).

$q_{\rm EA}=N^{-1}\sum_i\langle Z_i\rangle^2$;
$\chi_{\rm SG,conn}=N^{-1}\sum_{ij}(\langle Z_iZ_j\rangle-\langle Z_i\rangle\langle Z_j\rangle)^2$;
$\langle q^2\rangle=N^{-2}\sum_{ij}\langle Z_iZ_j\rangle^2$;
$\chi_{\rm SG,raw}=N\langle q^2\rangle$, N=108.
All use the same selected state, include i=j, and are equal-time observables.
Raw susceptibility and raw q2 are not independent evidence.

Every mean requires all eight qualified disorders at that field. Bars are
95% whole-disorder bootstrap intervals (5,000 resamples, seed 20261105),
using the same indices across fields and observables. Individual curves
have no bars. Missing or excluded points break lines.

Both RG levels remain truncated. Retained-space solver qualification does
not establish physical retention convergence. At nonzero longitudinal field,
polarization contributes to qEA and raw overlap. This single-size scan does
not determine a thermodynamic critical point or exponent.

The expensive larger-size/high-retention campaign and D1024 diagnostic were
stopped at the user's direction; saved artifacts remain on AutoDL.
'''
    cells = [nbformat.v4.new_markdown_cell(intro + '\nSnapshot: **' + now + '**. Extraction and plotting: **AutoDL**.')]
    figures = []
    def save(fig, name, caption):
        path = out/(name+'.png')
        fig.savefig(path,dpi=140);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
        figures.append(path)
        cells.extend([nbformat.v4.new_markdown_cell(caption), figure_cell(path)])
    fig, axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,obs,label in zip(axes.flat,OBS,LABELS):
        avg = [mean_lookup[g]['observables'][obs]['mean'] if g in mean_lookup else np.nan for g in GAMMAS]
        ax.plot(GAMMAS,avg,'o-',markersize=3,lw=1.2,color='#0072B2')
        for item in means:
            ax.vlines(item['gamma'],*item['observables'][obs]['interval95'],color='#0072B2',alpha=.45)
        ax.set(xlabel=r'$\Gamma$',ylabel=label,xlim=(.075,1.525));ax.grid(alpha=.2)
    fig.suptitle('L6, m=6, chi=3 | Eight matched disorders | 95% disorder bootstrap')
    save(fig,'L6_dense_means','## Eight-disorder field curves\n\nOnly fields with all eight qualified disorders appear.')
    fig, axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,obs,label in zip(axes.flat,OBS,LABELS):
        for sid in IDS:
            values=[qualified[g,sid]['observables'][obs] if (g,sid) in qualified else np.nan for g in GAMMAS]
            ax.plot(GAMMAS,values,'.-',markersize=3,lw=1,label=str(sid))
        ax.set(xlabel=r'$\Gamma$',ylabel=label,xlim=(.075,1.525));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2)
    save(fig,'L6_dense_individual','## Individual disorders\n\nNo error bars; disorder colors are consistent across panels.')
    fig,ax=plt.subplots(figsize=(12,3.5),layout='constrained')
    coverage=np.array([[1 if (g,sid) in qualified else (0 if (g,sid) in lookup else -1) for g in GAMMAS] for sid in IDS])
    from matplotlib.colors import ListedColormap, BoundaryNorm
    image=ax.imshow(coverage,aspect='auto',origin='lower',cmap=ListedColormap(['#dddddd','#D55E00','#009E73']),
                    norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),extent=(.0875,1.5125,-.5,7.5))
    ax.set(yticks=range(8),yticklabels=IDS,xlabel=r'$\Gamma$',ylabel='Disorder ID',title=f'Qualified {len(qualified)}/456 cases')
    bar=fig.colorbar(image,ax=ax,ticks=[-1,0,1]);bar.ax.set_yticklabels(['unfinished / failed','excluded','qualified'])
    save(fig,'L6_dense_coverage','## Coverage and exclusions\n\nUnfinished, failed and unqualified cases contribute no plotted scientific points.')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for obs,label in (('chi_sg_connected_equal_time','total'),('chi_sg_connected_diagonal','diagonal'),('chi_sg_connected_off_diagonal','off-diagonal')):
        axes[0].plot(GAMMAS,[mean_lookup[g]['observables'][obs]['mean'] if g in mean_lookup else np.nan for g in GAMMAS],label=label)
    axes[0].set(xlabel=r'$\Gamma$',ylabel='Connected susceptibility');axes[0].legend();axes[0].grid(alpha=.2)
    obs='magnetization'
    axes[1].plot(GAMMAS,[mean_lookup[g]['observables'][obs]['mean'] if g in mean_lookup else np.nan for g in GAMMAS],color='#0072B2')
    for item in means:axes[1].vlines(item['gamma'],*item['observables'][obs]['interval95'],color='#0072B2',alpha=.45)
    axes[1].set(xlabel=r'$\Gamma$',ylabel=r'$M_z$');axes[1].grid(alpha=.2)
    save(fig,'L6_dense_connected_context','## Connected-correlation and polarization context\n\n'
         'Left: mean total, diagonal and off-diagonal contributions. The diagonal is N^-1 sum_i (1-<Z_i>²)². '
         'Right: mean magnetization with disorder-only intervals.')
    peaks={}
    for obs in ('chi_sg_connected_equal_time','chi_sg_connected_off_diagonal'):
        if means:
            peak=max(means,key=lambda r:r['observables'][obs]['mean'])
            peaks[obs]=dict(gamma=peak['gamma'],mean=peak['observables'][obs]['mean'],interval95=peak['observables'][obs]['interval95'])
    if len(qualified) == 456:
        individual_peaks = [max((r for r in rows if r['sample_id'] == sid),
                               key=lambda r:r['observables']['chi_sg_connected_equal_time']) for sid in IDS]
        peak = peaks['chi_sg_connected_equal_time']
        start, end = (mean_lookup[g]['observables'] for g in (GAMMAS[0], GAMMAS[-1]))
        cells.insert(1, nbformat.v4.new_markdown_cell('## Completed scan: main findings\n\n'
            f"All 456 cases pass qualification: 416 new solves and 40 archived points, across all 57 fields. "
            f"The disorder-mean connected susceptibility has its sampled maximum at Gamma={peak['gamma']:.3f}, "
            f"with mean {peak['mean']:.5f} and pointwise 95% disorder interval "
            f"[{peak['interval95'][0]:.5f}, {peak['interval95'][1]:.5f}]. "
            'This interval describes susceptibility at that sampled field, not uncertainty in the peak location. '
            f"Individual-disorder maxima span Gamma={min(r['gamma'] for r in individual_peaks):.3f} "
            f"to {max(r['gamma'] for r in individual_peaks):.3f}.\n\n"
            f"Across Gamma=0.1 to 1.5, mean qEA changes from {start['q_ea_finite']['mean']:.6f} "
            f"to {end['q_ea_finite']['mean']:.6f}, and raw q2 from {start['q2_raw']['mean']:.6f} "
            f"to {end['q2_raw']['mean']:.6f}. "
            'These field trends and sample-dependent maxima characterize the fixed retained-space approximation. '
            'Neither the precise grid maximum nor sharp individual features establish a phase transition.'))
    cells.append(nbformat.v4.new_markdown_cell('## Snapshot summary\n\n```json\n'+json.dumps(dict(counts=counts,grid_maxima=peaks),indent=2)+'\n```\n\n'
        'Grid maxima describe this retained model and the currently complete fields; they are not critical-field estimates. '
        'No finite-size scaling fit is attempted.'))
    excluded=[dict(gamma=r['gamma'],sample_id=r['sample_id'],checks=r['checks']) for r in rows if not r['qualified']]
    cells.append(nbformat.v4.new_markdown_cell('## Excluded cases\n\n```json\n'+json.dumps(excluded,indent=2)+'\n```\n\n'
        'All raw states, isometries, disorder arrays and failures remain remote. '
        'Local notebook: notebook/triangle_L6_m6_chi3_dense.ipynb. No notebooks are staged in data/.'))
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(execution_location='AutoDL',snapshot_utc=now,
        kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbformat.validate(nb);notebook=out/'triangle_L6_m6_chi3_dense.ipynb';nbformat.write(nb,notebook)
    write(out/'summary.json',dict(utc=now,counts=counts,means=means,grid_maxima=peaks,excluded=excluded,failures=failures))
    write(out/'brief.json',dict(utc=now,counts=counts,grid_maxima=peaks,excluded=excluded))
    write(out/'delivery.json',dict(utc=now,files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in [notebook,out/'brief.json',*figures]},
        source_sha256=digest(Path(__file__)),bootstrap_sha256=digest(out/'bootstrap_indices.npy'),
        manifest_sha256=digest(root/'manifest.json'),audits={p.name:digest(p) for p in sorted((root/'audits').glob('*.json'))}))
    print(json.dumps(dict(counts=counts,grid_maxima=peaks)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.root,args.output)
