"""Hash-verified q_EA and equal-time susceptibility from saved physical moments.

This is a separate analysis snapshot: running solver sources are never edited.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_correlation_observables import overlap_observables

PRIMARY = ('q_ea_finite', 'chi_sg_connected_equal_time')
LABELS = (r'$q_{\rm EA}^{(N)}$', r'$\chi_{\rm SG,conn}^{\rm eq}$', r'$M_z$')
SOLVER_Q_ATOL = .001
SOLVER_CHI_RTOL = .005
SOLVER_Q2_RTOL = .005
RETENTION_Q_ATOL = .01
RETENTION_CHI_RTOL = .02
RETENTION_Q2_RTOL = .02


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_moments(path, expected, bonds, disorder_key=None):
    if digest(path) != expected:
        raise ValueError(f'Moment hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as a:
        values = overlap_observables(a['magnetization'], a['zz'])
        if 'spectra' in a:
            np.testing.assert_allclose(values[PRIMARY[1]], a['spectra'][0], atol=1e-10, rtol=1e-10)
        # Check disorder identity across fields, m, chi and starts at each size.
        signature = hashlib.sha256(a['bonds'].tobytes()+a['J'].tobytes()).hexdigest() if 'bonds' in a and 'J' in a else None
        if disorder_key is not None:
            if signature is None:
                raise ValueError(f'Missing disorder arrays: {path}')
            if disorder_key in bonds and bonds[disorder_key] != signature:
                raise ValueError(f'Disorder mismatch: {path}')
            bonds[disorder_key] = signature
    return values, dict(path=str(path), sha256=expected, bonds_sha256=signature)


def difference(a, b):
    return dict(q_absolute=abs(a[PRIMARY[0]]-b[PRIMARY[0]]),
                q2_relative=abs(a['q2_raw']-b['q2_raw'])/
                max(abs(a['q2_raw']),abs(b['q2_raw']),1e-10),
                chi_relative=abs(a[PRIMARY[1]]-b[PRIMARY[1]])/
                max(abs(a[PRIMARY[1]]), abs(b[PRIMARY[1]]), 1e-10))


def within(d, q_tolerance=SOLVER_Q_ATOL, chi_tolerance=SOLVER_CHI_RTOL,
           q2_tolerance=SOLVER_Q2_RTOL):
    return (d['q_absolute'] <= q_tolerance and d['chi_relative'] <= chi_tolerance
            and d['q2_relative'] <= q2_tolerance)


def collect(roots, archive_index):
    rows, pending, statuses, bonds = [], [], [], {}
    for root in roots:
        status = root/'status.json'
        statuses.append(dict(root=str(root), status=json.loads(status.read_text()) if status.exists() else {}))
        for folder in sorted((root/'cases').glob('*')):
            final = folder/'result.json'
            if not final.exists():
                pending.append(dict(path=str(folder), state='resource_deferred' if (folder/'resource_unresolved.json').exists() else 'unfinished'))
                continue
            row = json.loads(final.read_text())
            if row['h'] != .15 or row['lambda_rg'] != 1:
                raise ValueError(f'Unexpected physical parameters: {final}')
            records, provenance = [], []
            for record in row['records']:
                name = f"seed{record['seed']}_D{record['D_MPS']}.npz"
                value, source = load_moments(folder/name, row['hashes'][name], bonds, (row['L'],row['sample_id']))
                records.append(dict(record, observables=value))
                provenance.append(source)
            ends = [next(r for r in reversed(records) if r['seed'] == seed) for seed in (0, 1)]
            ladder = []
            for end in ends:
                history = [r for r in records if r['seed'] == end['seed']]
                ladder.append(difference(history[-1]['observables'], history[-2]['observables']) if len(history) >= 2 else None)
            starts = difference(ends[0]['observables'], ends[1]['observables'])
            checks = dict(ladder=ladder, starts=starts)
            qualified = all(d is not None and within(d) for d in ladder) and within(starts, chi_tolerance=.01, q2_tolerance=.01)
            if row['exact'] is not None:
                exact, source = load_moments(folder/'exact.npz', row['hashes']['exact.npz'], bonds)
                provenance.append(source)
                checks['exact'] = [difference(r['observables'], exact) for r in ends]
                qualified = qualified and all(within(d, chi_tolerance=.01, q2_tolerance=.01) for d in checks['exact'])
            selected = min(ends, key=lambda r: r['energy'])
            rows.append(dict({k:row[k] for k in ('L','gamma','sample_id','m','chi','h','lambda_rg','solver_passed')},
                analysis_qualified=bool(row['solver_passed'] and qualified),
                observables=selected['observables'], selected_seed=selected['seed'],
                D_MPS=selected['D_MPS'], checks=checks, seconds=row['seconds'],
                peak_rss_kib=row['peak_rss_kib'], provenance=provenance,
                result_path=str(final), result_sha256=digest(final), kind='dmrg'))
    if archive_index:
        for item in json.loads(archive_index.read_text()):
            path = Path(item['source'])
            original = json.loads(path.read_text())
            if original['h'] != .15 or original['lambda_rg'] != 1:
                raise ValueError(f'Unexpected archived parameters: {path}')
            if original['raw_sha256'] != item['raw_sha256']:
                raise ValueError(f'Archive index disagrees with source: {path}')
            value, source = load_moments(path.with_suffix('.npz'), item['raw_sha256'], bonds, (item['L'],item['sample_id']))
            rows.append(dict({k:item[k] for k in ('L','gamma','sample_id','m','chi')},
                observables=value, kind='archived_exact_retained_space',
                analysis_qualified=True, solver_passed=None, h=.15, lambda_rg=1.,
                provenance=[source], result_path=str(path), result_sha256=digest(path)))
    return rows, pending, statuses


def figure_cell(path):
    return nbformat.v4.new_markdown_cell(f'![{path.stem}](attachment:{path.name})',
        attachments={path.name: {'image/png':base64.b64encode(path.read_bytes()).decode()}})


def report(rows, pending, statuses, output):
    output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    good = [r for r in rows if r['analysis_qualified']]
    l8 = [r for r in good if r['L'] == 8 and r['kind'] == 'dmrg']
    fields, ms, chis, ids = [.1,.5,1.], [4,6,8], [2,3,4,8,16,32], [60000,60001]
    lookup = {(r['gamma'],r['sample_id'],r['m'],r['chi']):r for r in l8}
    means, comparisons = [], []
    for g in fields:
        for m in ms:
            for chi in chis:
                matched = [lookup.get((g,s,m,chi)) for s in ids]
                if all(r is not None for r in matched):
                    means.append(dict(gamma=g,m=m,chi=chi,disorders=ids,
                        observables={k:float(np.mean([r['observables'][k] for r in matched])) for k in matched[0]['observables']}))
    for row in l8:
        for axis, ladder in [('chi',chis),('m',ms)]:
            pos = ladder.index(row[axis])
            if not pos:
                continue
            key = (row['gamma'],row['sample_id'],row['m'] if axis == 'chi' else ladder[pos-1],
                   row['chi'] if axis == 'm' else ladder[pos-1])
            prior = lookup.get(key)
            if prior:
                delta = difference(row['observables'],prior['observables'])
                comparisons.append(dict(gamma=row['gamma'],sample_id=row['sample_id'],axis=axis,
                    settings=[[prior['m'],prior['chi']],[row['m'],row['chi']]], **delta,
                    within_prospective_tolerance=bool(within(delta,RETENTION_Q_ATOL,RETENTION_CHI_RTOL,RETENTION_Q2_RTOL))))
    cells = [nbformat.v4.new_markdown_cell(r'''# RG + DMRG: $q_{\rm EA}$ and susceptibility

Primary targets are $q_{\rm EA}^{(N)}=N^{-1}\sum_i\langle Z_i\rangle^2$ and
$\chi_{\rm SG,conn}^{\rm eq}=N^{-1}\sum_{ij}(\langle Z_iZ_j\rangle-\langle Z_i\rangle\langle Z_j\rangle)^2$.
Physical Pauli spins, including $i=j$; $\chi_{\rm SG,conn}^{\rm eq}=S_R(0)=NQ_{\rm conn}$.
The diagonal contribution is $N^{-1}\sum_i(1-\langle Z_i\rangle^2)^2$;
an uncorrelated transverse product state has susceptibility 1. Growth alone
from a strongly polarized state therefore does not establish criticality.
Magnetization $M_z=N^{-1}\sum_i\langle Z_i\rangle$ provides polarization context.
At $h=0.15$, positive finite-size $q_{\rm EA}$ alone does not establish glass order.
This susceptibility is equal-time. A zero-frequency response requires additional calculations.
Raw $\langle q^2\rangle=N^{-2}\sum_{ij}\langle Z_iZ_j\rangle^2$ is saved separately;
$q_{\rm EA}^{(N)}$ is not $\sqrt{\langle q^2\rangle}$.

$K\sim U(0.5,1.5)$, $\lambda=1$, $h=0.15$. All extraction and figure generation run on AutoDL.
'''+f'\nSnapshot: **{now}**. {len(l8)}/108 planned L8 cases pass both the original solver gates and the new observable checks. Dense scan held.'),
        nbformat.v4.new_markdown_cell('## L8 retention sensitivity\n\n'
        'Matched disorders 60000 and 60001. Each row is a field; columns show each disorder and the mean of both. '
        'Means require both qualified disorders. Missing points break lines. Two disorders provide sensitivity checks; '
        'no error bars are drawn. m=8 retains every triangle state; chi=32 remains partial superblock retention.')]
    for observable, label in zip((*PRIMARY,'magnetization'),LABELS):
        fig, axes = plt.subplots(3,3,figsize=(12,9),layout='constrained')
        for i,g in enumerate(fields):
            for j in range(3):
                ax = axes[i,j]
                for m,color in zip(ms,('#0072B2','#D55E00','#009E73')):
                    values=[]
                    for chi in chis:
                        r = lookup.get((g,ids[j],m,chi)) if j < 2 else next((r for r in means if (r['gamma'],r['m'],r['chi']) == (g,m,chi)),None)
                        values.append(r['observables'][observable] if r else np.nan)
                    ax.plot(range(len(chis)),values,'o-',color=color,label=f'm={m}')
                ax.set(title=f'Gamma={g:g} | '+(f'disorder {ids[j]}' if j<2 else 'two-disorder mean'),
                    xlabel='retained superblock states chi',ylabel=label,xticks=range(len(chis)),xticklabels=chis)
                ax.grid(alpha=.2); ax.legend(fontsize=8)
        path=output/f'L8_{observable}.png'
        fig.savefig(path,dpi=150); fig.savefig(path.with_suffix('.pdf')); plt.close(fig)
        cells.append(figure_cell(path))
    archives=[r for r in good if r['kind'] == 'archived_exact_retained_space']
    archive_means=[]
    if archives:
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        rng=np.random.default_rng(20261105)
        archive_ids=sorted({r['sample_id'] for r in archives})
        indices=rng.integers(len(archive_ids),size=(5000,len(archive_ids)))
        np.save(output/'archive_bootstrap_indices.npy',indices)
        for m,chi in sorted({(r['m'],r['chi']) for r in archives if r['L']==6}):
            groups=[]
            for g in sorted({r['gamma'] for r in archives if r['L']==6}):
                group=sorted([r for r in archives if (r['L'],r['gamma'],r['m'],r['chi'])==(6,g,m,chi)],key=lambda r:r['sample_id'])
                if [r['sample_id'] for r in group] != archive_ids:
                    continue
                item=dict(L=6,gamma=g,m=m,chi=chi,disorders=archive_ids,observables={})
                for key in (*PRIMARY,'magnetization'):
                    data=np.array([r['observables'][key] for r in group])
                    item['observables'][key]=dict(mean=float(data.mean()),interval95=np.quantile(data[indices].mean(axis=1),[.025,.975]).tolist())
                groups.append(item); archive_means.append(item)
            for ax,key in zip(axes,(*PRIMARY,'magnetization')):
                line,=ax.plot([r['gamma'] for r in groups],[r['observables'][key]['mean'] for r in groups],'o-',label=f'm={m}, chi={chi}')
                for r in groups:
                    ax.vlines(r['gamma'],*r['observables'][key]['interval95'],alpha=.5,color=line.get_color())
        for ax,label in zip(axes,LABELS):
            ax.set(xlabel='Gamma',ylabel=label); ax.grid(alpha=.2); ax.legend(fontsize=8)
        path=output/'L6_archived_overlap.png'
        fig.savefig(path,dpi=150); fig.savefig(path.with_suffix('.pdf')); plt.close(fig)
        cells.append(nbformat.v4.new_markdown_cell('## Existing L6 field scan\n\n'
            f'{len(archives)} hash-verified L4/L6 archived cases. L6 curves use all eight matched disorder IDs. '
            'Bars: 95% disorder bootstrap intervals, with the same resampling indices across fields and settings. '
            'These exact solutions of retained Hamiltonians still carry physical retention bias.'))
        cells.append(figure_cell(path))
        settings=sorted({(r['m'],r['chi']) for r in archives if r['L']==6})
        fig,axes=plt.subplots(2,len(settings),figsize=(13,7),layout='constrained',squeeze=False)
        for i,(m,chi) in enumerate(settings):
            for j,key in enumerate(PRIMARY):
                ax=axes[j,i]
                for sid in archive_ids:
                    group=sorted([r for r in archives if (r['L'],r['m'],r['chi'],r['sample_id'])==(6,m,chi,sid)],key=lambda r:r['gamma'])
                    ax.plot([r['gamma'] for r in group],[r['observables'][key] for r in group],'o-',markersize=3,label=str(sid))
                ax.set(title=f'L6 | m={m}, chi={chi}',xlabel='Gamma',ylabel=LABELS[j])
                ax.grid(alpha=.2)
                if i==0:
                    ax.legend(fontsize=7,ncol=2)
        path=output/'L6_individual_disorders.png'
        fig.savefig(path,dpi=150); fig.savefig(path.with_suffix('.pdf')); plt.close(fig)
        cells.append(nbformat.v4.new_markdown_cell('Individual L6 disorders, with no error bars:'))
        cells.append(figure_cell(path))
    excluded=[{k:r[k] for k in ('L','gamma','sample_id','m','chi','solver_passed','analysis_qualified','checks')} for r in rows if not r['analysis_qualified']]
    summary=dict(utc=now,counts=dict(extracted=len(rows),qualified_L8=len(l8),planned_L8=108),
        two_disorder_means=means,retention_comparisons=comparisons,archive_means=archive_means,
        excluded=excluded,unfinished=pending,campaigns=statuses,
        thresholds=dict(solver_q_absolute=SOLVER_Q_ATOL,solver_chi_relative=SOLVER_CHI_RTOL,
            solver_q2_relative=SOLVER_Q2_RTOL, starts_and_exact_q2_relative=.01,
            retention_q2_relative=RETENTION_Q2_RTOL,
            starts_and_exact_chi_relative=.01,retention_q_absolute=RETENTION_Q_ATOL,retention_chi_relative=RETENTION_CHI_RTOL),
        retention_qualified=False,interpretation='Finite-ladder sensitivity; no thermodynamic-order conclusion')
    n_sensitive=sum(not r['within_prospective_tolerance'] for r in comparisons)
    cells.append(nbformat.v4.new_markdown_cell('## Snapshot assessment\n\n'
        f'{n_sensitive} of {len(comparisons)} available adjacent-setting comparisons exceed at least one '
        'prospective retention tolerance. Counts describe this incomplete snapshot and do not constitute '
        'independent statistical tests. No retention setting is qualified yet. '
        'Inspect individual disorders because their shifts can cancel in a mean. '
        'The high-chi ladder and remaining Gamma=1 solves are needed before a practical choice.'))
    cells.append(nbformat.v4.new_markdown_cell('## Qualification and next decision\n\n'
        'The original solver gates remain prerequisites. Additional analysis checks require absolute q_EA drift <=0.001 '
        'and relative susceptibility drift <=0.5% on each final bond-dimension step, with both starts and exact '
        'benchmarks agreeing within 0.001 and 1%. These prospective tolerances are analysis checks. '
        'No failed original solve is rescued by them.\n\n'
        'For practical retention selection, inspect each disorder at fixed m across adjacent chi and at fixed chi '
        'across adjacent m. Provisional tolerances are absolute q_EA change <=0.01 and relative susceptibility change '
        '<=2%. Require the two highest available chi increments and m=6 to 8 comparison at all three fields for both '
        'disorders before selecting a candidate; then check block-partition sensitivity and larger disorder coverage. '
        'A finite plateau supports a practical setting; it does not establish complete physical convergence. '
        'Choose using saved wall times and peak RSS as well. Dense scan and L10 follow this assessment.'))
    cells.append(nbformat.v4.new_markdown_cell('## Excluded completed cases\n\n```json\n'+json.dumps(excluded,indent=2)+'\n```'))
    notebook=nbformat.v4.new_notebook(cells=cells,metadata=dict(execution_location='AutoDL',
        snapshot_utc=now,error_bar_policy='disorder only',kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbformat.validate(notebook)
    nbformat.write(notebook,output/'triangle_overlap_retention.ipynb')
    write(output/'cases.json',rows); write(output/'summary.json',summary)
    write(output/'provenance.json',dict(utc=now,sources={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('triangle_correlation_observables.py'))}))
    print(json.dumps(summary['counts']),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--roots',type=Path,nargs='+',required=True)
    parser.add_argument('--archive-index',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report(*collect(args.roots,args.archive_index),args.output)


if __name__ == '__main__':
    main()
