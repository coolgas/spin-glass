"""Live matched-disorder scaling report; crossings only from complete groups."""
import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np

from .triangle_correlation_observables import disorder_summary, ratio
from .finite_size_crossings import crossing_brackets
from .triangle_correlation_grid import GAMMAS


def report(root):
    rows = [json.loads(p.read_text()) for p in (root/'cases').glob('*/result.json')]
    passed = [r for r in rows if r['solver_passed']]
    manifest = root/'manifest.json'
    fields_planned = (json.loads(manifest.read_text())['gammas'] if manifest.exists() else GAMMAS)
    fields_planned = sorted(set(fields_planned) | {r['gamma'] for r in rows})
    field_index = {g: i for i, g in enumerate(fields_planned)}
    lookup = {(r['L'],r['gamma'],r['sample_id'],r['chi']):r for r in passed}
    ids = list(range(60000,60008))
    # One resampling matrix for all fields and sizes: disorder IDs are matched.
    indices = np.random.default_rng(20261105).integers(8,size=(5000,8))
    averages=[]; bootstrap={}; retention=[]
    for L,g,chi in sorted({(r['L'],r['gamma'],r['chi']) for r in passed}):
        if not all((L,g,s,chi) in lookup for s in ids):continue
        group=[lookup[L,g,s,chi] for s in ids]
        best=[min((next(x for x in reversed(r['records']) if x['seed']==seed) for seed in (0,1)),key=lambda x:x['energy']) for r in group]
        values=np.array([r['spectra'] for r in best])
        q2=[]
        for case, chosen in zip(group, best):
            path=root/'cases'/f"L{L}_g{g:g}_s{case['sample_id']}_m6_c{chi}"/f"seed{chosen['seed']}_D{chosen['D_MPS']}.npz"
            with np.load(path) as raw:
                zz=raw['zz']
                q2.append(float(np.sum(zz**2)/(3*L*L)**2))
        result=disorder_summary(values,indices)
        result.update(L=L,gamma=g,chi=chi,m=6,Q_conn=float(values[:,0].mean()/(3*L*L)),
                      q2=float(np.mean(q2)), q2_disorder_se=float(np.std(q2,ddof=1)/np.sqrt(8)))
        averages.append(result);bootstrap[L,g,chi]=ratio(values[indices].mean(axis=1))
    for L,g,chi in list(bootstrap):
        if chi==2 and (L,g,3) in bootstrap:
            delta=bootstrap[L,g,3]-bootstrap[L,g,2]
            valid=np.isfinite(delta).all()
            interval=np.quantile(delta,[.025,.975]).tolist() if valid else None
            retention.append(dict(L=L,gamma=g,delta_R_interval95=interval,
                stable=bool(valid and interval[0]>=-.01 and interval[1]<=.01)))
    crossings=[]
    for chi in (2,3):
        for a,b in ((4,6),(6,8),(8,10)):
            fields=sorted({r['gamma'] for r in averages if r['L']==a and r['chi']==chi} &
                          {r['gamma'] for r in averages if r['L']==b and r['chi']==chi})
            if len(fields)<2:continue
            ga=[next(r for r in averages if (r['L'],r['gamma'],r['chi'])==(a,g,chi)) for g in fields]
            gb=[next(r for r in averages if (r['L'],r['gamma'],r['chi'])==(b,g,chi)) for g in fields]
            if any(r['R'] is None for r in ga+gb):continue
            # Never interpolate across unfinished points in the campaign grid.
            brackets=[]
            for j in range(len(fields)-1):
                if field_index[fields[j+1]] != field_index[fields[j]]+1:continue
                brackets.extend(crossing_brackets(fields[j:j+2],
                    [r['R'] for r in ga[j:j+2]],[r['R'] for r in gb[j:j+2]]))
            unique={x['field']:x for x in brackets}
            for bracket in unique.values():
                lo,hi=bracket['lower'],bracket['upper']
                draws=[]
                if lo==hi:continue
                for k in range(5000):
                    va=[bootstrap[a,g,chi][k] for g in (lo,hi)]
                    vb=[bootstrap[b,g,chi][k] for g in (lo,hi)]
                    if not np.isfinite(va+vb).all():continue
                    hit=crossing_brackets([lo,hi],va,vb)
                    if len(hit)==1:draws.append(hit[0]['field'])
                gates=[next((r['stable'] for r in retention if r['L']==L and r['gamma']==g),False)
                       for L in (a,b) for g in (lo,hi)]
                crossings.append(dict(sizes=[a,b],chi=chi,**bracket,
                    bootstrap_crossing_fraction=len(draws)/5000,
                    conditional_interval95=np.quantile(draws,[.025,.975]).tolist() if draws else None,
                    retention_stable=all(gates), interpretation='candidate only; interval conditional on crossing in bracket'))
    destination=root/'report';destination.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharex=True,sharey=True,layout='constrained')
    for ax,chi in zip(axes,(2,3)):
        for L,color in zip((4,6,8,10),('#0072B2','#D55E00','#009E73','#CC79A7')):
            group=sorted([r for r in averages if r['L']==L and r['chi']==chi],key=lambda x:x['gamma'])
            # NaN gaps prevent lines across unfinished fields.
            fields=fields_planned
            data={r['gamma']:r for r in group}
            ax.plot(fields,[data[g]['R'] if g in data else np.nan for g in fields],
                    'o-',color=color,label=f'L={L}')
            for r in group:
                if r['R_interval95']:ax.vlines(r['gamma'],*r['R_interval95'],color=color,alpha=.6)
        ax.set(xlabel='Gamma',ylabel='R = xi_R / L',title=f'm=6, chi={chi}');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Eight-disorder groups only | disorder 95% intervals | RG retention under review')
    fig.savefig(destination/'R_scaling.png',dpi=160);fig.savefig(destination/'R_scaling.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharex=True,layout='constrained')
    for ax,key,label in zip(axes,('q2','Q_conn'),('Raw <q^2>','Q_conn = S_R(0) / N')):
        for L,color in zip((4,6,8,10),('#0072B2','#D55E00','#009E73','#CC79A7')):
            group=sorted([r for r in averages if r['L']==L and r['chi']==3],key=lambda x:x['gamma'])
            data={r['gamma']:r for r in group}
            fields=fields_planned
            ax.plot(fields,[data[g][key] if g in data else np.nan for g in fields],
                    'o-',color=color,label=f'L={L}')
            for r in group:
                uncertainty=(r['q2_disorder_se'] if key=='q2' else
                             r['spectrum_disorder_se'][0]/(3*L*L))
                ax.vlines(r['gamma'],r[key]-uncertainty,r[key]+uncertainty,color=color,alpha=.6)
        ax.set(xlabel='Gamma',ylabel=label);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Chi=3 | complete eight-disorder groups only | bars: disorder standard error')
    fig.savefig(destination/'overlap_companions.png',dpi=160)
    fig.savefig(destination/'overlap_companions.pdf');plt.close(fig)
    pilot=[]
    for L,g,chi in sorted({(r['L'],r['gamma'],r['chi']) for r in passed}):
        if not all((L,g,s,chi) in lookup for s in (60000,60001)):
            continue
        group=[lookup[L,g,s,chi] for s in (60000,60001)]
        best=[min((next(x for x in reversed(r['records']) if x['seed']==seed) for seed in (0,1)),
                  key=lambda x:x['energy']) for r in group]
        values=np.array([r['spectra'] for r in best])
        mean_R=float(ratio(values.mean(axis=0)))
        if np.isfinite(mean_R):
            pilot.append(dict(L=L,gamma=g,chi=chi,R=mean_R,disorders=[60000,60001]))
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharex=True,sharey=True,layout='constrained')
    for ax,chi in zip(axes,(2,3)):
        for L,color in zip((4,6,8,10),('#0072B2','#D55E00','#009E73','#CC79A7')):
            group=sorted([r for r in pilot if (r['L'],r['chi'])==(L,chi)],key=lambda x:x['gamma'])
            if group:
                # Only join consecutive points on the campaign grid.
                for a,b in zip(group,group[1:]):
                    if field_index[b['gamma']] == field_index[a['gamma']]+1:
                        ax.plot([a['gamma'],b['gamma']],[a['R'],b['R']],color=color)
                ax.plot([r['gamma'] for r in group],[r['R'] for r in group],
                        'o',color=color,label=f'L={L}')
        ax.set(xlabel='Gamma',ylabel='R = xi_R / L',title=f'm=6, chi={chi}')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Two matched disorders only | solver-passing cases | no uncertainty or crossing inference')
    fig.savefig(destination/'feasibility_pilot.png',dpi=160);fig.savefig(destination/'feasibility_pilot.pdf');plt.close(fig)
    content=dict(completed=len(rows),solver_passed=len(passed),gammas=fields_planned,complete_disorder_groups=averages,
                 feasibility_two_disorders=pilot,retention=retention,candidate_crossings=crossings)
    (destination/'summary.json').write_text(json.dumps(content,indent=2,allow_nan=False)+'\n')
    image=base64.b64encode((destination/'R_scaling.png').read_bytes()).decode()
    pilot_image=base64.b64encode((destination/'feasibility_pilot.png').read_bytes()).decode()
    companion_image=base64.b64encode((destination/'overlap_companions.png').read_bytes()).decode()
    cells=[nbformat.v4.new_markdown_cell('# Four-size correlation-length scan\n\n'
        f'{len(rows)} completed cases; {len(passed)} pass reduced-solver checks; {len(averages)} complete eight-disorder groups. '
        f'Target: L4,L6,L8,L10; {len(fields_planned)} planned fields, '
        f'Gamma={fields_planned[0]:g} to {fields_planned[-1]:g}; m=6 and chi=2,3. '
        'Original bonds, lambda=1, h=0.15. Blank regions are unfinished. '
        'Production panels require complete disorder groups; bars are disorder-bootstrap 95% intervals. '
        'Solver consistency does not establish RG convergence.'),
        nbformat.v4.new_markdown_cell('![Scaling curves](attachment:R_scaling.png)',attachments={'R_scaling.png':{'image/png':image}}),
        nbformat.v4.new_markdown_cell('## Overlap companions\n\nRaw <q^2> contains the finite-field polarization background; '
            'Q_conn uses connected physical-spin correlations. Collapse assessment awaits common complete groups '
            'at all four sizes and RG retention checks.\n\n'
            '![Overlap companions](attachment:overlap_companions.png)',
            attachments={'overlap_companions.png':{'image/png':companion_image}}),
        nbformat.v4.new_markdown_cell('## Feasibility subset\n\nTwo matched disorders, no error bars or crossing inference. '
            'Points require solver checks; missing points are unresolved or unfinished.\n\n'
            '![Two-disorder pilot](attachment:feasibility_pilot.png)',
            attachments={'feasibility_pilot.png':{'image/png':pilot_image}}),
        nbformat.v4.new_markdown_cell('## Candidate crossings\n\n'+
            ('No crossing established from completed groups.' if not crossings else
             'Exploratory brackets; conditional intervals exclude draws without a crossing and are not unconditional confidence intervals.\n\n```json\n'+json.dumps(crossings,indent=2)+'\n```')),
        nbformat.v4.new_markdown_cell('## Retention checks\n\nPaired chi=2 to 3 changes, with the same disorder bootstrap. '
            'The +/-0.01 criterion is an empirical stability check.\n\n```json\n'+json.dumps(retention,indent=2)+'\n```')]
    # Show available measurements before production panels with incomplete coverage.
    cells.insert(1,cells.pop(3))
    note=root/'campaign_note.md'
    if note.exists():
        cells.insert(2,nbformat.v4.new_markdown_cell(note.read_text()))
    nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(execution_location='AutoDL',
        kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    nbformat.validate(nb);nbformat.write(nb,destination/'triangle_correlation_scaling.ipynb')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);report(p.parse_args().root)
