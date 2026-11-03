"""Exploratory disorder-average scaling figures, with unresolved points visible."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .collective_campaign import write
from .correlation_lengths import length


def report(root):
    root=Path(root);review=json.loads((root/'review.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text());target=len(manifest['disorders'])
    figures=root/'figures';figures.mkdir(exist_ok=True)
    for kind in manifest['kinds']:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for L in manifest['sizes']:
            rows=sorted([r for r in review['rows'] if r['kind']==kind and r['L']==L and r['audit']=='main'],key=lambda r:r['gamma'])
            for ax,key,interval in [(axes[0],'xi_over_L','xi_interval'),(axes[1],'q','q_interval')]:
                x=[r['gamma'] for r in rows];y=[r[key] if r[key] is not None else np.nan for r in rows]
                line,=ax.plot(x,y,'o--',mfc='none',label=f'L={L}, pilot')
                for r in rows:
                    if r[key] is None:continue
                    ci=r[interval]
                    if ci:ax.vlines(r['gamma'],*ci,color=line.get_color())
                    ax.annotate(f"D={len(r['disorders'])}/{target}",(r['gamma'],r[key]),fontsize=7,xytext=(3,4),textcoords='offset points')
                ax.set_xlabel('Gamma');ax.grid(alpha=.2)
        axes[0].set_ylabel('xi_R/L (ratio of disorder-mean spectra)')
        axes[1].set_ylabel('mean_i <Zi>² (includes field polarization)')
        axes[0].legend();axes[1].legend()
        fig.suptitle(f'{kind}, h=0.15: unqualified finite-projection pilot; no collapse fit')
        fig.tight_layout();fig.savefig(figures/(kind+'_scaling.png'),dpi=180)
        fig.savefig(figures/(kind+'_scaling.pdf'));plt.close(fig)
    records=[json.loads(p.read_text()) for p in (root/'projector/runs').glob('*.json')]
    comparisons=[]
    for L in manifest['sizes']:
        for g in manifest['fields']:
            for kind in manifest['kinds']:
                for audit in ['population','time','start']:
                    subsets=[]
                    for sid in manifest['audit_ids']:
                        common=dict(L=L,gamma=g,kind=kind,sample_id=sid)
                        a=[r for r in records if all(r['task'][k]==v for k,v in common.items()) and r['task']['audit']==audit]
                        b=[r for r in records if all(r['task'][k]==v for k,v in common.items()) and r['task']['audit']=='main']
                        if len(a)==len(b)==2:subsets.append((sid,a,b))
                    if not subsets:continue
                    sa=np.mean([np.mean([r['spectrum'] for r in a],axis=0) for _,a,b in subsets],axis=0)
                    sb=np.mean([np.mean([r['spectrum'] for r in b],axis=0) for _,a,b in subsets],axis=0)
                    delta=length(sa,L)/L-length(sb,L)/L
                    comparisons.append(dict(L=L,gamma=g,kind=kind,audit=audit,paired_disorders=[s for s,a,b in subsets],
                        delta_xi_over_L=float(delta) if np.isfinite(delta) else None,
                        delta_q=float(np.mean([np.mean([r['q'] for r in a])-np.mean([r['q'] for r in b]) for _,a,b in subsets])),
                        delta_spectrum=(sa-sb).tolist(),scope='paired subset sensitivity; not a bias confidence bound'))
    write(root/'sensitivity.json',dict(rows=comparisons))
    text=['# Fixed-h disorder scaling pilot','',
        'All displayed points are unqualified finite-projection estimates. No critical exponents or collapse have been fitted.',
        'q is the polarization-containing mean overlap; xi/L uses the ratio of disorder-mean connected spectra.',
        'Intervals are exploratory whole-disorder bootstrap intervals; small pilot ensembles cannot establish precision.',
        '',f"Completed runs: {review['completed_runs']}",
        '', '| L | Gamma | Guide | D | q | xi/L | ancestry alarms |', '|---:|---:|---|---:|---:|---:|---:|']
    for r in review['rows']:
        if r['audit']=='main':text.append(f"| {r['L']} | {r['gamma']} | {r['kind']} | {len(r['disorders'])} | {r['q']:.6g} | {r['xi_over_L']} | {r['ancestry_alarms']} |")
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    report(p.parse_args().root)
