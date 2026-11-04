"""Higher-accuracy worldline study, with offline thermal and guided-GFMC comparisons."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits
from kagome_qea.exact import operator_matrix
from .physics import make_af_sample,basis
from .production_protocol import identity,geometry,exact_spectra
from .correlation_lengths import length,sha
from .correlation_remedy_campaign import write
from .validate_correlation_remedy import label
from .worldline_campaign import run_one
from .worldline_qmc import run
from .assess_worldline import assess
from .independent_replica_projector import spectrum_from_moments


CONTEXTS=[(.25,50025,[64.,128.,256.]),(.25,50059,[64.,128.,256.]),
          (.5,50043,[128.,256.,512.]),(.75,50004,[16.,32.,64.])]


def tasks():
    result=[]
    for g,sid,betas in CONTEXTS:
        settings=[('beta',b,.0625,64000) for b in betas]
        settings += [('fine',betas[-1],.03125,64000),('long',betas[-1],.03125,128000)]
        for kind,beta,dt,sweeps in settings:
            for r in range(4):
                result.append(dict(kind=kind,L=2,gamma=g,sample_id=sid,beta=beta,dt=dt,
                    temperatures=80,warmup=16000,sweeps=sweeps,thin=4,blocks=64,
                    replica=r,start=[0,1,-1,0][r]))
    return result


def references(output,old,corrected):
    """Low-energy spectral references: never passed to the sampling kernel."""
    output=Path(output);rows=[]
    for g,sid,betas in CONTEXTS:
        ham=make_af_sample(2,sid).hamiltonian(.15);matrix=operator_matrix(ham,g)
        with threadpool_limits(limits=1):
            e,v=eigsh(matrix,k=64,which='SA',tol=1e-12,v0=np.ones(matrix.shape[0]))
        ix=np.argsort(e);e=e[ix];v=v[:,ix]
        residual=float(np.max(np.linalg.norm(matrix@v-v*e,axis=0)));assert residual<1e-9
        spins=basis(12);lat=geometry(2);thermal=[]
        for beta in sorted(set([16.,32.,*betas])):
            weights=np.exp(-beta*(e-e[0]));z=weights.sum()
            # Omitted eigenvalues are no lower than the last retained eigenvalue.
            tail=(matrix.shape[0]-len(e))*np.exp(-beta*(e[-1]-e[0]))/z
            assert tail<1e-9,('Increase retained eigenstates',g,sid,beta,tail)
            prob=(v*v)@(weights/z)
            spectrum=exact_spectra(prob,spins,lat)['connected_spectrum']
            thermal.append(dict(beta=beta,spectrum=spectrum.tolist(),xi=float(length(spectrum,2)),
                energy=float(weights@e/z),omitted_probability_upper_bound=float(tail)))
        ground=exact_spectra(v[:,0]**2,spins,lat)['connected_spectrum']
        gp=Path(corrected)/'guides'/(label(g,sid)+'.json');gm=json.loads(gp.read_text())
        np.testing.assert_allclose(ground,gm['exact_spectrum'],atol=1e-9)
        gfmc=[]
        for setting in ['production','population','time']:
            paths=[Path(corrected)/'runs'/(label(g,sid)+f'_{setting}_r{r}.json') for r in range(4)]
            if not all(p.exists() for p in paths):continue
            records=[json.loads(p.read_text()) for p in paths]
            spectra=np.array([r['result']['connected_spectrum'] for r in records]);mean=spectra.mean(0)
            rng=np.random.default_rng(1102);boot=spectra[rng.integers(4,size=(4000,4))].mean(1)
            gfmc.append(dict(method='enumerated-corrected guided GFMC',setting=setting,spectrum=mean.tolist(),
                spectrum_se=boot.std(0,ddof=1).tolist(),xi=float(length(mean,2)),
                relative_bias=((mean-ground)/ground).tolist(),
                ancestry_alarms=sum(r['result']['ancestry_alarm'] for r in records),
                total_seconds=sum(r['seconds'] for r in records),
                guide_fit_seconds=gm['seconds'],input_sha256={str(p):sha(p) for p in paths}))
        old_rows=[r for r in json.loads((Path(old)/'review.json').read_text())['rows']
                  if r['L']==2 and r['gamma']==g and r['sample_id']==sid]
        for row in old_rows:
            target=next(t for t in thermal if t['beta']==row['beta'])
            row['relative_bias_to_thermal']=((np.array(row['spectrum'])-target['spectrum'])/target['spectrum']).tolist()
        rows.append(dict(gamma=g,sample_id=sid,gap=float(e[1]-e[0]),eigenpair_max_residual=residual,
            ground_spectrum=ground.tolist(),ground_xi=float(length(ground,2)),thermal=thermal,
            guided_gfmc=gfmc,old_worldline=old_rows))
    write(output/'references.json',dict(rows=rows,scope='Evaluation-only truncated spectral references, with explicit omitted-weight bound. Not used in worldline transitions.'))


def compare(output,corrected):
    root=Path(output);review=assess(root,Path(corrected)/'guides')
    refs=json.loads((root/'references.json').read_text());rows=[]
    for row in review['rows']:
        ref=next(r for r in refs['rows'] if (r['gamma'],r['sample_id'])==(row['gamma'],row['sample_id']))
        target=next(t for t in ref['thermal'] if t['beta']==row['beta'])
        row['thermal_spectrum']=target['spectrum'];row['thermal_xi']=target['xi']
        row['relative_bias_to_thermal']=((np.array(row['spectrum'])-target['spectrum'])/target['spectrum']).tolist()
        row['guided_gfmc']=ref['guided_gfmc'];rows.append(row)
        selected=[]
        for p in (root/'runs').glob('*.json'):
            record=json.loads(p.read_text());t=record['task']
            if (t['kind'],t['gamma'],t['sample_id'],t['beta'],t['dt'])==(row['kind'],row['gamma'],row['sample_id'],row['beta'],row['dt']):
                selected.append((t['replica'],p))
        selected.sort();data=[]
        for _,p in selected:
            with np.load(p.with_suffix('.npz')) as a:
                data.append((a['magnetization_blocks'].copy(),a['pair_blocks'].copy()))
        m=np.array([d[0] for d in data]);q=np.array([d[1] for d in data]);lat=geometry(2)
        row['block_length_sensitivity']=[]
        for merge in [1,2,4]:
            mb=m.reshape(4,-1,merge,12).mean(2);qb=q.reshape(4,-1,merge,12,12).mean(2)
            rng=np.random.default_rng(1102);boot=[]
            for _ in range(2000):
                ix=rng.integers(mb.shape[1],size=(4,mb.shape[1]))
                boot.append(spectrum_from_moments(mb[np.arange(4)[:,None],ix].mean(1),
                    qb[np.arange(4)[:,None],ix].mean(1),lat))
            boot=np.array(boot);lens=length(boot,2);valid=np.isfinite(lens)
            row['block_length_sensitivity'].append(dict(merged_blocks=merge,blocks_per_replica=mb.shape[1],
                spectrum_se=boot.std(0,ddof=1).tolist(),invalid_length_fraction=float(1-valid.mean()),
                xi_se=float(lens[valid].std(ddof=1)) if valid.sum()>1 else None))
        row['replica_site_magnetization']=m.mean(1).tolist()
    write(root/'matched_comparison.json',dict(rows=rows,completed=review['completed'],expected=review['expected'],
        scope='Matched L2 contexts; corrected GFMC has enumerated guides. No scalability claim from this comparison.'))
    lines=['# Worldline accuracy versus guided GFMC','',
        'Errors below are against the exact ground state. Thermal references are recorded separately. Errors and intervals remain subject to block-length and independent-start checks.', '',
        '| Γ | Disorder | Setting | β | Δτ | Worldline ξ | Ground ξ | Max spectral error | GFMC ξ (production) |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        gf=next(x for x in r['guided_gfmc'] if x['setting']=='production')
        xi='unresolved' if r['xi'] is None else f"{r['xi']:.6f}"
        lines.append(f"| {r['gamma']} | {r['sample_id']} | {r['kind']} | {r['beta']} | {r['dt']} | {xi} | {r['exact_xi']:.6f} | {max(map(abs,r['spectral_relative_bias'])):.3%} | {gf['xi']:.6f} |")
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(g,sid,_) in zip(axes.flat,CONTEXTS):
        ref=next(r for r in refs['rows'] if (r['gamma'],r['sample_id'])==(g,sid))
        ax.axhline(ref['ground_xi'],color='black',label='Exact ground state')
        thermal=ref['thermal'];ax.plot([r['beta'] for r in thermal],[r['xi'] for r in thermal],
            '--',color='tab:green',label='Thermal reference')
        gf=next(r for r in ref['guided_gfmc'] if r['setting']=='production')
        ax.axhline(gf['xi'],color='tab:purple',linestyle=':',label='Corrected guided GFMC')
        for kind,marker in [('beta','o'),('fine','s'),('long','D')]:
            rs=sorted([r for r in rows if r['gamma']==g and r['sample_id']==sid and r['kind']==kind],key=lambda r:r['beta'])
            if rs:ax.errorbar([r['beta'] for r in rs],[np.nan if r['xi'] is None else r['xi'] for r in rs],
                yerr=[np.nan if r['block_length_sensitivity'][-1]['xi_se'] is None else r['block_length_sensitivity'][-1]['xi_se'] for r in rs],
                fmt=marker,capsize=3,label='Worldline '+kind)
        ax.set(xscale='log',xlabel='Inverse temperature β',ylabel='Connected overlap length ξ',
            title=f'Γ={g}, disorder {sid}')
    axes.flat[0].legend(fontsize=7);fig.tight_layout()
    fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf');plt.close(fig)


def campaign(output,workers,corrected):
    import numba
    output=Path(output);ts=tasks();code=Path(__file__).parent
    manifest=dict(tasks=ts,h=.15,numba_version=numba.__version__,state_enumeration=False,guide=False,
        source_sha256={n:sha(code/n) for n in ['worldline_accuracy.py','worldline_qmc.py','worldline_campaign.py','physics.py','assess_worldline.py']},
        scope='Higher-accuracy L2 validation; reference data used only after simulations')
    mp=output/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:write(mp,manifest)
    write(output/'status.json',dict(state='compiling',completed=0,total=len(ts)))
    try:
        run(make_af_sample(2,50004).hamiltonian(.15),.75,beta=1.,dt=.25,temperatures=2,
            warmup=2,sweeps=8,thin=1,blocks=2,seed=0)
        # Longest tasks first to avoid a long serial tail on the worker pool.
        ts=sorted(ts,key=lambda t:t['temperatures']*(t['beta']/t['dt'])*(t['warmup']+t['sweeps']),reverse=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(run_one,t,str(output)) for t in ts]
            for count,future in enumerate(as_completed(futures),1):
                row=future.result();print(json.dumps(dict(completed=count,total=len(ts),result=row)),flush=True)
                write(output/'status.json',dict(state='running',completed=count,total=len(ts),updated_unix=time.time()))
        compare(output,corrected)
        write(output/'status.json',dict(state='completed',completed=len(ts),total=len(ts)))
    except Exception as e:
        write(output/'status.json',dict(state='failed',error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['references','run','compare'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--corrected',type=Path,required=True)
    p.add_argument('--old',type=Path);p.add_argument('--workers',type=int,default=18);a=p.parse_args()
    if a.stage=='references':references(a.output,a.old,a.corrected)
    elif a.stage=='compare':compare(a.output,a.corrected)
    else:campaign(a.output,a.workers,a.corrected)
