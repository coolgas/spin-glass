"""Deterministic qualification panel and paired retention report."""
import json
from pathlib import Path
import numpy as np
from .geometry import sample
from .results import json_write
from .convergence import compare,cdf_distance


def panel():
    # Unbiased prespecified IDs; stress IDs are additional, explicitly labeled.
    regular=list(range(12))
    geometries=[sample(4,sid) for sid in range(128)]
    row_max=np.array([g['interactions'].sum(1).max() for g in geometries])
    minima=np.array([g['distances'].min() for g in geometries])
    stress=[]
    for ranking in (np.argsort(-row_max),np.argsort(minima)):
        for sid in ranking:
            if int(sid) not in regular+stress:
                stress.append(int(sid))
                if len(stress)%2==0: break
    tasks=[]
    # Interleave matched retention levels so unrelated panel cases proceed.
    for sid in regular+stress:
        for omega in (0.015625,.262144,1.):
            for delta in (2.,3.15):
                for model in ('full','nn_nnn','nn'):
                    for chi in (8,12,16):
                        tasks.append(dict(id=f's{sid:03d}_{model}_d{delta:g}_g{omega:g}_c{chi}',
                            sample=sid,model=model,delta=delta,omega=omega,chi=chi,
                            dimensions=[32,64] if chi==8 else [32,64,128,256],pairs=20000,
                            panel_role='stress' if sid in stress else 'prespecified'))
    return tasks,dict(prespecified_ids=regular,stress_ids=stress,
                      stress_selection='two extreme max row sums and two smallest distances, excluding earlier IDs',
                      retention_stage=[8,12,16],higher_retention='24,32 after measured runtime and sensitivity assessment')


def report(root):
    root=Path(root);tasks=json.loads((root/'tasks.json').read_text())
    results={}
    for task in tasks:
        path=root/'cases'/task['id']/'result.json'
        if path.exists(): results[task['id']]=json.loads(path.read_text())
    paired=[]
    for task in tasks:
        if task['chi'] not in (8,12): continue
        higher=12 if task['chi']==8 else 16
        key=task['id']; other=key.rsplit('_c',1)[0]+f'_c{higher}'
        if key not in results or other not in results: continue
        a=min(results[key]['endpoints'],key=lambda r:r['energy'])
        b=min(results[other]['endpoints'],key=lambda r:r['energy'])
        diff=compare(a,b,48,energy_tol=1e-5,qea_tol=.002)
        with np.load(root/'cases'/key/'pq.npz') as x,np.load(root/'cases'/other/'pq.npz') as y:
            cdf=cdf_distance(x['p'],y['p'])
        paired.append(dict(case=key,higher_chi=higher,**diff,cdf_distance=cdf,
                           lower_solver_pass=results[key]['solver_pass'],higher_solver_pass=results[other]['solver_pass']))
    result=dict(planned=len(tasks),completed=len(results),
                solver_passes=sum(r['solver_pass'] for r in results.values()),
                sampling_moment_passes=sum(r['sampling_moment_check'] for r in results.values()),
                paired_retention=paired,production_qualified=False,
                note='Paired CDF differences include sampling noise; no automatic RG qualification.')
    json_write(root/'qualification-report.json',result)
    return result
