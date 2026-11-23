"""Export closed matched disorder curves, preserving signed histograms and flags."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import tarfile
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run,out):
    run=Path(run).expanduser().resolve(); out=Path(out).expanduser().resolve()
    if out.is_relative_to(run): raise ValueError('Export outside the source run')
    tasks=json.loads((run/'tasks.json').read_text())
    if not tasks: raise ValueError('Empty manifest')
    adaptive=all(t.get('adaptive',False) for t in tasks)
    if any(t.get('adaptive',False) for t in tasks)!=adaptive: raise ValueError('Mixed campaign modes')
    L=6 if adaptive else 4; nspin=3*L*L; nbins=nspin+1
    models=['full'] if adaptive else ['full','nn_nnn','nn']
    deltas=[2.,3.15]; drives=np.geomspace(.03,10.,30)
    expected={(m,d,i) for m in models for d in deltas for i in range(30)}
    groups={}
    seen_ids=set()
    for task in tasks:
        if task['id'] in seen_ids: raise ValueError('Duplicate case ID')
        seen_ids.add(task['id'])
        index=int(np.argmin(abs(drives-task['omega'])))
        if not np.isclose(drives[index],task['omega'],rtol=1e-14,atol=0): raise ValueError('Unexpected drive')
        if task.get('L',L)!=L or task['chi']!=12: raise ValueError('Unexpected lattice or chi')
        sid=task['sample']; key=(task['model'],task['delta'],index)
        cohort=groups.setdefault(sid,{})
        if key in cohort: raise ValueError('Duplicate disorder/parameter point')
        cohort[key]=task
    # A truncated manifest must never make an incomplete curve appear matched.
    if any(set(group)!=expected for group in groups.values()):
        raise ValueError('Incomplete parameter grid in manifest')
    ids=sorted(sid for sid,group in groups.items() if all(
        (run/'cases'/t['id']/'result.json').exists() for t in group.values()))
    if not ids: raise ValueError('No complete matched disorder curves')
    shape=(len(ids),len(models),len(deltas),len(drives))
    counts=np.zeros(shape+(nbins,),dtype=np.int64)
    qea=np.zeros(shape); q2=np.zeros(shape); passed=np.zeros(shape,dtype=bool)
    final_D=np.zeros(shape,dtype=np.int64)
    flag_names=['solver_pass','sampling_moment_check','rg_converged','production_qualified',
                'production_selected','bond_dimension_verified']
    numeric_flags={name:np.zeros(shape,dtype=bool) for name in flag_names}
    manifest=[]; flags=[]; pairs_set=set()
    for si,sid in enumerate(ids):
        for (model,delta,index),task in groups[sid].items():
            folder=run/'cases'/task['id']; rf=folder/'result.json'; result=json.loads(rf.read_text())
            pf=folder/'pq.npz'
            if not result.get('completed',True): raise ValueError('Incomplete result')
            for name,digest in result['hashes'].items():
                path=(folder/name).resolve()
                if not path.is_relative_to(folder) or sha(path)!=digest: raise ValueError('Artifact hash mismatch')
            inputs=result['inputs']
            for a,b in [('sample_id','sample'),('model','model'),('delta','delta'),('omega','omega'),('chi','chi')]:
                if inputs[a]!=task[b]: raise ValueError('Result/manifest mismatch')
            pairs=task.get('pairs',20000); pairs_set.add(pairs)
            if inputs['pairs']!=pairs: raise ValueError('Replica count mismatch')
            best=min(result['endpoints'],key=lambda e:e['energy'])
            k=(si,models.index(model),deltas.index(delta),index)
            with np.load(pf) as hist:
                if not np.array_equal(hist['q'],np.linspace(-1,1,nbins)): raise ValueError('Signed bins mismatch')
                if hist['counts'].dtype.kind not in 'iu' or (hist['counts']<0).any() or hist['counts'].sum()!=pairs:
                    raise ValueError('Incomplete histogram counts')
                np.testing.assert_allclose(hist['p'],hist['counts']/pairs,rtol=0,atol=1e-15)
                counts[k]=hist['counts']
            qea[k]=best['observables']['qea']; q2[k]=best['observables']['q2']
            final_D[k]=best['D']
            for name in flag_names: numeric_flags[name][k]=result[name]
            passed[k]=result['solver_pass'] and result['sampling_moment_check']
            manifest.append(dict(case=task['id'],result_sha256=sha(rf),pq_sha256=sha(pf),
                geometry_sha256=result['geometry_sha256'],final_D=best['D'],
                sweeps=[e['sweeps'] for e in result['endpoints']],
                decisions=result.get('decisions',[]),canonical_diagnostics=result.get('canonical_diagnostics'),
                start_comparison=result['start_comparison'],storage=result.get('storage'),
                numerical_flags={name:result[name] for name in flag_names}))
            if not passed[k]: flags.append(dict(case=task['id'],sample=sid,model=model,delta=delta,
                omega=task['omega'],solver_pass=result['solver_pass'],sampling_pass=result['sampling_moment_check']))
    if len(pairs_set)!=1: raise ValueError('Mixed sampling counts require a weighted export format')
    # Normalize within each disorder first; every complete disorder has equal weight.
    probabilities=counts/counts.sum(axis=-1,keepdims=True)
    q=np.linspace(-1,1,nbins)
    m2=np.sum(probabilities*q**2,axis=-1); m4=np.sum(probabilities*q**4,axis=-1)
    binder=.5*(3-m4.mean(axis=0)/m2.mean(axis=0)**2)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'histograms.npz',counts=counts,qea=qea,q2=q2,passed=passed,
        final_D=final_D,**numeric_flags,sample_ids=ids,models=models,deltas=deltas,drives=drives,q=q,
        m2=m2,m4=m4,binder=binder)
    metadata=dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_run=str(run),
        source_release='adaptive-l6' if adaptive else 'rydberg-kagome-20261117-v5',L=L,N=nspin,chi=12,
        D='adaptive' if adaptive else 128,sample_ids=ids,
        cohort_rule='Complete matched disorder curves; flagged cases retained; equal disorder weighting',
        binder_definition='0.5*(3-mean_dis(m4)/mean_dis(m2)**2)',
        bootstrap_unit='whole disorder curve across all models, detunings and drives',
        case_count=len(manifest),flags=flags,pairs_per_case=next(iter(pairs_set)),
        files={'histograms.npz':sha(out/'histograms.npz')})
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'case-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out/'snapshot.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['histograms.npz','metadata.json','case-manifest.json']: tar.add(out/name,arcname=name)
    return dict(directory=str(out),archive=str(archive),archive_sha256=sha(archive),
                archive_bytes=archive.stat().st_size,case_count=len(manifest),sample_ids=ids)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); print(json.dumps(export(args.run,args.output),indent=2))
