"""Finite, resumable cases. Qualification results never imply RG convergence."""
import json
import os
import resource
import time
import traceback
from pathlib import Path
import numpy as np
from .geometry import sample,masks
from .superblock import build
from . import dmrg
from .observables import summarize,vector_moments
from .sampling import mps_samples,overlap_histogram
from .convergence import compare,sampling_check
from .results import json_write,checkpoint,load_checkpoint,array_hash,source_hashes,digest


def run_case(output,L,sid,model_name,delta,omega,chi,dimensions=(32,64,128,256),
             pairs=20000,max_sweeps=120,validate=True,fixed_production=False):
    if not Path(output).is_absolute():
        raise ValueError("Output path must be absolute")
    root=Path(output).resolve()
    if any(s in root.parts for s in ('Dropbox','CloudStorage')):
        raise ValueError("Use an explicit non-synced output path")
    if omega<=0 or not np.isfinite([omega,delta]).all() or pairs<2:
        raise ValueError("Finite parameters, positive drive and >=2 replica pairs required")
    if fixed_production and (chi!=12 or list(dimensions)!=[128]):
        raise ValueError("Selected production requires chi=12 and direct D=128")
    if (len(dimensions)<2 and not fixed_production) or list(dimensions)!=sorted(set(dimensions)):
        raise ValueError("Increasing D ladder with at least two levels required")
    root.mkdir(parents=True,exist_ok=True)
    inputs=dict(L=L,sample_id=sid,model=model_name,delta=delta,omega=omega,chi=chi,
                dimensions=list(dimensions),pairs=pairs,max_sweeps=max_sweeps,
                validate=validate,fixed_production=fixed_production,sources=source_hashes())
    if (root/'inputs.json').exists():
        if json.loads((root/'inputs.json').read_text())!=inputs:
            raise ValueError("Resume source/config mismatch; use new release/output")
    else: json_write(root/'inputs.json',inputs)
    if (root/'result.json').exists():
        result=json.loads((root/'result.json').read_text())
        for name,sha in result['hashes'].items():
            if digest(root/name)!=sha: raise ValueError(f"Corrupt completed output: {name}")
        return result
    lock=root/'owner.json'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        raise RuntimeError(f"Case already owned or interrupted: inspect {lock}")
    with os.fdopen(fd,'w') as f: json.dump(dict(pid=os.getpid(),started=time.time()),f)
    start=time.monotonic()
    try:
        return _case(root,inputs,start)
    except BaseException:
        json_write(root/f'failure-{time.time_ns()}.json',dict(traceback=traceback.format_exc(),inputs=inputs))
        raise
    finally:
        lock.unlink()


def _case(root,inputs,start):
    L,sid=inputs['L'],inputs['sample_id']; n=3*L*L
    geometry=sample(L,sid)
    v=geometry['interactions']*masks(L)[inputs['model']]
    suffix='.pkl.gz' if inputs['fixed_production'] else '.pkl'
    model_path=root/('model'+suffix)
    if model_path.exists():
        if digest(model_path)!=json.loads((root/'model_hash.json').read_text())['sha256']:
            raise ValueError("Corrupt model checkpoint")
        rg=load_checkpoint(model_path)
    else:
        rg=build(v,inputs['delta'],inputs['omega'],L,inputs['chi'])
        sha=checkpoint(model_path,rg)
        json_write(root/'model_hash.json',dict(sha256=sha))
    np.savez_compressed(root/'geometry.npz',points=geometry['points'],
                        displacements=geometry['displacements'],interactions=v,sites=rg['sites'])
    mpo,order=dmrg.build_mpo(rg)
    checkpoint_sites=mpo.lat.mps_sites() if inputs['fixed_production'] else ()
    exact=None
    if inputs['validate']:
        if (root/'exact.json').exists(): exact=json.loads((root/'exact.json').read_text())
        else:
            solutions=[rg['operator'].solve(seed) for seed in (0,1)]
            ref=min(solutions,key=lambda s:s['energy'])
            m,zz=vector_moments(rg,ref['vector'])
            exact=dict(energy=ref['energy']+rg['constant'],observables=summarize(m,zz),
                       residual=ref['residual'],gap=ref['gap'],
                       start_energy_difference=abs(solutions[0]['energy']-solutions[1]['energy']))
            # Independent MPO comparison on random MPS, not just the optimized state.
            errors=[]
            for seed in (211,713):
                p=dmrg.initial_state(mpo,seed); vec=dmrg.full_vector(p,order,rg['chi'])
                errors.append(float(abs(mpo.H_MPO.expectation_value(p)-vec @ rg['operator'].action(vec))))
            if max(errors)>1e-8: raise RuntimeError(f"MPO mismatch {errors}")
            exact['mpo_errors']=errors
            np.savez_compressed(root/'exact.npz',state=ref['vector'],m=m,zz=zz)
            json_write(root/'exact.json',exact)
    endpoints=[]; checkpoints=[]; all_records=[]
    for seed in (0,1):
        psi=dmrg.initial_state(mpo,seed); previous=None; endpoint=None
        for D in inputs['dimensions']:
            record_file=root/f'start{seed}-D{D}.json'
            state_file=root/f'start{seed}-D{D}{suffix}'
            if record_file.exists():
                row=json.loads(record_file.read_text())
                if digest(state_file)!=row['checkpoint_sha256']: raise ValueError("Checkpoint hash mismatch")
                psi=load_checkpoint(state_file,checkpoint_sites)
            else:
                out=dmrg.optimize(psi,mpo,rg,order,D,inputs['max_sweeps'])
                m,zz,x=out.pop('m'),out.pop('zz'),out.pop('x')
                row=dict(**out,seed=seed,D=D,observables=summarize(m,zz),transverse_mean=float(x.mean()))
                if exact is not None:
                    row['exact_comparison']=compare(row,exact,n)
                    vec=dmrg.full_vector(psi,order,rg['chi'])
                    direct_m,direct_zz=vector_moments(rg,vec)
                    if max(np.max(abs(m-direct_m)),np.max(abs(zz-direct_zz)))>1e-8:
                        raise RuntimeError("MPS physical reconstruction mismatch")
                    action=rg['operator'].action(vec)
                    e=np.vdot(vec,action).real
                    row['projected_residual']=float(np.linalg.norm(action-e*vec))
                row['D_comparison']=compare(row,previous,n) if previous is not None else None
                row['bond_dimension_verified']=bool(row['D_comparison'] and row['D_comparison']['passed'])
                row['solver_pass']=bool(row['sweep_pass'] and
                    (inputs['fixed_production'] or row['bond_dimension_verified']) and
                    (exact is None or row['exact_comparison']['passed']))
                row['checkpoint_sha256']=checkpoint(state_file,psi,checkpoint_sites)
                np.savez_compressed(root/f'start{seed}-D{D}-moments.npz',m=m,zz=zz,x=x)
                json_write(record_file,row)
            endpoint=row; all_records.append(row)
            if row['solver_pass']: break
            previous=row
        endpoints.append(endpoint); checkpoints.append(state_file)
    agreed=compare(*endpoints,n)
    winner=int(endpoints[1]['energy']<endpoints[0]['energy'])
    hist=None; sampling_pass=False
    if inputs['pairs']:
        psi=load_checkpoint(checkpoints[winner],checkpoint_sites)
        samples=mps_samples(psi,rg,order,2*inputs['pairs'],20261117+sid)
        hist=overlap_histogram(samples)
        sampling_pass=sampling_check(hist,endpoints[winner]['observables'])
        np.savez_compressed(root/'pq.npz',counts=hist.pop('counts'),p=hist.pop('p'),q=hist.pop('q'))
    result=dict(inputs=inputs,geometry_sha256=array_hash(geometry['points']),
                attempts=geometry['attempts'],distance_version=geometry['distance_version'],
                block_cut_gaps=rg['cut_gaps'],block_residuals=rg['block_residuals'],
                exact=exact,endpoints=endpoints,start_comparison=agreed,
                sampling=hist,sampling_moment_check=sampling_pass,
                solver_pass=bool(all(r['solver_pass'] for r in endpoints) and agreed['passed']),
                rg_converged=False,production_qualified=False,
                production_selected=inputs['fixed_production'],
                approximation_status='user-selected chi12 D128; RG sensitivity retained' if inputs['fixed_production'] else 'qualification',
                seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['hashes']={p.name:digest(p) for p in root.iterdir()
                      if p.suffix in ('.npz','.pkl','.gz')}
    json_write(root/'result.json',result)
    return result
