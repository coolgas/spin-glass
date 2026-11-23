"""Explicit adaptive case path; immutable completed results and verified resume."""
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import time
import traceback
import numpy as np
from . import adaptive, dmrg
from .geometry import sample,masks
from .superblock import build
from .results import checkpoint,load_checkpoint,json_write,digest,array_hash,source_hashes
from .observables import summarize
from .sampling import mps_samples,overlap_histogram
from .convergence import compare,sampling_check


def retire_committed_restart(root,result):
    """Finish a committed retention decision; never glob-delete failure evidence."""
    for name,sha in result.get('retired_artifacts',{}).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError('Retirement outside case')
        if path.exists():
            if digest(path)!=sha: raise ValueError('Retired artifact hash mismatch')
            path.unlink()
    folder=root/'restart'
    if folder.exists() and not any(folder.iterdir()): folder.rmdir()


def run_case(output,L,sid,model_name,delta,omega,chi=12,dimensions=(128,256,384,512),
             pairs=20000,max_sweeps=120,seconds=None,invocation_sweeps=None,
             memory_gib=8.,model_cache=None,checkpoint_interval=30,final_state_retention='full'):
    import scipy,tenpy
    root=Path(output)
    if not root.is_absolute() or any(s in root.resolve().parts for s in ('Dropbox','CloudStorage')):
        raise ValueError('Explicit non-synced output required')
    root=root.resolve()
    if max_sweeps!=120 or list(dimensions)!=sorted(set(dimensions)) or not dimensions or min(dimensions)<1:
        raise ValueError('Adaptive cases require 120 total sweeps and increasing positive dimensions')
    if omega<=0 or not np.isfinite([omega,delta]).all() or pairs<2 or not 1<=memory_gib<=64:
        raise ValueError('Invalid case parameters')
    if seconds is not None and seconds<=0 or invocation_sweeps is not None and invocation_sweeps<1:
        raise ValueError('Invocation budgets must be positive')
    if final_state_retention not in ('full','analysis'):
        raise ValueError('Final state retention must be full or analysis')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'case.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: lock.close(); raise RuntimeError('Case has a live worker')
    # The kernel lock releases on process death; owner records are append-only evidence.
    owner=root/f'owner-{time.time_ns()}.json'
    json_write(owner,dict(pid=os.getpid(),started=time.time()))
    started=time.monotonic()
    try:
        inputs=dict(L=L,sample_id=sid,model=model_name,delta=delta,omega=omega,chi=chi,
            dimensions=list(dimensions),pairs=pairs,max_sweeps=120,adaptive=True,
            checkpoint_interval=checkpoint_interval,
            final_state_retention=final_state_retention,
            fixed_production=False,validate=False,sources=source_hashes(),
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,tenpy=tenpy.__version__))
        path=root/'inputs.json'
        if path.exists():
            if json.loads(path.read_text())!=inputs: raise ValueError('Resume source/config/environment mismatch')
        else: json_write(path,inputs)
        if (root/'result.json').exists():
            result=json.loads((root/'result.json').read_text())
            for name,sha in result['hashes'].items():
                if digest(root/name)!=sha: raise ValueError('Completed output hash mismatch')
            retire_committed_restart(root,result)
            return result
        geometry=sample(L,sid); v=geometry['interactions']*masks(L)[model_name]
        model_path=root/'model.pkl.gz'; receipt=root/'model_hash.json'
        if model_path.exists() and receipt.exists():
            if digest(model_path)!=json.loads(receipt.read_text())['sha256']:
                raise ValueError('Model checkpoint hash mismatch')
            model=load_checkpoint(model_path)
        else:
            if model_path.exists():
                # A killed model creation may leave an uncommitted payload; preserve it.
                model_path.rename(root/f'model-uncommitted-{time.time_ns()}.pkl.gz')
            if model_cache:
                cache=Path(model_cache)
                previous=json.loads((cache/'inputs.json').read_text())
                for key in ('L','sample_id','model','delta','omega','chi'):
                    if previous[key]!=inputs[key]: raise ValueError('Cached model inputs mismatch')
                source=cache/'model.pkl.gz'; expected=json.loads((cache/'model_hash.json').read_text())['sha256']
                if digest(source)!=expected: raise ValueError('Cached model hash mismatch')
                model=load_checkpoint(source)
                shutil.copyfile(source,model_path)
                if digest(model_path)!=expected: raise ValueError('Cached copy hash mismatch')
                # Match an independent rebuild's geometry and actual projected couplings.
                original=np.load(cache/'geometry.npz')
                with original:
                    if not np.array_equal(original['interactions'],v): raise ValueError('Cached geometry mismatch')
                sha=expected
                json_write(root/'cache-provenance.json',dict(source=str(cache),sha256=sha))
            else:
                model=build(v,delta,omega,L,chi); sha=checkpoint(model_path,model)
            json_write(receipt,dict(sha256=sha))
        geometry_path=root/'geometry.npz'
        if not geometry_path.exists():
            tmp=root/'geometry.tmp.npz'
            np.savez_compressed(tmp,points=geometry['points'],displacements=geometry['displacements'],
                                interactions=v,sites=model['sites'])
            os.replace(tmp,geometry_path)
        else:
            with np.load(geometry_path) as g:
                if not np.array_equal(g['points'],geometry['points']) or not np.array_equal(g['interactions'],v):
                    raise ValueError('Saved geometry mismatch')
        mpo,order=dmrg.build_mpo(model,interactions=v)
        identity=dict(inputs=inputs,model_sha256=digest(model_path),geometry_sha256=digest(geometry_path),
                      mpo='shared-Z-exact-v1')
        remaining=max(.001,seconds-(time.monotonic()-started)) if seconds else None
        engines,state,outcome=adaptive.optimize(root,mpo,model,order,dimensions,identity,
            seconds=remaining,invocation_sweeps=invocation_sweeps,memory_gib=memory_gib,
            checkpoint_interval=checkpoint_interval)
        if not state['complete']:
            return dict(state=outcome,completed=False,sweeps_per_start=[len(h) for h in state['history']],
                        final_D=state['D'],production_qualified=False)
        endpoints=[]
        for seed,engine in enumerate(engines):
            adaptive.disable_mixer(engine); psi=engine.psi
            psi.canonical_form()
            m,zz,x=dmrg.moments(psi,model,order)
            row=dict(seed=seed,D=state['D'],sweeps=len(state['history'][seed]),
                energy=float(mpo.H_MPO.expectation_value(psi)+model['constant']),
                observables=summarize(m,zz),transverse_mean=float(x.mean()),
                canonical_error=float(np.linalg.norm(psi.norm_test())),
                sweep_pass=bool(state['solver_pass']),solver_pass=bool(state['solver_pass']),
                bond_dimension_verified=False,D_comparison=None)
            final=root/f'start{seed}-final.pkl.gz'
            final_receipt=root/f'start{seed}-final-checkpoint.json'
            if final_state_retention=='full':
                if final.exists() and final_receipt.exists():
                    if digest(final)!=json.loads(final_receipt.read_text())['sha256']:
                        raise ValueError('Final checkpoint hash mismatch')
                else:
                    if final.exists(): final.rename(root/f'start{seed}-uncommitted-{time.time_ns()}.pkl.gz')
                    sha=checkpoint(final,psi,mpo.lat.mps_sites())
                    json_write(final_receipt,dict(sha256=sha))
            moments_path=root/f'start{seed}-final-moments.npz'
            tmp=root/f'start{seed}-final-moments.tmp.npz'
            np.savez_compressed(tmp,m=m,zz=zz,x=x); os.replace(tmp,moments_path)
            json_write(root/f'start{seed}-final.json',row); endpoints.append(row)
        agreed=compare(*endpoints,3*L*L); winner=int(endpoints[1]['energy']<endpoints[0]['energy'])
        samples=mps_samples(engines[winner].psi,model,order,2*pairs,20261117+sid)
        hist=overlap_histogram(samples); sampling_pass=sampling_check(hist,endpoints[winner]['observables'])
        tmp=root/'pq.tmp.npz'
        np.savez_compressed(tmp,counts=hist.pop('counts'),p=hist.pop('p'),q=hist.pop('q')); os.replace(tmp,root/'pq.npz')
        # Retain the full optimization record without keeping its tensor payloads.
        json_write(root/'optimization-history.json',state)
        resume=json.loads((root/'resume.json').read_text())
        result=dict(inputs=inputs,state=outcome,completed=True,endpoints=endpoints,start_comparison=agreed,
            sampling=hist,sampling_moment_check=sampling_pass,solver_pass=bool(all(e['solver_pass'] for e in endpoints) and agreed['passed']),
            rg_converged=False,production_qualified=False,production_selected=True,bond_dimension_verified=False,
            approximation_status='chi12 adaptive D; RG and D sensitivity unresolved' if chi==12 else 'adaptive validation fixture',
            decisions=state['decisions'],canonical_diagnostics=dict(interval_sweeps=5,
                definition='numpy.linalg.norm(psi.norm_test())',threshold=1e-8,
                starts=[[dict(sweep=int(k),D=v['D'],mixer=v['mixer'],canonical_error=v['canonical_error'])
                         for k,v in sorted(m.items(),key=lambda item:int(item[0]))]
                        for m in state['moments']]),geometry_sha256=array_hash(geometry['points']),
            attempts=geometry['attempts'],distance_version=geometry['distance_version'],
            block_cut_gaps=model['cut_gaps'],block_residuals=model['block_residuals'],
            seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resume_generation=resume['generation'],storage=dict(final_state_retention=final_state_retention,
                wavefunctions_retained=final_state_retention=='full',
                restart_retained=not bool(sampling_pass),
                resampling_requires_rerun=final_state_retention=='analysis' and bool(sampling_pass)))
        names=['inputs.json','model.pkl.gz','model_hash.json','geometry.npz','pq.npz',
               'optimization-history.json']
        for seed in (0,1):
            names += [f'start{seed}-final-moments.npz',f'start{seed}-final.json']
            if final_state_retention=='full':
                names += [f'start{seed}-final.pkl.gz',f'start{seed}-final-checkpoint.json']
        result['hashes']={name:digest(root/name) for name in names}
        # Verify analysis data before committing any retirement, including signed
        # histogram completeness. A failed sampling check keeps the restart pair.
        with np.load(root/'pq.npz') as saved:
            if int(saved['counts'].sum())!=pairs or len(saved['q'])!=3*L*L+1:
                raise ValueError('Incomplete signed overlap histogram')
        for seed in (0,1):
            with np.load(root/f'start{seed}-final-moments.npz') as saved:
                if not all(np.isfinite(saved[key]).all() for key in ('m','zz','x')):
                    raise ValueError('Nonfinite final moments')
        result['retired_artifacts']={}
        if sampling_pass:
            retired=[entry['file'] for entry in resume['files']]
            retired += [f"restart/{resume['generation']}.json"]
            if final_state_retention=='analysis': retired.append('resume.json')
            result['retired_artifacts']={name:digest(root/name) for name in retired}
        json_write(root/'result.json',result)
        retire_committed_restart(root,result)
        return result
    except BaseException:
        json_write(root/f'failure-{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        lock.close()
