"""Two-start adaptive solver with immutable, manifest-committed restart state."""
import json
import time
import uuid
from pathlib import Path
import numpy as np
from . import dmrg
from .results import checkpoint, load_checkpoint, digest, json_write
from .adaptive_policy import growth_reason, locally_stable, final_window_pass
from .adaptive_runtime import disable_mixer, increase_dimension, one_sweep
from .adaptive_memory import snapshot, permitted, growth_bytes, GIB


def engine_state(engine):
    return dict(resume=engine.get_resume_data(),options=engine.options.as_dict(),
                mixer=engine.mixer,lanczos=engine.lanczos_params.as_dict(),
                sweep_stats=engine.sweep_stats,update_stats=engine.update_stats,
                entropy=engine._entropy_approx)


def restore_engine(record,mpo):
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    engine=TwoSiteDMRGEngine(record['resume']['psi'],mpo,record['options'],resume_data=record['resume'])
    # Do not pre_run_initialize(): it reactivates/reset the saved mixer.
    engine.mixer=record['mixer']
    engine.options.touch('mixer')
    engine.lanczos_params.update(record['lanczos'])
    engine.sweep_stats=record['sweep_stats']; engine.update_stats=record['update_stats']
    engine._entropy_approx=record['entropy']
    return engine


def save_restart(root, engines, state, identity):
    root=Path(root); folder=root/'restart'; folder.mkdir(exist_ok=True)
    generation=uuid.uuid4().hex
    files=[]
    for seed,engine in enumerate(engines):
        path=folder/f'{generation}-start{seed}.pkl.gz'
        sha=checkpoint(path,engine_state(engine),engine.psi.sites)
        files.append(dict(file=str(path.relative_to(root)),sha256=sha))
    record=dict(schema=1,generation=generation,identity=identity,files=files,state=state)
    # Immutable complete manifest first; pointer changes only after both states verify.
    json_write(folder/f'{generation}.json',record)
    json_write(root/'resume.json',record)
    return record


def load_restart(root,mpo,identity):
    root=Path(root)
    with (root/'resume.json').open() as f: record=json.load(f)
    if record['schema']!=1 or record['identity']!=identity:
        raise ValueError('Adaptive resume identity mismatch')
    for entry in record['files']:
        path=(root/entry['file']).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path)!=entry['sha256']:
            raise ValueError('Adaptive checkpoint hash mismatch')
    engines=[restore_engine(load_checkpoint(root/e['file'],mpo.lat.mps_sites()),mpo) for e in record['files']]
    state=record['state']
    for seed,engine in enumerate(engines):
        if engine.sweeps!=len(state['history'][seed]):
            raise ValueError('Resume sweep count mismatch')
        if engine.trunc_params['chi_max']!=state['D']:
            raise ValueError('Resume dimension mismatch')
    return engines,state


def physical_m(psi,model,order):
    m=np.zeros(model['sites'].size)
    physical=np.asarray([model['sites'][b] for b in order])
    for a in range(12): m[physical[:,a]]=np.real(psi.expectation_value(f'Z{a}'))
    return m


def optimize(root,mpo,model,order,dimensions,identity,seconds=None,
             invocation_sweeps=None,memory_gib=8.):
    """At most 120 cumulative sweeps/start; invocation budget only pauses work.

    Checkpoints at every completed start-sweep preserve partially completed pairs.
    Generation retention is explicit: no older scientific checkpoint is deleted.
    """
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    root=Path(root); started=time.monotonic(); deadline=started+seconds if seconds else float('inf')
    if (root/'resume.json').exists():
        engines,state=load_restart(root,mpo,identity)
    else:
        state=dict(D=dimensions[0],level=0,stage_start=0,history=[[],[]],moments=[{},{}],
                   decisions=[],disagreement={},handled=[],solver_pass=False,complete=False)
        engines=[]
        for seed in (0,1):
            psi=dmrg.initial_state(mpo,seed)
            engine=TwoSiteDMRGEngine(psi,mpo,dict(mixer=True,N_sweeps_check=1,
                mixer_params=dict(amplitude=1e-3,decay=2.,disable_after=15),
                trunc_params=dict(chi_max=state['D'],svd_min=1e-12)))
            engine.pre_run_initialize(); engines.append(engine)
        save_restart(root,engines,state,identity)
    initial=sum(len(h) for h in state['history'])
    n=model['sites'].size; reason=None
    def save(): save_restart(root,engines,state,identity)
    def progress(phase,**extra):
        json_write(root/'adaptive-status.json',dict(updated=time.time(),phase=phase,D=state['D'],
            sweeps=[len(h) for h in state['history']],**extra))
    while not state['complete']:
        counts=[len(h) for h in state['history']]
        sweep=min(counts)
        # Process paired decision once; checkpointing afterwards makes resume idempotent.
        if counts[0]==counts[1] and sweep>0 and sweep not in state['handled']:
            state['handled'].append(sweep)
            if sweep%5==0:
                moments=[{int(k):v for k,v in m.items()} for m in state['moments']]
                de=abs(state['history'][0][-1]['energy']-state['history'][1][-1]['energy'])/n
                dq=abs(moments[0][sweep]['qea']-moments[1][sweep]['qea'])
                state['disagreement'][str(sweep)]=(de>=1e-7 or dq>=1e-3) and all(
                    locally_stable(state['history'][s],moments[s],sweep,n) for s in (0,1))
                stable=[final_window_pass(state['history'][s],moments[s],sweep,n,
                    float(np.linalg.norm(engines[s].psi.norm_test()))) for s in (0,1)]
                if all(stable) and de<1e-7 and dq<1e-3:
                    state['solver_pass']=True; state['complete']=True
                if sweep in (20,40,60) and not state['complete']:
                    reasons=[growth_reason(state['history'][s],moments[s],state['D'],sweep,n) for s in (0,1)]
                    if sweep==60 and state['disagreement'].get('40') and state['disagreement'].get('60'):
                        for row in reasons: row.append('persistent_start_disagreement')
                    grow=any(reasons) and state['level']+1<len(dimensions)
                    event=dict(sweep=sweep,D=state['D'],reasons=reasons,next_D=dimensions[state['level']+1] if grow else state['D'])
                    state['decisions'].append(event)
                    if grow:
                        state['pending_growth']=event['next_D']
            if sweep>=120: state['complete']=True
            save()
        if state['complete']: break
        if time.monotonic()>=deadline or (invocation_sweeps is not None and sum(counts)-initial>=invocation_sweeps):
            reason='paused'; break
        if 'pending_growth' in state:
            mem=snapshot(); target=state['pending_growth']
            forecast=growth_bytes(mem['rss_bytes'],state['D'],target,mem['peak_rss_bytes'])
            if not permitted(mem['rss_bytes'],forecast,mem['available_bytes'],memory_gib*GIB*.85):
                reason='memory_deferred'; break
            for engine in engines: increase_dimension(engine,target)
            state['D']=target; state['level']+=1; state['stage_start']=sweep
            state['decisions'][-1]['growth_executed']=True
            del state['pending_growth']; save()
        mem=snapshot()
        if mem['available_bytes']<16*GIB or mem['rss_bytes']>memory_gib*GIB*.85:
            reason='memory_deferred'; break
        seed=0 if counts[0]<=counts[1] else 1
        engine=engines[seed]; sweep=counts[seed]+1
        limit=15 if state['stage_start']==0 else 5
        if sweep-state['stage_start']>limit and engine.mixer is not None: disable_mixer(engine)
        mixed=engine.mixer is not None
        progress('sweeping',start=seed,next_sweep=sweep)
        t=time.monotonic(); energy,_=one_sweep(engine)
        row=dict(sweep=sweep,D=state['D'],mixer=mixed,energy=float(energy+model['constant']),
                 max_bond=int(max(engine.psi.chi)),discarded_weight=float(engine.sweep_stats['max_trunc_err'][-1]),
                 seconds=time.monotonic()-t)
        if not np.isfinite([row['energy'],row['discarded_weight']]).all(): raise FloatingPointError('Nonfinite sweep')
        state['history'][seed].append(row)
        if sweep%5==0:
            m=physical_m(engine.psi,model,order)
            if not np.isfinite(m).all(): raise FloatingPointError('Nonfinite moments')
            state['moments'][seed][str(sweep)]=dict(D=state['D'],mixer=mixed,m=m.tolist(),qea=float(np.mean(m*m)))
        progress('checkpointing'); save()
    reason=reason or ('converged' if state['solver_pass'] else 'sweep_limit')
    progress(reason)
    return engines,state,reason
