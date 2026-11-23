"""Time-bounded actual L6 DMRG demo. This is not a production case runner."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from adaptive_d_policy import growth_reason, final_window_pass, locally_stable
from adaptive_d_runtime import disable_mixer, increase_dimension, one_sweep
from adaptive_memory import snapshot, growth_bytes, permitted, GIB


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-case',type=Path,required=True)
    p.add_argument('--mode',choices=['reduced','target'],required=True)
    p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--memory-gib',type=float,default=8.)
    a=p.parse_args()
    if not 120<=a.seconds<=900 or not 1<=a.memory_gib<=32:
        raise ValueError('Pilot requires 120..900 seconds and 1..32 GiB')
    if not a.output.is_absolute() or any(s in a.output.resolve().parts for s in ('Dropbox','CloudStorage')):
        raise ValueError('Explicit non-synced output required')
    root=a.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(int(a.memory_gib*GIB),int(a.memory_gib*GIB)))
    import numpy as np
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    from rydberg_kagome import dmrg
    from rydberg_kagome.results import digest, load_checkpoint, checkpoint, json_write
    started=time.monotonic(); deadline=started+a.seconds-60
    ladder=[16,32,64,128] if a.mode=='reduced' else [128,256,384,512]
    with (a.source_case/'inputs.json').open() as f: inputs=json.load(f)
    if (inputs['L'],inputs['model'],inputs['chi']) != (6,'full',12):
        raise ValueError('Demo requires L6 full chi12 model')
    with (a.source_case/'model_hash.json').open() as f: expected=json.load(f)['sha256']
    model_path=a.source_case/'model.pkl.gz'
    if digest(model_path)!=expected: raise RuntimeError('Source model hash mismatch')
    meta=dict(mode=a.mode,L=6,model='full',chi=12,delta=inputs['delta'],omega=inputs['omega'],
              sample_id=inputs['sample_id'],ladder=ladder,starts=[0,1],sweep_limit_per_start=120,
              seconds_limit=a.seconds,memory_limit_gib=a.memory_gib,mpo_builder='shared-Z-exact-v1',source_case=str(a.source_case),source_model_sha256=expected,
              production_campaign=False,production_qualified=False,rg_converged=False,
              source_scripts={str(q):digest(q) for q in (Path(__file__),Path(__file__).with_name('adaptive_d_policy.py'),Path(__file__).with_name('adaptive_d_runtime.py'),Path(__file__).with_name('adaptive_memory.py'),Path(dmrg.__file__))})
    json_write(root/'inputs.json',meta)
    def status(**kw):
        json_write(root/'status.json',dict(updated=time.time(),pid=os.getpid(),elapsed=time.monotonic()-started,**kw))
    status(state='loading_model',sweep=0,D=ladder[0])
    model=load_checkpoint(model_path)
    status(state='building_mpo',sweep=0,D=ladder[0])
    geometry_path=a.source_case/'geometry.npz'
    with (a.source_case/'result.json').open() as f: source_result=json.load(f)
    if digest(geometry_path)!=source_result['hashes']['geometry.npz']:
        raise RuntimeError('Source interaction hash mismatch')
    with np.load(geometry_path) as g: interactions=g['interactions']
    mpo,order=dmrg.build_mpo(model,interactions=interactions)
    json_write(root/'mpo-profile.json',dict(bond_dimensions=list(map(int,mpo.H_MPO.chi)),**snapshot()))
    sites=mpo.lat.mps_sites(); n=model['sites'].size
    status(state='initializing_states',sweep=0,D=ladder[0])
    psis=[dmrg.initial_state(mpo,seed) for seed in (0,1)]
    histories=[[],[]]; moments=[{},{}]; decisions=[]; engines=[]
    def make_engine(psi,cap,activation):
        engine=TwoSiteDMRGEngine(psi,mpo,dict(mixer=True,
            mixer_params=dict(amplitude=1e-3,decay=2.,disable_after=activation),
            N_sweeps_check=1,trunc_params=dict(chi_max=cap,svd_min=1e-12)))
        engine.pre_run_initialize()
        return engine
    cap=ladder[0]; level=0; stage_start=0; disagree={}; passed=False; completed=0; memory_blocked=False
    status(state='initializing_engines',sweep=0,D=cap)
    engines=[make_engine(psi,cap,15) for psi in psis]
    def magnetizations(psi):
        m=np.zeros(n); physical=np.asarray([model['sites'][b] for b in order])
        for j in range(12): m[physical[:,j]]=np.real(psi.expectation_value(f'Z{j}'))
        return m
    with (root/'trace.jsonl').open('x',buffering=1) as trace:
        for sweep in range(1,121):
            if time.monotonic()>=deadline: break
            mem=snapshot()
            if mem['available_bytes']<16*GIB or mem['rss_bytes']>a.memory_gib*GIB*.85:
                memory_blocked=True; break
            for seed,engine in enumerate(engines):
                if time.monotonic()>=deadline: break
                active_limit=15 if stage_start==0 else 5
                if sweep-stage_start>active_limit and engine.mixer is not None: disable_mixer(engine)
                mixed=engine.mixer is not None
                status(state='sweeping',sweep=sweep,start=seed,D=cap,decisions=decisions)
                t=time.monotonic(); energy,_=one_sweep(engine)
                h=dict(start=seed,sweep=sweep,D=cap,mixer=mixed,energy=float(energy+model['constant']),
                       max_bond=int(max(psis[seed].chi)),discarded_weight=float(engine.sweep_stats['max_trunc_err'][-1]),
                       seconds=time.monotonic()-t)
                if not np.isfinite([h['energy'],h['discarded_weight']]).all():
                    raise FloatingPointError('Nonfinite sweep diagnostics')
                histories[seed].append(h); trace.write(json.dumps(h)+'\n')
                status(state='running',sweep=sweep,start=seed,D=cap,decisions=decisions)
                if sweep%5==0:
                    status(state='measuring',sweep=sweep,start=seed,D=cap,decisions=decisions)
                    m=magnetizations(psis[seed])
                    moments[seed][sweep]=dict(D=cap,mixer=mixed,m=m.tolist(),qea=float(np.mean(m*m)))
                    json_write(root/f'start{seed}-diagnostics.json',dict(history=histories[seed],moments=moments[seed]))
            else:
                completed=sweep
                if sweep%5==0:
                    de=abs(histories[0][-1]['energy']-histories[1][-1]['energy'])/n
                    dq=abs(moments[0][sweep]['qea']-moments[1][sweep]['qea'])
                    disagree[sweep]=(de>=1e-7 or dq>=1e-3) and all(
                        locally_stable(histories[s],moments[s],sweep,n) for s in (0,1))
                    ok=[final_window_pass(histories[s],moments[s],sweep,n,
                        float(np.linalg.norm(psis[s].norm_test()))) for s in (0,1)]
                    if all(ok) and de<1e-7 and dq<1e-3: passed=True; break
                if sweep in (20,40,60):
                    reasons=[growth_reason(histories[s],moments[s],cap,sweep,n) for s in (0,1)]
                    if sweep==60 and disagree.get(40) and disagree.get(60):
                        for reason in reasons: reason.append('persistent_start_disagreement')
                    grow=any(reasons) and level+1<len(ladder)
                    event=dict(sweep=sweep,D=cap,reasons=reasons,next_D=ladder[level+1] if grow else cap)
                    decisions.append(event); trace.write(json.dumps(dict(decision=event))+'\n')
                    status(state='checkpointing',sweep=sweep,D=cap,decisions=decisions)
                    # Persistent references avoid copying huge MPO site operators.
                    for seed,engine in enumerate(engines):
                        disable_mixer(engine)
                        file=root/f'start{seed}-sweep{sweep}-D{cap}.pkl.gz'
                        sha=checkpoint(file,psis[seed],sites)
                        restored=load_checkpoint(file,sites)
                        delta=float(np.max(abs(magnetizations(restored)-magnetizations(psis[seed]))))
                        if delta>1e-12: raise RuntimeError('Checkpoint moment mismatch')
                        json_write(file.with_suffix('.receipt.json'),dict(sha256=sha,sweep=sweep,D=cap,start=seed,moment_restore_error=delta))
                        del restored
                    if grow:
                        mem=snapshot()
                        forecast=growth_bytes(mem['rss_bytes'],cap,ladder[level+1],mem['peak_rss_bytes'])
                        event['memory_admission']=dict(**mem,forecast_bytes=forecast)
                        if not permitted(mem['rss_bytes'],forecast,mem['available_bytes'],a.memory_gib*GIB*.85):
                            event['growth_executed']=False
                            trace.write(json.dumps(dict(decision_outcome=event))+'\n')
                            memory_blocked=True
                            status(state='memory_deferred',sweep=sweep,D=cap,decisions=decisions)
                            break
                        level+=1; cap=ladder[level]; stage_start=sweep
                        for engine in engines: increase_dimension(engine,cap)
                        event['growth_executed']=True
                        trace.write(json.dumps(dict(decision_outcome=event))+'\n')
                        status(state='dimension_increased',sweep=sweep,D=cap,decisions=decisions)
                continue
            break
    for seed in (0,1):
        json_write(root/f'start{seed}-diagnostics.json',dict(history=histories[seed],moments=moments[seed]))
    outcome='memory_deferred' if memory_blocked else 'converged' if passed else 'sweep_limit' if completed==120 else 'time_limit'
    json_write(root/'result.json',dict(**meta,state=outcome,solver_pass=passed,sweeps_per_start=[len(h) for h in histories],
        final_D=cap,decisions=decisions,seconds=time.monotonic()-started,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        sampling_performed=False,bond_dimension_verified=False))
    status(state=outcome,sweep=completed,D=cap,decisions=decisions)


if __name__=='__main__': main()
