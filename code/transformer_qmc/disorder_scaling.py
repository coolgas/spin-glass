"""Sampled-guide, independent-population CT GFMC disorder scaling pilot.

Training runs on CUDA; frozen-guide projectors run in CPU workers. Every
disorder has equal weight. Results remain unqualified until sensitivity tests.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .collective_campaign import write,checkpoint
from .correlation_lengths import sha,length
from .independent_replica_projector import spectrum_from_moments
from .physics import make_af_sample
from .production_protocol import geometry,identity
from .projector import propagate,rates_and_local
from .sampled_simple_guides import train
from .simple_guides import Jastrow,RBM,SimpleGuide


def observables(m,p,L):
    """Products involve distinct physical populations, including q and q2."""
    pairs=list(combinations(range(4),2))
    q=np.mean([np.mean(m[a]*m[b]) for a,b in pairs])
    q2=np.mean([np.mean(p[a]*p[b]) for a,b in pairs])
    lattice=geometry(L)
    spectrum=spectrum_from_moments(m,p,lattice)[[0]+lattice['kmin_indices']]
    return dict(q=float(q),q2=float(q2),spectrum=spectrum.tolist())


def project_task(task,model_path,output):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        return _project(task,Path(model_path),Path(output))


def _project(task,model_path,root):
    key=identity(task)[:24];dest=root/'runs'/(key+'.json');cp=root/'checkpoints'/(key+'.npz')
    provenance=dict(model_sha256=sha(model_path),sources={n:sha(Path(__file__).with_name(n)) for n in
        ['disorder_scaling.py','projector.py','simple_guides.py','independent_replica_projector.py','physics.py']})
    if dest.exists():
        record=json.loads(dest.read_text());assert record['provenance']==provenance
        assert sha(dest.with_suffix('.npz'))==record['npz_sha256'];return record
    L=task['L'];n=3*L*L;w=task['walkers'];dt=task['delta']
    ham=make_af_sample(L,task['sample_id']).hamiltonian(.15)
    model=Jastrow(n) if task['kind']=='jastrow' else RBM(n,2)
    model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=False)['model'])
    guide=SimpleGuide(model,sweeps=200)
    rng=np.random.default_rng(int(identity(task)[:16],16))
    equil=round(task['equil']/dt);forward=round(task['forward']/dt)
    started=time.monotonic();before=0.
    if cp.exists():
        with np.load(cp) as data:
            meta=json.loads(str(data['metadata']));assert meta['task']==task and meta['provenance']==provenance
            rng.bit_generator.state=meta['rng'];step=meta['step'];before=meta['seconds']
            states=data['states'].copy();shifts=data['shifts'].copy();tagged=data['tagged'].copy()
            ancestors=data['ancestors'].copy();ms=list(data['m']);ps=list(data['p'])
            ess=list(data['ess']);lineage=list(data['lineage']);weight=list(data['weight'])
    else:
        states=(guide.sample(4*w,int(rng.integers(2**32))).reshape(4,w,n) if task['start']=='guide'
                else (1-2*rng.integers(2,size=(4,w,n))).astype(np.int8))
        shifts=np.array([rates_and_local(ham,task['gamma'],guide,s[:,None,:])[1].mean() for s in states])
        tagged=states.astype(float);ancestors=np.tile(np.arange(w),(4,1))
        ms=[];ps=[];ess=[];lineage=[];weight=[];step=0
    total=equil+task['cycles']*forward
    def save():
        cp.parent.mkdir(parents=True,exist_ok=True)
        checkpoint(cp,dict(states=states,shifts=shifts,tagged=tagged,ancestors=ancestors,
            m=np.array(ms),p=np.array(ps),ess=np.array(ess),lineage=np.array(lineage),weight=np.array(weight)),
            dict(task=task,provenance=provenance,rng=rng.bit_generator.state,step=step,
                 seconds=before+time.monotonic()-started))
        write(root/'progress'/(key+'.json'),dict(task=task,step=step,total=total,cycles=len(ms)))
    while step<total:
        weights=[]
        for r in range(4):
            s,children,shifts[r],_,_,we=propagate(ham,task['gamma'],guide,states[r,:,None,:],shifts[r],dt,rng)
            states[r]=s[:,0];weights.append(we)
            if step>=equil:ancestors[r]=ancestors[r,children]
        step+=1;weight.append(weights)
        boundary=step==equil or (step>equil and (step-equil)%forward==0)
        if step>equil and boundary:
            m=[];p=[];e=[];la=[]
            for r in range(4):
                count=np.bincount(ancestors[r],minlength=w);prob=count/w
                m.append(prob@tagged[r]);p.append(tagged[r].T@(prob[:,None]*tagged[r]))
                e.append(float(w*w/(count@count)));la.append(float(count.max()/w))
            ms.append(m);ps.append(p);ess.append(e);lineage.append(la)
        if boundary:
            tagged=states.astype(float);ancestors=np.tile(np.arange(w),(4,1))
        if boundary or step%100==0:save()
    ms=np.array(ms);ps=np.array(ps)
    result=observables(ms.mean(0),ps.mean(0),L)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest.with_suffix('.npz'),magnetization_cycles=ms,pair_cycles=ps,
        descendant_ess=ess,largest_lineage=lineage,weight_ess=weight)
    record=dict(task=task,provenance=provenance,**result,
        minimum_descendant_ess=float(np.min(ess)),ancestry_alarm=bool(np.min(ess)<8 or np.max(lineage)>.5),
        seconds=before+time.monotonic()-started,npz_sha256=sha(dest.with_suffix('.npz')),
        ground_state_qualified=False)
    write(dest,record);return record


def reduce_results(root):
    records=[json.loads(p.read_text()) for p in (root/'projector/runs').glob('*.json')]
    groups={}
    for r in records:
        t=r['task'];groups.setdefault((t['L'],t['gamma'],t['kind'],t['audit']),{}).setdefault(t['sample_id'],[]).append(r)
    summary=[]
    for (L,g,kind,audit),disorders in sorted(groups.items()):
        # Only completed repeat groups enter a displayed partial mean. Explicit
        # IDs and completion counts prevent partial coverage looking final.
        complete={sid:rs for sid,rs in disorders.items() if len(rs)==2}
        if not complete:continue
        ids=sorted(complete)
        sp=np.array([np.mean([r['spectrum'] for r in complete[s]],axis=0) for s in ids])
        qs=np.array([np.mean([r['q'] for r in complete[s]]) for s in ids])
        q2=np.array([np.mean([r['q2'] for r in complete[s]]) for s in ids])
        rng=np.random.default_rng(871);indices=rng.integers(len(ids),size=(2000,len(ids)))
        boot=length(sp[indices].mean(1),L)/L;valid=np.isfinite(boot);xi=length(sp.mean(0),L)/L
        summary.append(dict(L=L,gamma=g,kind=kind,audit=audit,disorders=ids,
            spectrum=sp.mean(0).tolist(),q=float(qs.mean()),q2=float(q2.mean()),
            q_interval=np.quantile(qs[indices].mean(1),[.025,.975]).tolist() if len(ids)>1 else None,
            xi_over_L=float(xi) if np.isfinite(xi) else None,
            xi_interval=np.quantile(boot[valid],[.025,.975]).tolist() if valid.any() and len(ids)>1 else None,
            invalid_bootstrap_fraction=float(1-valid.mean()),
            ancestry_alarms=sum(r['ancestry_alarm'] for rs in complete.values() for r in rs),
            ground_state_qualified=False))
    write(root/'review.json',dict(rows=summary,completed_runs=len(records),
        uncertainty='Exploratory whole-disorder bootstrap of repeat means; D=8 pilot, D=2 audits. No precision qualification.',
        q_definition='N^-1 sum_i <Zi>^2 includes field polarization',
        xi_definition='ratio of disorder-mean spectra; |kmin|=2pi/(sqrt(3)L); invalid ratios retained'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--hours',type=float,default=12)
    p.add_argument('--sizes',type=int,nargs='+',default=[4,6])
    p.add_argument('--fields',type=float,nargs='+',default=[.6,.65,.7,.85,1.])
    p.add_argument('--disorders',type=int,nargs='+',default=list(range(60000,60008)))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if any(L<3 for L in a.sizes):raise ValueError('This pilot requires L>=3; use exact L2 validation separately')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    manifest=dict(sizes=a.sizes,fields=a.fields,disorders=a.disorders,h=.15,kinds=['jastrow','rbm_2'],
        seeds=[0,1],updates_per_fit=1200,selection='lowest frozen mean energy across random/up/down diagnostics',
        training='parallel tempering and decreasing field; no enumeration',
        audit_ids=a.disorders[:2],sources={n:sha(Path(__file__).with_name(n)) for n in
            ['disorder_scaling.py','sampled_simple_guides.py','simple_guides.py','projector.py',
             'physics.py','production_protocol.py','independent_replica_projector.py','correlation_lengths.py',
             'report_disorder_scaling.py']})
    path=a.output/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:write(path,manifest)
    training=a.output/'training';training.mkdir(exist_ok=True)
    deadline=time.monotonic()+3600*a.hours;futures={};completed=0;fits=0;paused=False
    # Spawn isolates CUDA initialization from CPU projector workers.
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        def harvest(block=False):
            nonlocal completed
            for future in list(futures):
                if block or future.done():
                    task=futures.pop(future)
                    try:future.result()
                    except Exception as error:
                        write(a.output/('failure_'+identity(task)[:16]+'.json'),dict(task=task,error=repr(error)));raise
                    completed+=1
            reduce_results(a.output)
        for L,g,sid,kind in [(L,g,s,k) for L in a.sizes for g in a.fields
                            for s in a.disorders for k in ['jastrow','rbm_2']]:
            if time.monotonic()>=deadline:paused=True;break
            candidates=[]
            stages=len([v for v in sorted(set([1.5,.75,g]),reverse=True) if v>=g])
            settings=argparse.Namespace(device='cuda',chains=256,temperatures=8,steps=1200//stages,
                sweeps_per_step=2,diagnostic_sweeps=200,inputs=a.output/'inputs')
            for seed in [0,1]:
                task=dict(L=L,gamma=g,sample_id=sid,kind=kind,seed=seed,mode='anneal')
                try:r=train(task,settings,training,deadline)
                except Exception as error:
                    write(a.output/('failure_training_'+identity(task)[:16]+'.json'),dict(task=task,error=repr(error)));raise
                if r['status']!='completed':paused=True;break
                candidates.append(r);fits+=1
            if paused:break
            best=min(candidates,key=lambda r:np.mean([d['energy'] for d in r['diagnostics']]))
            name='L{L}_s{sample_id}_g{gamma:g}_{kind}_{mode}_seed{seed}'.format(**best['task']).replace('.','p')
            model_path=training/(name+'.pt');assert sha(model_path)==best['checkpoint_sha256']
            write(a.output/'selections'/(name+'.json'),dict(selected=best['task'],
                candidate_energies=[dict(seed=r['task']['seed'],energy=float(np.mean([d['energy'] for d in r['diagnostics']]))) for r in candidates],
                checkpoint_sha256=sha(model_path)))
            configs=[('main',512,16.,16.,'guide')]
            if sid in a.disorders[:2]:configs += [('population',1024,16.,16.,'guide'),('time',512,32.,32.,'guide'),('start',512,16.,16.,'uniform')]
            for audit,w,eq,fw,start in configs:
                for repeat in [0,1]:
                    t=dict(L=L,gamma=g,sample_id=sid,kind=kind,training_seed=best['task']['seed'],
                        audit=audit,walkers=w,equil=eq,forward=fw,start=start,delta=.1,cycles=8,repeat=repeat)
                    futures[pool.submit(project_task,t,str(model_path),str(a.output/'projector'))]=t
            harvest()
            write(a.output/'status.json',dict(state='running',completed_fits=fits,completed_runs=completed,
                queued_or_running_runs=len(futures),last_context=[L,g,sid,kind]))
        harvest(block=True)
    from .report_disorder_scaling import report
    report(a.output)
    write(a.output/'status.json',dict(state='paused_training_budget' if paused else 'completed',
        completed_fits=fits,completed_runs=completed))


if __name__=='__main__':main()
