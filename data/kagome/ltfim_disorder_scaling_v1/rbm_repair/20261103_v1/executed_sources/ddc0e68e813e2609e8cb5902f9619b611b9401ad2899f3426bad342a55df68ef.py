"""Bounded strong-field repair validation. Run computations on AutoDL only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from .collective_campaign import label, write
from .correlation_lengths import sha
from .physics import basis, exact_reference, make_af_sample
from .projector import propagate, rates_and_local
from .sampled_simple_guides import train
from .simple_guides import Jastrow, RBM, SimpleGuide


def training_case(case, output):
    torch.set_num_threads(1)
    root = Path(output)
    task = {k: case[k] for k in ['L', 'gamma', 'sample_id', 'kind', 'seed', 'mode']}
    inputs = root/'inputs'
    if task['L'] == 2:
        ham = make_af_sample(2, task['sample_id']).hamiltonian(.15)
        truth = exact_reference(ham, task['gamma'])
        gp = inputs/'guides'/(label(task['gamma'], task['sample_id'])+'.npz')
        gp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gp, exact_amplitude=truth['amplitude'])
    settings = argparse.Namespace(device='cuda', chains=256, temperatures=8,
        steps=case['steps'], sweeps_per_step=2, diagnostic_sweeps=100, inputs=inputs)
    dest = root/'training'
    dest.mkdir(parents=True, exist_ok=True)
    return train(task, settings, dest, time.monotonic()+1800)


def project_case(case, output):
    """Four separately branching populations; save raw tags and descendants."""
    torch.set_num_threads(1)
    root = Path(output)
    key = case['name']
    completed=root/'projector'/(key+'.json')
    if completed.exists():
        record=json.loads(completed.read_text())
        assert record['case']==case and record['checkpoint_sha256']==sha(Path(case['checkpoint']))
        assert record['raw_sha256']==sha(completed.with_suffix('.npz'))
        return {k:v for k,v in record.items() if k!='numerical_trace'}
    ham = make_af_sample(case['L'], case['sample_id']).hamiltonian(.15)
    model = Jastrow(ham.n_sites) if case['kind'] == 'jastrow' else RBM(ham.n_sites, 2)
    model.load_state_dict(torch.load(case['checkpoint'], map_location='cpu', weights_only=False)['model'])
    guide = SimpleGuide(model, sweeps=200)
    rng = np.random.default_rng(case['seed'])
    w, n, gamma, dt = case['walkers'], ham.n_sites, case['gamma'], .05
    states = (guide.sample(4*w, int(rng.integers(2**32))).reshape(4,w,n)
              if case['start'] == 'guide' else (1-2*rng.integers(2,size=(4,w,n))).astype(np.int8))
    shifts = np.array([rates_and_local(ham,gamma,guide,s[:,None,:])[1].mean() for s in states])
    energy, growth, weight, descendants, lineages, tags, counts, traces = [], [], [], [], [], [], [], []
    equil, forward = round(case['equil']/dt), round(case['forward']/dt)
    ancestors = np.tile(np.arange(w),(4,1))
    tagged = states.copy()
    started = time.monotonic()
    for step in range(equil+case['cycles']*forward):
        e, g, ess, diagnostic = [], [], [], []
        for r in range(4):
            diag = {}
            s, children, shifts[r], mixed, eg, we = propagate(ham,gamma,guide,states[r,:,None,:],shifts[r],dt,rng,diag)
            states[r] = s[:,0]
            if step >= equil:
                ancestors[r] = ancestors[r,children]
            e.append(mixed);g.append(eg);ess.append(we);diagnostic.append(diag)
        energy.append(e);growth.append(g);weight.append(ess);traces.append(diagnostic)
        boundary = step+1 == equil or (step+1 > equil and (step+1-equil) % forward == 0)
        if step+1 > equil and boundary:
            count = np.array([np.bincount(a,minlength=w) for a in ancestors])
            tags.append(tagged.copy());counts.append(count)
            descendants.append(w*w/(count*count).sum(axis=1));lineages.append(count.max(axis=1)/w)
        if boundary:
            tagged=states.copy();ancestors=np.tile(np.arange(w),(4,1))
        if (step+1) % 20 == 0:
            write(root/'projector'/(key+'_progress.json'),dict(case=case,step=step+1,total=equil+case['cycles']*forward))
    tags, counts = np.array(tags), np.array(counts)
    local = np.einsum('crw,crwn->crn',counts/w,tags).mean(0)
    # Convert int8 tags before matrix products to prevent integer overflow.
    pairs = np.array([[s.astype(float).T@((c/w)[:,None]*s) for s,c in zip(t,co)]
                      for t,co in zip(tags,counts)]).mean(0)
    q = np.mean([np.mean(local[a]*local[b]) for a,b in combinations(range(4),2)])
    q2 = np.mean([np.mean(pairs[a]*pairs[b]) for a,b in combinations(range(4),2)])
    record = dict(case=case,energy=float(np.mean(energy[equil:])),growth_energy=float(np.mean(growth[equil:])),
        mean_q=float(q),q2=float(q2),minimum_weight_ess=float(np.min(weight)),
        minimum_descendant_ess=float(np.min(descendants)),largest_lineage=float(np.max(lineages)),
        seconds=time.monotonic()-started,ground_state_qualified=False,
        checkpoint_sha256=sha(Path(case['checkpoint'])),
        log_weight_min=min(d['min_log_weight'] for block in traces for d in block),
        log_weight_max=max(d['max_log_weight'] for block in traces for d in block))
    record['projection_sources']={name:sha(Path(__file__).with_name(name)) for name in
                                 ['strong_field_validation.py','projector.py','simple_guides.py','physics.py']}
    record['numerics_sha256']=sha(Path(__file__).parent.parent/'qmc_numerics.py')
    record['ancestry_alarm'] = record['minimum_descendant_ess'] < 8 or record['largest_lineage'] > .5
    if case['L'] == 2:
        truth = exact_reference(ham,gamma)
        record['exact'] = {k:float(truth[k]) for k in ['energy','q_mean','q2','residual']}
        record['energy_error'] = record['energy']-truth['energy']
    dest = root/'projector'/(key+'.npz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,tags=tags,counts=counts,energy=energy,growth=growth,
                        weight_ess=weight,descendant_ess=descendants,largest_lineage=lineages,
                        final_states=states,shifts=shifts,rng=json.dumps(rng.bit_generator.state))
    record['raw_sha256'] = sha(dest)
    record['numerical_trace'] = traces
    write(dest.with_suffix('.json'),record)
    return {k:v for k,v in record.items() if k!='numerical_trace'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['training','projection'],required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--projection-kinds',nargs='+',default=['rbm_2','jastrow'],choices=['rbm_2','jastrow'])
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    ctx=mp.get_context('spawn')
    cases=[dict(L=L,gamma=g,kind=k,steps=steps,sample_id=60000,seed=0,mode=mode) for L,g,k,steps,mode in
           [(2,4.,'rbm_2',300,'direct'),(6,4.,'rbm_2',400,'direct'),
            (6,4.,'jastrow',400,'direct'),(6,1.,'rbm_2',400,'anneal')]]
    if args.phase=='training':
        write(root/'protocol.json',dict(cases=cases,h=.15,boundary='periodic kagome',N='3 L^2',
              bonds='K iid U(.5,1.5); SeedSequence([20261027,L,60000,1]); J=-K',
              sources={name:sha(Path(__file__).with_name(name)) for name in
                       ['strong_field_validation.py','sampled_simple_guides.py','simple_guides.py','projector.py','physics.py']},
              numerics_sha256=sha(Path(__file__).parent.parent/'qmc_numerics.py'),
              scope='One prespecified disorder; controls only, no disorder ensemble qualification'))
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx) as pool:
            futures=[pool.submit(training_case,c,str(root)) for c in cases]
            results=[]
            for case,future in zip(cases,futures):
                try:
                    r=future.result();results.append(dict(case=case,status=r['status'],sampling_alarms=r.get('sampling_alarms')))
                except Exception as error:
                    results.append(dict(case=case,error=repr(error)))
                write(root/'training_summary.json',dict(results=results))
        if any('error' in r or r['status']!='completed' for r in results):
            raise RuntimeError('Incomplete training; see retained training_summary.json')
    else:
        projections=[]
        prefix='projection' if set(args.projection_kinds)=={'rbm_2','jastrow'} else 'projection_'+'_'.join(args.projection_kinds)
        for c in cases:
            if c['gamma']!=4. or c['kind'] not in args.projection_kinds:continue
            stem=('L{L}_s{sample_id}_g{gamma:g}_{kind}_{mode}_seed{seed}'.format(**c)).replace('.','p')
            checkpoint=root/'training'/(stem+'.pt')
            for repeat in [0,1]:
                configs=[('main',128,.5,.5,'guide'),('start',128,.5,.5,'uniform'),
                         ('population',256,.5,.5,'guide'),('time',128,1.,1.,'guide')]
                for audit,w,eq,fw,start in configs:
                    projections.append(dict(L=c['L'],gamma=4.,sample_id=60000,kind=c['kind'],
                        name=f"L{c['L']}_{c['kind']}_{audit}_r{repeat}",checkpoint=str(checkpoint),
                        walkers=w,equil=eq,forward=fw,cycles=4,start=start,seed=8700+repeat,repeat=repeat,audit=audit))
        results=[]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx) as pool:
            futures=[pool.submit(project_case,c,str(root)) for c in projections]
            for case,future in zip(projections,futures):
                try:results.append(future.result())
                except Exception as error:results.append(dict(case=case,error=repr(error)))
                write(root/(prefix+'_summary.json'),dict(results=results))
        if any('error' in r for r in results):
            raise RuntimeError('Projection failed; see retained projection_summary.json')
    prefix='training' if args.phase=='training' else prefix
    write(root/(prefix+'_complete.json'),dict(completed=len(results),ground_state_qualified=False))


if __name__=='__main__':main()
