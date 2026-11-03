"""Checkpointed matched L2 GFMC proposal benchmark, with finite-time references."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply, LinearOperator, eigsh
from threadpoolctl import threadpool_limits

from kagome_qea.exact import operator_matrix
from .collective_projector import SplitProposal
from .triangle_projector import TriangleProposal
from .correlation_lengths import length, sha
from .independent_replica_projector import spectrum_from_moments
from .model import EnumeratedGuide
from .physics import basis, diagonal, make_af_sample
from .production_protocol import exact_spectra, geometry, identity
from .projector import propagate, rates_and_local


CONTEXTS = [(.25, 50025), (.25, 50059), (.5, 50043), (.75, 50004)]
METHODS = [('ct', .1, 0.), ('natural', .1, 0.), ('natural', .05, 0.),
           ('guided', .1, 0.), ('guided', .05, 0.),
           ('broad', .1, .1), ('broad', .05, .1)]
SCHEMA = 2


@lru_cache(maxsize=1)
def source_hashes():
    source = Path(__file__).parent
    names = ['collective_campaign.py', 'collective_projector.py', 'projector.py',
             'independent_replica_projector.py', 'physics.py', 'model.py',
             'production_protocol.py', 'correlation_lengths.py',
             'triangle_projector.py', 'triangle_campaign.py']
    return {name: sha(source/name) for name in names}


def write(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def label(g, sid):return f'L2_s{sid}_g{g:g}_h0p15'.replace('.', 'p')


def load_guide(root, task):
    path = Path(root) / 'guides' / (label(task['gamma'], task['sample_id']) + '.npz')
    with np.load(path) as data:
        logs = data['original_logs' if task['guide'] == 'original' else 'logs'].copy()
    return EnumeratedGuide(logs, 12), path


def tasks(stage):
    result = []
    for gamma, sid in CONTEXTS:
        for guide in ['original', 'corrected']:
            if stage == 'pilot':
                settings = METHODS; repeats = 1
                sizes = [('pilot', 512, 8., 8., 4, 'guide')]
            elif stage == 'main':
                settings = METHODS; repeats = 8
                sizes = [('main', 1024, 32., 32., 16, 'guide')]
            else:
                settings = [('ct', .1, 0.), ('guided', .05, 0.), ('broad', .05, .1),
                            ('guided', .025, 0.), ('broad', .025, .1)]
                repeats = 4
                sizes = [('population', 4096, 32., 32., 16, 'guide'),
                         ('time', 1024, 64., 64., 16, 'guide'),
                         ('start', 1024, 32., 32., 16, 'uniform')]
            for method, delta, epsilon in settings:
                for audit, walkers, equil, forward, cycles, start in sizes:
                    for repeat in range(repeats):
                        result.append(dict(L=2, h=.15, gamma=gamma, sample_id=sid, guide=guide,
                            method=method, delta=delta, epsilon=epsilon, walkers=walkers,
                            equil=equil, forward=forward, cycles=cycles, start=start,
                            repeat=repeat, audit=audit))
    return result


def checkpoint(path, arrays, meta):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays, metadata=np.array(json.dumps(meta)))
    temporary.replace(path)


def run_one(task, output, inputs, max_seconds=3600, deadline=None):
    with threadpool_limits(limits=1):
        return _run(task, output, inputs, max_seconds, deadline)


def _run(task, output, inputs, max_seconds, deadline=None):
    root = Path(output); key = identity(task)[:24]
    path = root/'runs'/(key+'.json'); cp = root/'checkpoints'/(key+'.npz')
    progress = root/'progress'/(key+'.json')
    begin = time.monotonic(); guide, gp = load_guide(inputs, task)
    provenance = dict(schema=SCHEMA, guide_sha256=sha(gp), source_sha256=source_hashes())
    if path.exists():
        record = json.loads(path.read_text()); assert record['task'] == task
        assert record['provenance'] == provenance
        assert sha(path.with_suffix('.npz')) == record['npz_sha256']
        return record
    ham = make_af_sample(2, task['sample_id']).hamiltonian(.15)
    seed = int(identity(['collective-gfmc-v1', task])[:16], 16)
    rng = np.random.default_rng(seed)
    w, n = task['walkers'], ham.n_sites
    delta = task['delta']; equil_steps = round(task['equil']/delta)
    forward_steps = round(task['forward']/delta)
    assert abs(equil_steps*delta-task['equil']) < 1e-10
    assert abs(forward_steps*delta-task['forward']) < 1e-10 and forward_steps > 0
    if task['method'] == 'triangle':
        kernel = TriangleProposal(ham, task['gamma'], guide, delta=delta, cache=True)
    else:
        kernel = None if task['method'] == 'ct' else SplitProposal(
            ham, task['gamma'], guide, delta=delta, epsilon=task['epsilon'], cache=True,
            natural=task['method'] == 'natural')
    if cp.exists():
        with np.load(cp) as saved:
            meta = json.loads(str(saved['metadata']))
            assert meta['task'] == task and meta['provenance'] == provenance
            rng.bit_generator.state = meta['rng']
            states = saved['states'].copy(); shifts = saved['shifts'].copy()
            tagged = saved['tagged'].copy(); ancestors = saved['ancestors'].copy()
            moments = list(saved['moments']); pairs = list(saved['pairs'])
            descendant = list(saved['descendant']); largest = list(saved['largest'])
            transport = list(saved['transport'])
            proposed = saved['proposed'].copy(); surviving = saved['surviving'].copy()
            ess_sum = saved['ess_sum'].copy(); ess_min = saved['ess_min'].copy()
            step = meta['step']; elapsed_before = meta['seconds']; updates = meta['updates']
    else:
        states = (guide.sample(4*w, int(rng.integers(2**32))).reshape(4, w, n)
                  if task['start']=='guide' else (1-2*rng.integers(2,size=(4,w,n))).astype(np.int8))
        shifts = np.array([rates_and_local(ham, task['gamma'], guide, s[:,None,:])[1].mean() for s in states])
        tagged=states.copy().astype(float);ancestors=np.tile(np.arange(w),(4,1))
        moments=[];pairs=[];descendant=[];largest=[];transport=[]
        proposed=np.zeros(n+1,dtype=np.int64);surviving=proposed.copy()
        ess_sum=np.zeros(4);ess_min=np.full(4,np.inf);step=0;updates=0;elapsed_before=0.
    total = equil_steps + task['cycles']*forward_steps
    last_progress = time.monotonic()
    def save(state):
        seconds=elapsed_before+time.monotonic()-begin
        cp.parent.mkdir(parents=True,exist_ok=True)
        checkpoint(cp,dict(states=states,shifts=shifts,tagged=tagged,ancestors=ancestors,
                   moments=np.array(moments),pairs=np.array(pairs),descendant=np.array(descendant),
                   largest=np.array(largest),transport=np.array(transport),proposed=proposed,
                   surviving=surviving,ess_sum=ess_sum,ess_min=ess_min),
                   dict(task=task,provenance=provenance,rng=rng.bit_generator.state,step=step,
                        updates=updates,seconds=seconds))
        write(progress,dict(state=state,task=task,step=step,total_steps=total,
            completed_cycles=len(moments),seconds=seconds,pid=os.getpid(),updated_unix=time.time(),
            minimum_weight_ess=None if updates==0 else float(ess_min.min())))
    save('running')
    while step < total:
        if time.monotonic()-begin > max_seconds or (deadline is not None and time.monotonic() >= deadline):
            save('paused_time_cap')
            return dict(task=task,status='paused_time_cap',seconds=elapsed_before+time.monotonic()-begin)
        if kernel is None:
            children=np.empty_like(ancestors);next_states=states.copy();ess=np.zeros(4)
            for r in range(4):
                a,c,shifts[r],_,_,ess[r]=propagate(ham,task['gamma'],guide,states[r,:,None,:],shifts[r],delta,rng)
                next_states[r]=a[:,0];children[r]=c
            states=next_states
        else:
            states,children,stats=kernel.step(states,rng);ess=stats['ess']
            proposed+=stats['proposed'];surviving+=stats['surviving']
        ess_sum+=ess;ess_min=np.minimum(ess_min,ess);updates+=1
        if step >= equil_steps:
            ancestors=ancestors[np.arange(4)[:,None],children]
        step+=1
        boundary=step==equil_steps or (step>equil_steps and (step-equil_steps)%forward_steps==0)
        if step>equil_steps and boundary:
            m=[];q=[];de=[];la=[];tr=[]
            for r in range(4):
                counts=np.bincount(ancestors[r],minlength=w);prob=counts/w
                m.append(prob@tagged[r]);q.append(tagged[r].T@(prob[:,None]*tagged[r]))
                de.append(float(w*w/(counts@counts)));la.append(float(counts.max()/w))
                overlap=(states[r]*tagged[r][ancestors[r]]).mean(1)
                tr.append([float(overlap.mean()),float(np.mean(overlap<0))])
            moments.append(m);pairs.append(q);descendant.append(de);largest.append(la);transport.append(tr)
        if boundary:
            tagged=states.copy().astype(float);ancestors=np.tile(np.arange(w),(4,1))
        if boundary or time.monotonic()-last_progress>20:
            save('running');last_progress=time.monotonic()
    m=np.array(moments).transpose(1,0,2);q=np.array(pairs).transpose(1,0,2,3)
    spectrum=spectrum_from_moments(m.mean(1),q.mean(1),geometry(2))
    arrays=dict(magnetization_cycles=m,pair_cycles=q,descendant_ess=np.array(descendant).T,
                largest_lineage=np.array(largest).T,transport=np.array(transport),
                proposed_flip_counts=proposed,surviving_flip_counts=surviving,
                final_states=states,mean_weight_ess=ess_sum/updates,minimum_weight_ess=ess_min)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.tmp.npz'),**arrays)
    path.with_suffix('.tmp.npz').replace(path.with_suffix('.npz'))
    xi=length(spectrum,2)
    record=dict(task=task,status='completed',seed=seed,seconds=elapsed_before+time.monotonic()-begin,
        provenance=provenance,guide_sha256=sha(gp),enumeration='L2 frozen-guide lookup only; not scalable guide fitting',
        spectrum=spectrum.tolist(),xi_over_L=float(xi/2) if np.isfinite(xi) else None,
        minimum_descendant_ess=float(np.min(descendant)),maximum_lineage=float(np.max(largest)),
        ancestry_alarm=bool(np.min(descendant)<8 or np.max(largest)>.5),
        mean_weight_ess=(ess_sum/updates).tolist(),npz_sha256=sha(path.with_suffix('.npz')),
        ground_state_qualified=False)
    write(path,record);save('completed');return record


def split_action(vector, potential, gamma, delta):
    """Apply the symmetric split operator in O(N 2**N), for reference only."""
    result = np.exp(-delta*potential/2)*vector
    c, s = np.cosh(gamma*delta), np.sinh(gamma*delta)
    for site in range(int(np.log2(len(vector)))):
        blocks = result.reshape(-1, 2, 1 << site)
        a, b = blocks[:, 0].copy(), blocks[:, 1].copy()
        blocks[:, 0] = c*a+s*b; blocks[:, 1] = s*a+c*b
    return np.exp(-delta*potential/2)*result


def finite_moments(left, right, advance, cycles, spins, lattice):
    ms, qs = [], []
    for cycle in range(cycles):
        probability = left*right; probability /= probability.sum()
        ms.append(probability@spins)
        qs.append(spins.T@(probability[:, None]*spins))
        if cycle+1 < cycles:
            left = advance(left); left /= np.linalg.norm(left)
    m, q = np.mean(ms, axis=0), np.mean(qs, axis=0)
    return spectrum_from_moments(np.tile(m, (4, 1)), np.tile(q, (4, 1, 1)), lattice)


def reference_one(context, inputs):
    """Evaluation only: exact finite projection and split ground-state targets."""
    with threadpool_limits(limits=1):
        gamma, sid = context
        spins=basis(12);lattice=geometry(2)
        task=dict(gamma=gamma,sample_id=sid)
        gp=Path(inputs)/'guides'/(label(gamma,sid)+'.npz')
        with np.load(gp) as data:exact=data['exact_amplitude'].copy()
        ham=make_af_sample(2,sid).hamiltonian(.15);matrix=operator_matrix(ham,gamma)
        energy=float(exact@(matrix@exact));residual=float(np.linalg.norm(matrix@exact-energy*exact))
        assert residual<1e-9
        spectrum=exact_spectra(exact**2,spins,lattice)['connected_spectrum']
        row=dict(gamma=gamma,sample_id=sid,ground_spectrum=spectrum.tolist(),
                 ground_xi_over_L=float(length(spectrum,2)/2),residual=residual,finite_projection=[],split=[])
        shifted=matrix-energy*eye(len(spins),format='csr');trace=float(shifted.diagonal().sum())
        schedules = [('pilot',8.,8.,4,'guide'), ('main',32.,32.,16,'guide'),
                     ('time',64.,64.,16,'guide'), ('start',32.,32.,16,'uniform')]
        for guide_kind in ['original','corrected']:
            task['guide']=guide_kind;guide,_=load_guide(inputs,task)
            amplitude=np.exp(guide.logs-guide.logs.max());amplitude/=np.linalg.norm(amplitude)
            for audit,equil,forward,cycles,start in schedules:
                initial = amplitude if start == 'guide' else 1/amplitude
                initial = initial/np.linalg.norm(initial)
                left=expm_multiply(-equil*shifted,initial,traceA=-equil*trace)
                right=expm_multiply(-forward*shifted,amplitude,traceA=-forward*trace)
                advance = lambda x: expm_multiply(-forward*shifted,x,traceA=-forward*trace)
                target=finite_moments(left,right,advance,cycles,spins,lattice)
                row['finite_projection'].append(dict(guide=guide_kind,audit=audit,spectrum=target.tolist()))
        potential = diagonal(ham, spins)
        for delta in [.1, .05, .025]:
            # Scalar shift controls norm growth; it leaves eigenvectors unchanged.
            v = potential-energy
            action = lambda x: split_action(x, v, gamma, delta)
            operator = LinearOperator((len(spins),len(spins)),matvec=action,dtype=float)
            values, vectors = eigsh(operator,k=1,which='LA',v0=exact,tol=1e-12)
            ground = vectors[:,0]
            residual_split = float(np.linalg.norm(action(ground)-values[0]*ground))
            assert residual_split < 1e-9
            row['split'].append(dict(delta=delta,eigenvalue=float(values[0]),residual=residual_split,
                ground_spectrum=exact_spectra(ground**2,spins,lattice)['connected_spectrum'].tolist()))
            def advance_steps(x, duration):
                for _ in range(round(duration/delta)):
                    x = action(x)
                    x /= np.linalg.norm(x)
                return x
            for guide_kind in ['original','corrected']:
                task['guide']=guide_kind;guide,_=load_guide(inputs,task)
                amplitude=np.exp(guide.logs-guide.logs.max());amplitude/=np.linalg.norm(amplitude)
                for audit,equil,forward,cycles,start in schedules:
                    initial=amplitude if start=='guide' else 1/amplitude
                    initial=initial/np.linalg.norm(initial)
                    left=advance_steps(initial,equil);right=advance_steps(amplitude,forward)
                    target=finite_moments(left,right,lambda x:advance_steps(x,forward),cycles,spins,lattice)
                    row['finite_projection'].append(dict(guide=guide_kind,audit=audit,delta=delta,spectrum=target.tolist()))
        return row


def references(inputs, output):
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(reference_one, CONTEXTS, [str(inputs)]*len(CONTEXTS)))
    write(Path(output)/'references.json',dict(rows=rows,source_sha256=source_hashes(),
        scope='Exact references are offline evaluation only.'))


def assess(output):
    root=Path(output);ref=json.loads((root/'references.json').read_text())
    records=[]
    for p in sorted((root/'runs').glob('*.json')):
        r=json.loads(p.read_text());assert r['npz_sha256']==sha(p.with_suffix('.npz'))
        with np.load(p.with_suffix('.npz')) as a:
            m=a['magnetization_cycles'];q=a['pair_cycles']
            np.testing.assert_allclose(spectrum_from_moments(m.mean(1),q.mean(1),geometry(2)),r['spectrum'],atol=1e-12)
        records.append(r)
    groups={}
    for r in records:
        t=r['task'];k=tuple(t[x] for x in ['gamma','sample_id','guide','method','delta','epsilon','audit'])
        groups.setdefault(k,[]).append(r)
    rows=[];rng=np.random.default_rng(20261102)
    for key,rs in sorted(groups.items()):
        gamma,sid,guide,method,delta,epsilon,audit=key
        target=next(r for r in ref['rows'] if (r['gamma'],r['sample_id'])==(gamma,sid))
        raw=np.array([r['spectrum'] for r in rs]);mean=raw.mean(0);truth=np.array(target['ground_spectrum'])
        reference_audit = 'main' if audit == 'population' else audit
        finite_target = np.array(next(r['spectrum'] for r in target['finite_projection']
            if r['guide']==guide and r['audit']==reference_audit
            and (r.get('delta') is None if method=='ct' else r.get('delta')==delta)))
        split_target = truth if method=='ct' else np.array(next(
            r['ground_spectrum'] for r in target['split'] if r['delta']==delta))
        boot=raw[rng.integers(len(rs),size=(2000,len(rs)))].mean(1)
        se=boot.std(0,ddof=1) if len(rs)>1 else np.full_like(mean,np.nan)
        lens=length(boot,2)/2;valid=np.isfinite(lens);xi=length(mean,2)/2
        rows.append(dict(gamma=gamma,sample_id=sid,guide=guide,method=method,delta=delta,
            epsilon=epsilon,audit=audit,repeats=len(rs),spectrum=mean.tolist(),
            spectrum_se=se.tolist() if len(rs)>1 else None,
            relative_bias=((mean-truth)/truth).tolist(),max_relative_bias=float(np.max(abs(mean-truth)/truth)),
            finite_projection_spectrum=finite_target.tolist(),
            relative_sampling_error=((mean-finite_target)/finite_target).tolist(),
            split_ground_relative_bias=((split_target-truth)/truth).tolist(),
            finite_projection_relative_bias=((finite_target-split_target)/split_target).tolist(),
            relative_95_bounds=((abs(mean-truth)+1.96*se)/truth).tolist() if len(rs)>1 else None,
            xi_over_L=float(xi) if np.isfinite(xi) else None,exact_xi_over_L=target['ground_xi_over_L'],
            xi_interval=np.quantile(lens[valid],[.025,.975]).tolist() if len(rs)>1 and valid.any() else None,
            invalid_length_fraction=float(1-valid.mean()),alarms=sum(r['ancestry_alarm'] for r in rs),
            minimum_descendant_ess=min(r['minimum_descendant_ess'] for r in rs),
            maximum_lineage=max(r['maximum_lineage'] for r in rs),
            total_seconds=sum(r['seconds'] for r in rs),ground_state_qualified=False))
    report=dict(completed=len(records),rows=rows,uncertainty='Whole-run bootstrap; pointwise approximate intervals. No disorder averaging of selected contexts. No automatic equilibration claim.')
    write(root/'review.json',report)
    lines=['# Collective-flip GFMC benchmark','',report['uncertainty'],'',
           '| Gamma | Disorder | Guide | Method | delta | Audit | Runs | Max spectral bias | Alarms | CPU s |',
           '|---:|---:|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['gamma']} | {r['sample_id']} | {r['guide']} | {r['method']} | {r['delta']} | {r['audit']} | {r['repeats']} | {r['max_relative_bias']:.2%} | {r['alarms']} | {r['total_seconds']:.1f} |")
    (root/'REVIEW.md').write_text('\n'.join(lines)+'\n');return report


def campaign(output, inputs, workers, stages, hours):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _campaign(root, inputs, workers, stages, hours)


def _campaign(root, inputs, workers, stages, hours):
    start=time.monotonic();deadline=start+hours*3600
    manifest=dict(contexts=CONTEXTS,methods=METHODS,workers=workers,stages=stages,hours_cap=hours,
        schema=SCHEMA,source_sha256=source_hashes(),
        input_sha256={label(g,s):sha(Path(inputs)/'guides'/(label(g,s)+'.npz')) for g,s in CONTEXTS},
        scope='Matched L2 algorithm development; no exact labels enter proposals or guide fitting.')
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        for field in ['schema','source_sha256','input_sha256']:
            assert previous[field]==manifest[field], f'Campaign {field} changed; use a new output directory'
    else:write(manifest_path,manifest)
    for stage in stages:
        ts=tasks(stage);done=0;attempts={};pending=iter(ts)
        write(root/'status.json',dict(state='running',stage=stage,completed=done,total=len(ts),updated_unix=time.time()))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={}
            def submit(task):
                futures[pool.submit(run_one,task,str(root),str(inputs),1800,deadline)]=task
            for _ in range(workers):
                task=next(pending,None)
                if task is not None:submit(task)
            while futures:
                ready,_=wait(futures,timeout=20,return_when=FIRST_COMPLETED)
                for future in ready:
                    task=futures.pop(future)
                    try:record=future.result()
                    except Exception as error:
                        write(root/'failure.json',dict(task=task,error=repr(error),updated_unix=time.time()))
                        raise
                    if record['status']=='completed':done+=1
                    print(json.dumps(dict(stage=stage,completed=done,total=len(ts),task=task,status=record['status'],seconds=record['seconds'])),flush=True)
                    if time.monotonic()<deadline:
                        if record['status']!='completed':
                            key=identity(task);attempts[key]=attempts.get(key,0)+1
                            if attempts[key]>16:raise RuntimeError('Task failed to finish after 16 checkpoint resumes')
                            submit(task)
                        else:
                            task=next(pending,None)
                            if task is not None:submit(task)
                write(root/'status.json',dict(state='running',stage=stage,completed=done,total=len(ts),active=len(futures),updated_unix=time.time()))
        if stage=='pilot':
            median=float(np.median([json.loads(p.read_text())['seconds'] for p in (root/'runs').glob('*.json')]))
            write(root/'pilot_timing.json',dict(median_seconds=median,scope='512 walkers, 4 short cycles; cost only.'))
        if time.monotonic()>=deadline:
            write(root/'status.json',dict(state='paused_time_cap',stage=stage,completed=done,total=len(ts),updated_unix=time.time()));return
    if not (root/'references.json').exists():references(inputs,root)
    assess(root)
    write(root/'status.json',dict(state='completed',completed_runs=len(list((root/'runs').glob('*.json'))),updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['pilot','main','audit','all','assess'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--workers',type=int,default=18);p.add_argument('--hours',type=float,default=8.)
    a=p.parse_args()
    if a.stage=='assess':assess(a.output)
    else:campaign(a.output,a.inputs,a.workers,['pilot','main','audit'] if a.stage=='all' else [a.stage],a.hours)
