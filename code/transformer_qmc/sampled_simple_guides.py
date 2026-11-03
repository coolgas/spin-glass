"""CUDA sampled VMC with tempered chains and transverse-field continuation.

No basis enumeration enters training. Tempering targets psi(s)**(2*beta), with
beta=0 providing uniform refreshes. Exact L2 diagnostics are evaluation only,
after freezing the trained parameters. Larger-L records are feasibility probes.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .collective_campaign import CONTEXTS, label, write
from .correlation_lengths import sha
from .physics import make_af_sample
from .simple_guides import Jastrow, RBM


def exchange_ratio(beta_a, beta_b, log_a, log_b):
    return 2*(beta_a-beta_b)*(log_b-log_a)


class TemperedChains:
    """Batched local Metropolis, inversion, uniform hot refresh, replica exchange.

    Cached fields are rebuilt every sweep because VMC changes the parameters.
    Population dimension consists of independent chains, not GFMC walkers.
    Fixed-parameter diagnostics must be separate from adaptive training traces.
    """
    def __init__(self, model, chains=256, temperatures=8, seed=0, start='random'):
        if chains < 2 or temperatures < 2:
            raise ValueError('Need at least two chains and temperatures')
        self.model=model;self.device=next(model.parameters()).device
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        self.betas=torch.linspace(0,1,temperatures,device=self.device,dtype=torch.float64)
        self.states=(2*torch.randint(2,(temperatures,chains,model.n_sites),
            device=self.device,generator=self.rng)-1).double()
        if start in ['up','down']:self.states.fill_(1 if start=='up' else -1)
        self.labels=torch.arange(temperatures,device=self.device)[:,None].expand(-1,chains).clone()
        self.phase=torch.zeros((temperatures,chains),device=self.device,dtype=torch.long)
        self.trips=torch.zeros_like(self.phase)
        self.accepted=torch.zeros(temperatures-1,device=self.device,dtype=torch.long)
        self.attempted=torch.zeros_like(self.accepted)
        self.sweeps=0

    @torch.no_grad()
    def sweep(self):
        model=self.model
        t,c,n=self.states.shape
        # beta=0 has exactly uniform probability, independent of the guide.
        self.states[0]=(2*torch.randint(2,(c,n),device=self.device,generator=self.rng)-1).double()
        spins=self.states.reshape(-1,n)
        beta=self.betas[:,None].expand(-1,c).reshape(-1)
        if isinstance(model,RBM):
            field=spins@model.w+model.b
        else:
            matrix=torch.zeros((n,n),device=self.device,dtype=spins.dtype)
            matrix[model.pairs[:,0],model.pairs[:,1]]=model.j
            matrix=matrix+matrix.T
            field=spins@matrix+model.a
        for site in torch.randperm(n,device=self.device,generator=self.rng):
            if isinstance(model,RBM):
                delta=-2*spins[:,site,None]*model.w[site]
                updated=field+delta
                lr=-2*spins[:,site]*model.a[site]+(
                    torch.logaddexp(updated,-updated)-torch.logaddexp(field,-field)).sum(1)
            else:
                lr=-2*spins[:,site]*field[:,site]
                delta=-2*spins[:,site,None]*matrix[site]
            take=torch.log(torch.rand(len(spins),device=self.device,generator=self.rng))<2*beta*lr
            field.add_(delta*take[:,None])
            spins[:,site]*=torch.where(take,-1.,1.)
        logs=model(spins)
        inverted=model(-spins)
        take=torch.log(torch.rand(len(spins),device=self.device,generator=self.rng))<2*beta*(inverted-logs)
        spins.mul_(torch.where(take,-1.,1.)[:,None])
        logs=torch.where(take,inverted,logs).reshape(t,c)
        for a in range(self.sweeps%2,t-1,2):
            lr=exchange_ratio(self.betas[a],self.betas[a+1],logs[a],logs[a+1])
            take=torch.log(torch.rand(c,device=self.device,generator=self.rng))<lr
            old=self.states[a,take].clone()
            self.states[a,take]=self.states[a+1,take];self.states[a+1,take]=old
            old=self.labels[a,take].clone()
            self.labels[a,take]=self.labels[a+1,take];self.labels[a+1,take]=old
            self.accepted[a]+=take.sum();self.attempted[a]+=c
        col=torch.arange(c,device=self.device)
        hot=self.labels[0];cold=self.labels[-1]
        self.trips[hot,col]+=(self.phase[hot,col]==2).long()
        self.phase[hot,col]=1
        self.phase[cold,col]=torch.where(self.phase[cold,col]==1,2,self.phase[cold,col])
        self.sweeps+=1
        return self.states[-1]

    def diagnostics(self):
        return dict(sweeps=self.sweeps,
            exchange_fraction=(self.accepted/self.attempted.clamp_min(1)).cpu().tolist(),
            round_trips=int(self.trips.sum()),
            labels_with_round_trips=int((self.trips>0).sum()),
            warning='Round trips measure temperature transport, not independent valley equilibration')


@torch.no_grad()
def local_energy(model, states, ham, gamma):
    bonds=torch.tensor(ham.bonds,device=states.device)
    coupling=torch.tensor(ham.J,device=states.device)
    diagonal=-ham.h*states.sum(1)-(states[:,bonds[:,0]]*states[:,bonds[:,1]])@coupling
    ratios=model.flip_log_ratios(states)
    energy=diagonal-gamma*torch.exp(ratios).sum(1)
    if not torch.isfinite(energy).all():
        raise FloatingPointError('Nonfinite local energy; no ratio clipping is allowed')
    return energy


def energy_loss(logs, energies, n_sites):
    """VMC covariance gradient, with detached local energies; not energy itself."""
    centered=energies.detach()-energies.detach().mean()
    return 2*((logs-logs.mean())*centered).mean()/n_sites


def learning_rate(model, requested=None):
    """Conservative large-RBM step size; explicit overrides enter provenance.

    A global gradient-norm cap does not bound Adam's per-coordinate steps.
    The N=108 matched diagnostic is recorded by rbm_optimizer_diagnostic.py.
    """
    value = requested if requested is not None else (
        .001 if isinstance(model, RBM) and model.n_sites > 48 else .01)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('learning rate must be positive and finite')
    return float(value)


def sampling_alarms(diagnostic, chains):
    """Necessary transport/start checks only, never a convergence certificate."""
    alarms = []
    for row in diagnostic:
        if row['round_trips'] == 0:
            alarms.append(row['start']+': no temperature round trips')
        if min(row['exchange_fraction']) < .01:
            alarms.append(row['start']+': exchange bottleneck below 1%')
        if not np.isfinite([row['energy'], row['variance'], row['chain_mean_energy_sd']]).all():
            alarms.append(row['start']+': nonfinite frozen diagnostic')
    for i, a in enumerate(diagnostic):
        for b in diagnostic[i+1:]:
            se = np.hypot(a['chain_mean_energy_sd'], b['chain_mean_energy_sd'])/np.sqrt(chains)
            if abs(a['energy']-b['energy']) > max(5*se, 1e-10):
                alarms.append(a['start']+'/'+b['start']+': energy discrepancy exceeds five chain SE')
    return alarms


def select_sampled_guide(candidates):
    """Rank only fits passing necessary independent-start sampling checks.

    A failed context must remain a visible missing ensemble member; it is
    never replaced by another disorder or hidden from the completion count.
    """
    eligible = [r for r in candidates if r['status'] == 'completed' and not
                sampling_alarms(r['diagnostics'], r['signature']['chains'])]
    if not eligible:
        raise RuntimeError('No candidate passes frozen sampler checks; retraining required')
    return min(eligible, key=lambda r: np.mean([d['energy'] for d in r['diagnostics']]))


def freeze_diagnostics(model,ham,gamma,chains,temperatures,sweeps,seed):
    rows=[];arrays={}
    for k,start in enumerate(['random','up','down']):
        sampler=TemperedChains(model,chains,temperatures,seed+10000+k,start)
        energies=[];mag=[];pairs=[];samples=[]
        for step in range(2*sweeps):
            states=sampler.sweep()
            if step>=sweeps and step%4==0:
                energy=local_energy(model,states,ham,gamma)
                energies.append(energy.cpu().numpy())
                mag.append(states.mean(0).cpu().numpy())
                pairs.append((states.T@states/len(states)).cpu().numpy())
                if ham.n_sites==12:samples.append(states.cpu().numpy().astype(np.int8))
        values=np.array(energies)
        arrays[start+'_energy']=values
        arrays[start+'_magnetization']=np.array(mag)
        arrays[start+'_pairs']=np.array(pairs)
        if samples:arrays[start+'_samples']=np.array(samples)
        rows.append(dict(start=start,energy=float(values.mean()),
            variance=float(values.var()),chain_mean_energy_sd=float(values.mean(0).std()),
            **sampler.diagnostics()))
    return rows,arrays


def train(task,args,root,deadline):
    L,gamma,sid,kind,seed,mode=(task[k] for k in ['L','gamma','sample_id','kind','seed','mode'])
    ham=make_af_sample(L,sid).hamiltonian(.15)
    model=(Jastrow(ham.n_sites,seed) if kind=='jastrow' else RBM(ham.n_sites,2,seed)).to(args.device)
    sampler=TemperedChains(model,args.chains,args.temperatures,seed+100)
    lr=learning_rate(model,getattr(args,'learning_rate',None))
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    schedule=sorted(set([1.5,.75,gamma]),reverse=True) if mode=='anneal' else [gamma]
    schedule=[g for g in schedule if g>=gamma]
    name=f'L{L}_s{sid}_g{gamma:g}_{kind}_{mode}_seed{seed}'.replace('.','p')
    final=root/(name+'.json');checkpoint=root/(name+'.pt')
    signature=dict(task=task,chains=args.chains,temperatures=args.temperatures,steps=args.steps,
        optimizer='Adam',learning_rate=lr,gradient_norm_cap=1.,
        sweeps_per_step=args.sweeps_per_step,diagnostic_sweeps=args.diagnostic_sweeps,
        sources={p:sha(Path(__file__).with_name(p)) for p in ['sampled_simple_guides.py','simple_guides.py']})
    if final.exists():
        record=json.loads(final.read_text());assert record['signature']==signature
        return record
    history=[];stage=0;step=0;seconds_before=0.
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=args.device,weights_only=False)
        assert saved['signature']==signature
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        for attr in ['states','labels','phase','trips','accepted','attempted','sweeps']:
            setattr(sampler,attr,saved['sampler'][attr])
        sampler.rng.set_state(saved['rng'].cpu())
        history=saved['history'];stage=saved['stage'];step=saved['step'];seconds_before=saved['seconds']
    started=time.monotonic()
    def save():
        temporary=checkpoint.with_suffix('.tmp.pt')
        torch.save(dict(signature=signature,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler={a:getattr(sampler,a) for a in ['states','labels','phase','trips','accepted','attempted','sweeps']},
            rng=sampler.rng.get_state(),history=history,stage=stage,step=step,
            seconds=seconds_before+time.monotonic()-started),temporary)
        temporary.replace(checkpoint)
    while stage<len(schedule):
        field=schedule[stage]
        while step<args.steps:
            if time.monotonic()>=deadline:
                save();return dict(status='paused_time_cap',task=task)
            for _ in range(args.sweeps_per_step):states=sampler.sweep()
            energy=local_energy(model,states,ham,field)
            optimizer.zero_grad()
            loss=energy_loss(model(states),energy,ham.n_sites)
            loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite VMC gradient')
            optimizer.step();step+=1
            if step%25==0:
                row=dict(stage=stage,gamma=field,step=step,energy=float(energy.mean()),
                    variance=float(energy.var()),gradient_norm=float(gradient))
                history.append(row);save()
                write(root/'status.json',dict(state='training',task=task,**row))
        stage+=1;step=0;save()
    diagnostic,arrays=freeze_diagnostics(model,ham,gamma,args.chains,args.temperatures,
        args.diagnostic_sweeps,seed)
    record=dict(task=task,signature=signature,status='completed',history=history,
        diagnostics=diagnostic,training_sampler=sampler.diagnostics(),
        sampling_alarms=sampling_alarms(diagnostic,args.chains),
        seconds=seconds_before+time.monotonic()-started,ground_state_qualified=False,
        scope='Sampled energy-only VMC; adaptive training samples are not convergence evidence')
    if L==2:
        # Frozen post-fit evaluation; neither exact states nor full-basis gradients
        # are available in the optimization loop above.
        from .physics import basis
        from kagome_qea.exact import operator_matrix
        from .production_protocol import exact_spectra,geometry
        spins=basis(12)
        with torch.no_grad():logs=model(torch.tensor(spins,device=args.device,dtype=torch.float64)).cpu().numpy()
        amplitude=np.exp(logs-logs.max());amplitude/=np.linalg.norm(amplitude)
        matrix=operator_matrix(ham,gamma)
        gp=Path(args.inputs)/'guides'/(label(gamma,sid)+'.npz')
        with np.load(gp) as data:exact=data['exact_amplitude'].copy()
        energy=float(amplitude@(matrix@amplitude))
        truth=exact_spectra(exact**2,spins,geometry(2))['connected_spectrum']
        spectrum=exact_spectra(amplitude**2,spins,geometry(2))['connected_spectrum']
        record['exact_evaluation']=dict(energy=energy,energy_error=energy-float(exact@(matrix@exact)),
            variance=float(np.linalg.norm(matrix@amplitude-energy*amplitude)**2),
            fidelity=float((exact@amplitude)**2),max_spectral_error=float(np.max(abs(spectrum-truth)/truth)),
            spectrum=spectrum.tolist(),exact_spectrum=truth.tolist(),input_sha256=sha(gp))
        for row in diagnostic:
            sample=arrays[row['start']+'_samples'].reshape(-1,12)
            ids=((1-sample.astype(np.int64))//2)@(1<<np.arange(12))
            histogram=np.bincount(ids,minlength=4096)/len(ids)
            row['tv_from_exact_guide_probability']=float(abs(histogram-amplitude**2).sum()/2)
            row['energy_minus_exact_guide_energy']=row['energy']-energy
        arrays['logs']=np.log(amplitude)
    np.savez_compressed(root/(name+'.npz'),**arrays)
    record['npz_sha256']=sha(root/(name+'.npz'));record['checkpoint_sha256']=sha(checkpoint)
    write(final,record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=400)
    p.add_argument('--learning-rate',type=float,default=None,help='Default .001 for RBM N>48; .01 otherwise')
    p.add_argument('--chains',type=int,default=256);p.add_argument('--temperatures',type=int,default=8)
    p.add_argument('--sweeps-per-step',type=int,default=2);p.add_argument('--diagnostic-sweeps',type=int,default=200)
    p.add_argument('--hours',type=float,default=3);p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--sizes',type=int,nargs='+',default=[2,4])
    p.add_argument('--kinds',nargs='+',default=['jastrow','rbm_2'])
    p.add_argument('--modes',nargs='+',default=['direct','anneal'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    deadline=time.monotonic()+args.hours*3600
    tasks=[dict(L=L,gamma=g,sample_id=s,kind=k,seed=seed,mode=mode)
        for L in args.sizes for g,s in (CONTEXTS if L==2 else [(.25,50025),(.5,50043)])
        for k in args.kinds for seed in range(args.seeds) for mode in args.modes]
    rows=[]
    for task in tasks:
        try:record=train(task,args,args.output,deadline)
        except Exception as error:
            write(args.output/'failure.json',dict(task=task,error=repr(error)));raise
        if record['status']!='completed':
            write(args.output/'status.json',dict(state=record['status'],task=task,completed=len(rows),total=len(tasks)))
            return
        rows.append(record);write(args.output/'review.json',dict(rows=rows))
        print(json.dumps(dict(task=task,seconds=record['seconds'],evaluation=record.get('exact_evaluation'),diagnostics=record['diagnostics'])),flush=True)
    write(args.output/'status.json',dict(state='completed',completed=len(rows),total=len(tasks)))


if __name__=='__main__':main()
