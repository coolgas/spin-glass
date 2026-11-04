"""L2 ansatz/optimization diagnostic: energy-only fits, exact evaluation afterward.

Enumeration removes training-chain mixing as a confound. This is NOT scalable
training or a claim of sampled VMC performance. No exact eigenvector is read
until fitting completes. Fits start from scratch, not the corrected transformer.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.sparse.linalg import expm_multiply

from kagome_qea.exact import operator_matrix
from .collective_campaign import CONTEXTS, label, write
from .correlation_lengths import sha
from .physics import basis, diagonal, make_af_sample
from .production_protocol import exact_spectra, geometry
from .simple_guides import Jastrow, RBM


def objective(model, spins, potential, flips, gamma):
    logs = model(spins)
    amplitude = torch.exp(logs-logs.max())
    amplitude = amplitude / torch.linalg.vector_norm(amplitude)
    applied = potential*amplitude-gamma*amplitude[flips].sum(1)
    return amplitude @ applied, amplitude, applied


def fit(ham, gamma, kind, seed, device, adam_steps=800, lbfgs_steps=200):
    torch.manual_seed(seed)
    model = (Jastrow(ham.n_sites,seed) if kind=='jastrow' else
             RBM(ham.n_sites,alpha=int(kind.split('_')[1]),seed=seed)).to(device)
    spins = torch.tensor(basis(ham.n_sites),dtype=torch.float64,device=device)
    potential = torch.tensor(diagonal(ham,spins.cpu().numpy()),device=device)
    flips = torch.arange(len(spins),device=device)[:,None] ^ (1<<torch.arange(ham.n_sites,device=device))
    history = []; started=time.monotonic()
    optimizer = torch.optim.Adam(model.parameters(),lr=.015)
    best_energy = float('inf'); best_state = None
    def save_best(energy):
        nonlocal best_energy, best_state
        value = float(energy.detach())
        if not np.isfinite(value): raise FloatingPointError('Nonfinite variational energy')
        if value < best_energy:
            best_energy = value
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        return value
    for step in range(adam_steps):
        optimizer.zero_grad()
        energy,_,_ = objective(model,spins,potential,flips,gamma)
        value = save_best(energy)
        energy.backward(); optimizer.step()
        if step%100==0:
            history.append(dict(stage='adam',step=step,energy=value))
    model.load_state_dict(best_state)
    optimizer = torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=lbfgs_steps,
        tolerance_grad=1e-9,tolerance_change=1e-13,line_search_fn='strong_wolfe',history_size=40)
    def closure():
        optimizer.zero_grad()
        energy,_,_ = objective(model,spins,potential,flips,gamma)
        save_best(energy); energy.backward()
        return energy
    optimizer.step(closure)
    energy,_,_=objective(model,spins,potential,flips,gamma);save_best(energy)
    model.load_state_dict(best_state)
    with torch.no_grad():
        energy,amplitude,applied = objective(model,spins,potential,flips,gamma)
        variance = float(((applied-energy*amplitude)**2).sum())
        logs = model(spins).cpu().numpy()
        logs -= np.logaddexp.reduce(2*logs)/2
    return model,logs,dict(energy=float(energy),variance=variance,history=history,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started,
        training='Full-basis L2 variational expectation, no eigenvector labels',exact_labels_used=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--adam-steps',type=int,default=800);p.add_argument('--lbfgs-steps',type=int,default=200)
    p.add_argument('--kinds',nargs='+',default=['jastrow','rbm_1','rbm_2'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    sources={name:sha(Path(__file__).with_name(name)) for name in ['simple_guides.py','simple_guide_benchmark.py']}
    manifest=dict(source_sha256=sources,settings=vars(args).copy(),selection='All seeds retained; select by variational energy only')
    manifest['settings']={k:str(v) if isinstance(v,Path) else v for k,v in manifest['settings'].items()}
    write(args.output/'manifest.json',manifest)
    rows=[]
    for gamma,sid in CONTEXTS:
        ham=make_af_sample(2,sid).hamiltonian(.15);matrix=operator_matrix(ham,gamma)
        spins=basis(12)
        for kind in args.kinds:
            for seed in range(args.seeds):
                name=f'{label(gamma,sid)}_{kind}_seed{seed}'
                path=args.output/(name+'.json')
                if path.exists():
                    row=json.loads(path.read_text());assert row['source_sha256']==sources
                else:
                    write(args.output/'status.json',dict(state='fitting',gamma=gamma,sample_id=sid,kind=kind,seed=seed,completed=len(rows)))
                    model,logs,record=fit(ham,gamma,kind,seed,args.device,args.adam_steps,args.lbfgs_steps)
                    torch.save(model.state_dict(),args.output/(name+'.pt'))
                    # Exact amplitudes are evaluation only, loaded after the fit.
                    gp=args.inputs/'guides'/(label(gamma,sid)+'.npz')
                    with np.load(gp) as data:
                        exact=data['exact_amplitude'].copy()
                        oldlogs={key:data[key].copy() for key in ['logs','original_logs']}
                    amplitude=np.exp(logs)
                    exact_energy=float(exact@(matrix@exact))
                    truth=exact_spectra(exact**2,spins,geometry(2))['connected_spectrum']
                    spectrum=exact_spectra(amplitude**2,spins,geometry(2))['connected_spectrum']
                    forward=[]
                    for tau in [8.,32.,64.]:
                        projected=expm_multiply(-tau*(matrix-exact_energy*__import__('scipy').sparse.eye(4096)),amplitude)
                        projected/=np.linalg.norm(projected)
                        sp=exact_spectra(projected**2,spins,geometry(2))['connected_spectrum']
                        forward.append(dict(tau=tau,fidelity=float((exact@projected)**2),
                            max_spectral_relative_error=float(np.max(abs(sp-truth)/truth))))
                    np.savez_compressed(args.output/(name+'.npz'),logs=logs,**{'transformer_'+k:v for k,v in oldlogs.items()})
                    row=dict(gamma=gamma,sample_id=sid,kind=kind,seed=seed,**record,
                        energy_error=record['energy']-exact_energy,fidelity=float((exact@amplitude)**2),
                        max_spectral_relative_error=float(np.max(abs(spectrum-truth)/truth)),
                        spectrum=spectrum.tolist(),exact_spectrum=truth.tolist(),projection=forward,
                        source_sha256=sources,input_sha256=sha(gp),npz_sha256=sha(args.output/(name+'.npz')),
                        ground_state_qualified=False)
                    write(path,row)
                rows.append(row)
                write(args.output/'review.json',dict(rows=rows,scope='L2 full-basis training diagnostic; not sampled-VMC scaling'))
                print(json.dumps({k:row[k] for k in ['gamma','sample_id','kind','seed','energy_error','fidelity','max_spectral_relative_error','seconds']}),flush=True)
    write(args.output/'status.json',dict(state='completed',completed=len(rows)))


if __name__=='__main__':main()
