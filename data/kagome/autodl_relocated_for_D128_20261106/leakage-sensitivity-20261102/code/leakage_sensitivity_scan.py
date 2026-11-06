"""Equal-quota leakage response maps on saved geometries (GPU, float64).

Keeps 100 geometries in each site/bin. Derivative convergence is checked on
every geometry over the entire occupation window, with adaptive step halving.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from geometry_qfi_binned import Solver

C6S = [65536,130728,260767,520160,1037582,2069702,4128510,8235289,16427232,32768000]
TIMES = np.arange(2000, dtype=float)*.01
SNAPSHOTS = np.array([5.,10.,15.,19.99])
# Equal-duration windows for comparing the same rate at different times;
# original consecutive windows and cumulative windows are secondary outputs.
WINDOWS = np.array([[1,5],[6,10],[11,15],[15.99,19.99],
                    [5,10],[10,15],[15,19.99],[1,10],[1,15],[1,19.99]])
ENDPOINTS = np.rint(WINDOWS/.01).astype(int)


def integrate(values):
    return np.sum((values[...,1:]+values[...,:-1])*.005,axis=-1)/19.99


@torch.inference_mode()
def responses(solver, offsets, site, c6, step, times=TIMES, flux=False):
    offsets = torch.as_tensor(offsets,dtype=torch.float64,device=solver.device)
    shifts = (0,step,-step,step/2,-step/2)
    specs=[]
    for shift in shifts:
        probe=offsets.clone();probe[:,site]+=shift
        specs.append(solver.spectrum(probe,c6))
    outputs=[[] for _ in shifts]
    velocities=[]
    norm_error=0.
    for start in range(0,len(times),250):
        ts=torch.tensor(times[start:start+250],device=solver.device,dtype=torch.float64)
        for j,spec in enumerate(specs):
            psi=solver.states(spec,ts)
            prob=psi.abs().square()
            norm_error=max(norm_error,float((prob.sum(1)-1).abs().max()))
            outputs[j].append(prob[:,solver.outside].sum(1).cpu().numpy())
            if flux and j==0:
                # -i H psi = exact time derivative of the spectral expansion.
                w,v,coeff=spec
                hpsi=v @ (w[:,:,None]*torch.exp(-1j*w[:,:,None]*ts[None,None,:])*coeff[:,:,None])
                velocity=2*(psi[:,solver.outside].conj()*(-1j*hpsi[:,solver.outside])).real.sum(1)
                velocities.append(velocity.cpu().numpy())
    values=[np.concatenate(parts,axis=1) for parts in outputs]
    full=(values[1]-values[2])/(2*step)
    half=(values[3]-values[4])/step
    assert norm_error<1e-9
    assert all(np.isfinite(v).all() and v.min()>-1e-10 and v.max()<1+1e-10 for v in values)
    return values[0],full,half,norm_error,np.concatenate(velocities,axis=1) if flux else None


def converged(solver,offsets,site,c6,batch=20):
    leakage=[]; derivative=[]; errors=[]; steps=[]; norms=[]
    for lo in range(0,len(offsets),batch):
        probe=offsets[lo:lo+batch]
        step=1e-4
        for attempt in range(9):
            ell,d1,d2,norm,_=responses(solver,probe,site,c6,step)
            err=np.sqrt(np.mean((d1-d2)**2,axis=1))
            scale=np.sqrt(np.mean(d2**2,axis=1))
            # Full-window and pointwise convergence; retain error traces for
            # conservative endpoint masks as well as an aggregate check.
            ok=(err <= 1e-8+1e-3*scale) & (np.max(abs(d1-d2),axis=1)<=1e-7+3e-3*np.max(abs(d2),axis=1))
            if np.all(ok):break
            step/=2
        else:raise RuntimeError(f'Derivative did not converge: L={solver.length} site={site} C6={c6}')
        assert np.max(abs(ell[:,0]))<1e-10 and np.max(abs(d2[:,0]))<1e-8
        leakage.append(ell);derivative.append(d2);errors.append(abs(d1-d2));steps.extend([step/2]*len(probe));norms.append(norm)
    return np.concatenate(leakage),np.concatenate(derivative),np.concatenate(errors),np.array(steps),max(norms)


def summarize_bin(ell,chi,error,seed):
    d=np.sqrt(np.mean(chi**2,axis=0))
    numerical=np.sqrt(np.mean(error**2,axis=0))
    endpoint_d=d[ENDPOINTS]
    # Do not replace zero or unresolved amplitudes with an arbitrary epsilon.
    resolved=np.all(endpoint_d>np.maximum(1e-7,10*numerical[ENDPOINTS]),axis=1)
    signed=np.full(len(WINDOWS),np.nan)
    signed[resolved]=np.log(endpoint_d[resolved,1]/endpoint_d[resolved,0])/np.diff(WINDOWS,axis=1).ravel()[resolved]
    per_s2=integrate(chi**2)
    per_l=integrate(ell)
    rng=np.random.default_rng(seed)
    resample=rng.integers(0,100,(300,100))
    boot_s=np.sqrt(per_s2[resample].mean(1))
    boot_l=per_l[resample].mean(1)
    # Resample whole matched trajectories, not times.
    boot_d=np.sqrt((chi[:,ENDPOINTS]**2)[resample].mean(1))
    with np.errstate(divide='ignore',invalid='ignore'):
        boot_rate=np.log(boot_d[:,:,1]/boot_d[:,:,0])/np.diff(WINDOWS,axis=1).ravel()
    boot_rate[:,~resolved]=np.nan
    ci_signed=np.full((len(WINDOWS),2),np.nan);ci_absolute=ci_signed.copy()
    if np.any(resolved):
        ci_signed[resolved]=np.quantile(boot_rate[:,resolved],[.025,.975],axis=0).T
        ci_absolute[resolved]=np.quantile(abs(boot_rate[:,resolved]),[.025,.975],axis=0).T
    return dict(mean_leakage=ell.mean(0),D=d,numerical_error=numerical,
                mean_leakage_integral=per_l.mean(),S=np.sqrt(per_s2.mean()),
                per_geometry_leakage=per_l,per_geometry_S=np.sqrt(per_s2),
                lambda_signed=signed,lambda_absolute=abs(signed),resolved=resolved,
                S_ci=np.quantile(boot_s,[.025,.975]),leakage_ci=np.quantile(boot_l,[.025,.975]),
                lambda_signed_ci=ci_signed,lambda_absolute_ci=ci_absolute)


def validate(out,device):
    from scipy.linalg import expm
    from src.model import RydSimFull1D
    records=[]
    for length in (6,7):
        solver=Solver(length,device)
        for site,a,c6 in ((0,-2.8,65536),(length//2,1.,2069702),(length-1,2.8,32768000)):
            noise=np.random.default_rng(182+site).normal(0,.02,(1,length));noise[0,site]=a
            times=np.array([0.,1.,5.,19.99])
            ell,d1,d2,norm,velocity=responses(solver,noise,site,c6,1e-4,times,flux=True)
            expected=[]
            reference=RydSimFull1D(length,2.5,0,c6,8)
            H,_=reference.build_Ham(noise[0])
            outside=solver.outside.cpu().numpy()
            for t in times:
                psi=expm(-1j*H*t)[:,0]
                expected.append(np.sum(abs(psi[outside])**2))
            np.testing.assert_allclose(ell[0],expected,atol=2e-10,rtol=1e-8)
            dt=1e-5
            nearby=np.ravel(np.stack([times[1:]-dt,times[1:]+dt],axis=1))
            local=responses(solver,noise,site,c6,1e-4,nearby)[0][0].reshape(-1,2)
            np.testing.assert_allclose(velocity[0,1:],np.diff(local,axis=1).ravel()/(2*dt),atol=2e-7,rtol=1e-4)
            records.append(dict(length=length,site=site,c6=c6,matrix_exponential_error=float(np.max(abs(ell[0]-expected))),norm_error=norm))
    report=dict(status='PASS',checks=records)
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print('VALIDATION_PASS',flush=True)


def run(args):
    start=time.perf_counter()
    config=dict(quota=100,lengths=[6,7],sites='all, separate',bins=20,c6_values=C6S,
                time_grid='0:0.01:19.99',windows=WINDOWS.tolist(),snapshot_times=SNAPSHOTS.tolist(),primary_windows='first four; fixed duration 4 ending at each snapshot',input_defect_sigma=1.,background_sigma=.02,
                x_coordinate='RMS over all nearest-neighbor bonds and all 100 geometries of bond minus 8 micrometers',
                projector='no adjacent excitations, fixed across all parameters',
                precision='float64 and complex128',derivative='central difference, accepted half step, adaptive convergence on every geometry',
                bootstrap='300 whole-geometry resamples, percentile 95% intervals',
                rate='absolute logarithmic endpoint growth of ensemble RMS displacement derivative',
                numerical_mask='both endpoint D > max(1e-7, 10*RMS derivative-step discrepancy)',
                original_source=str(args.source),status='running')
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    for length in (6,7):
        solver=Solver(length,args.device)
        for site in range(length):
            with h5py.File(args.source/f'L{length}_site{site}.h5') as src:
                offsets=src['positions_offset'][:]
                assert offsets.shape==(20,100,length)
                np.testing.assert_array_equal(src['count'][:],np.full(20,100))
                edges=src['bin_edges'][:]
                previous_leakage=src['leakage'][:]
                np.testing.assert_array_equal(src['times'][:],np.array([5.,10.,15.,20.]))
                membership=np.searchsorted(edges,abs(offsets[:,:,site]),side='right')-1
                np.testing.assert_array_equal(membership,np.broadcast_to(np.arange(20)[:,None],(20,100)))
            bond_delta=np.diff(offsets,axis=-1)
            sigma=np.sqrt(np.mean(bond_delta**2,axis=(1,2)))
            digest=hashlib.sha256(offsets.tobytes()).hexdigest()
            path=args.out/f'L{length}_site{site}.h5'
            with h5py.File(path,'a') as f:
                if 'geometry_sha256' in f.attrs:assert f.attrs['geometry_sha256']==digest
                f.attrs.update(length=length,site=site,geometry_sha256=digest)
                if 'positions_offset' not in f:
                    for k,v in dict(positions_offset=offsets,sigma_b=sigma,mean_abs_displacement=abs(offsets[:,:,site]).mean(1),bin_edges=edges,c6_values=C6S,times=TIMES,windows=WINDOWS,count=np.full(20,100)).items():f.create_dataset(k,data=v)
                for ci,c6 in enumerate(C6S):
                    for m in range(20):
                        key=f'c{ci}/b{m}'
                        if key in f and f[key].attrs.get('complete',False):continue
                        if key in f:del f[key]  # Only an unfinished group owned by this resumable job.
                        ell,chi,err,steps,norm=converged(solver,offsets[m],site,c6,args.batch)
                        np.testing.assert_allclose(ell[:,[500,1000,1500]],previous_leakage[ci,m,:,:3],atol=3e-9,rtol=1e-7)
                        summary=summarize_bin(ell,chi,err,20261102+length*10000+site*100+m)
                        g=f.create_group(key)
                        for k,v in summary.items():g.create_dataset(k,data=v)
                        for k,v in dict(leakage=ell,chi=chi).items():
                            g.create_dataset(k,data=v.astype('f4'),compression='gzip',compression_opts=1,shuffle=True)
                        g.create_dataset('accepted_step',data=steps)
                        g.attrs.update(complete=True,norm_error=norm,max_derivative_rms_discrepancy=float(np.sqrt(np.mean(err**2,axis=1)).max()))
                        f.flush()
                        print(f'L={length} site={site} C6={c6} bin={m+1}/20 n=100 elapsed={time.perf_counter()-start:.1f}s',flush=True)
                f.attrs['complete']=True
    config.update(status='complete',seconds=time.perf_counter()-start)
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    print('SCAN_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('validate','run'),required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch',type=int,default=20)
    args=p.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='validate':validate(args.out,args.device)
    else:run(args)


if __name__=='__main__':main()
