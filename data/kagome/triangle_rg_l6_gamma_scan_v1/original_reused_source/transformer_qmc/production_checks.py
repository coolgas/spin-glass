"""Resumable full-grid direct-guide checks and exact production estimator references.

GPU and CPU stages have independent locks and immutable source/checkpoint identity.
Pointwise precision agreement is explicitly distinct from ensemble qualification.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply

from kagome_qea.exact import operator_matrix, moments_from_probability
from .diagnose_shared import reference
from .direct_gfmc_pilot import CachedDirectGuide
from .experiment import write_json
from .frozen_scan import SELECTED_SHA256, key, model_from, stage_lock
from .physics import basis, correlations, make_af_sample
from .production_protocol import GAMMAS, FIELDS, exact_spectra, geometry, source_hashes, identity


def contexts():
    return [(sid, g, h) for sid in range(50000, 50008) for g, h in itertools.product(GAMMAS, FIELDS)]


def check_record(path, signature):
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    if data.get('signature') != signature or data.get('status') != 'completed':
        raise RuntimeError(f'invalid existing record: {path}')
    return True


def gpu_checks(args, signature):
    high = model_from(args.checkpoint, 'cuda')
    low = model_from(args.checkpoint, 'cuda').float().eval()
    torch.set_float32_matmul_precision('highest')
    spins = basis(12)
    tensor = torch.as_tensor(spins, device='cuda', dtype=torch.long)
    flips = np.arange(4096)[:, None] ^ (1 << np.arange(12))
    destination = args.output/'direct_checks'
    destination.mkdir(exist_ok=True)
    for number, (sid, gamma, h) in enumerate(contexts(), 1):
        path = destination/(key(2, sid, gamma, h, 1.)+'.json')
        if check_record(path, signature):
            continue
        start = time.monotonic()
        sample = make_af_sample(2, sid); ham = sample.hamiltonian(h)
        high.set_hamiltonian(ham, gamma); low.set_hamiltonian(ham, gamma)
        logs64 = high.log_amplitudes(tensor).double().cpu().numpy()
        logs32 = low.log_amplitudes(tensor).double().cpu().numpy()
        expected = logs64[flips]-logs64[:, None]
        ratios64 = high.all_flip_log_ratios(tensor).double().cpu().numpy()
        ratios32 = low.all_flip_log_ratios(tensor).double().cpu().numpy()
        if not all(np.isfinite(x).all() for x in (logs64, logs32, ratios64, ratios32)):
            raise FloatingPointError('nonfinite direct guide output')
        error64 = float(abs(ratios64-expected).max())
        norm64 = float(abs(np.exp(2*logs64).sum()-1))
        error32 = float(abs(ratios32-ratios64).max())
        norm32 = float(abs(np.exp(2*logs32).sum()-1))
        if error64 > 1e-10 or norm64 > 1e-10 or error32 > 1e-3 or norm32 > 1e-4:
            raise AssertionError((error64, norm64, error32, norm32))
        # Exercise small cache eviction, repeats, and Hamiltonian reconstruction.
        cached = CachedDirectGuide(high, max_entries=16)
        for indices in (np.arange(8), np.arange(8,24), np.arange(8)):
            np.testing.assert_allclose(cached.all_flip_log_ratios(spins[indices]), expected[indices], atol=1e-10, rtol=0)
        alternate = make_af_sample(2, sid+1000).hamiltonian(.05 if h != .05 else .1)
        high.set_hamiltonian(alternate, gamma+.1)
        fresh = CachedDirectGuide(high)
        probe = fresh.all_flip_log_ratios(spins[:4])
        alt_logs = fresh.log_amplitudes(spins)
        np.testing.assert_allclose(probe, (alt_logs[flips]-alt_logs[:,None])[:4], atol=1e-10, rtol=0)
        high.set_hamiltonian(ham, gamma)
        np.testing.assert_allclose(CachedDirectGuide(high).all_flip_log_ratios(spins[:4]),expected[:4],atol=1e-10,rtol=0)
        npz = path.with_suffix('.npz'); tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, logs64=logs64, logs32=logs32, ratios64=ratios64, ratios32=ratios32)
        tmp.replace(npz)
        write_json(path,dict(status='completed', signature=signature, sample_id=sid, L=2,
                             gamma=gamma, h=h, checkpoint_sha256=SELECTED_SHA256,
                             direct64_error=error64, normalization64_error=norm64,
                             max_log_ratio_precision_error=error32, normalization32_error=norm32,
                             max_relative_rate_error=float(abs(np.expm1(ratios32-ratios64)).max()),
                             cache_eviction_and_context_checks=True, precision_ensemble_qualified=False,
                             npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                             seconds=time.monotonic()-start))
        write_json(args.output/'gpu_check_status.json',dict(state='running', completed=number, expected=len(contexts()),
                                                          updated_unix=time.time(), pid=os.getpid()))
        print('DIRECT_CHECK',number,path.name,flush=True)
    write_json(args.output/'gpu_check_status.json',dict(state='completed',completed=len(contexts()),
                                                       expected=len(contexts()),updated_unix=time.time()))


def exact_context(payload):
    output, baseline, signature, sid, gamma, h = payload
    torch.set_num_threads(1)
    output, baseline = Path(output), Path(baseline)
    name = key(2,sid,gamma,h,1.)
    path=output/'exact'/(name+'.json')
    if check_record(path,signature):return str(path)
    start=time.monotonic();sample=make_af_sample(2,sid);ham=sample.hamiltonian(h)
    matrix=operator_matrix(ham,gamma);spins=basis(12);lattice=geometry(2)
    original=baseline/'cache'/(name+'.json')
    reused=False
    if original.exists():
        info=json.loads(original.read_text()); npz=original.with_suffix('.npz')
        if hashlib.sha256(npz.read_bytes()).hexdigest()!=info['cache_sha256']:
            raise RuntimeError('baseline cache identity mismatch')
        with np.load(npz) as data:
            psi=data['exact'].copy()
            np.testing.assert_array_equal(data['bonds'],sample.bonds)
            np.testing.assert_array_equal(data['J'],sample.J)
        energy=info['energy']; reused=True
    else:
        energy,psi,_=reference(matrix,h)
    residual=float(np.linalg.norm(matrix@psi-energy*psi))
    if residual>1e-9 or abs(float(psi@psi)-1)>1e-10:
        raise RuntimeError('exact reference residual/normalization failed')
    p=psi**2
    obs=correlations(p,spins,sample.points,2)
    obs.update(moments_from_probability(p,12));obs.update(exact_spectra(p,spins,lattice))
    npz=path.with_suffix('.npz');tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,amplitude=psi,bonds=sample.bonds,J=sample.J);tmp.replace(npz)
    write_json(path,dict(status='completed',signature=signature,L=2,sample_id=sid,gamma=gamma,h=h,
                         energy=energy,residual=residual,observables=obs,reused_baseline=reused,
                         npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),seconds=time.monotonic()-start))
    return str(path)


def cpu_checks(args,signature):
    (args.output/'exact').mkdir(exist_ok=True)
    tasks=[(str(args.output),str(args.baseline),signature,sid,g,h) for sid,g,h in contexts()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(exact_context,t) for t in tasks]
        for count,f in enumerate(as_completed(futures),1):
            print('EXACT_REFERENCE',count,f.result(),flush=True)
            write_json(args.output/'exact_status.json',dict(state='completed' if count==len(tasks) else 'running',
                                                           completed=count,expected=len(tasks),updated_unix=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--stage',choices=['gpu','exact'],required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2 if args.stage=='gpu' else 1);torch.set_num_interop_threads(1)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=SELECTED_SHA256:
        raise RuntimeError('checkpoint hash mismatch')
    sources=source_hashes(Path(__file__).resolve().parents[1])
    signature=identity(dict(source=sources,checkpoint=SELECTED_SHA256,contexts=contexts()))
    manifest=dict(signature=signature,source_sha256=sources,checkpoint_sha256=SELECTED_SHA256,
                  contexts=contexts(),precision='float64 comparison; float32 candidate; double projector')
    with stage_lock(args.output,'checks_'+args.stage):
        target=args.output/('manifest_'+args.stage+'.json')
        if target.exists() and json.loads(target.read_text())!=json.loads(json.dumps(manifest)):
            raise RuntimeError('source identity changed')
        write_json(target,manifest)
        try:
            (gpu_checks if args.stage=='gpu' else cpu_checks)(args,signature)
        except BaseException as error:
            write_json(args.output/(args.stage+'_failure.json'),dict(error=repr(error),updated_unix=time.time()))
            raise


if __name__=='__main__':
    main()
