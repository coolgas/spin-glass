"""Deterministic infinite-population targets at production forward tag times."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply

from kagome_qea.exact import operator_matrix
from transformer_qmc.experiment import write_json
from transformer_qmc.frozen_scan import digest
from transformer_qmc.physics import basis, correlations, make_af_sample


def context(root, path):
    torch.set_num_threads(1)
    root, path = Path(root), Path(path)
    target_path = root/'finite_forward'/path.name
    if target_path.exists():
        return str(target_path)
    exact = json.loads(path.read_text())
    if digest(path.with_suffix('.npz')) != exact['cache_sha256']:
        raise RuntimeError('cache identity mismatch')
    h,g,sid = exact['h'],exact['gamma'],exact['sample_id']
    H = operator_matrix(make_af_sample(2,sid).hamiltonian(h),g)
    shifted = H-exact['energy']*eye(H.shape[0],format='csr')
    trace = float(shifted.diagonal().sum())
    spins = basis(12)
    with np.load(path.with_suffix('.npz')) as cache:
        logs = cache['logs'].copy()
    rows = []
    for kind in ('constant','network'):
        psi = np.ones(len(spins)) if kind=='constant' else np.exp(logs)
        psi /= np.linalg.norm(psi)
        # All tag times: main 40+8c, eq audit 72+8c, forward audit 40+16c.
        left = expm_multiply(-shifted,psi,start=40.,stop=280.,num=31,traceA=-trace)
        right = {f:expm_multiply(-f*shifted,psi,traceA=-f*trace) for f in (8.,16.)}
        for setting,indices,f in [('main',range(16),8.),('equilibration',range(4,20),8.),
                                  ('forward',range(0,31,2),16.)]:
            values = []
            for i in indices:
                p = left[i]*right[f]
                if p.min() < -1e-12:
                    raise RuntimeError('negative exact forward probability')
                c = correlations(np.maximum(p,0),spins)
                values.append([c['q_mean'],c['q2'],12*c['q2'],c['chi_connected']])
            values = np.asarray(values)
            rows.append(dict(guide=kind,setting=setting,
                             mean=dict(zip(['q','q2','chi_sg','chi_connected'],values.mean(0))),
                             cycles=values))
    write_json(target_path,dict(h=h,gamma=g,sample_id=sid,rows=rows,
                               cache_sha256=exact['cache_sha256'],source_sha256=digest(__file__)))
    return str(target_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    (args.results/'finite_forward').mkdir(exist_ok=True)
    paths = sorted((args.results/'cache').glob('*.json'))
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(context,str(args.results),str(p)) for p in paths]
        for count,f in enumerate(as_completed(futures),1):
            print('REFERENCE',f.result(),flush=True)
            write_json(args.results/'reference_status.json',dict(completed=count,expected=len(paths),
                        state='completed' if count==len(paths) else 'running',
                        elapsed_seconds=time.time()-started,updated_unix=time.time()))


if __name__ == '__main__':
    main()
