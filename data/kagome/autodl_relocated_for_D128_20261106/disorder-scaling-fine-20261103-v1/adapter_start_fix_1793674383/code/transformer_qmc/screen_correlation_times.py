"""Deterministic low-field forward-time escalation on frozen campaign inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply
from threadpoolctl import threadpool_limits

from kagome_qea.exact import operator_matrix
from transformer_qmc.physics import make_af_sample, basis
from transformer_qmc.production_protocol import exact_spectra, geometry
from transformer_qmc.correlation_lengths import length


def one(task):
    root,sid=task
    root=Path(root)
    stem=f'L2_s{sid}_g0p5_h0p15'
    ep=root/'exact'/(stem+'.json');gp=root/'guides'/(stem+'.json')
    e=json.loads(ep.read_text());g=json.loads(gp.read_text())
    archive=gp.with_suffix('.npz')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==g['npz_sha256']
    with np.load(archive) as d:psi=np.exp(d['logs']-d['logs'].max())
    h=operator_matrix(make_af_sample(2,sid).hamiltonian(.15),.5)-e['energy']*eye(4096,format='csr')
    tr=float(h.diagonal().sum());spins=basis(12);lattice=geometry(2);rows=[]
    with threadpool_limits(limits=1):
        for fw in [64.,128.]:
            left=expm_multiply(-34*h,psi,traceA=-34*tr)
            right=expm_multiply(-fw*h,psi,traceA=-fw*tr)
            spectra=[]
            for cycle in range(16):
                spectra.append(exact_spectra(left*right,spins,lattice)['connected_spectrum'])
                if cycle<15:left=expm_multiply(-fw*h,left,traceA=-fw*tr)
            rows.append(dict(forward=fw,spectrum=np.mean(spectra,axis=0).tolist()))
    return dict(sample_id=sid,exact=e['spectrum'],rows=rows,guide_sha256=g['npz_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();manifest=json.loads((a.root/'manifest.json').read_text())
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        records=list(pool.map(one,[(str(a.root),s) for s in manifest['development_ids']]))
    exact=np.mean([r['exact'] for r in records],axis=0);summary=[]
    for index,fw in enumerate([64.,128.]):
        s=np.mean([r['rows'][index]['spectrum'] for r in records],axis=0)
        summary.append(dict(forward=fw,xi=float(length(s,2)),exact_xi=float(length(exact,2)),
                            spectral_relative_bias=((s-exact)/exact).tolist()))
    result=dict(equilibration=32.,measurement=2.,cycles=16,gamma=.5,h=.15,records=records,summary=summary,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.root/'long_time_screen.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
