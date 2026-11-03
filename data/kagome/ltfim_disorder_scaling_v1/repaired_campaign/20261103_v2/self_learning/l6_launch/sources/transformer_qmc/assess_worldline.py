"""Offline diagnostics for sampled worldline pilots; exact references are evaluation-only."""
import argparse
import json
from pathlib import Path
import numpy as np
from .correlation_lengths import length,sha
from .correlation_remedy_campaign import write
from .production_protocol import geometry,identity
from .independent_replica_projector import spectrum_from_moments
from .validate_correlation_remedy import label


def assess(root,references):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    expected={identity(t)[:24]:t for t in manifest['tasks']};groups={};records=[]
    for p in sorted((root/'runs').glob('*.json')):
        r=json.loads(p.read_text());t=r['task'];assert expected[p.stem]==t
        assert sha(p.with_suffix('.npz'))==r['npz_sha256']
        with np.load(p.with_suffix('.npz')) as a:data={k:a[k].copy() for k in a.files}
        assert data['magnetization_blocks'].shape==(t['blocks'],3*t['L']**2)
        assert np.all(data['counts']>0)
        assert all(np.isfinite(data[k]).all() for k in data)
        key=(t['kind'],t['L'],t['gamma'],t['sample_id'],t['beta'],t['dt'])
        groups.setdefault(key,{})[t['replica']]=(r,data);records.append(r)
    rows=[]
    for key,group in sorted(groups.items()):
        if set(group)!=set(range(4)):continue
        kind,L,g,sid,beta,dt=key;rs=[group[i][0] for i in range(4)];ds=[group[i][1] for i in range(4)]
        m=np.array([d['magnetization_blocks'] for d in ds]);q=np.array([d['pair_blocks'] for d in ds]);lattice=geometry(L)
        spectrum=spectrum_from_moments(m.mean(1),q.mean(1),lattice)
        select=[0,*lattice['kmin_indices']];spec=spectrum[select]
        # length() expects [S0, shortest three momenta] and uses the physical size L.
        xi=float(length(spec,L));row=dict(kind=kind,L=L,gamma=g,sample_id=sid,beta=beta,dt=dt,
            spectrum=spec.tolist(),xi=xi if np.isfinite(xi) else None,
            mean_seconds=float(np.mean([r['seconds'] for r in rs])),
            worldline_bytes=rs[0]['worldline_bytes'],
            minimum_exchange_acceptance=min(r['minimum_exchange_acceptance'] for r in rs),
            round_trips=[r['round_trips'] for r in rs],
            replica_energy=[r['energy'] for r in rs],ground_state_qualified=False)
        if L==2:
            ep=Path(references)/(label(g,sid)+'.json');exact=json.loads(ep.read_text())
            target=np.array(exact['exact_spectrum']);row['exact_spectrum']=target.tolist()
            row['spectral_relative_bias']=((spec-target)/target).tolist()
            row['exact_xi']=float(length(target,2));row['exact_energy']=exact['exact_energy']
            rng=np.random.default_rng(1102);boots=[]
            for _ in range(1000):
                ix=rng.integers(m.shape[1],size=(4,m.shape[1]))
                mm=m[np.arange(4)[:,None],ix].mean(1);qq=q[np.arange(4)[:,None],ix].mean(1)
                boots.append(spectrum_from_moments(mm,qq,lattice))
            boots=np.array(boots);row['block_bootstrap_spectral_se']=boots.std(0,ddof=1).tolist()
            row['spectral_relative_95_bounds']=((abs(spec-target)+1.96*boots.std(0,ddof=1))/target).tolist()
            row['halves']=[spectrum_from_moments(m[:,sl].mean(1),q[:,sl].mean(1),lattice).tolist()
                           for sl in [slice(None,m.shape[1]//2),slice(m.shape[1]//2,None)]]
            row['uncertainty']='Provisional block bootstrap; block-length and independent-run stability still required.'
        rows.append(row)
    review=dict(completed=len(records),expected=len(expected),rows=rows,
        qualification='Development results only. No field is automatically qualified from one block bootstrap or a cost pilot.')
    write(root/'review.json',review);return review


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--references',type=Path,required=True)
    a=p.parse_args();r=assess(a.root,a.references);print(json.dumps(r,indent=2))
