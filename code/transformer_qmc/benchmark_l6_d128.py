"""Identical saved-state, two-sweep D128 CPU timing; no production outputs."""
import argparse
import json
import pickle
from pathlib import Path
import time

import numpy as np
from tenpy.algorithms import dmrg
from threadpoolctl import threadpool_limits

from .triangle_correlation_dmrg import digest, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    old=json.loads((args.case/'result.json').read_text())
    for filename in ['model.pkl','seed0_D64.pkl']:
        assert digest(args.case/filename)==old['hashes'][filename]
    saved=pickle.loads((args.case/'model.pkl').read_bytes())
    state=(args.case/'seed0_D64.pkl').read_bytes()
    records=[]
    for threads in (1,2,4,8):
        with threadpool_limits(limits=threads):
            psi=pickle.loads(state)['psi']
            engine=dmrg.TwoSiteDMRGEngine(psi,saved['mpo'],dict(mixer=True,
                mixer_params=dict(amplitude=1e-3,decay=2.,disable_after=15),
                min_sweeps=2,max_sweeps=2,max_E_err=0.,max_S_err=0.,max_trunc_err=None,
                trunc_params=dict(chi_max=128,svd_min=1e-12)))
            begin=time.monotonic()
            energy,psi=engine.run()
            records.append(dict(threads=threads,seconds=time.monotonic()-begin,
                                energy=float(np.real(energy)),max_bond=max(psi.chi)))
            print(json.dumps(records[-1]),flush=True)
    disagreement=max(r['energy'] for r in records)-min(r['energy'] for r in records)
    assert disagreement/108 < 1e-8, disagreement
    write(args.output,dict(records=records,energy_disagreement_per_spin=disagreement/108,
          parent_checkpoint_sha256=old['hashes']['seed0_D64.pkl'],
          scope='Two sweeps for timing only; original production optimizer unchanged'))


if __name__=='__main__':
    main()
