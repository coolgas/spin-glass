"""Compare BLAS threads on identical saved DMRG states without modifying them."""
import argparse
import json
import pickle
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from tenpy.algorithms import dmrg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dimension', type=int, default=64)
    args = parser.parse_args()
    with (args.source/'model.pkl').open('rb') as stream:
        mpo = pickle.load(stream)['mpo']
    checkpoint = (args.source/f'seed0_D{args.dimension}.pkl').read_bytes()
    records = []
    # Keep every timing on the same checkpoint and use the same two sweeps.
    for threads in (1, 2, 4, 8):
        with threadpool_limits(limits=threads):
            psi = pickle.loads(checkpoint)['psi']
            engine = dmrg.TwoSiteDMRGEngine(psi, mpo, dict(mixer=False,
                min_sweeps=2, max_sweeps=2, max_E_err=0., max_S_err=0.,
                trunc_params=dict(chi_max=args.dimension, svd_min=1e-12)))
            started = time.monotonic()
            engine.run()
            seconds = time.monotonic()-started
            records.append(dict(threads=threads, seconds=seconds,
                energy=float(np.real(mpo.H_MPO.expectation_value(psi))),
                threadpools=threadpool_info()))
            args.output.write_text(json.dumps(dict(source=str(args.source),
                scope=f'Two DMRG sweeps at D{args.dimension}; identical seed0 checkpoint',
                records=records), indent=2)+'\n')
            print(json.dumps({k:records[-1][k] for k in ('threads','seconds','energy')}), flush=True)


if __name__ == '__main__':
    main()
