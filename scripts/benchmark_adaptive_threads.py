"""Replay one committed start-sweep without modifying the scientific case."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
    import numpy as np
    from rydberg_kagome import adaptive,dmrg
    from rydberg_kagome.results import load_checkpoint,digest,source_hashes
    record=json.loads((a.case/'resume.json').read_text()); identity=record['identity']
    if source_hashes()!=identity['inputs']['sources']: raise ValueError('Source mismatch')
    model=load_checkpoint(a.case/'model.pkl.gz')
    with np.load(a.case/'geometry.npz') as g: mpo,_=dmrg.build_mpo(model,interactions=g['interactions'])
    entry=record['files'][0]
    if digest(a.case/entry['file'])!=entry['sha256']: raise ValueError('Payload mismatch')
    engine=adaptive.restore_engine(load_checkpoint(a.case/entry['file'],mpo.lat.mps_sites()),mpo)
    before=engine.sweeps; started=time.monotonic(); energy,_=adaptive.one_sweep(engine)
    a.output.write_text(json.dumps(dict(generation=record['generation'],start=0,
        before=before,after=engine.sweeps,seconds=time.monotonic()-started,
        energy=float(energy+model['constant']),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n')


if __name__=='__main__': main()
