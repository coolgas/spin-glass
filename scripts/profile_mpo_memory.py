"""Bounded-process MPO representation benchmark; no optimization or sampling."""
import argparse
import json
import time
import resource
from pathlib import Path
import numpy as np
from rydberg_kagome import dmrg
from rydberg_kagome.results import load_checkpoint,digest
from adaptive_memory import snapshot

p=argparse.ArgumentParser()
p.add_argument('--case',type=Path,required=True)
p.add_argument('--builder',choices=['legacy','compact'],required=True)
a=p.parse_args()
resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
t=time.monotonic()
with (a.case/'model_hash.json').open() as f: expected=json.load(f)['sha256']
assert digest(a.case/'model.pkl.gz')==expected
model=load_checkpoint(a.case/'model.pkl.gz')
with (a.case/'result.json').open() as f: recorded=json.load(f)
assert digest(a.case/'geometry.npz')==recorded['hashes']['geometry.npz']
with np.load(a.case/'geometry.npz') as g: v=g['interactions']
mpo,order=dmrg.build_mpo(model,interactions=v if a.builder=='compact' else None)
setup=time.monotonic()-t
energies=[]
for seed in (211,713):
    psi=dmrg.initial_state(mpo,seed)
    energies.append(float(mpo.H_MPO.expectation_value(psi)))
print(json.dumps(dict(builder=a.builder,setup_seconds=setup,total_seconds=time.monotonic()-t,
    bond_dimensions=list(map(int,mpo.H_MPO.chi)),random_state_energies=energies,
    peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,**snapshot())),flush=True)
