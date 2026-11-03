"""Reproduce the preserved failed start and compare log-domain arithmetic."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from .correlation_lengths import sha
from .physics import make_af_sample
from .production_protocol import identity
from .projector import propagate, rates_and_local
from .simple_guides import RBM, SimpleGuide


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.bundle
    torch.set_num_threads(1)
    source=root/'preserved_failure/projector.py'
    spec=importlib.util.spec_from_file_location('transformer_qmc.old_projector',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model_path=root/'preserved_failure/L6_s60000_g0p6_rbm_2_anneal_seed0.pt'
    model=RBM(108,2)
    model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=False)['model'])
    guide=SimpleGuide(model);ham=make_af_sample(6,60000).hamiltonian(.15)
    task=dict(L=6,gamma=.6,sample_id=60000,kind='rbm_2',training_seed=0,audit='start',
              walkers=512,equil=16.,forward=16.,start='uniform',delta=.1,cycles=8,repeat=0)
    rng=np.random.default_rng(int(identity(task)[:16],16))
    states=(1-2*rng.integers(2,size=(4,512,108))).astype(np.int8)
    shift=float(rates_and_local(ham,.6,guide,states[0,:,None,:])[1].mean())
    seedstate=rng.bit_generator.state
    record=dict(task=task,source_sha256=sha(source),model_sha256=sha(model_path),initial_shift=shift)
    for name,fn in [('original',module.propagate),('log_weight',propagate)]:
        rng.bit_generator.state=seedstate;t=time.monotonic();diag={}
        try:
            inputs=(ham,.6,guide,states[0,:,None,:],shift,.1,rng)
            result=fn(*inputs,diagnostics=diag) if name=='log_weight' else fn(*inputs)
            record[name]=dict(mixed=result[3],growth=result[4],ess=result[5],diagnostics=diag,seconds=time.monotonic()-t)
        except Exception as error:
            record[name]=dict(error=repr(error),seconds=time.monotonic()-t)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
