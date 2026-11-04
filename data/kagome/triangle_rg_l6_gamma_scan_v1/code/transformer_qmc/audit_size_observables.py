"""Independent higher-statistics diagonal-observable check of frozen final guides."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .experiment import load_checkpoint, write_json
from .physics import make_af_sample
from .projector import tags
from .unified import UnifiedTransformer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    rows=json.loads((args.results/'summary.json').read_text())
    if len(rows)!=6:
        raise RuntimeError('audit requires all six completed refinements')
    output=[]
    for run in rows:
        state=load_checkpoint(args.results/run['final_checkpoint'],'cuda')
        ham=make_af_sample(run['L'],0).hamiltonian(.3)
        model=UnifiedTransformer(ham,.5,**state['metadata']['architecture']).cuda()
        model.load_state_dict(state['model'])
        model.eval()
        seed=170000+run['L']*100+run['seed']
        generator=torch.Generator(device='cuda').manual_seed(seed)
        values={key:[] for key in ('q','q2','chi_connected','magnetization')}
        for _ in range(64):
            spins=model.sample(512*4,generator).cpu().numpy().reshape(512,4,ham.n_sites)
            measured=tags(spins)
            for key in ('q','q2','chi_connected'):
                values[key].append(float(measured[key].mean()))
            values['magnetization'].append(float(spins.mean()))
        result=dict(name=run['name'],L=run['L'],seed=run['seed'],evaluation_seed=seed,
                    replica_groups=64*512,batches=64,
                    observables={key:dict(mean=float(np.mean(v)),
                                          se=float(np.std(v,ddof=1)/np.sqrt(len(v))),batches=v)
                                 for key,v in values.items()})
        output.append(result)
        write_json(args.results/'observable_audit.json',output)
        print(run['name'],{k:(v['mean'],v['se']) for k,v in result['observables'].items()},flush=True)


if __name__=='__main__':
    main()
