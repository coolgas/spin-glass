"""Fine-grid pilot with full before/after histograms retained at measurement time."""
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .preliminary_overlap_notebook import digest, dump
from .production_protocol import identity
from .projected_overlap_histograms import run_one
from .sampled_simple_guides import TemperedChains
from .simple_guides import Jastrow, RBM


@torch.no_grad()
def sample_before(task, model_path, output):
    n = 3*task['L']**2
    root = Path(output)
    dest = root/'guide_histograms'/(identity(task)[:24]+'.json')
    if dest.exists():
        return
    model = Jastrow(n) if task['kind']=='jastrow' else RBM(n,2)
    model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=False)['model'])
    arrays = {}; starts = []
    for start in ['random','up','down']:
        seed = int(identity(dict(task,start=start))[:8],16)
        sampler = TemperedChains(model,128,8,seed,start)
        values = []
        for step in range(300):
            spins = sampler.sweep()
            if step>=100 and (step-100)%4==0:
                values.append((spins[:64]!=spins[64:]).sum(1).numpy())
        values = np.asarray(values,dtype=np.int16)
        arrays[start+'_mismatches'] = values
        mass = np.bincount(values.ravel(),minlength=n+1)[::-1]/values.size
        starts.append(dict(start=start,mass=mass.tolist(),**sampler.diagnostics()))
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest.with_suffix('.npz'),**arrays)
    dump(dest,dict(task=task,q_grid=np.linspace(-1,1,n+1).tolist(),starts=starts,
                   model_sha256=digest(Path(model_path)),raw_sha256=digest(dest.with_suffix('.npz')),
                   raw_file=str(dest.with_suffix('.npz').relative_to(root)),
                   protocol='128 columns, eight temperatures, 100 burn-in + 200 retained-window sweeps, stride four; disjoint halves form independent replicas; random/up/down starts'))


def project_with_histogram(task,model_path,output):
    torch.set_num_threads(1)
    spec = dict(task=task,model=model_path)
    result = run_one(spec,output)
    if task['audit']=='main' and task['repeat']==0:
        with threadpool_limits(limits=1):
            sample_before(task,model_path,output)
    return result


if __name__=='__main__':
    from . import parallel_disorder_scaling as campaign
    campaign.project_task = project_with_histogram
    campaign.main()
