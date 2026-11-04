"""Audit a downloaded trained guide's exact symmetries and checkpoint identity."""

import argparse
import hashlib
import json
import numpy as np
import torch

from .experiment import load_checkpoint, write_json
from .physics import basis, make_af_sample
from .symmetric import SymmetricTransformer, translate_hamiltonian, translation_permutation


@torch.no_grad()
def main():
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.checkpoint:
        checkpoint=args.checkpoint
        verification=json.loads((args.results/'transfer_verification.json').read_text())
        if verification['checkpoint'] != checkpoint.name:
            raise RuntimeError('transfer identity mismatch')
        evaluation=verification
    else:
        evaluation=json.loads((args.results/'evaluation.json').read_text())
        checkpoint=args.results/evaluation['checkpoint']
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest!=evaluation['sha256']:
        raise RuntimeError('checkpoint hash mismatch')
    state=load_checkpoint(checkpoint,'cpu')
    model=SymmetricTransformer(make_af_sample(2).hamiltonian(0.),.5,
                               **state['metadata']['architecture']).double().eval()
    model.load_state_dict(state['model'])
    weights={k:v.clone() for k,v in model.state_dict().items()}
    rows=[]
    for L in (2,4,6,8):
        for s in (0,10000):
            for h in (0.,.3):
                ham=make_af_sample(L,s).hamiltonian(h)
                model.set_hamiltonian(ham,.5)
                spins=(torch.tensor(basis(12),dtype=torch.long) if L==2 else
                       model.sample(16,torch.Generator().manual_seed(900+L+s)))
                logp=model.log_prob(spins)
                parity_error=float((logp-model.log_prob(-spins)).abs().max()) if h==0 else None
                if h==0:
                    assert parity_error < 1e-10
                if L==2:
                    assert abs(float(logp.exp().sum())-1)<1e-10
                for dx,dy in ((1,0),(0,1),(1,1)):
                    permutation=translation_permutation(L,dx,dy)
                    moved=torch.empty_like(spins);moved[:,permutation]=spins
                    model.set_hamiltonian(translate_hamiltonian(ham,permutation),.5)
                    error=float((logp-model.log_prob(moved)).abs().max())
                    assert error < 1e-10
                    rows.append(dict(L=L,sample_id=s,h=h,dx=dx,dy=dy,
                                     max_translation_logp_error=error,
                                     max_parity_logp_error=parity_error,
                                     configurations=len(spins)))
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,weights[k],atol=0,rtol=0)
    write_json(args.results/'symmetry_audit.json',dict(checkpoint=checkpoint.name,
                sha256=digest,precision='float64',rows=rows,passed=True))
    print('Trained-checkpoint translation/parity audit passed across all sizes.')


if __name__=='__main__':
    main()
