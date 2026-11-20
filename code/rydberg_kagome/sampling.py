"""Joint physical Born sampling via right-canonical MPS and block POVM trees."""
import numpy as np


def sample_tensors(tensors,decoders,sites,spins,draws,seed,batch_size=128):
    """Tensors have axes (left, physical retained, right), right canonical.

    Contract prefix POVMs, then condition the virtual state on each full
    12-spin outcome. Never assume independent block marginals.
    """
    rng=np.random.default_rng(seed)
    trees=[]
    for W in decoders:
        levels=[np.einsum('ri,rj->rij',W.conj(),W)]
        for _ in range(int(np.log2(len(W)))):
            level=levels[-1]; levels.append(level.reshape(-1,2,*level.shape[1:]).sum(1))
        trees.append(levels)
    output=np.empty((draws,sum(map(len,sites))),dtype=np.int8)
    for start in range(0,draws,batch_size):
        length=min(batch_size,draws-start); left=np.ones((length,1))
        for T,W,block,levels in zip(tensors,decoders,sites,trees):
            conditioned=np.einsum('bl,lpr->bpr',left,T)
            rho=np.einsum('bpr,bqr->bpq',conditioned,conditioned.conj())
            prefix=np.zeros(length,dtype=int)
            parent=np.trace(rho,axis1=1,axis2=2).real
            for depth in range(1,len(levels)):
                povm=levels[-1-depth][2*prefix]
                prob_left=np.einsum('bpq,bpq->b',rho,povm).real
                ratio=prob_left/parent
                if np.any(ratio < -1e-8) or np.any(ratio>1+1e-8):
                    raise RuntimeError("Invalid conditional Born probability")
                right=rng.random(length)>=np.clip(ratio,0,1)
                prefix=2*prefix+right
                parent=np.where(right,parent-prob_left,prob_left)
            left=np.einsum('bp,bpr->br',W[prefix],conditioned)
            norms=np.linalg.norm(left,axis=1)
            if np.any(norms<=0): raise RuntimeError("Zero sampled Born weight")
            left/=norms[:,None]
            output[start:start+length,block]=spins[prefix]
    return output


def mps_samples(psi,model,order,draws,seed):
    psi.canonical_form()
    tensors=[psi.get_B(i,form='B').transpose(['vL','p','vR']).to_ndarray() for i in range(len(order))]
    return sample_tensors(tensors,[model['decoders'][b] for b in order],
                          [model['sites'][b] for b in order],model['spins'],draws,seed)


def overlap_histogram(samples):
    if len(samples)%2: raise ValueError("Disjoint replica pairs required")
    n=samples.shape[1]
    equal=np.sum(samples[::2]==samples[1::2],axis=1)
    counts=np.bincount(equal,minlength=n+1)
    q=np.linspace(-1,1,n+1); p=counts/counts.sum()
    values=q[equal]
    return dict(counts=counts,p=p,q=q,pairs=len(equal),
                qmean=float(values.mean()),qmean_se=float(values.std(ddof=1)/np.sqrt(len(values))),
                q2=float(np.mean(values**2)),q2_se=float(np.std(values**2,ddof=1)/np.sqrt(len(values))),
                q4=float(np.mean(values**4)),cdf95_dkw=float(np.sqrt(np.log(40)/(2*len(equal)))))
