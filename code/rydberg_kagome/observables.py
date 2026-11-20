"""Physical observables, with same-block products projected before multiplication."""
import numpy as np
from .exact import apply_axis


def summarize(m, zz):
    m, zz = np.asarray(m), np.asarray(zz)
    if zz.shape != (len(m), len(m)) or not np.allclose(zz,zz.T) or not np.allclose(np.diag(zz),1):
        raise ValueError("Invalid physical ZZ matrix")
    connected = zz-np.outer(m,m)
    return dict(qea=float(np.mean(m*m)), q2=float(np.mean(zz*zz)),
                magnetization=float(m.mean()), density=float((1+m.mean())/2),
                local_moment_variance=float(np.var(m)),
                chi_connected=float(np.sum(connected**2)/len(m)),
                chi_connected_diagonal=float(np.mean(np.diag(connected)**2)))


def vector_moments(model, vector):
    chi, count = model['chi'], len(model['sites'])
    state = vector.reshape((chi,)*count)
    n = sum(map(len,model['sites']))
    m, zz, inserted = np.zeros(n), np.eye(n), []
    for b, (W, sites) in enumerate(zip(model['decoders'],model['sites'])):
        flat = np.moveaxis(state,b,0).reshape(chi,-1)
        rho = flat @ flat.conj().T
        ops = model['zops'][b]
        inserted.append([apply_axis(state, op, b).ravel() for op in ops])
        for a,i in enumerate(sites):
            m[i] = np.trace(rho @ ops[a]).real
            for c in range(a):
                j=sites[c]
                product = W.T @ ((model['spins'][:,a]*model['spins'][:,c])[:,None]*W)
                zz[i,j]=zz[j,i]=np.trace(rho @ product).real
    for b in range(count):
        for c in range(b):
            cross=(np.asarray(inserted[b]).conj() @ np.asarray(inserted[c]).T).real
            zz[np.ix_(model['sites'][b],model['sites'][c])]=cross
            zz[np.ix_(model['sites'][c],model['sites'][b])]=cross.T
    return m, zz


def physical_vector_moments(vector, spins):
    p=np.abs(vector)**2
    return p @ spins, (spins.T*p) @ spins
