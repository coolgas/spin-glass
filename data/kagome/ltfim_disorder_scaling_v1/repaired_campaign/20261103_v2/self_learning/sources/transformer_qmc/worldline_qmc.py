"""Sign-free discrete-time TFIM QMC with temporal clusters and beta exchange.

H=-sum J_ij Z_i Z_j-h sum Z_i-Gamma sum X_i (J may be frustrated).
Weight exp[(beta/M)(sum J ss+h sum s)+Kt sum s(t)s(t+1)],
Kt=-log(tanh(beta*Gamma/M))/2. No basis enumeration or wavefunction guide.
Finite beta, finite M and mixing must be checked independently.
"""
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda fn: fn


@njit(cache=True)
def temporal_sweep(states, dt, kt, neighbors, couplings, h):
    """FK bonds only along time; conditional component signs use exact heat baths."""
    n,m=states.shape
    field=np.empty(m);links=np.empty(m,dtype=np.bool_);indices=np.empty(m,dtype=np.int64)
    for site in np.random.permutation(n):
        for t in range(m):
            value=h
            for j in range(neighbors.shape[1]):
                neighbor=neighbors[site,j]
                if neighbor>=0:value+=couplings[site,j]*states[neighbor,t]
            field[t]=dt*value
        cut=-1
        for t in range(m):
            links[t]=(states[site,t]==states[site,(t+1)%m] and np.random.random()<1-np.exp(-2*kt))
            if not links[t]:cut=t
        if cut<0:
            total=field.sum()
            sign=1 if np.random.random()<.5*(1+np.tanh(total)) else -1
            states[site,:]=sign
        else:
            count=0;total=0.
            for step in range(m):
                t=(cut+1+step)%m;indices[count]=t;count+=1;total+=field[t]
                if not links[t]:
                    sign=1 if np.random.random()<.5*(1+np.tanh(total)) else -1
                    for j in range(count):states[site,indices[j]]=sign
                    count=0;total=0.


@njit(cache=True)
def sufficient_statistics(states,bonds,J,h):
    n,m=states.shape;spatial=0.;temporal=0.
    for t in range(m):
        for i in range(n):
            spatial+=h*states[i,t]
            temporal+=states[i,t]*states[i,(t+1)%m]
        for b in range(len(J)):spatial+=J[b]*states[bonds[b,0],t]*states[bonds[b,1],t]
    return spatial,temporal


@njit(cache=True)
def exchange_log_ratio(fa,ta,fb,tb,dta,kta,dtb,ktb):
    return (dta-dtb)*(fb-fa)+(kta-ktb)*(tb-ta)


@njit(cache=True)
def simulate(neighbors,couplings,bonds,J,h,gamma,betas,m,warmup,sweeps,thin,blocks,seed,start):
    np.random.seed(seed);nt=len(betas);n=len(neighbors)
    states=np.ones((nt,n,m),dtype=np.int8)
    if start==0:
        for a in range(nt):
            for i in range(n):
                for t in range(m):states[a,i,t]=1 if np.random.random()<.5 else -1
    elif start==-1:states[:,:,:]=-1
    dt=betas/m;kt=-.5*np.log(np.tanh(dt*gamma))
    mag=np.zeros((blocks,n));pair=np.zeros((blocks,n,n));count=np.zeros(blocks,dtype=np.int64)
    energy=np.zeros(blocks);accept=np.zeros(nt-1,dtype=np.int64);attempt=np.zeros(nt-1,dtype=np.int64)
    labels=np.arange(nt);visits=np.zeros((nt,2),dtype=np.int64)
    # Each label must reach hot -> cold -> hot for a counted round trip.
    phase=np.zeros(nt,dtype=np.int64);trips=np.zeros(nt,dtype=np.int64)
    for sweep in range(warmup+sweeps):
        for a in range(nt):temporal_sweep(states[a],dt[a],kt[a],neighbors,couplings,h)
        f=np.empty(nt);u=np.empty(nt)
        for a in range(nt):f[a],u[a]=sufficient_statistics(states[a],bonds,J,h)
        for a in range(sweep%2,nt-1,2):
            b=a+1;lr=exchange_log_ratio(f[a],u[a],f[b],u[b],dt[a],kt[a],dt[b],kt[b])
            attempt[a]+=1
            if np.log(np.random.random())<min(0.,lr):
                tmp=states[a].copy();states[a]=states[b];states[b]=tmp
                lab=labels[a];labels[a]=labels[b];labels[b]=lab;accept[a]+=1
        hot=labels[0];cold=labels[-1]
        visits[hot,0]+=1;visits[cold,1]+=1
        if phase[hot]==2:trips[hot]+=1
        phase[hot]=1
        if phase[cold]==1:phase[cold]=2
        if sweep>=warmup and (sweep-warmup)%thin==0:
            block=min(blocks-1,(sweep-warmup)*blocks//sweeps)
            # A random equal-time slice is unbiased; slices are never independent SE samples.
            t=np.random.randint(m);s=states[-1,:,t];count[block]+=1
            for i in range(n):
                mag[block,i]+=s[i]
                for j in range(n):pair[block,i,j]+=s[i]*s[j]
            ff,uu=sufficient_statistics(states[-1],bonds,J,h)
            x=dt[-1]*gamma
            # Derivative of the full transfer-matrix weight, including its prefactor.
            energy[block]+=-ff/m-n*gamma/np.tanh(2*x)+gamma*uu/(m*np.sinh(2*x))
    for b in range(blocks):
        mag[b]/=count[b];pair[b]/=count[b];energy[b]/=count[b]
    return mag,pair,energy,count,accept,attempt,trips,visits,states[-1]


def adjacency(ham):
    degree=np.bincount(ham.bonds.ravel(),minlength=ham.n_sites)
    neighbors=np.full((ham.n_sites,int(degree.max())),-1,dtype=np.int64)
    couplings=np.zeros_like(neighbors,dtype=float);used=np.zeros(ham.n_sites,dtype=int)
    for (i,j),value in zip(ham.bonds,ham.J):
        neighbors[i,used[i]]=j;couplings[i,used[i]]=value;used[i]+=1
        neighbors[j,used[j]]=i;couplings[j,used[j]]=value;used[j]+=1
    return neighbors,couplings


def run(ham,gamma,*,beta=16.,dt=.125,temperatures=16,warmup=3000,sweeps=12000,
        thin=4,blocks=24,seed=0,start=0):
    if gamma<=0 or beta<=0 or dt<=0 or temperatures<2 or warmup<0 or thin<1 or sweeps<blocks*thin:
        raise ValueError('Invalid worldline parameters')
    if start not in (-1,0,1):raise ValueError('start must be random(0), up(1), or down(-1)')
    m=max(2,int(np.ceil(beta/dt)));betas=np.geomspace(min(.25,beta/2),beta,temperatures)
    neighbors,couplings=adjacency(ham)
    result=simulate(neighbors,couplings,ham.bonds,ham.J,ham.h,gamma,betas,m,warmup,sweeps,thin,blocks,seed,start)
    mag,pair,energy,count,accept,attempt,trips,visits,last=result
    return dict(magnetization_blocks=mag,pair_blocks=pair,energy_blocks=energy,counts=count,
                exchange_accepted=accept,exchange_attempts=attempt,round_trips=trips,
                endpoint_visits=visits,last_worldline=last,betas=betas,M=np.array(m))
