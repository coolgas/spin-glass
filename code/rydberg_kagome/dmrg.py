"""TeNPy solver adapted from historical triangle_effective_mps, without imports."""
import numpy as np


def build_mpo(model):
    from tenpy.linalg import np_conserved as npc
    from tenpy.models.lattice import Lattice
    from tenpy.models.model import CouplingModel, MPOModel
    from tenpy.networks.site import Site
    count=len(model['sites']); width=int(round(np.sqrt(count)))
    order=[y*width+x for y in range(width) for x in
           (range(width) if y%2==0 else reversed(range(width)))]
    inverse={b:i for i,b in enumerate(order)}
    sites=[Site(npc.LegCharge.from_trivial(model['chi']), sort_charge=False) for _ in order]
    for i,b in enumerate(order):
        W=model['decoders'][b]; spins=model['spins']
        for a in range(12):
            sites[i].add_op(f'Z{a}',model['zops'][b][a])
            xW=W[np.arange(4096) ^ (1<<a)]
            sites[i].add_op(f'X{a}',W.T @ xW)
            for c in range(a):
                sites[i].add_op(f'P{a}_{c}',W.T @ ((spins[:,a]*spins[:,c])[:,None]*W))
    terms=[]
    for k,(coefficient,factors) in enumerate(model['operator'].terms):
        converted=[]
        for b,matrix in factors:
            i=inverse[b]; name=f'T{k}_{i}'
            sites[i].add_op(name,matrix)
            converted.append((i,name))
        terms.append((coefficient,sorted(converted)))
    coupling=CouplingModel(Lattice([1],sites,bc='open',bc_MPS='finite'))
    for coefficient,factors in terms:
        if len(factors)==1:
            i,name=factors[0]; coupling.add_onsite_term(coefficient,i,name)
        else:
            (i,a),(j,b)=factors
            coupling.add_coupling_term(coefficient,i,j,a,b)
    return MPOModel(coupling.lat,coupling.calc_H_MPO()), order


def initial_state(mpo,seed):
    from tenpy.networks.mps import MPS
    rng=np.random.default_rng(seed); sites=mpo.lat.mps_sites()
    chi=sites[0].dim; count=len(sites)
    bonds=[min(16,chi**min(i,count-i)) for i in range(count+1)]
    tensors=[rng.normal(size=(s.dim,bonds[i],bonds[i+1])) for i,s in enumerate(sites)]
    psi=MPS.from_Bflat(sites,tensors,form=None,dtype=float,unit_cell_width=1)
    psi.canonical_form()
    return psi


def moments(psi,model,order):
    physical=np.asarray([model['sites'][b] for b in order]); n=physical.size
    m,zz,x=np.zeros(n),np.eye(n),np.zeros(n)
    for a in range(12):
        m[physical[:,a]]=np.real(psi.expectation_value(f'Z{a}'))
        x[physical[:,a]]=np.real(psi.expectation_value(f'X{a}'))
        for c in range(a):
            val=np.real(psi.expectation_value(f'P{a}_{c}'))
            zz[physical[:,a],physical[:,c]]=val
            zz[physical[:,c],physical[:,a]]=val
        for c in range(12):
            val=np.real(psi.correlation_function(f'Z{a}',f'Z{c}',autoJW=False))
            for i in range(len(order)):
                for j in range(i):
                    p,q=physical[i,a],physical[j,c]
                    zz[p,q]=zz[q,p]=val[i,j]
    return m,zz,x


def optimize(psi,mpo,model,order,D,max_sweeps=120):
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    if max_sweeps < 60:
        raise ValueError("Need >=60 sweeps for independent confirmation")
    engine=TwoSiteDMRGEngine(psi,mpo,dict(mixer=True,
        mixer_params=dict(amplitude=1e-3,decay=2.,disable_after=15),
        min_sweeps=40,max_sweeps=40,max_E_err=0.,max_S_err=0.,
        trunc_params=dict(chi_max=D,svd_min=1e-12)))
    engine.run()
    n=model['sites'].size
    energies=list(engine.sweep_stats['E'])
    old_m,_,_=moments(psi,model,order)
    stable=False; drift=float('inf')
    used=40
    while used+20<=max_sweeps:
        confirm=TwoSiteDMRGEngine(psi,mpo,dict(mixer=False,min_sweeps=20,max_sweeps=20,
            max_E_err=0.,max_S_err=0.,trunc_params=dict(chi_max=D,svd_min=1e-12)))
        confirm.run(); used+=20
        energies.extend(confirm.sweep_stats['E'])
        m,zz,x=moments(psi,model,order)
        drift=float(np.max(abs(m-old_m)))
        stable=len(energies)>=51 and np.max(abs(np.diff(energies[-51:])))/n<1e-8 and drift<1e-4
        old_m=m
        if stable: break
    psi.canonical_form()
    canonical=float(np.linalg.norm(psi.norm_test()))
    return dict(energy=float(np.real(mpo.H_MPO.expectation_value(psi)))+model['constant'],
                sweeps=used,energy_history=[float(e) for e in energies],moment_drift=drift,
                canonical_error=canonical,sweep_pass=bool(stable and canonical<1e-8),
                max_bond=int(max(psi.chi)),m=m,zz=zz,x=x,
                truncation_max=float(max(confirm.sweep_stats['max_trunc_err'])))


def full_vector(psi,order,chi):
    if chi**len(order)>2**22:
        raise ValueError("Full-state extraction restricted to bounded references")
    tensor=psi.get_theta(0,n=len(order)).to_ndarray().reshape((chi,)*len(order))
    return tensor.transpose(np.argsort(order)).ravel()
