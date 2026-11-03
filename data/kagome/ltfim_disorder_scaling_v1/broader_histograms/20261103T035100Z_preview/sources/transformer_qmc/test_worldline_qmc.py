"""Small deterministic checks of the worldline target and exchange kernel."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
from .worldline_qmc import exchange_log_ratio,temporal_sweep,run


class WorldlineTests(unittest.TestCase):
    def test_single_spin_transfer_matrix(self):
        # Enumerate only the test's imaginary-time paths; production never enumerates.
        beta,gamma,h,m=1.3,.7,.2,8;dt=beta/m
        paths=1-2*((np.arange(2**m)[:,None]>>np.arange(m))&1)
        kt=-.5*np.log(np.tanh(dt*gamma))
        w=np.exp(dt*h*paths.sum(1)+kt*(paths*np.roll(paths,-1,axis=1)).sum(1))
        prefactor=(np.sinh(2*dt*gamma)/2)**(m/2)
        z=np.diag([1.,-1.]);x=np.array([[0.,1.],[1.,0.]])
        transfer=expm(dt*h*z/2)@expm(dt*gamma*x)@expm(dt*h*z/2)
        density=np.linalg.matrix_power(transfer,m)
        np.testing.assert_allclose(prefactor*w.sum(),np.trace(density),rtol=1e-12)
        np.testing.assert_allclose(w@paths[:,0]/w.sum(),np.trace(z@density)/np.trace(density),atol=1e-12)

    def test_exchange_detailed_balance(self):
        fa,ta,fb,tb=2.,8.,-3.,4.;da,ka,db,kb=.1,1.3,.2,.8
        before=da*fa+ka*ta+db*fb+kb*tb
        after=da*fb+ka*tb+db*fa+kb*ta
        self.assertAlmostEqual(exchange_log_ratio(fa,ta,fb,tb,da,ka,db,kb),after-before)

    def test_temporal_cluster_conditional_distribution(self):
        # Seed the Python implementation, so the test is identical with/without Numba.
        sweep=getattr(temporal_sweep,'py_func',temporal_sweep)
        np.random.seed(1102);m=4;dt=.4;kt=.7;h=.3
        states=np.ones((1,m),dtype=np.int8)
        neighbors=np.full((1,1),-1,dtype=np.int64);couplings=np.zeros((1,1))
        paths=1-2*((np.arange(2**m)[:,None]>>np.arange(m))&1)
        w=np.exp(dt*h*paths.sum(1)+kt*(paths*np.roll(paths,-1,axis=1)).sum(1));w/=w.sum()
        target=np.array([w@paths.mean(1),w@(paths*np.roll(paths,-1,axis=1)).mean(1)])
        samples=[]
        for t in range(14000):
            sweep(states,dt,kt,neighbors,couplings,h)
            if t>=2000:samples.append([states.mean(),(states*np.roll(states,-1,axis=1)).mean()])
        np.testing.assert_allclose(np.mean(samples,axis=0),target,atol=.035)

    def test_frustrated_triangle_full_sampler(self):
        # Independent enumeration of a tiny space-time system, including AF
        # frustration, field, periodic time and the full exchange sampler.
        n,m=3,4;beta,gamma,h=1.2,.8,.17;dt=beta/m
        bonds=np.array([[0,1],[1,2],[0,2]],dtype=np.int64)
        J=np.array([-.7,-1.,-.9])
        ham=SimpleNamespace(n_sites=n,bonds=bonds,J=J,h=h)
        paths=(1-2*((np.arange(2**(n*m))[:,None]>>np.arange(n*m))&1)).reshape(-1,n,m)
        spatial=h*paths.sum((1,2))
        for (i,j),coupling in zip(bonds,J):spatial+=coupling*(paths[:,i]*paths[:,j]).sum(1)
        temporal=(paths*np.roll(paths,-1,axis=2)).sum((1,2))
        def partition(b):
            step=b/m;kt=-.5*np.log(np.tanh(step*gamma))
            weights=np.exp(step*spatial+kt*temporal)
            return weights.sum()*(np.sinh(2*step*gamma)/2)**(n*m/2),weights/weights.sum()
        _,probability=partition(beta)
        target_m=probability@paths[:,:,0]
        target_q=np.einsum('s,si,sj->ij',probability,paths[:,:,0],paths[:,:,0])
        eps=1e-5
        derivative=-(np.log(partition(beta+eps)[0])-np.log(partition(beta-eps)[0]))/(2*eps)
        energies=-spatial/m-n*gamma/np.tanh(2*dt*gamma)+gamma*temporal/(m*np.sinh(2*dt*gamma))
        np.testing.assert_allclose(probability@energies,derivative,atol=1e-8)
        result=run(ham,gamma,beta=beta,dt=dt,temperatures=3,warmup=2000,
                   sweeps=24000,thin=1,blocks=48,seed=731,start=-1)
        np.testing.assert_allclose(result['magnetization_blocks'].mean(0),target_m,atol=.035)
        np.testing.assert_allclose(result['pair_blocks'].mean(0),target_q,atol=.035)
        self.assertLess(abs(result['energy_blocks'].mean()-derivative),.05)


if __name__=='__main__':unittest.main()
