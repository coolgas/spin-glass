"""Independent sampler targets and energy-gradient checks."""
import unittest
import numpy as np
import torch

from .physics import basis,make_af_sample
from .simple_guides import Jastrow,RBM
from .sampled_simple_guides import TemperedChains,exchange_ratio,local_energy,energy_loss
from kagome_qea.exact import operator_matrix


class SampledTests(unittest.TestCase):
    def test_exchange_detailed_balance(self):
        for a,b,x,y in [(0.,1.,-3.,2.),(.4,.8,1.,-.2)]:
            before=2*(a*x+b*y);after=2*(a*y+b*x)
            lr=exchange_ratio(a,b,x,y)
            self.assertAlmostEqual(before+min(0,lr),after+min(0,-lr))

    def test_tempered_frozen_marginal(self):
        for model in [Jastrow(3,4),RBM(3,2,4)]:
            with torch.no_grad():model.a.copy_(torch.tensor([.25,-.1,.15]))
            sampler=TemperedChains(model,chains=12000,temperatures=4,seed=81)
            for _ in range(50):sampler.sweep()
            spins=torch.tensor(basis(3),dtype=torch.float64)
            expected=torch.softmax(2*model(spins),0).detach().numpy()
            sample=sampler.states[-1].numpy().astype(int)
            ids=((1-sample)//2)@(1<<np.arange(3))
            observed=np.bincount(ids,minlength=8)/len(ids)
            np.testing.assert_allclose(observed,expected,atol=.018,rtol=0)

    def test_local_energy_and_covariance_gradient(self):
        ham=make_af_sample(2,50025).hamiltonian(.15)
        spins=torch.tensor(basis(12),dtype=torch.float64)
        matrix=operator_matrix(ham,.25)
        for model in [Jastrow(12,5),RBM(12,2,5)]:
            logs=model(spins);amplitude=torch.exp(logs-logs.max())
            amplitude=amplitude/torch.linalg.vector_norm(amplitude)
            e=local_energy(model,spins,ham,.25)
            np.testing.assert_allclose((e*amplitude).detach().numpy(),matrix@amplitude.detach().numpy(),atol=1e-12)
            p=amplitude.detach()**2
            covariance=2*(p*(e-(p*e).sum())*logs).sum()
            covariance.backward();gradient=model.a.grad.clone()
            model.zero_grad()
            # Independent differentiable dense expectation for one parameter.
            dense=torch.tensor(matrix.toarray(),dtype=torch.float64)
            logs=model(spins);v=torch.softmax(2*logs,0).sqrt()
            (v@(dense@v)).backward()
            torch.testing.assert_close(model.a.grad,gradient,atol=1e-11,rtol=1e-11)


if __name__=='__main__':unittest.main()
