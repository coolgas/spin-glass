"""Check fast ratios and the doubled-hidden RBM probability convention."""
import unittest
import numpy as np
import torch
from .physics import basis
from .simple_guides import Jastrow, RBM
from .simple_guide_benchmark import objective
from .physics import make_af_sample, diagonal
from kagome_qea.exact import operator_matrix


class SimpleTests(unittest.TestCase):
    def test_training_energy_and_gradient(self):
        ham=make_af_sample(2,50025).hamiltonian(.15)
        spins=torch.tensor(basis(12),dtype=torch.float64)
        potential=torch.tensor(diagonal(ham,spins.numpy()))
        flips=torch.arange(4096)[:,None] ^ (1<<torch.arange(12))
        matrix=operator_matrix(ham,.25)
        for model in [Jastrow(12,3),RBM(12,1,3)]:
            energy,amplitude,applied=objective(model,spins,potential,flips,.25)
            np.testing.assert_allclose(applied.detach().numpy(),matrix@amplitude.detach().numpy(),atol=1e-13)
            energy.backward()
            derivative=float(model.a.grad[0]);step=1e-5
            with torch.no_grad():
                model.a[0]+=step
                plus=float(objective(model,spins,potential,flips,.25)[0])
                model.a[0]-=2*step
                minus=float(objective(model,spins,potential,flips,.25)[0])
                model.a[0]+=step
            self.assertAlmostEqual(derivative,(plus-minus)/(2*step),places=7)

    def test_analytic_flip_ratios(self):
        spins=torch.tensor(basis(4),dtype=torch.float64)
        for model in [Jastrow(4,9),RBM(4,2,9)]:
            fast=model.flip_log_ratios(spins)
            for site in range(4):
                flipped=spins.clone();flipped[:,site]*=-1
                torch.testing.assert_close(fast[:,site],model(flipped)-model(spins),atol=1e-12,rtol=1e-12)

    def test_rbm_squared_amplitude_gibbs_invariance(self):
        model=RBM(2,1,7)
        with torch.no_grad():
            model.a.copy_(torch.tensor([.2,-.3]));model.w.mul_(5)
        visible=torch.tensor(basis(2),dtype=torch.float64)
        hidden=torch.tensor(basis(4),dtype=torch.float64)
        fields=visible@model.w+model.b
        joint_log=2*(visible@model.a)[:,None]+fields@(hidden[:,:2]+hidden[:,2:]).T
        joint=torch.softmax(joint_log.flatten(),0).reshape(4,16)
        marginal=joint.sum(1)
        expected=torch.softmax(2*model(visible),0)
        torch.testing.assert_close(marginal,expected,atol=1e-13,rtol=1e-13)
        h_given_s=joint/joint.sum(1)[:,None]
        s_given_h=joint/joint.sum(0)[None,:]
        transition=h_given_s@s_given_h.T
        torch.testing.assert_close(marginal@transition,marginal,atol=1e-13,rtol=1e-13)
        torch.testing.assert_close(marginal[:,None]*transition,(marginal[:,None]*transition).T,atol=1e-13,rtol=1e-13)


if __name__=='__main__':unittest.main()
