"""Independent finite-state checks of the enhanced feedback kernels."""
from itertools import product
import unittest
import numpy as np
import torch
from .feedback_samplers import tempered_log_density, tempered_exchange, bounded_step, TwoCopyTempering
from .physics import basis
from .simple_guides import RBM


class TemperedFeedbackTests(unittest.TestCase):
    def model(self):
        model = RBM(2, 1, 3)
        with torch.no_grad():
            model.a[:] = torch.tensor([.31, -.2]); model.b[:] = torch.tensor([.12, -.22])
            model.w[:] = torch.tensor([[.4, -.31], [.27, .18]])
        return model

    def test_exact_tempered_gibbs_marginal_and_invariance(self):
        model = self.model(); x = torch.tensor(basis(2), dtype=torch.float64)
        hidden = torch.tensor(list(product([-1., 1.], repeat=4)), dtype=torch.float64).reshape(-1, 2, 2)
        for beta in [0., .37, 1.]:
            transition = np.zeros((4, 4)); explicit_marginal = []
            for i, s in enumerate(x):
                ph = torch.sigmoid(2*beta*(s@model.w+model.b))
                weights = []
                for h in hidden:
                    mass = torch.where(h > 0, ph, 1-ph).prod()
                    pv = torch.sigmoid(2*beta*(2*model.a+h.sum(0)@model.w.T))
                    transition[i] += (mass*torch.where(x > 0, pv, 1-pv).prod(1)).detach().numpy()
                    weights.append(torch.exp(beta*(2*s@model.a+(h*(s@model.w+model.b)).sum())))
                explicit_marginal.append(torch.stack(weights).sum())
            p = torch.softmax(tempered_log_density(model, x, beta), 0).detach().numpy()
            weights = torch.stack(explicit_marginal).detach().numpy(); weights /= weights.sum()
            np.testing.assert_allclose(weights, p, atol=1e-12)
            np.testing.assert_allclose(p@transition, p, atol=1e-12)
            if beta == 1:
                np.testing.assert_allclose(p, torch.softmax(2*model(x),0).detach().numpy(), atol=1e-12)

    def test_exchange_and_inversion_detailed_balance(self):
        model = self.model(); x = torch.tensor(basis(2), dtype=torch.float64)
        for beta_a, beta_b in [(0., .3), (.3, 1.)]:
            pa = torch.softmax(tempered_log_density(model,x,beta_a),0).detach().numpy()
            pb = torch.softmax(tempered_log_density(model,x,beta_b),0).detach().numpy()
            for i, a in enumerate(x):
                for j, b in enumerate(x):
                    log_ratio = float(tempered_exchange(model, a, b, beta_a, beta_b).detach())
                    self.assertAlmostEqual(pa[i]*pb[j]*min(1,np.exp(log_ratio)),
                                           pa[j]*pb[i]*min(1,np.exp(-log_ratio)), places=13)
        for beta in [0., .3, 1.]:
            p = torch.softmax(tempered_log_density(model,x,beta),0).detach().numpy()
            for i,s in enumerate(x):
                j = int(torch.where((x == -s).all(1))[0][0])
                lr = float((tempered_log_density(model,-s,beta)-tempered_log_density(model,s,beta)).detach())
                self.assertAlmostEqual(p[i]*min(1,np.exp(lr)),p[j]*min(1,np.exp(-lr)),places=13)

    def test_empirical_step_cap_and_rejection_restore(self):
        model = self.model(); x = torch.tensor(basis(2), dtype=torch.float64)
        old = [p.detach().clone() for p in model.parameters()]; logs = model(x).detach()
        with torch.no_grad():
            model.a.add_(.5)
        result = bounded_step(model,old,x,logs,.025,.15)
        self.assertGreater(result[0],0);self.assertLess(result[0],1)
        delta = 2*(model(x).detach()-logs);delta -= delta.mean()
        self.assertLessEqual(float(delta.square().mean().sqrt()),.025)
        self.assertLessEqual(float(delta.abs().max()),.15)
        with torch.no_grad():
            model.a.add_(1000.)
        result = bounded_step(model,old,x,logs,1e-12,1e-12)
        self.assertEqual(result[0],0)
        for p,v in zip(model.parameters(),old):
            torch.testing.assert_close(p,v,atol=0,rtol=0)

    def test_full_sampler_matches_exact_cold_density(self):
        model = self.model(); generator = torch.Generator().manual_seed(1929)
        sampler = TwoCopyTempering(model,2048,4,generator)
        counts = np.zeros(4)
        for step in range(100):
            samples = sampler.sweep().numpy().astype(int)
            if step >= 20:
                ids = ((1-samples)//2)@(1 << np.arange(2))
                counts += np.bincount(ids,minlength=4)
        x = torch.tensor(basis(2),dtype=torch.float64)
        truth = torch.softmax(2*model(x),0).detach().numpy()
        self.assertLess(np.abs(counts/counts.sum()-truth).max(),.01)
        self.assertGreater(sampler.diagnostics()['round_trips'],0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
