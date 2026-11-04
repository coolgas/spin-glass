"""Independent sampler targets and energy-gradient checks."""
import unittest
import numpy as np
import torch

from .physics import basis,make_af_sample
from .simple_guides import Jastrow,RBM
from .sampled_simple_guides import TemperedChains,exchange_ratio,local_energy,energy_loss,sampling_alarms,select_sampled_guide
from kagome_qea.exact import operator_matrix


class SampledTests(unittest.TestCase):
    def test_guide_selection_rejects_spuriously_low_unmixed_energy(self):
        from copy import deepcopy
        good=dict(status='completed',signature=dict(chains=256),diagnostics=[
            dict(start=s,energy=-100.,variance=1.,chain_mean_energy_sd=.1,round_trips=50,
                 exchange_fraction=[.3,.4]) for s in ['random','up','down']])
        bad=deepcopy(good)
        for row in bad['diagnostics']:
            row.update(energy=-200.,round_trips=0,exchange_fraction=[0.,.4])
        self.assertIs(select_sampled_guide([bad,good]),good)
        with self.assertRaisesRegex(RuntimeError,'No candidate'):
            select_sampled_guide([bad])

    def test_low_variance_does_not_override_sampling_alarm(self):
        rows=[dict(start=s,energy=-80.,variance=1e-10,chain_mean_energy_sd=1e-6,
                   round_trips=0,exchange_fraction=[0.,.9]) for s in ['random','up','down']]
        self.assertEqual(len(sampling_alarms(rows,256)),6)
        for r in rows:
            r.update(round_trips=10,exchange_fraction=[.5,.5],chain_mean_energy_sd=.1)
        self.assertEqual(sampling_alarms(rows,256),[])
        rows[1]['energy']+=1
        self.assertEqual(len(sampling_alarms(rows,256)),2)

    def test_exchange_detailed_balance(self):
        for a,b,x,y in [(0.,1.,-3.,2.),(.4,.8,1.,-.2)]:
            before=2*(a*x+b*y);after=2*(a*y+b*x)
            lr=exchange_ratio(a,b,x,y)
            self.assertAlmostEqual(before+min(0,lr),after+min(0,-lr))

    def test_tempered_frozen_marginal(self):
        for model in [Jastrow(3,4),RBM(3,2,4)]:
            with torch.no_grad():
                model.a.copy_(torch.tensor([.25,-.1,.15]))
                if isinstance(model, RBM):
                    model.w.mul_(8)
                    model.b.mul_(10)
                else:
                    model.j.mul_(30)
            sampler=TemperedChains(model,chains=12000,temperatures=4,seed=81)
            for _ in range(50):sampler.sweep()
            spins=torch.tensor(basis(3),dtype=torch.float64)
            for beta, population in zip(sampler.betas, sampler.states):
                expected=torch.softmax(2*beta*model(spins),0).detach().numpy()
                sample=population.numpy().astype(int)
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
