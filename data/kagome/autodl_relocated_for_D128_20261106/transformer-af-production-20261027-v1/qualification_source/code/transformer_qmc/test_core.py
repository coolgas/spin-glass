"""Exact identities and stochastic projector checks: python -m transformer_qmc.test_core."""

import unittest
import numpy as np
import torch
from scipy.linalg import expm

from ising import Hamiltonian
from kagome_qea.exact import operator_matrix
from .physics import make_af_sample, basis, exact_reference, diagonal, correlations
from .model import CellTransformer, EnumeratedGuide, TransformerGuide
from .projector import rates_and_local, propagate, project, tags
from .vmc import local_energy
from .unified import UnifiedTransformer


class CoreTests(unittest.TestCase):
    def test_unified_size_transfer_and_causality(self):
        torch.manual_seed(17)
        model = UnifiedTransformer(make_af_sample(2).hamiltonian(.3), .5,
                                   width=8, layers=1, heads=2).double()
        torch.nn.init.normal_(model.output.weight, std=.05)
        parameters = {k:tuple(v.shape) for k,v in model.state_dict().items()}
        states = torch.tensor(basis(12), dtype=torch.long)
        self.assertAlmostEqual(float(model.log_prob(states).exp().sum().detach()), 1., places=10)
        tokens = torch.randint(8,(2,4))
        tokens[1,:2] = tokens[0,:2]
        torch.testing.assert_close(model.logits(tokens)[0,:3],model.logits(tokens)[1,:3])
        for L in (4,6,8,3):
            model.set_hamiltonian(make_af_sample(L,1).hamiltonian(.1),1.)
            self.assertEqual(parameters,{k:tuple(v.shape) for k,v in model.state_dict().items()})
            draws = model.sample(3)
            self.assertEqual(tuple(draws.shape),(3,3*L*L))
            self.assertTrue(torch.isfinite(model.log_prob(draws)).all())
        model.set_hamiltonian(make_af_sample(2).hamiltonian(.3),.5)
        p = model.log_prob(states).exp()
        e = local_energy(model,make_af_sample(2).hamiltonian(.3),.5,states)
        model.zero_grad()
        (p.detach()*(e-(p*e).sum()).detach()*model.log_prob(states)).sum().backward()
        gradient = model.output.bias.grad.clone()
        model.zero_grad()
        psi = model.log_prob(states).mul(.5).exp()
        # Sparse Hamiltonian action avoids allocating a dense 4096² matrix.
        flips = torch.arange(4096)[:,None] ^ (1 << torch.arange(12))
        diag = torch.tensor(diagonal(make_af_sample(2).hamiltonian(.3),states.numpy()))
        (psi*(diag*psi-.5*psi[flips].sum(1))).sum().backward()
        torch.testing.assert_close(gradient,model.output.bias.grad,atol=1e-9,rtol=1e-8)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_af_identity(self):
        sample = make_af_sample()
        self.assertTrue(np.all((-sample.J > 0.5) & (-sample.J < 1.5)))
        np.testing.assert_array_equal(sample.J, make_af_sample().J)
        states = basis(12)
        ham = sample.hamiltonian(0.3)
        expected = (states[:, sample.bonds[:, 0]] * states[:, sample.bonds[:, 1]]) @ (-sample.J) - .3 * states.sum(1)
        np.testing.assert_allclose(diagonal(ham, states), expected)
        for invalid in (-.1, 1., np.nan):
            with self.assertRaises(ValueError):
                make_af_sample(width=invalid)

    def test_transformer_causality_normalization_gradient(self):
        ham = Hamiltonian(6, [[0,1],[1,2],[2,0],[2,3],[3,4],[4,5]], -np.ones(6), .3)
        torch.manual_seed(5)
        model = CellTransformer(ham, .8, width=8, layers=1, heads=2).double()
        torch.nn.init.normal_(model.output.weight, std=.05)
        states = torch.tensor(basis(6), dtype=torch.long)
        p = model.log_prob(states).exp()
        self.assertAlmostEqual(float(p.sum()), 1., places=10)
        tokens = model.tokens(states[:1]).repeat(2,1)
        tokens[1, 1] = (tokens[0,1] + 1) % 8
        torch.testing.assert_close(model.logits(tokens)[0,0], model.logits(tokens)[1,0])
        model.zero_grad()
        e = local_energy(model, ham, .8, states)
        mean = (p * e).sum()
        loss = (p.detach() * (e - mean).detach() * model.log_prob(states)).sum()
        loss.backward()
        gradient = model.output.bias.grad[0].item()
        matrix = torch.tensor(operator_matrix(ham, .8).toarray(), dtype=torch.double)
        model.zero_grad()
        psi = model.log_prob(states).mul(.5).exp()
        (psi @ matrix @ psi).backward()
        self.assertAlmostEqual(gradient, model.output.bias.grad[0].item(), places=9)
        adapter = TransformerGuide(model)
        table = EnumeratedGuide(adapter.log_amplitudes(states.numpy()), 6)
        np.testing.assert_allclose(adapter.all_flip_log_ratios(states.numpy()),
                                   table.all_flip_log_ratios(states.numpy()), atol=1e-10)
        samples = model.sample(20000, torch.Generator().manual_seed(9)).numpy()
        freq = np.bincount(table.indexes(samples), minlength=64) / len(samples)
        self.assertLess(np.max(np.abs(freq - p.detach().numpy())), .006)

    def test_four_replica_connected_identity(self):
        # Independent polarized spins provide an analytic four-copy benchmark.
        rng = np.random.default_rng(11)
        m = np.array([.2,.5,-.7])
        states = np.where(rng.random((300000,4,3)) < (1+m)/2, 1, -1)
        got = tags(states)["chi_connected"]
        expected = np.mean((1-m*m)**2)
        self.assertLess(abs(got.mean()-expected), 5*got.std()/np.sqrt(len(got)))

    def test_exact_guide_and_short_time_generator(self):
        ham = Hamiltonian(3, [[0,1],[1,2],[0,2]], [-.8,-1.1,-.9], .25)
        ref = exact_reference(ham, .7)
        guide = EnumeratedGuide(np.log(ref['amplitude']), 3)
        states = basis(3)
        _, local = rates_and_local(ham,.7,guide,states[:,None,:])
        np.testing.assert_allclose(local,ref['energy'],atol=1e-11)
        # Non-exact guide: weighted stochastic distribution vs exact propagator.
        trial = EnumeratedGuide(np.linspace(-.2,.3,8),3)
        initial = np.repeat(states[[0],None,:],100000,axis=0)
        evolved, _, _, _, _, _ = propagate(ham,.7,trial,initial,0.,.1,np.random.default_rng(7))
        freq=np.bincount(trial.indexes(evolved[:,0]),minlength=8)/len(initial)
        expected=expm(-.1*operator_matrix(ham,.7).toarray())[:,0]*np.exp(trial.logs-trial.logs[0])
        expected/=expected.sum()
        np.testing.assert_allclose(freq,expected,atol=.004)
        records=[project(ham,.7,guide,n_walkers=512,n_replicas=4,n_equilibrate=5,
                         n_measure=5,n_cycles=4,tau_forward=.3,seed=i) for i in range(8)]
        np.testing.assert_allclose([r['energy'] for r in records],ref['energy'],atol=1e-10)
        for name in ('q2','chi_connected'):
            values=np.array([r[name] for r in records])
            self.assertLess(abs(values.mean()-ref[name]),max(.01,5*values.std(ddof=1)/np.sqrt(8)))
        self.assertTrue(all(np.min(r['descendant_ess'])>511 for r in records))


if __name__ == '__main__':
    unittest.main(verbosity=2)
