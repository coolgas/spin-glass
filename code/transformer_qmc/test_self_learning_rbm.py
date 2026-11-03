"""Check the amplitude convention and likelihood gradient independently."""
import unittest
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from .physics import basis
from . import self_learning_rbm as feedback
from .self_learning_rbm import likelihood_loss
from .simple_guides import RBM


class SelfLearningTests(unittest.TestCase):
    def test_likelihood_gradient_against_normalized_exact_objective(self):
        model = RBM(3, 2, 9)
        states = torch.tensor(basis(3), dtype=torch.float64)
        data = states[[0, 0, 1, 2, 7]]
        logs = model(states)
        probability = torch.softmax(2*logs.detach(), 0)
        exact = torch.logsumexp(2*logs, 0)-2*model(data).mean()
        truth = torch.autograd.grad(exact, tuple(model.parameters()))
        # Weight all negative states by the exactly normalized density.
        surrogate = 2*((probability*model(states)).sum()-model(data).mean())
        actual = torch.autograd.grad(surrogate, tuple(model.parameters()))
        for a, b in zip(actual, truth):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        # Production helper must use twice the log-amplitude derivative.
        actual = torch.autograd.grad(likelihood_loss(model, data, states), tuple(model.parameters()))
        expected = torch.autograd.grad(2*(model(states).mean()-model(data).mean()), tuple(model.parameters()))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_mixed_distribution_update_has_ground_state_fixed_point(self):
        psi0 = np.array([.1, .3, .7, 1.2])
        guide = np.array([1.2, .4, .2, .1])
        errors = []
        for _ in range(8):
            delta = np.log(guide/psi0); errors.append(np.ptp(delta))
            mixed = guide*psi0; mixed /= mixed.sum()
            guide = np.sqrt(mixed)
        np.testing.assert_allclose(np.array(errors[1:])/errors[:-1], .5, atol=1e-12)
        fixed = np.sqrt(psi0**2/np.sum(psi0**2))
        np.testing.assert_allclose(fixed/fixed[0], psi0/psi0[0])

    def test_two_copy_gibbs_preserves_squared_amplitude_density(self):
        # Independent exact transition matrix for the two-hidden-copy Gibbs move.
        from itertools import product
        model = RBM(2, 1, 3)
        with torch.no_grad():
            model.a[:] = torch.tensor([.31, -.2]); model.b[:] = torch.tensor([.12, -.22])
            model.w[:] = torch.tensor([[.4, -.31], [.27, .18]])
        x = torch.tensor(basis(2), dtype=torch.float64)
        hidden = torch.tensor(list(product([-1., 1.], repeat=4)), dtype=torch.float64).reshape(-1, 2, 2)
        transition = np.zeros((4, 4))
        for i, s in enumerate(x):
            ph = torch.sigmoid(2*(s@model.w+model.b))
            for h in hidden:
                mass = torch.where(h > 0, ph, 1-ph).prod()
                pv = torch.sigmoid(2*(2*model.a+h.sum(0)@model.w.T))
                transition[i] += (mass*torch.where(x > 0, pv, 1-pv).prod(1)).detach().numpy()
        p = torch.softmax(2*model(x), 0).detach().numpy()
        np.testing.assert_allclose(p@transition, p, atol=1e-12)

    def test_interrupted_fit_resumes_identical_trajectory(self):
        self.check_interrupted_fit('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_interrupted_fit_resumes_identical_trajectory(self):
        self.check_interrupted_fit('cuda')

    def check_interrupted_fit(self, device):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(L=2, gamma=.6, sample_id=60000, seed=0,
                stints=2, walkers=8, equil=.1, collect=.2, delta=.1,
                sample_walkers=8, updates=30, batch=8, gibbs=1, initial_sweeps=1,
                lr=.001, device=device, initial=None, output=Path(temporary)/'full')
            feedback.run(args)
            full = torch.load(args.output/'checkpoint.pt', weights_only=False)
            args.output = Path(temporary)/'resumed'
            original = feedback.save_checkpoint
            def interrupt(path, state):
                original(path, state)
                if state.get('phase') == 'fit' and state.get('update') == 25:
                    raise InterruptedError('intentional restart regression')
            with patch.object(feedback, 'save_checkpoint', interrupt):
                with self.assertRaises(InterruptedError):
                    feedback.run(args)
            feedback.run(args)
            resumed = torch.load(args.output/'checkpoint.pt', weights_only=False)
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], atol=0, rtol=0)
            torch.testing.assert_close(full['negative'], resumed['negative'], atol=0, rtol=0)
            torch.testing.assert_close(full['torch_rng'], resumed['torch_rng'], atol=0, rtol=0)
            self.assertEqual(full['rng'], resumed['rng'])
            self.assertEqual(full['rows'], resumed['rows'])
            np.testing.assert_array_equal(full['states'], resumed['states'])
            self.assertEqual(full['optimizer']['param_groups'], resumed['optimizer']['param_groups'])
            for index, values in full['optimizer']['state'].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, resumed['optimizer']['state'][index][name],
                                               atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
