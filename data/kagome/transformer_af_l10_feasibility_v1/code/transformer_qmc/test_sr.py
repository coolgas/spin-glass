"""Numerical checks for the SR score, shared solve, and physical descent."""

import unittest
import numpy as np
import torch

from .physics import basis, make_af_sample
from .symmetric import SymmetricTransformer
from .sr import score_matrix, context_rows, solve_direction, apply_direction
from .vmc import local_energy, enumerate_model


class SRTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        self.ham = make_af_sample(2, 13).hamiltonian(0.)
        self.model = SymmetricTransformer(self.ham, .5, width=4, layers=1, heads=1).double()

    def test_scores_and_shared_dense_solve(self):
        generator = torch.Generator().manual_seed(19)
        blocks, targets = [], []
        for h in (0., .3):
            ham = make_af_sample(2, 13).hamiltonian(h)
            self.model.set_hamiltonian(ham, .5)
            spins = self.model.sample(9, generator)
            scores = score_matrix(self.model, spins, chunk=4)
            for index in (0, 8):
                values = torch.autograd.grad(.5*self.model.log_prob(spins[index:index+1]).sum(),
                                             tuple(self.model.parameters()))
                expected = torch.cat([v.flatten() for v in values])
                torch.testing.assert_close(scores[index], expected, atol=2e-12, rtol=2e-12)
            rows, force = context_rows(scores, local_energy(self.model, ham, .5, spins), 12, 2)
            blocks.append(rows); targets.append(force)
        rows, force = torch.cat(blocks), torch.cat(targets)
        direction, residual = solve_direction(rows, force, .02)
        dense = rows.T@rows + .02*torch.eye(rows.shape[1], dtype=rows.dtype)
        reference = torch.linalg.solve(dense, rows.T@force)
        torch.testing.assert_close(direction, reference, atol=1e-11, rtol=1e-10)
        self.assertLess(residual, 1e-11)

    def test_exact_weighted_force_and_descent(self):
        spins = torch.as_tensor(basis(12), dtype=torch.long)
        scores = score_matrix(self.model, spins, chunk=64)
        probability = self.model.log_prob(spins).exp().detach()
        energy = local_energy(self.model, self.ham, .5, spins)
        centered = scores-(probability[:, None]*scores).sum(0)
        rows = centered*probability.sqrt()[:, None]/np.sqrt(12)
        force = 2*(energy-probability@energy)*probability.sqrt()/np.sqrt(12)
        direction, _ = solve_direction(rows, force, .01)
        exact = enumerate_model(self.model, self.ham, .5)
        amplitude = (.5*self.model.log_prob(spins)).exp()
        # Direct differentiable Rayleigh quotient, independent of sampled formula.
        from kagome_qea.exact import operator_matrix
        matrix = torch.tensor(operator_matrix(self.ham, .5).toarray(), dtype=torch.float64)
        quotient = amplitude@(matrix@amplitude)/(amplitude@amplitude)/12
        derivatives = torch.autograd.grad(quotient, tuple(self.model.parameters()))
        gradient = torch.cat([v.flatten() for v in derivatives])
        torch.testing.assert_close(rows.T@force, gradient, atol=1e-11, rtol=1e-10)
        report = apply_direction(self.model, direction, [scores], learning_rate=.01, trust=.01)
        after = enumerate_model(self.model, self.ham, .5)
        self.assertLess(after['energy'], exact['energy'])
        self.assertLessEqual(report['predicted_max_logpsi_rms'], .01000001)
        self.assertAlmostEqual(after['normalization'], 1., places=11)
        torch.testing.assert_close(self.model.log_prob(spins), self.model.log_prob(-spins))

    def test_all_size_serialized_resume(self):
        import io
        from .train_remedies import update, choose_context
        generator = torch.Generator().manual_seed(81)
        rng = np.random.default_rng(82)

        def advance(model, generator, rng, step):
            sample_id, gamma, h = choose_context(step, rng, [100, 101], True)
            return update(model, generator, sample_id=sample_id, gamma=gamma, h=h,
                          batch=4, method='sr', score_chunk=2)

        advance(self.model, generator, rng, 0)
        buffer = io.BytesIO()
        torch.save(dict(model=self.model.state_dict(), generator=generator.get_state(),
                        numpy_rng=rng.bit_generator.state), buffer)
        expected = advance(self.model, generator, rng, 1)
        resumed = SymmetricTransformer(self.ham, .5, width=4, layers=1, heads=1).double()
        buffer.seek(0)
        state = torch.load(buffer, weights_only=False)
        resumed.load_state_dict(state['model'])
        other_generator = torch.Generator(); other_generator.set_state(state['generator'])
        other_rng = np.random.default_rng(); other_rng.bit_generator.state = state['numpy_rng']
        actual = advance(resumed, other_generator, other_rng, 1)
        self.assertEqual(expected, actual)
        for a, b in zip(self.model.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_float32_refinement_with_large_nullspace(self):
        rows = torch.randn(256, 23)*3
        force = torch.randn(256)
        direction, residual = solve_direction(rows, force, .01)
        reference, _ = solve_direction(rows, force, .01, precision='double')
        torch.testing.assert_close(direction, reference, atol=2e-7, rtol=2e-6)
        self.assertLess(residual, 1e-7)


if __name__ == '__main__':
    unittest.main()
