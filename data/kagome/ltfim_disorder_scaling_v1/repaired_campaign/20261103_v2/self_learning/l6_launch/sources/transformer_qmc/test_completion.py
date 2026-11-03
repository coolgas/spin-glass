"""Causality, sign and normalization of the independent-spin completion input."""

import unittest
import numpy as np
import torch

from .physics import basis, make_af_sample
from .symmetric import SymmetricTransformer, translate_hamiltonian, translation_permutation


class CompletionTests(unittest.TestCase):
    def test_direct_enumeration_causality_and_symmetry(self):
        torch.set_num_threads(2)
        ham = make_af_sample(2, 17).hamiltonian(.3)
        model = SymmetricTransformer(ham, .5, width=8, layers=1, heads=2, completion=True).double()
        with torch.no_grad():
            model.base.completion_scale.fill_(.3)
        decoder = model.base
        tokens = torch.tensor([[0, 1, 2, 3], [0, 1, 6, 7]])
        torch.testing.assert_close(decoder.logits(tokens)[0, :3], decoder.logits(tokens)[1, :3])
        canonical = decoder.future_coupling.numpy()
        spins = decoder.spin_options[tokens[0]].numpy().reshape(-1)
        options = decoder.spin_options.numpy()
        actual = decoder.completion_energy(tokens).detach().numpy()
        for cell in range(4):
            expected = np.zeros(8)
            for site in range(3*(cell+1), 12):
                if not np.any(canonical[site, 3*cell:3*cell+3]):
                    continue
                past = canonical[site, :3*cell] @ spins[:3*cell]-.3/.5
                change = options @ canonical[site, 3*cell:3*cell+3]
                expected += -np.sqrt(1+(past+change)**2)+np.sqrt(1+past**2)
            np.testing.assert_allclose(actual[0, cell], expected, atol=2e-14)
        states = torch.tensor(basis(12), dtype=torch.long)
        for h in (.3, 0.):
            ham = make_af_sample(2, 17).hamiltonian(h)
            model.set_hamiltonian(ham, .5)
            before = model.log_prob(states)
            self.assertAlmostEqual(float(before.detach().exp().sum()), 1., places=12)
            if h == 0:
                torch.testing.assert_close(before, model.log_prob(-states))
            permutation = translation_permutation(2, 1, 1)
            moved = torch.empty_like(states); moved[:, permutation] = states
            model.set_hamiltonian(translate_hamiltonian(ham, permutation), .5)
            torch.testing.assert_close(before, model.log_prob(moved), atol=2e-13, rtol=0)
        loss = model.log_prob(states[:8]).sum()
        derivative, = torch.autograd.grad(loss, [model.base.completion_scale])
        self.assertTrue(torch.isfinite(derivative))

    def test_completion_scores(self):
        from .sr import score_matrix
        torch.set_num_threads(2)
        ham = make_af_sample(2, 17).hamiltonian(.3)
        model = SymmetricTransformer(ham, .5, width=4, layers=1, heads=1, completion=True).double()
        states = torch.tensor(basis(12)[:5], dtype=torch.long)
        scores = score_matrix(model, states, chunk=2)
        derivatives = torch.autograd.grad(.5*model.log_prob(states[:1]).sum(), tuple(model.parameters()))
        torch.testing.assert_close(scores[0], torch.cat([v.flatten() for v in derivatives]), atol=2e-12, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
