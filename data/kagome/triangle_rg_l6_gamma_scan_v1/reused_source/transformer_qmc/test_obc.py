"""OBC topology, conditioning, normalized parity mixture and projector checks."""
import unittest
import numpy as np
import torch

from .obc import OpenBoundaryTransformer
from .physics import make_af_sample, basis
from .vmc import local_energy
from .model import EnumeratedGuide
from .projector import project
from .guided_overlap_demo import histogram_tags
from kagome_qea.exact import operator_matrix


class OpenGuideTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.ham = make_af_sample(2, 71000, periodic=False).hamiltonian(0.)
        self.model = OpenBoundaryTransformer(self.ham, .5, width=16, layers=1, heads=2).double()
        self.spins = torch.tensor(basis(12), dtype=torch.long)

    def test_topology_and_open_geometry(self):
        for L in (2, 4, 6, 8, 10):
            ham = make_af_sample(L, 71000, periodic=False).hamiltonian(.05)
            self.assertEqual(len(ham.bonds), 6*L*L-4*L+1)
            self.model.set_hamiltonian(ham, .5)
            features = self.model.base.features.detach().numpy()
            self.assertEqual(int(features[:, 6:12].sum()), len(ham.bonds))
            self.assertTrue(np.all(features[:, :6][features[:, 6:12] == 0] == 0))
            self.assertEqual(float(self.model.base.geometry_bias[0, 0, -1]),
                             -float(self.model.base.slopes[0])*2*(L-1))
        with self.assertRaisesRegex(ValueError, 'open-patch'):
            self.model.set_hamiltonian(make_af_sample(2, 71000, periodic=True).hamiltonian(0.), .5)

    def test_normalization_parity_ratios_local_energy_and_sampling(self):
        with torch.no_grad():
            logs = self.model.log_amplitudes(self.spins).numpy()
            p = np.exp(2*logs)
            self.assertAlmostEqual(p.sum(), 1., places=12)
            np.testing.assert_allclose(logs, logs[::-1], atol=1e-12)
            selected = np.array([0, 73, 991, 4095])
            ratios = self.model.all_flip_log_ratios(self.spins[selected]).numpy()
            np.testing.assert_allclose(ratios, logs[selected[:,None] ^ (1 << np.arange(12))]-logs[selected,None], atol=1e-12)
            direct = local_energy(self.model, self.ham, .5, self.spins[selected]).numpy()
            psi = np.exp(logs)
            expected = operator_matrix(self.ham, .5) @ psi / psi
            np.testing.assert_allclose(direct, expected[selected], atol=1e-11)
            draws = self.model.sample(12000, torch.Generator().manual_seed(917)).numpy()
            # Compare two nontrivial moments with exact probability and a 6-SE bound.
            for values, measured in [(basis(12)[:, 0], draws[:, 0]),
                                      (basis(12)[:, 0]*basis(12)[:, 1], draws[:, 0]*draws[:, 1])]:
                mean = p @ values
                se = np.sqrt((p @ values**2-mean**2)/len(draws))
                self.assertLess(abs(measured.mean()-mean), 6*se+.001)
        result = project(self.ham, .5, EnumeratedGuide(logs,12), n_walkers=32, n_replicas=4,
                         tau_block=.05, n_equilibrate=2, n_measure=2, tau_forward=.1,
                         n_cycles=2, seed=719, tagger=histogram_tags)
        q = np.linspace(-1,1,13)
        np.testing.assert_allclose(np.asarray(result['pq_cycles']) @ q**2, result['q2_cycles'], atol=1e-12)

    def test_context_refresh_and_optimizer(self):
        for L, h, gamma in [(2,.3,2.), (4,0.,.5), (2,0.,.5)]:
            ham = make_af_sample(L,71001,periodic=False).hamiltonian(h)
            self.model.set_hamiltonian(ham,gamma)
            spins = self.model.sample(4,torch.Generator().manual_seed(9))
            energy = local_energy(self.model,ham,gamma,spins)
            loss = ((energy-energy.mean()).detach()*self.model.log_prob(spins)).mean()
            self.model.zero_grad(); loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
