"""Independent dense-kernel checks for collective GFMC proposal correctness."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm

from .collective_projector import SplitProposal
from .collective_campaign import split_action
from .model import EnumeratedGuide
from .physics import basis, diagonal


class CollectiveTests(unittest.TestCase):
    def setUp(self):
        self.ham = SimpleNamespace(n_sites=3, bonds=np.array([[0, 1], [1, 2], [0, 2]]),
                                   J=np.array([-.7, -1., -.9]), h=.15)
        self.states = basis(3)
        self.guide = EnumeratedGuide(-.2 * diagonal(self.ham, self.states), 3)

    def test_endpoint_probabilities_and_weights(self):
        rng = np.random.default_rng(7)
        v = diagonal(self.ham, self.states)
        kinetic = np.zeros((8, 8))
        for s in range(8):
            for i in range(3):kinetic[s ^ (1 << i), s] = .5
        half = np.diag(np.exp(-.1 * v))
        target = half @ expm(.2 * kinetic) @ half
        target *= np.exp(self.guide.logs[:, None] - self.guide.logs[None, :])
        for cache in [False, True]:
            proposal = SplitProposal(self.ham, .5, self.guide, delta=.2, epsilon=.3, cache=cache)
            for s in range(8):
                source = np.repeat(self.states[s:s+1], 8, axis=0)
                _, log_g = proposal.guided(source, rng, destination=self.states)
                np.testing.assert_allclose(np.exp(log_g).sum(), 1., atol=1e-13)
                x = np.count_nonzero(source != self.states, axis=1)
                log_b = -np.log(3.) - proposal.log_choose[x]
                log_b[x == 0] = -np.inf
                log_q = np.logaddexp(np.log(.7) + log_g, np.log(.3) + log_b)
                w = (-.1 * (v[s] + v) + 3 * proposal.log_cosh + x * proposal.log_tanh
                     + self.guide.logs - self.guide.logs[s] - log_q)
                np.testing.assert_allclose(np.exp(log_q + w), target[:, s], atol=1e-13)

    def test_random_draw_reconstructs_kernel_column(self):
        rng = np.random.default_rng(19)
        proposal = SplitProposal(self.ham, .5, self.guide, delta=.2, epsilon=.3, cache=True)
        source = np.repeat(self.states[0:1], 80000, axis=0)
        destinations, logs, _ = proposal.draw(source, rng)
        ix = self.guide.indexes(destinations)
        empirical = np.bincount(ix, weights=np.exp(logs), minlength=8) / len(source)
        v = diagonal(self.ham, self.states)
        x = np.count_nonzero(self.states != self.states[0], axis=1)
        exact = (np.exp(-.1 * (v + v[0]) + self.guide.logs - self.guide.logs[0])
                 * np.sinh(.1)**x * np.cosh(.1)**(3-x))
        np.testing.assert_allclose(empirical, exact, atol=.007)

    def test_natural_and_cached_direct_equivalence(self):
        source = np.tile(self.states, (300, 1))
        for natural in [False, True]:
            for epsilon in [0., .3]:
                direct = SplitProposal(self.ham,.5,self.guide,.2,epsilon,cache=False,natural=natural)
                cached = SplitProposal(self.ham,.5,self.guide,.2,epsilon,cache=True,natural=natural)
                a=direct.draw(source,np.random.default_rng(73))
                b=cached.draw(source,np.random.default_rng(73))
                for left,right in zip(a,b):np.testing.assert_allclose(left,right,atol=1e-13)
        proposal=SplitProposal(self.ham,.5,self.guide,.2,natural=True)
        _, logs=proposal.guided(np.repeat(self.states[:1],8,axis=0),None,destination=self.states)
        count=np.count_nonzero(self.states!=self.states[0],axis=1)
        p=np.tanh(.1)/(1+np.tanh(.1))
        np.testing.assert_allclose(np.exp(logs),p**count*(1-p)**(3-count),atol=1e-14)

    def test_split_reference_action(self):
        potential=diagonal(self.ham,self.states)
        kinetic=np.zeros((8,8))
        for source in range(8):
            for site in range(3):kinetic[source^(1<<site),source]=.5
        half=np.diag(np.exp(-.1*potential))
        target=half@expm(.2*kinetic)@half
        for source in range(8):
            np.testing.assert_allclose(split_action(np.eye(8)[:,source],potential,.5,.2),target[:,source],atol=1e-14)


if __name__ == '__main__':unittest.main()
