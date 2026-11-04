"""Physical checks for the symmetric probability guide and its VMC gradient."""

import unittest
import numpy as np
import torch

from ising import Hamiltonian
from .physics import basis, diagonal, make_af_sample
from .symmetric import SymmetricTransformer, translate_hamiltonian, translation_permutation
from .vmc import local_energy
from .model import TransformerGuide, EnumeratedGuide


class SymmetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, h=0., clean=False):
        ham=make_af_sample(2,13).hamiltonian(h)
        if clean:
            ham=Hamiltonian(12,ham.bonds,-np.ones(len(ham.bonds)),h)
        torch.manual_seed(91)
        model=SymmetricTransformer(ham,.7,width=8,layers=1,heads=2).double()
        torch.nn.init.normal_(model.base.output.weight,std=.2)
        return ham,model

    def test_covariance_parity_normalization_and_sizes(self):
        spins=torch.as_tensor(basis(12),dtype=torch.long)
        for h,clean in ((0.,False),(.3,False),(0.,True)):
            ham,model=self.model(h,clean)
            with torch.no_grad():
                original=model.log_prob(spins)
                self.assertAlmostEqual(float(original.exp().sum()),1.,places=11)
                if h==0:
                    torch.testing.assert_close(original,model.log_prob(-spins),atol=1e-12,rtol=0)
                else:
                    self.assertGreater(float((original-model.log_prob(-spins)).abs().max()),.01)
                for dx,dy in ((1,0),(0,1),(1,1)):
                    permutation=translation_permutation(2,dx,dy)
                    shifted=translate_hamiltonian(ham,permutation)
                    moved=torch.empty_like(spins);moved[:,permutation]=spins
                    model.set_hamiltonian(shifted,.7)
                    torch.testing.assert_close(original,model.log_prob(moved),atol=1e-12,rtol=0)
                if clean:
                    self.assertEqual(len(model.to_original),4)
        saved={k:v.clone() for k,v in model.state_dict().items()}
        for L in (3,4,6,8):
            ham=make_af_sample(L,17).hamiltonian(0.)
            model.set_hamiltonian(ham,.5)
            draws=model.sample(3)
            self.assertEqual(tuple(draws.shape),(3,3*L*L))
            self.assertTrue(torch.isfinite(model.log_prob(draws)).all())
            # L>2 translations are not generally their own inverses.
            original=model.log_prob(draws)
            permutation=translation_permutation(L,1,0)
            moved=torch.empty_like(draws);moved[:,permutation]=draws
            model.set_hamiltonian(translate_hamiltonian(ham,permutation),.5)
            torch.testing.assert_close(original,model.log_prob(moved),atol=1e-12,rtol=0)
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v,saved[k],atol=0,rtol=0)

    def test_gradient_ratios_and_sampling(self):
        ham,model=self.model()
        spins=torch.as_tensor(basis(12),dtype=torch.long)
        logs=model.log_prob(spins)
        p=logs.exp()
        e=local_energy(model,ham,.7,spins)
        (p.detach()*(e-(p.detach()*e).sum())*logs).sum().backward()
        grad=model.base.output.weight.grad.clone()
        model.zero_grad()
        psi=model.log_prob(spins).mul(.5).exp()
        flips=torch.arange(4096)[:,None] ^ (1 << torch.arange(12))
        diag=torch.as_tensor(diagonal(ham,spins.numpy()))
        (psi*(diag*psi-.7*psi[flips].sum(1))).sum().backward()
        torch.testing.assert_close(grad,model.base.output.weight.grad,atol=1e-10,rtol=1e-8)
        guide=TransformerGuide(model)
        table=EnumeratedGuide(guide.log_amplitudes(spins.numpy()),12)
        np.testing.assert_allclose(guide.all_flip_log_ratios(spins[:19].numpy()),
                                   table.all_flip_log_ratios(spins[:19].numpy()),atol=1e-11)
        # Test actual sampling against nontrivial one- and two-spin moments.
        draws=model.sample(40000,torch.Generator().manual_seed(173)).numpy()
        exact=p.detach().numpy()
        for observable,sampled in ((spins[:,0].numpy(),draws[:,0]),
                                   ((spins[:,0]*spins[:,4]).numpy(),draws[:,0]*draws[:,4])):
            target=float(exact @ observable)
            self.assertLess(abs(sampled.mean()-target),6*np.sqrt((1-target**2)/len(draws)))

    def test_physical_decoder_causality(self):
        _,model=self.model()
        tokens=torch.randint(8,(2,4))
        tokens[1,:2]=tokens[0,:2]
        torch.testing.assert_close(model.base.logits(tokens)[0,:3],
                                   model.base.logits(tokens)[1,:3],atol=1e-12,rtol=0)

    def test_serialized_resume_next_update(self):
        import io
        ham,model=self.model()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        generator=torch.Generator().manual_seed(45)
        def update(model,optimizer,generator):
            optimizer.zero_grad()
            spins=model.sample(32,generator)
            energies=local_energy(model,ham,.7,spins)
            ((energies-energies.mean())*model.log_prob(spins)).mean().backward()
            optimizer.step()
            return spins
        update(model,optimizer,generator)
        buffer=io.BytesIO()
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                        generator=generator.get_state()),buffer)
        expected=update(model,optimizer,generator)
        _,resumed=self.model()
        opt=torch.optim.Adam(resumed.parameters(),lr=.001)
        buffer.seek(0)
        state=torch.load(buffer,weights_only=False)
        resumed.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        rng=torch.Generator();rng.set_state(state['generator'])
        actual=update(resumed,opt,rng)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for a,b in zip(model.parameters(),resumed.parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__=='__main__':
    unittest.main(verbosity=2)
