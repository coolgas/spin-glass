import unittest
import numpy as np
from rydberg_kagome.hamiltonian import physical_matrix,occupation_matrix,spin_table
from rydberg_kagome.geometry import sample,masks


class HamiltonianTests(unittest.TestCase):
    def test_two_atom_factor(self):
        v=np.array([[0,2.3],[2.3,0]])
        np.testing.assert_allclose(physical_matrix(v,0,0).diagonal(),[2.3,0,0,0],atol=1e-14)

    def test_independent_occupation(self):
        s=sample(2,3)
        for mask in masks(2).values():
            for delta in (2.,3.15):
                a=physical_matrix(s['interactions']*mask,delta,.73)
                b=occupation_matrix(s['interactions']*mask,delta,.73)
                diff=(a-b).data
                self.assertLess(np.max(abs(diff),initial=0),1e-12)

    def test_high_drive_product(self):
        from rydberg_kagome.exact import ground_state
        v=np.zeros((4,4)); result=ground_state(physical_matrix(v,0,2))
        p=abs(result['vector'])**2
        np.testing.assert_allclose(p,np.ones(16)/16,atol=1e-12)
        self.assertAlmostEqual(result['energy'],-4)
