import numpy as np
import itertools

from typing import List, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # ../

class RydSimFull1D:
    def __init__(self, L: int, Omega: float, Delta: float, C6: float, R0: float):
        """
        Inputs:
            L: The chain length.
            Omega: The Rabi frequency.
            Delta: The detuning.
            C6: Interaction strength.
            R: The lattice constant.
        """
        self.L = L
        self.Omega = Omega
        self.Delta = Delta
        self.C6 = C6
        self.R0 = R0

    @property
    def basis(self):
        combinations = np.array(list(itertools.product([0, 1], repeat=self.L)), dtype=int)
        return combinations
    
    def compute_bonds(self, noises: List[float]) -> np.ndarray:
        """Compute inter-site bond lengths without building the full Hamiltonian.

        Suitable for large L where the dense Hamiltonian would be intractable
        (e.g. L > 14 with TDVP-based dynamics).
        """
        L = self.L
        assert L == len(noises), "The lenth of the noises should match the chain length."
        r = np.array([i * self.R0 + noises[i] for i in range(L)], dtype=float)
        bonds = np.diff(r)
        return bonds

    def build_Ham(self, noises: List[float]) -> Tuple[np.ndarray, List]:
        L = self.L
        assert L == len(noises), "The lenth of the noises should match the chain length."

        Omega = self.Omega
        Delta = self.Delta
        C6 = self.C6
        R0 = self.R0

        r = [i*R0+noises[i] for i in range(L)]
        bonds = np.diff(r)

        basis = self.basis
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=float)

        # One body terms
        for i in range(dim):
            b = basis[i]
            for j in range(L):
                H[i, i] += Delta if b[j] == 1 else 0
                b_cpy = np.copy(b)
                b_cpy[j] = (b[j] + 1) % 2
                idx = np.where((basis==b_cpy).all(axis=1))[0][0]
                H[idx, i] += Omega/2
        
        # Two body terms
        for i in range(dim):
            b = basis[i]
            for j in range(L-1):
                for k in range(j+1, L):
                    if b[j] == b[k] == 1:
                        dij = np.abs(r[k]-r[j]) # the distance between site k and j
                        Vij = C6/dij**6
                        H[i, i] += Vij

        return H, bonds

