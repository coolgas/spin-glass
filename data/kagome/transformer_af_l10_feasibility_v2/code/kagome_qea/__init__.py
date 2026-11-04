"""Phase-1 code for the Kagome quantum Edwards–Anderson model.

The research plan lives in the notebook. This package is the solver side:
geometry, stored disorder, observables, exact diagonalization for N <= 12,
continuous-time projector QMC, and continuous-time path-integral Monte Carlo.
"""

__version__ = "0.1.0"

from kagome_qea.lattice import build_kagome
from kagome_qea.hamiltonian import Hamiltonian
from kagome_qea.sample import load_sample, make_sample, save_ensemble

__all__ = [
    "Hamiltonian",
    "build_kagome",
    "load_sample",
    "make_sample",
    "save_ensemble",
]
