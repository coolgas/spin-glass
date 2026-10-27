"""Independent Ising MILP and a variational upper bound for the quantum energy."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "issue9-energy2048-20261026" / "source"
sys.path.insert(0, str(SOURCE / "code"))
from kagome_qea.sample import make_sample


def run(sample_id=0, time_limit=120):
    """Binary x_i and y_ij=x_i XOR x_j; s_i=1-2*x_i, J convention -J*s_i*s_j."""
    start = time.monotonic()
    ham = make_sample(6, True, sample_id).hamiltonian(0.0)
    n, m = ham.n_sites, len(ham.J)
    matrix = lil_matrix((4*m, n+m), dtype=float)
    upper = np.zeros(4*m)
    for k, (i, j) in enumerate(ham.bonds):
        # y >= x_i-x_j; y >= x_j-x_i; y <= x_i+x_j; y <= 2-x_i-x_j.
        matrix[4*k, [i, j, n+k]] = [1, -1, -1]
        matrix[4*k+1, [i, j, n+k]] = [-1, 1, -1]
        matrix[4*k+2, [i, j, n+k]] = [-1, -1, 1]
        matrix[4*k+3, [i, j, n+k]] = [1, 1, 1]
        upper[4*k+3] = 2
    objective = np.r_[np.zeros(n), 2*ham.J]
    constant = -float(ham.J.sum())
    lower_bounds = np.zeros(n+m)
    upper_bounds = np.ones(n+m)
    # h=0 gives global spin inversion symmetry; fix the first spin.
    upper_bounds[0] = 0
    result = milp(objective, integrality=np.ones(n+m),
                  bounds=Bounds(lower_bounds, upper_bounds),
                  constraints=LinearConstraint(matrix.tocsr(), -np.inf, upper),
                  options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    if result.x is None:
        raise RuntimeError(str(result))
    spins = (1-2*np.rint(result.x[:n])).astype(np.int8)
    classical_energy = ham.diagonal_energy(spins)
    np.testing.assert_allclose(classical_energy, result.fun+constant, atol=1e-7)
    assert np.max(matrix@np.rint(result.x)-upper) < 1e-7
    gamma = 0.05

    def product_energy(theta):
        z = spins * np.cos(theta)
        return -np.dot(ham.J, z[ham.bonds[:, 0]]*z[ham.bonds[:, 1]])-gamma*np.sin(theta).sum()

    # Product state with <Z_i>=s_i cos(theta_i), <X_i>=sin(theta_i).
    # Any such normalized state gives E0 <= <H>, without a guide or QMC assumption.
    optimized = minimize(product_energy, np.full(n, 0.03), method="L-BFGS-B",
                         bounds=[(0, np.pi/2)]*n,
                         options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8})
    variational = min(classical_energy, float(optimized.fun))
    output = dict(sample_id=sample_id, n_sites=n, gamma=gamma,
                  classical_energy=classical_energy,
                  classical_dual_bound=float(result.mip_dual_bound+constant),
                  classical_mip_gap=float(result.mip_gap),
                  classical_solver_status=int(result.status),
                  classical_solver_message=str(result.message),
                  spins=spins.tolist(), product_angles=optimized.x.tolist(),
                  product_energy=float(optimized.fun),
                  quantum_variational_upper_bound=variational,
                  quantum_lower_bound=float(result.mip_dual_bound+constant-n*gamma),
                  variational_optimization_success=bool(optimized.success),
                  elapsed_seconds=time.monotonic()-start,
                  interpretation="E0 <= variational upper bound. A higher QMC estimate can falsify accuracy; agreement cannot certify it.")
    (ROOT/f"bound_sample{sample_id}.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in output.items() if k not in ("spins", "product_angles")}), flush=True)
    return output


if __name__ == "__main__":
    run()
