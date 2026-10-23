#!/usr/bin/env python3
"""Ground-state energy of short TFIM chains from flat-guide GFMC.

From the repository root:

    python code/benchmark_tfim_gfmc.py

The reference is exact diagonalization. A row passes when the block standard
error is at most 0.03 and both energy estimators lie within
max(0.03, 3 standard errors) of that reference. Each run is written under
``data/tfim/gfmc/{lattice}/{boundary}/{size}/J…/gamma…/h…/seed….npz``.
"""

import math
import sys
from pathlib import Path

import numpy as np

from gfmc import GFMC, save_run
from ising import chain, square
from kagome_qea.exact import ground_state

ROOT = Path(__file__).resolve().parents[1]


ATOL = 0.03
SETTINGS = dict(n_walkers=2500, tau_block=0.2, n_equilibrate=40, n_measure=50)


def main():
    report = _Report()
    print("TFIM GFMC, flat guide, J = 1")
    _check_graphs(report)
    _compare(
        report, "one spin, h = 0", chain(1), 1.0, -1.0, seed=1,
        lattice="chain", size="n1", periodic=False, configurations=np.array([1]),
    )
    _compare(
        report, "one spin, h = 0.3", chain(1), 0.8,
        -math.sqrt(0.3 ** 2 + 0.8 ** 2), seed=2,
        lattice="chain", size="n1", periodic=False, h=0.3,
    )
    _compare(
        report, "open chain N = 8", chain(8, periodic=False), 1.0, None, seed=3,
        lattice="chain", size="n8", periodic=False,
    )
    _compare(
        report, "periodic chain N = 8", chain(8, periodic=True), 1.0, None, seed=4,
        lattice="chain", size="n8", periodic=True,
    )
    _compare(
        report, "open square 3×2", square(3, 2, periodic=False), 1.0, None, seed=5,
        lattice="square", size="nx3_ny2", periodic=False,
    )
    print(f"Passed {report.passed}. Failed {len(report.failed)}.")
    if report.failed:
        print("Failed checks: " + ", ".join(report.failed))
        return 1
    return 0


def _check_graphs(report):
    open_chain = chain(8, periodic=False)
    periodic_chain = chain(8, periodic=True)
    patch = square(3, 2, periodic=False)
    if open_chain.bonds.shape[0] == 7 and periodic_chain.bonds.shape[0] == 8 and patch.bonds.shape[0] == 7:
        report.ok("graphs", "chain 7/8 bonds, 3×2 square 7 bonds")
    else:
        report.fail(
            "graphs",
            f"bonds {open_chain.bonds.shape[0]}, {periodic_chain.bonds.shape[0]}, {patch.bonds.shape[0]}",
        )
    try:
        square(1, 4, periodic=True)
    except ValueError:
        report.ok("periodic square", "a side of length 1 is rejected")
    else:
        report.fail("periodic square", "a side of length 1 was accepted")


def _compare(
    report, name, hamiltonian, gamma, exact, seed, lattice, size, periodic,
    configurations=None, h=None,
):
    if h is not None:
        hamiltonian = hamiltonian.with_field(h)
    if exact is None:
        exact = ground_state(hamiltonian, gamma)["energy"]
    result = GFMC(hamiltonian, gamma, h=h).run(seed=seed, configurations=configurations, **SETTINGS)
    boundary = "pbc" if periodic else "obc"
    path = save_run(
        ROOT / "data",
        {
            "lattice": lattice,
            "boundary": boundary,
            "size": size,
            "n_sites": hamiltonian.n_sites,
            "J": 1.0,
            "gamma": gamma,
            "h": hamiltonian.h,
            "seed": seed,
            "guide": "constant",
            "exact_energy": exact,
            **SETTINGS,
            **result,
        },
    )
    for label, got, stderr in (
        ("mixed", result["energy"], result["energy_error"]),
        ("growth", result["energy_growth"], result["energy_growth_error"]),
    ):
        report.close(f"{name} {label}", got, exact, ATOL, stderr)
    print(
        f"  {name}: mixed {result['energy']:.4f}±{result['energy_error']:.4f}  "
        f"growth {result['energy_growth']:.4f}±{result['energy_growth_error']:.4f}  "
        f"exact {exact:.4f}"
    )
    print(f"    {path.relative_to(ROOT)}")


class _Report:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def ok(self, name, detail):
        self.passed += 1
        print(f"pass  {name}: {detail}")

    def fail(self, name, detail):
        self.failed.append(name)
        print(f"FAIL  {name}: {detail}")

    def close(self, name, got, target, atol, stderr):
        difference = abs(float(got) - float(target))
        if not math.isfinite(float(stderr)) or float(stderr) > atol:
            self.fail(name, f"{got:.6g} vs {target:.6g}, stderr {stderr} exceeds {atol}")
            return
        limit = max(float(atol), 3.0 * float(stderr))
        detail = f"{got:.6g} vs {target:.6g}, |diff| {difference:.3g}, stderr {float(stderr):.4g}"
        if difference <= limit:
            self.ok(name, detail)
        else:
            self.fail(name, detail)


if __name__ == "__main__":
    raise SystemExit(main())
