"""Phase-1 checks.

Fast checks cover geometry, matrix elements, disorder identity, exact moments,
and the continuous-time solvers on one and four spins. The full command also
compares both solvers with exact results on one ``L=2`` sample of each
boundary condition. That is an algorithm check, not a disorder average and
not a crossing.
"""

import math
import sys
from pathlib import Path

import numpy as np

from kagome_qea.exact import ground_state, operator_matrix, thermal_observables
from kagome_qea.hamiltonian import Hamiltonian
from kagome_qea.lattice import build_kagome
from kagome_qea.observables import (
    block_means,
    bootstrap_indices,
    crossings,
    disorder_binder,
    jackknife_disorder_binder,
    jackknife_mean_error,
)
from kagome_qea.pimc import (
    Worldline,
    flip_action_change,
    measure_energy,
    measure_magnetization,
    parallel_temper,
    segment_action_change,
    simulate,
    sweep,
)
from kagome_qea.pqmc import project
from kagome_qea.sample import (
    DEBUG_SAMPLE_COUNT,
    PRODUCTION_SAMPLE_COUNT,
    build_ensemble,
    load_sample,
    make_sample,
    save_ensemble,
)
from kagome_qea.scan import iter_jobs

CHAIN_BONDS = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
CHAIN_J = np.array([0.7, -1.2, 0.4])
# Dense diagonalization of this chain at Γ=0.8, h=0, from an independent session.
CHAIN_E0 = -3.833176055770127


def repository_root():
    return Path(__file__).resolve().parents[2]


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def ok(self, name, detail):
        self.passed += 1
        print(f"pass  {name}: {detail}")

    def fail(self, name, detail):
        self.failed.append(name)
        print(f"FAIL  {name}: {detail}")

    def close(self, name, got, target, atol, stderr=None):
        difference = abs(float(got) - float(target))
        limit = float(atol)
        noise = ""
        if stderr is not None and math.isfinite(float(stderr)):
            noise = f", stderr {float(stderr):.4g}"
            if float(stderr) > atol:
                self.fail(
                    name,
                    f"{got:.6g} vs {target:.6g}, |diff| {difference:.3g}{noise}, "
                    f"stderr exceeds atol {atol}",
                )
                return
            limit = max(limit, 3.0 * float(stderr))
        detail = f"{got:.6g} vs {target:.6g}, |diff| {difference:.3g}{noise}"
        if difference <= limit:
            self.ok(name, detail)
        else:
            self.fail(name, detail + f", atol {atol}")


def _reference_matrix(n_sites, bonds, couplings, gamma, field):
    """Explicit dense matrix, written as loops so it can disagree with the sparse builder."""
    dimension = 1 << n_sites
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    for basis in range(dimension):
        spins = [1 - 2 * ((basis >> site) & 1) for site in range(n_sites)]
        energy = -field * sum(spins)
        for (left, right), coupling in zip(bonds, couplings):
            energy -= coupling * spins[left] * spins[right]
        matrix[basis, basis] = energy
        for site in range(n_sites):
            matrix[basis, basis ^ (1 << site)] -= gamma
    return matrix


def check_lattice(report):
    from generate_kagome_4x4 import build_kagome as exported

    for size in (2, 3, 4):
        for periodic in (False, True):
            points, bonds, triangles, adjacency = build_kagome(size, periodic=periodic)
            again, bonds_again, _, _ = exported(size, periodic=periodic)
            degree = adjacency.sum(axis=1).astype(int)
            expected_bonds = 6 * size * size if periodic else 6 * size * size - 4 * size + 1
            expected_triangles = 2 * size * size if periodic else size * size + (size - 1) ** 2
            expected_degrees = (
                {4: 3 * size * size}
                if periodic
                else {2: 2 * size + 1, 3: 4 * (size - 1), 4: 3 * (size - 1) ** 2}
            )
            values, counts = np.unique(degree, return_counts=True)
            boundary = "PBC" if periodic else "OBC"
            name = f"lattice L={size} {boundary}"
            consistent = (
                len(points) == 3 * size * size
                and len(bonds) == expected_bonds
                and len(triangles) == expected_triangles
                and dict(zip(values.tolist(), counts.tolist())) == expected_degrees
                and np.array_equal(bonds, bonds_again)
                and np.allclose(points, again)
            )
            if consistent:
                report.ok(name, f"{len(points)} sites, {len(bonds)} bonds")
            else:
                report.fail(name, "count or re-export mismatch")
    try:
        build_kagome(1, periodic=True)
    except ValueError:
        report.ok("lattice L=1 PBC", "rejected")
    else:
        report.fail("lattice L=1 PBC", "the torus was accepted")


def check_matrix_elements(report):
    reference = _reference_matrix(4, CHAIN_BONDS, CHAIN_J, 0.8, 0.0)
    eigenvalues = np.linalg.eigvalsh(reference)
    report.close("chain reference energy", eigenvalues[0], CHAIN_E0, 1e-9)
    ham = Hamiltonian(4, CHAIN_BONDS, CHAIN_J, 0.0)
    sparse = operator_matrix(ham, 0.8).toarray()
    mismatch = float(np.max(np.abs(sparse - reference)))
    if mismatch < 1e-9:
        report.ok("chain sparse matrix", f"max abs diff {mismatch:.2e}")
    else:
        report.fail("chain sparse matrix", f"max abs diff {mismatch:.2e}")
    all_up = np.ones(4)
    # All spins +1: -(0.7 - 1.2 + 0.4) = 0.1.
    report.close("chain all-up diagonal", ham.diagonal_energy(all_up), 0.1, 1e-12)

    single = Hamiltonian(1, np.zeros((0, 2), dtype=np.int64), np.zeros(0), 0.3)
    exact = -math.sqrt(0.3 ** 2 + 0.8 ** 2)
    report.close("single-spin ground energy", ground_state(single, 0.8)["energy"], exact, 1e-10)

    rng = np.random.default_rng(7)
    for size, periodic in ((2, False), (2, True), (4, False), (4, True)):
        sample = make_sample(size, periodic, 0)
        points = sample.points
        separation = np.sum((points[sample.bonds[:, 0]] - points[sample.bonds[:, 1]]) ** 2, axis=1)
        wrapped = ~np.isclose(separation, 1.0, atol=1e-6)
        boundary = "PBC" if periodic else "OBC"
        if periodic and not np.any(wrapped):
            report.fail(f"wrap bond L={size}", "every PBC bond is a Euclidean neighbor")
        elif (not periodic) and np.any(wrapped):
            report.fail(f"open bond L={size}", "an OBC bond is not a Euclidean neighbor")
        else:
            report.ok(f"bond geometry L={size} {boundary}", f"wrap bonds {int(np.count_nonzero(wrapped))}")
        for field in (0.0, 0.4):
            operator = sample.hamiltonian(field)
            spins = rng.choice(np.array([-1.0, 1.0]), size=sample.n_sites)
            manual = -field * float(spins.sum())
            manual -= float(
                np.sum(sample.J * spins[sample.bonds[:, 0]] * spins[sample.bonds[:, 1]])
            )
            energy = operator.diagonal_energy(spins)
            worst = abs(energy - manual)
            for site in range(sample.n_sites):
                flipped = spins.copy()
                flipped[site] *= -1
                worst = max(
                    worst,
                    abs(operator.diagonal_energy(flipped) - energy - operator.flip_delta(spins, site)),
                )
            name = f"flip delta L={size} {boundary} h={field}"
            if worst < 1e-8:
                report.ok(name, f"max abs diff {worst:.2e}")
            else:
                report.fail(name, f"max abs diff {worst:.2e}")
        if not periodic and size == 4:
            operator = sample.hamiltonian(0.0)
            n_degree_two = sum(len(operator.neighbors[site]) == 2 for site in range(sample.n_sites))
            if n_degree_two == 2 * size + 1:
                report.ok("OBC degree-2 neighbors", f"{n_degree_two} sites")
            else:
                report.fail("OBC degree-2 neighbors", str(n_degree_two))


def check_exact_rows(report):
    sample = make_sample(2, True, 1)
    operator = sample.hamiltonian(0.25)
    matrix = operator_matrix(operator, 1.0)
    rng = np.random.default_rng(3)
    rows = rng.choice(1 << sample.n_sites, size=24, replace=False)
    worst = 0.0
    for basis in rows:
        spins = np.array([1 - 2 * ((int(basis) >> site) & 1) for site in range(sample.n_sites)])
        worst = max(worst, abs(float(matrix[int(basis), int(basis)]) - operator.diagonal_energy(spins)))
        for site in range(sample.n_sites):
            flipped = int(basis) ^ (1 << site)
            worst = max(worst, abs(float(matrix[int(basis), flipped]) + 1.0))
    if worst < 1e-8:
        report.ok("L=2 PBC matrix rows", f"max abs diff {worst:.2e}")
    else:
        report.fail("L=2 PBC matrix rows", f"max abs diff {worst:.2e}")
    state = ground_state(operator, 1.0)
    if state["residual"] < 1e-8 and state["degeneracy"] == 1:
        report.ok(
            "L=2 PBC ground state",
            f"E {state['energy']:.6f}, gap {state['gap']:.4f}, q2 {state['q2']:.4f}",
        )
    else:
        report.fail(
            "L=2 PBC ground state",
            f"residual {state['residual']}, degeneracy {state['degeneracy']}",
        )


def check_disorder_and_statistics(report):
    for periodic, flag in ((False, 0), (True, 1)):
        sample = make_sample(4, periodic, 0)
        entropy = [1000, 4, 0, flag]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
        drawn = rng.normal(loc=0.0, scale=1.0, size=len(sample.bonds))
        boundary = "PBC" if periodic else "OBC"
        if np.array_equal(sample.seed_entropy, entropy) and np.allclose(sample.J, drawn):
            report.ok(f"disorder seed {boundary}", f"{sample.J.shape[0]} couplings")
        else:
            report.fail(f"disorder seed {boundary}", "SeedSequence draw disagrees")
    root = repository_root()
    data_dir = root / "data"
    sample = make_sample(2, False, 0)
    path = save_ensemble(data_dir, build_ensemble(2, False, DEBUG_SAMPLE_COUNT))
    loaded = load_sample(path, 0)
    if np.allclose(loaded.J, sample.J) and np.array_equal(loaded.bonds, sample.bonds):
        report.ok("disorder roundtrip", path.name)
    else:
        report.fail("disorder roundtrip", "loaded record disagrees")
    for size in (2, 3, 4):
        for periodic in (False, True):
            for count in (DEBUG_SAMPLE_COUNT, PRODUCTION_SAMPLE_COUNT):
                save_ensemble(data_dir, build_ensemble(size, periodic, count))
    report.ok("disorder files", "debug 32 and production 128 for L=2,3,4, both boundaries")

    q2 = np.array([0.2, 0.4, 0.5])
    q4 = 2.0 * q2 ** 2
    binder, error = jackknife_disorder_binder(q2, q4)
    report.close("flat Binder jackknife", binder, 0.5, 1e-12)
    report.close("flat Binder jackknife error", error, 0.0, 1e-12)
    values = np.arange(1, 21, dtype=float)
    usual = float(values.std(ddof=1) / np.sqrt(values.size))
    report.close("jackknife mean error", jackknife_mean_error(values), usual, 1e-12)
    averaged, _ratio = disorder_binder(q2, q4)
    report.close("disorder Binder", averaged, 0.5, 1e-12)
    roots = crossings([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], [1.0, 0.0, 2.0])
    if np.allclose(roots, [0.5, 1.5]):
        report.ok("crossings", "0.5 and 1.5")
    else:
        report.fail("crossings", str(roots))
    if crossings([0.0, 1.0], [1.0, 2.0], [0.0, 0.0]) == []:
        report.ok("absent crossing", "empty")
    else:
        report.fail("absent crossing", "a root was reported")
    means = block_means(np.arange(10, dtype=float), 5)
    if np.allclose(means, [2.0, 7.0]):
        report.ok("block means", "two blocks")
    else:
        report.fail("block means", str(means))
    indexes = bootstrap_indices(4, 3, np.random.default_rng(1))
    if indexes.shape == (3, 4):
        report.ok("bootstrap indices", "shape (3, 4)")
    else:
        report.fail("bootstrap indices", str(indexes.shape))
    jobs = list(iter_jobs([sample], [0.4, 1.0], h_values=(0.0,)))
    if len(jobs) == 2 and jobs[0][2] == 0.0:
        report.ok("scan order", "field held outside the gamma loop")
    else:
        report.fail("scan order", str(jobs))


def _single_spin_quantities(beta, gamma, field):
    epsilon = math.sqrt(field * field + gamma * gamma)
    thermal = math.tanh(beta * epsilon)
    return -epsilon * thermal, (field / epsilon) * thermal, -epsilon


def check_worldline_action(report):
    ham = Hamiltonian(4, CHAIN_BONDS, CHAIN_J, 0.2)
    worldline = Worldline(4, 1.7)
    rng = np.random.default_rng(11)
    for _ in range(30):
        sweep(worldline, ham, 0.9, rng)
    before = worldline.classical_action(ham)
    predicted = flip_action_change(worldline, ham, 1)
    worldline.s0[1] = -worldline.s0[1]
    after = worldline.classical_action(ham)
    report.close("worldline flip action", after - before, predicted, 1e-8)
    # Plant a non-wrapping pair so the segment derivative does not depend on mixing.
    worldline.kinks[2] = [0.25, 0.7]
    passed = 1
    spin = worldline.s0[2] if passed % 2 == 0 else -worldline.s0[2]
    predicted = segment_action_change(worldline, ham, 2, 0.25, 0.7, spin)
    before = worldline.classical_action(ham)
    del worldline.kinks[2][0:2]
    after = worldline.classical_action(ham)
    report.close("worldline segment action", after - before, predicted, 1e-8)


def _kink_histogram(beta, gamma, n_equilibrate, n_measure, seed, stride=4):
    ham = Hamiltonian(1, np.zeros((0, 2), dtype=np.int64), np.zeros(0), 0.0)
    worldline = Worldline(1, beta)
    rng = np.random.default_rng(seed)
    for _ in range(n_equilibrate):
        sweep(worldline, ham, gamma, rng)
    counts = []
    magnetization = []
    energies = []
    for step in range(n_measure * stride):
        sweep(worldline, ham, gamma, rng)
        if (step + 1) % stride == 0:
            counts.append(worldline.kink_count)
            magnetization.append(worldline.integrate_spin(0, 0.0, beta) / beta)
            energies.append(measure_energy(worldline, ham))
    return np.asarray(counts), np.asarray(magnetization), np.asarray(energies)


def check_pimc_single_spin(report):
    beta, gamma = 4.0, 1.0
    counts, magnetization, energies = _kink_histogram(
        beta, gamma, 2000, 20000, seed=5, stride=1
    )
    partition = 2.0 * math.cosh(beta * gamma)
    for kinks in (0, 2, 4, 6, 8):
        expected = 2.0 * (beta * gamma) ** kinks / math.factorial(kinks) / partition
        observed = float(np.mean(counts == kinks))
        stderr = math.sqrt(max(observed * (1.0 - observed), 1e-12) / counts.size)
        report.close(f"kink P(K={kinks})", observed, expected, 0.012, stderr)
    energy, moment, _ground = _single_spin_quantities(beta, gamma, 0.0)
    report.close("PIMC single-spin energy", float(energies.mean()), energy, 0.02)
    report.close("PIMC single-spin mz", float(magnetization.mean()), moment, 0.02)

    field = 0.3
    gamma = 0.8
    beta = 4.0
    ham = Hamiltonian(1, np.zeros((0, 2), dtype=np.int64), np.zeros(0), field)
    result = simulate(
        ham, gamma, beta, n_replicas=1, n_sweeps=8000, n_equilibrate=1000, seed=6, block_size=200
    )
    energy, moment, _ground = _single_spin_quantities(beta, gamma, field)
    report.close("PIMC single-spin field energy", result["energy"], energy, 0.02, result["energy_error"])
    # Magnetization is recovered from a second short reader on the same update rules.
    worldline = Worldline(1, beta)
    rng = np.random.default_rng(6)
    for _ in range(1000):
        sweep(worldline, ham, gamma, rng)
    readings = []
    for _ in range(4000):
        sweep(worldline, ham, gamma, rng)
        readings.append(float(measure_magnetization(worldline)[0]))
    report.close("PIMC single-spin field mz", float(np.mean(readings)), moment, 0.03)
    pair = simulate(
        Hamiltonian(1, np.zeros((0, 2), dtype=np.int64), np.zeros(0), 0.0),
        1.0,
        4.0,
        n_replicas=2,
        n_sweeps=3000,
        n_equilibrate=400,
        seed=7,
        block_size=100,
    )
    report.close("PIMC single-spin pair q", pair["q"], 0.0, 0.05, pair["q_error"])
    report.close("PIMC single-spin pair q2", pair["q2"], 1.0, 1e-9)


def check_pqmc_small(report):
    single = Hamiltonian(1, np.zeros((0, 2), dtype=np.int64), np.zeros(0), 0.0)
    result = project(
        single,
        1.0,
        n_replicas=1,
        n_walkers=1500,
        tau_block=0.2,
        n_equilibrate=20,
        n_measure=40,
        n_forward_cycles=0,
        seed=8,
    )
    report.close("PQMC single-spin energy", result["energy"], -1.0, 0.015, result["energy_error"])
    field = Hamiltonian(1, np.zeros((0, 2), dtype=np.int64), np.zeros(0), 0.3)
    result = project(
        field,
        0.8,
        n_replicas=1,
        n_walkers=1500,
        tau_block=0.2,
        n_equilibrate=20,
        n_measure=40,
        n_forward_cycles=0,
        seed=9,
    )
    exact = -math.sqrt(0.3 ** 2 + 0.8 ** 2)
    report.close("PQMC single-spin field energy", result["energy"], exact, 0.02, result["energy_error"])

    dimer = Hamiltonian(2, np.array([[0, 1]]), np.array([0.7]), 0.0)
    exact = ground_state(dimer, 0.8)
    result = project(
        dimer,
        0.8,
        n_replicas=2,
        n_walkers=2000,
        tau_block=0.2,
        n_equilibrate=30,
        n_measure=40,
        tau_forward=4.0,
        n_forward_cycles=6,
        seed=10,
    )
    report.close("PQMC dimer energy", result["energy"], exact["energy"], 0.03, result["energy_error"])
    report.close("PQMC dimer q2", result["q2"], exact["q2"], 0.03, result["q2_error"])
    report.close("PQMC dimer q4", result["q4"], exact["q4"], 0.05, result["q4_error"])


def check_chain_solvers(report):
    ham = Hamiltonian(4, CHAIN_BONDS, CHAIN_J, 0.0)
    gamma = 0.8
    beta = 3.0
    thermal = thermal_observables(ham, gamma, beta)
    ground = ground_state(ham, gamma)
    if thermal["tail_bound"] > 1e-4:
        report.fail("chain thermal tail", str(thermal["tail_bound"]))
    else:
        report.ok("chain thermal tail", f"{thermal['tail_bound']:.2e}")
    pimc = simulate(
        ham, gamma, beta, n_replicas=2, n_sweeps=4000, n_equilibrate=800, seed=12, block_size=100
    )
    report.close("PIMC chain energy", pimc["energy"], thermal["energy"], 0.05, pimc["energy_error"])
    report.close("PIMC chain q2", pimc["q2"], thermal["q2"], 0.04, pimc["q2_error"])
    pqmc = project(
        ham,
        gamma,
        n_replicas=2,
        n_walkers=2500,
        tau_block=0.2,
        n_equilibrate=30,
        n_measure=50,
        tau_forward=5.0,
        n_forward_cycles=6,
        seed=13,
    )
    report.close("PQMC chain energy", pqmc["energy"], ground["energy"], 0.04, pqmc["energy_error"])
    report.close("PQMC chain q2", pqmc["q2"], ground["q2"], 0.04, pqmc["q2_error"])


def check_parallel_temper(report):
    beta, gammas = 4.0, (0.5, 1.5)
    ham = Hamiltonian(1, np.zeros((0, 2), dtype=np.int64), np.zeros(0), 0.0)
    results, stats = parallel_temper(
        ham,
        gammas,
        beta,
        n_replicas=1,
        n_sweeps=12000,
        n_equilibrate=800,
        seed=16,
        block_size=150,
    )
    report.ok("tempering swaps", f"accept rate {stats['accept_rate']:.3f}")
    for result, gamma in zip(results, gammas):
        energy, _moment, _ground = _single_spin_quantities(beta, gamma, 0.0)
        report.close(
            f"tempered energy Γ={gamma}",
            result["energy"],
            energy,
            0.03,
            result["energy_error"],
        )


def check_l2(report):
    gamma = 1.0
    beta = 3.0
    field = 0.25
    for periodic in (False, True):
        sample = make_sample(2, periodic, 0)
        boundary = "PBC" if periodic else "OBC"
        operator = sample.hamiltonian(0.0)
        ground = ground_state(operator, gamma)
        thermal = thermal_observables(operator, gamma, beta)
        print(
            f"  {boundary} sample 0  E0 {ground['energy']:.4f}  "
            f"gap {ground['gap']:.4f}  q2 {ground['q2']:.4f}  "
            f"thermal E {thermal['energy']:.4f}  tail {thermal['tail_bound']:.2e}"
        )
        if ground["degeneracy"] != 1 or ground["residual"] > 1e-7:
            report.fail(f"L2 {boundary} ground state", "degenerate or residual")
        elif thermal["tail_bound"] > 1e-3:
            report.fail(f"L2 {boundary} thermal tail", str(thermal["tail_bound"]))
        else:
            report.ok(f"L2 {boundary} exact", f"gap {ground['gap']:.4f}")
        # A flat guide collapses lineages, so the overlap error is the scatter
        # across forward-walking cycles. Four units of forward time reaches
        # the ground-state overlap for sample 0: its first excited state is
        # orthogonal to the flat trial.
        pqmc = project(
            operator,
            gamma,
            n_replicas=2,
            n_walkers=5000,
            tau_block=0.2,
            n_equilibrate=30,
            n_measure=25,
            tau_forward=4.0,
            n_forward_cycles=24,
            seed=20 + int(periodic),
        )
        print(
            f"  PQMC {boundary} E {pqmc['energy']:.4f}±{pqmc['energy_error']:.4f}  "
            f"q2 {pqmc['q2']:.4f}±{pqmc['q2_error']:.4f}  "
            f"ancestors {pqmc['min_unique_ancestors']}"
        )
        report.close(
            f"PQMC L2 {boundary} energy",
            pqmc["energy"],
            ground["energy"],
            0.25,
            pqmc["energy_error"],
        )
        report.close(
            f"PQMC L2 {boundary} q2",
            pqmc["q2"],
            ground["q2"],
            0.06,
            pqmc["q2_error"],
        )
        pimc = simulate(
            operator,
            gamma,
            beta,
            n_replicas=2,
            n_sweeps=2500,
            n_equilibrate=600,
            seed=30 + int(periodic),
            block_size=50,
        )
        print(
            f"  PIMC {boundary} E {pimc['energy']:.4f}±{pimc['energy_error']:.4f}  "
            f"q2 {pimc['q2']:.4f}±{pimc['q2_error']:.4f}"
        )
        report.close(
            f"PIMC L2 {boundary} energy",
            pimc["energy"],
            thermal["energy"],
            0.25,
            pimc["energy_error"],
        )
        report.close(
            f"PIMC L2 {boundary} q2",
            pimc["q2"],
            thermal["q2"],
            0.06,
            pimc["q2_error"],
        )
    sample = make_sample(2, False, 0)
    operator = sample.hamiltonian(field)
    thermal = thermal_observables(operator, gamma, beta)
    ground = ground_state(operator, gamma)
    if abs(ground["q"]) <= 1e-8:
        report.fail("L2 OBC h>0 magnetization", "connected marker was not exercised")
    else:
        report.ok("L2 OBC h>0 magnetization", f"⟨q⟩ {ground['q']:.4f}")
    pimc = simulate(
        operator,
        gamma,
        beta,
        n_replicas=2,
        n_sweeps=2500,
        n_equilibrate=600,
        seed=40,
        block_size=50,
    )
    report.close(
        "PIMC L2 OBC field energy",
        pimc["energy"],
        thermal["energy"],
        0.3,
        pimc["energy_error"],
    )
    report.close(
        "PIMC L2 OBC field connected Binder",
        pimc["connected_binder"],
        thermal["connected_binder"],
        0.12,
        pimc["q2_error"],
    )


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    full = "--fast" not in arguments
    report = Report()
    print("Phase-1 checks")
    check_lattice(report)
    check_matrix_elements(report)
    check_exact_rows(report)
    check_disorder_and_statistics(report)
    check_worldline_action(report)
    check_pimc_single_spin(report)
    check_pqmc_small(report)
    check_chain_solvers(report)
    check_parallel_temper(report)
    if full:
        print("L=2 sample_id 0, gamma 1, beta 3. Algorithm check, one sample per boundary.")
        check_l2(report)
    else:
        print("Skipped L=2 solver comparison (--fast).")
    print(f"Passed {report.passed}. Failed {len(report.failed)}.")
    if report.failed:
        print("Failed checks: " + ", ".join(report.failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
