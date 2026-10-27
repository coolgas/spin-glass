"""Disorder records.

A sample is ``(L, periodic, sample_id)`` plus the ensemble name. Couplings
are drawn once and stored. Solver streams are separate from this stream.
Round 1 is ``gaussian_nn``: independent standard normals on the stored bonds.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kagome_qea.hamiltonian import Hamiltonian
from kagome_qea.lattice import build_kagome

SEED_PREFIX = 1000
GAUSSIAN_NN = "gaussian_nn"
DEBUG_SAMPLE_COUNT = 32
PRODUCTION_SAMPLE_COUNT = 128


@dataclass(frozen=True)
class Sample:
    """One stored realization. ``J[b]`` belongs to ``bonds[b]``."""

    L: int
    periodic: bool
    sample_id: int
    seed_entropy: np.ndarray
    points: np.ndarray
    bonds: np.ndarray
    J: np.ndarray
    ensemble: str

    def __post_init__(self):
        if self.J.shape != (self.bonds.shape[0],):
            raise ValueError("J must have one coupling per bond")
        if np.asarray(self.seed_entropy).shape != (4,):
            raise ValueError("seed_entropy must have shape (4,)")

    @property
    def n_sites(self):
        return int(self.points.shape[0])

    def hamiltonian(self, h=0.0):
        """Operator for this record. ``h`` is not part of the stored disorder."""
        return Hamiltonian(self.n_sites, self.bonds, self.J, h)


def seed_entropy(L, sample_id, periodic):
    """Entropy ``[1000, L, sample_id, int(periodic)]``."""
    return np.array(
        [SEED_PREFIX, int(L), int(sample_id), int(bool(periodic))],
        dtype=np.int64,
    )


def draw_couplings(ensemble, bonds, points, rng):
    """One coupling per stored bond.

    ``points`` is unused by ``gaussian_nn``. A later long-range drawer needs
    the positions, and antiferromagnetic couplings are stored as negative
    ``J`` because the operator is ``-∑ J σ^z σ^z``.
    """
    if ensemble == GAUSSIAN_NN:
        return rng.normal(loc=0.0, scale=1.0, size=len(bonds))
    raise NotImplementedError(
        f"Ensemble {ensemble!r} is outside phase 1. "
        "Add a drawer that returns one coupling per stored bond."
    )


def make_sample(L, periodic, sample_id, ensemble=GAUSSIAN_NN):
    """Draw one record. ``sample_id`` starts at 0."""
    if isinstance(sample_id, bool) or int(sample_id) < 0:
        raise ValueError("sample_id must be a non-negative integer")
    points, bonds, _triangles, _adjacency = build_kagome(int(L), periodic=bool(periodic))
    entropy = seed_entropy(L, sample_id, periodic)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy.tolist())))
    couplings = draw_couplings(ensemble, bonds, points, rng)
    return Sample(
        L=int(L),
        periodic=bool(periodic),
        sample_id=int(sample_id),
        seed_entropy=entropy,
        points=np.asarray(points, dtype=np.float64),
        bonds=np.asarray(bonds, dtype=np.int64),
        J=np.asarray(couplings, dtype=np.float64),
        ensemble=str(ensemble),
    )


def build_ensemble(L, periodic, n_samples, ensemble=GAUSSIAN_NN):
    """Samples ``0 .. n_samples-1`` for one graph."""
    if int(n_samples) < 1:
        raise ValueError("n_samples must be positive")
    return [
        make_sample(L, periodic, sample_id, ensemble)
        for sample_id in range(int(n_samples))
    ]


def ensemble_filename(L, periodic, n_samples, ensemble="gaussian_nn"):
    """``kagome/{ensemble}/{obc|pbc}/L{L}_nr{N}.npz`` under the data directory."""
    boundary = "pbc" if periodic else "obc"
    return Path(f"kagome/{ensemble}/{boundary}/L{int(L)}_nr{int(n_samples)}.npz")


def save_ensemble(directory, samples):
    """Write one ``(L, periodic, ensemble)`` file with a sample axis."""
    if not samples:
        raise ValueError("samples is empty")
    first = samples[0]
    for sample in samples:
        if (
            sample.L != first.L
            or sample.periodic != first.periodic
            or sample.ensemble != first.ensemble
        ):
            raise ValueError("a file holds one size, boundary, and ensemble")
        if sample.bonds.shape != first.bonds.shape or not np.array_equal(
            sample.bonds, first.bonds
        ):
            raise ValueError("samples in one file must share the bond list")
    directory = Path(directory)
    path = directory / ensemble_filename(
        first.L, first.periodic, len(samples), first.ensemble
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        L=np.int64(first.L),
        periodic=np.bool_(first.periodic),
        ensemble=np.array(first.ensemble),
        n_r=np.int64(len(samples)),
        sample_id=np.array([sample.sample_id for sample in samples], dtype=np.int64),
        seed_entropy=np.stack([sample.seed_entropy for sample in samples]),
        points=np.asarray(first.points, dtype=np.float64),
        bonds=np.asarray(first.bonds, dtype=np.int64),
        J=np.stack([np.asarray(sample.J, dtype=np.float64) for sample in samples]),
        numpy_version=np.array(np.__version__),
    )
    return path


def load_ensemble(path):
    """Read a file written by :func:`save_ensemble`."""
    with np.load(path, allow_pickle=False) as data:
        n_samples = int(data["n_r"])
        points = np.asarray(data["points"], dtype=np.float64)
        bonds = np.asarray(data["bonds"], dtype=np.int64)
        ensemble = str(data["ensemble"])
        samples = []
        for index in range(n_samples):
            samples.append(
                Sample(
                    L=int(data["L"]),
                    periodic=bool(data["periodic"]),
                    sample_id=int(data["sample_id"][index]),
                    seed_entropy=np.asarray(data["seed_entropy"][index], dtype=np.int64),
                    points=points,
                    bonds=bonds,
                    J=np.asarray(data["J"][index], dtype=np.float64),
                    ensemble=ensemble,
                )
            )
    return samples


def load_sample(path, sample_id):
    """One record from an ensemble file."""
    for sample in load_ensemble(path):
        if sample.sample_id == int(sample_id):
            return sample
    raise KeyError(f"sample_id {sample_id} is not in {path}")
