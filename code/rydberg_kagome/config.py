"""Frozen physical conventions. Energies are expressed in V0."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Physics:
    R_um: float = 8.0
    w_um: float = 1.0
    minimum_distance_um: float = 4.0
    c6_MHz_um6: float = 1074181.80463517
    seed_prefix: int = 20261116
    max_attempts: int = 100000

    @property
    def v0_MHz(self):
        return self.c6_MHz_um6 / self.R_um**6

    def validate(self):
        if min(self.R_um, self.minimum_distance_um, self.c6_MHz_um6) <= 0 or self.w_um < 0:
            raise ValueError("Invalid physical scale")
        if self.max_attempts < 1:
            raise ValueError("No geometry attempts allowed")
        return self


def drive_grid():
    return np.geomspace(0.03, 10.0, 30)


def specification():
    g = drive_grid()
    return dict(schema=1, physics=asdict(Physics()), L=4,
                sample_ids=list(range(128)), deltas=[2.0, 3.15],
                models=["full", "nn_nnn", "nn"], drives=g.tolist(),
                rb_over_R=(g**(-1/6)).tolist(), rb_um=(8*g**(-1/6)).tolist(),
                pair_convention="ordered i!=j: occupation V/2; ZZ V/8",
                distance_convention="euclidean-shortest-oblique-v1",
                chi_ladder=[8, 12, 16, 24, 32], D_ladder=[32, 64, 128, 256, 512],
                starts=[0, 1], replica_pairs=20000,
                production_chi=12,production_D=128,
                production_requires="user-selected approximation; per-case solver/start/sampling checks; all 128 samples; RG sensitivity reported")
