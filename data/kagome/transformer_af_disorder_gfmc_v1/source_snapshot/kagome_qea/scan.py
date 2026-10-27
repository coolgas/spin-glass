"""Job order for a parameter scan.

Each yielded triple is one equilibrium run. Samples do not interact, so a
batch can split the ``sample_id`` range across processes. A new ensemble
replaces the stored bonds and couplings. A field scan reuses those couplings
and changes ``h``.
"""


def iter_jobs(samples, gammas, h_values=(0.0,)):
    """Yield ``(sample, gamma, h)`` with the field held fixed across ``Γ``."""
    for sample in samples:
        for field in h_values:
            for gamma in gammas:
                yield sample, float(gamma), float(field)
