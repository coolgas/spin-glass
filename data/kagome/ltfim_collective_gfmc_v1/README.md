# Collective-flip guided GFMC development benchmark

This study compares the existing continuous-time propagator with corrected
finite-step natural, guide-aware, and broad-mixture endpoint proposals. The
four selected L=2 contexts are difficult-case algorithm diagnostics, not a
representative disorder ensemble. All use h=0.15 and frozen original/corrected
guides from `ltfim_correlation_expanded_v1`.

The remote campaign is `/root/autodl-tmp/collective-gfmc-20261103-v1` on AutoDL.
`results` contains the 56-task timing pilot and exact evaluation references;
`overnight-results` contains 56 short pilot tasks, 448 repeated main tasks, and
480 population/time/initialization audits. Up to 18 CPU workers run at once.
`gpu-results` measures direct neural-guide cost without enumeration through
L=10. These short GPU trajectories do not establish ground-state accuracy.

Local snapshots are stored under `pilot/`, `benchmark/`, and `gpu/`. A snapshot
can be incomplete while the remote jobs run. Each completed run has JSON
metadata and a hashed NPZ of moments, ancestry, weight ESS, and transport
diagnostics. Invalid length ratios and ancestry alarms remain in the records.
The campaign automatically validates raw results and writes `review.json` and
`REVIEW.md` after completing its task grid.

Source and frozen-guide hashes are recorded and enforced on checkpoint resume.
Checkpoint recovery was verified bit for bit after interruption partway through
forward walking for all four methods. Local and remote dense-kernel tests pass.
Exact split and continuous-time finite-projection references are evaluation
tools only. Their differences distinguish splitting, finite projection, and
population/sampling effects.

No result is automatically qualified as converged. In particular, good energy,
weight ESS, or short-time transport does not establish accurate connected
spectra. The accessible-size objective remains xi_R/L versus Gamma and L,
subject to population, time, guide, and disorder uncertainty checks.

The offline `low_modes.json` diagnosis identifies a concrete failure in the two
Gamma=0.25 corrected guides. Their squared ground-state overlaps are about
4.9e-10 and 4.2e-11. Their energy errors (0.07963 and 0.13623) almost equal the
first excitation gaps, despite local-energy variances around 1e-5. The guide
correction has concentrated on an excited/metastable sector in these cases.

An additional bounded CPU follow-up is queued behind the primary campaign's
exclusive controller lock. It mixes normalized original-guide probability
into corrected-guide probability at fractions 0.01, 0.1, and 0.5, then repeats
the same diagnostics. These fractions are trial settings, not optimized using
exact labels. `mix*/guide-definition.json` preserves formulas and input hashes.
This queue uses a blocking lock, with no background polling monitor.

The cancelled worldline campaign is a separate preserved study. The AutoDL
instance remains powered on after these jobs finish.
