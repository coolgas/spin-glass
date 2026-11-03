# Iterative PQMC–RBM feedback — 2026-11-03

The current L4 production, L6 training-budget anchors, and .85/1.00 references
use energy-only VMC followed by fixed-guide GFMC. They do not implement the
self-learning protocol in Brodoloni and Pilati, *Zero-temperature Monte Carlo
simulations of two-dimensional quantum spin glasses guided by neural network
states*, https://arxiv.org/abs/2407.05978, Sec. II.2. Increasing VMC updates does
not test that protocol. These campaigns remain useful as fixed-guide controls.

The feedback rule follows Pilati, Inack and Pieri, Phys. Rev. E 100, 043301
(2019), https://arxiv.org/abs/1907.00907, Secs. II.3–II.4. For guide psi_s,
equilibrated importance-sampled walkers follow f_s proportional to psi_s psi_0
in the infinite-population/projection limit. Fit a normalized density p_theta
to these walkers by minimizing KL(f_s || p_theta), then use sqrt(p_theta) as
the next guide. Ideally psi_(s+1) is proportional to sqrt(psi_s psi_0); the
log-amplitude error, modulo a constant, halves at each iteration. The mixed
walker distribution is a training target, not a pure-observable estimate.

Our RBM.forward represents log(psi_theta), so p_theta is proportional to
exp(2 RBM.forward). The negative phase samples psi_theta squared using the
existing two-hidden-copy Gibbs transition. The likelihood gradient is
2(E_model grad log psi - E_walkers grad log psi). Finite persistent Gibbs
chains approximate the negative expectation. This is the same ideal guide
update, with our existing squared-amplitude family rather than the paper's
single-RBM probability family; it is not an exact reproduction of their
architecture or training schedule. At h=.15 the visible and hidden biases
remain trainable; the EA paper's zero-field inversion symmetry cannot be imposed.

`self_learning_rbm.py` implements a finite resumable pilot. Each stint starts
a new population from the current guide, excludes initial equilibration, stores
mixed walker configurations, then fits the next guide. Restarting the population
avoids carrying a distribution with the previous guide into a new importance
transform. All projection steps and likelihood parameters/RNGs are checkpointed
every 25 steps, with immutable input/output models and raw walker/fit diagnostics
per completed stint. No physical measurements or forward tags span guide updates.
Training ancestry is recorded and does not trigger selection of favorable samples.

Initial validation: exact likelihood-gradient comparison on an enumerated tiny
system, exact invariance of the two-copy Gibbs kernel, and the idealized feedback
fixed point. Then run finite L2 kagome pilots at h=.15, Gamma=.60, disorder60000,
seeds0/1, six stints each, 1024 walkers, equilibration8 plus collection12,
delta=.1, 64 randomly selected walkers/block, 1000 Adam updates/stint, lr=.001,
256 persistent negative chains and ten Gibbs sweeps/update. Exact calculations
are evaluation only. Archive fidelity, variational energy, ground-probability TV,
and KL to each empirical target before/after fitting. These choices are a pilot,
not convergence settings or an assertion of monotonic exact improvement.

L6 deployment follows assessment of these finite pilots and restart validation.
Retain both disorders60000/60001 and seeds0/1 in any subsequent L6 comparison,
compare the same starting VMC checkpoints, and independently measure final guides
with the matched four-population overlap protocol. Check population, projection
time, training-negative-phase mixing, P(q), q2, connected spectra and ancestry
jointly. Finishing adaptive stints or lowering energy does not certify convergence.
The 2D EA results concern square-lattice Gaussian couplings at zero longitudinal
field; they motivate this method but do not qualify our kagome finite-field case.
