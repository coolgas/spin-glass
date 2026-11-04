# Simpler guiding wavefunctions: controlled comparison

The first comparison uses per-disorder positive real wavefunctions:

* Jastrow: log psi(s) = sum_i a_i s_i + sum_(i<j) u_ij s_i s_j.
* RBM: log psi(s) = sum_i a_i s_i + sum_mu log[2 cosh(b_mu + sum_i W_i_mu s_i)].

RBM hidden densities alpha=1 and 2 give 12 and 24 hidden units at L=2.
Both models have analytic single-spin flip ratios and no token ordering or
positional encoding. The current Jastrow includes all pairs, not just bonds.

## What the experiment isolates

Train from scratch on four selected disorders, with three initializations for
each model. Use 800 Adam steps and up to 200 L-BFGS iterations. Optimize only
the variational energy. Full-basis expectations remove MCMC training failure
as a confound **at L=2 only**. Exact ground-state amplitudes are loaded after
fitting, solely for evaluation. Keep every seed, including metastable failures.
Choose the lowest-energy seed before running matched GFMC.

This compares complete fitting pipelines. It does not isolate ALiBi, nor match
the transformer correction's training budget. An architecture ablation would
need matched optimization, data, parameter counts and compute, including a
transformer without ALiBi. Cheap small models justify a simpler baseline now.

Ground fidelity, spectra, local-energy variance, seed-to-seed failures and exact
finite projection are all recorded. Low variance alone is insufficient.
Selected-guide GFMC uses four independent populations, 1024 walkers each,
equilibration 32, forward time 32, 16 cycles and four repeats. It compares
continuous-time and triangle rules with identical guide and estimator settings.
GFMC raw moments and ancestry are preserved. These are selected L2 diagnostics,
not a disorder-averaged finite-size result or convergence certificate.

## Scalable interface, unqualified training

`SimpleGuide` evaluates either ansatz directly and exposes analytic ratios.
Its `sample()` is finite-length MCMC. For the RBM, psi squared is represented by
**two independent copies of the hidden spins** conditional on visibles. One
ordinary hidden copy would sample psi, not psi squared. Tests enumerate a tiny
joint distribution to verify the doubled-hidden convention and detailed balance.
Jastrow uses single-site Metropolis and global inversion.

Neither finite-chain initialization nor sampled training is qualified by the
full-basis study. Larger sizes require mixing diagnostics, independent starts,
and sampled-VMC optimization checks. Simpler ratios can reduce GFMC cost without
guaranteeing that the trained guide supports every important valley.

## Sampled training follow-up on AutoDL

`sampled_simple_guides.py` trains without enumerating the basis. It compares
direct energy minimization with continuation from Gamma=1.5 through 0.75 to the
target field. Both schedules use identical 400-step stages; continuation uses
more total updates, so this comparison does not isolate schedule from compute.
Two seeds, Jastrow and alpha=2 RBM, four L2 contexts, and two L4 contexts give
48 tasks. All fits are retained; no post-hoc fidelity selection is made.

Training samples come from eight parallel-tempering levels with density
psi^(2 beta), beta from 0 to 1, and 256 chains per level. Local Metropolis moves
use cached fields, global inversion is Metropolis corrected, beta=0 states are
refreshed uniformly, and exchanges use the exact tempered density ratio.
This sampled-training experiment does not use the doubled-hidden Gibbs kernel.

The VMC gradient is twice the covariance of local energy with log-amplitude
derivatives, divided by site count. Local energies are detached in the surrogate
loss. Adam uses learning rate 0.01 and a unit gradient-norm cap; no local-energy
or importance-ratio clipping is used. Parameters, optimizer, RNG, chain states,
temperature labels, and training history are checkpointed every 25 steps.

After freezing parameters, new random/up/down initializations undergo 200 warmup
and 200 measurement sweeps, recording every fourth measurement. Raw energies,
magnetizations, pairs, and L2 samples are saved. Temperature round trips and
exchange fractions do not establish valley equilibration. Exact L2 guide
probabilities and Hamiltonian expectations are evaluated after training to
measure sampling discrepancy independently. L4 results are feasibility probes.

`audit_simple_guides.py` separately checks the full-basis-trained selected
Jastrow and alpha=2 RBM guides using 4096-walker runs, doubled projection times,
uniform initialization, and triangle step 0.05. It retains invalid xi/L ratios
and whole-run bootstrap results. Exact CT finite-projection references use the
actual forward-walking schedule, unlike the symmetric-state illustration in
the initial training screen. Two audit repeats are sensitivity checks, not a
precision certificate.

All experiments, tests and notebook execution now run on AutoDL. Local use is
limited to source editing and transfers; the initial local runs are preserved
with their original provenance rather than relabeled as remote measurements.
