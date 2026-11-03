# Hamiltonian-aware triangle GFMC

For H = -sum J_ij Z_i Z_j - h sum Z_i - Gamma sum X_i, every Kagome bond
belongs to one elementary triangle and every site to two. Divide the triangles
into two vertex-disjoint layers. Each triangle Hamiltonian contains its three
bonds, half the longitudinal field, and half the transverse field on its sites.
Consequently H=A+B exactly. Terms within a layer commute.

Use G_delta = exp(-delta A/2) exp(-delta B) exp(-delta A/2). Each local block
is computed by an exact 8-by-8 matrix exponential. This retains the frustrated
local energy landscape during coordinated updates without adding collective
rates to the Hamiltonian. The effective-Hamiltonian error is O(delta^2).

## Proposal and path correction

For one local block K, hold surrounding spins fixed and evaluate the guide on
the eight triangle configurations. Sample q(y|x)=K[y,x] psi(y)/Z(x), where
Z(x)=sum_z K[z,x] psi(z). The weight is Z(x)/psi(x). Multiplying these weights
corrects the sampled path exactly: guide ratios telescope and summing over
intermediate states recovers diag(psi) G_delta diag(1/psi).

Different paths can have the same endpoint; an endpoint-probability shortcut
would be incorrect. We accumulate weights in log space and resample once per
complete step, independently in four physical populations. Existing pure
forward-walking moments, ancestry diagnostics and checkpoint semantics apply.
Flip histograms count net endpoint changes, not intermediate events.

## Scalability and limits

The default implementation needs a guide callable and eight candidates per
local factor, with 3 L^2 factors per full step. This is polynomial but can be
expensive for a nonlocal neural guide. The optional N<=16 table is only an L2
benchmark accelerator. Enumerated operator actions are offline evaluation.

Three-spin locality does not guarantee valley transport. At low Gamma the
physical amplitudes remain small and a metastable guide can suppress access to
the ground sector. Assess spectra, initialization, projection time, population,
ancestry and runtime, not energy or weight ESS alone.

## Validation and experiment

Tests independently check the full Hamiltonian decomposition, cubic one-step
error, symmetry, positivity, weighted path moments against deterministic
operator action, cached/direct equivalence, L=4,6,8,10 layer covers, and exact
checkpoint recovery during a forward walk.

The matched campaign compares continuous-time, guided single-site and triangle
rules on four selected L2 disorders with original and corrected frozen guides.
Triangle steps 0.05, 0.1 and 0.2 probe the accuracy/cost tradeoff. Main runs use
four repeats, 1024 walkers per population, equilibration 32 and forward time 32.
Audits use 4096 walkers, projection times 64, or uniform initialization.

Offline references use the actual triangle product, both its dominant
eigenvector and exact finite-projection schedule. Uniform walker initialization
corresponds to the initial unguided vector 1/psi; guide-probability initialization
corresponds to psi. This separates splitting, finite-time and sampling errors.
These selected cases do not establish a finite-size disorder ensemble.

Raw moments, invalid length ratios, ancestry alarms, source hashes and input
hashes are retained. AutoDL uses an isolated snapshot; existing campaigns retain
their original sources. Work remains uncommitted.
