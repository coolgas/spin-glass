# Fine overlap scan requested after the preliminary review

Fixed longitudinal field h=0.15. The L=4 target is Gamma=0.55, 0.56, ..., 0.70
(spacing 0.01), following the request to extend the upper bound to 0.70 to
investigate glassy behavior. The L=6 target is Gamma=0.55, 0.56, ..., 0.60.
Sizes remain L=4,6; larger sizes await user review.

The scan rationale is a possible finite-size shift toward larger Gamma at fixed
h: the region showing glassy behavior, if present, may move right as L increases.
This direction is a hypothesis to test, not an established phase boundary.
The L=6 fine grid above is a low-field supplement, not its full search window.
The original L=6 campaign already targets Gamma=0.60, 0.65, 0.70, 0.85 and 1.00;
retain that higher-field coverage to bracket any change, then resolve the
relevant interval with spacing 0.01. Choose the interval using matched complete
disorder ensembles, P(q), magnetization and connected correlations. A peak or
width change on a finite lattice is a crossover diagnostic, not by itself a
thermodynamic phase boundary. Keep common fields across sizes for comparisons.

Use the original eight disorders 60000–60007 at each size, both guide families,
both optimization seeds and energy-only seed selection. All bonds are quenched,
independent antiferromagnetic K_ij ~ Uniform(0.5,1.5) on periodic kagome nearest
neighbors; the Hamiltonian uses +sum K_ij ZiZj -h sum Zi -Gamma sum Xi.
All disorders retain equal weights, including narrow distributions and ancestry
alarms. Broadness is an observable to measure, never a sample-selection rule.

The new L=4 computation adds 0.55–0.59, 0.61–0.64 and 0.66–0.69. Existing
0.60, 0.65 and 0.70 training/projection records are reused by hash and task
identity; dedicated pure histograms are measured when absent. L=6 adds
0.55–0.59 and reuses 0.60. Each new guide uses
the original 1200-update continuation protocol, frozen random/up/down diagnostics,
256 chains and eight tempering levels. Main projection and the two-disorder
population/time/start audits retain the original pilot settings.

Report disorder-mean P(q) before and after projection with capped ±1 standard
error bars. Also record distribution variance, P(q<0), uniform magnetization,
and connected replicon spectra to distinguish overlap localization and field
polarization from uniform ferromagnetic order. A narrow positive P(q) alone is
insufficient to classify the phase. Finer Gamma spacing does not itself broaden
the distribution, and finite size/projection/mixing limitations remain.

The immediate compact notebook retains its explicitly dated 0.60/0.65 snapshot
until full eight-disorder histograms from the new range exist. It must explicitly
state that the requested scan is L=4: 0.55–0.70 and L=6: 0.55–0.60, both with
spacing 0.01. Update the
compact plots from the fine scan when both before/after ensembles are complete;
include exact and actual MC comparisons for L=2, also with eight disorders.
