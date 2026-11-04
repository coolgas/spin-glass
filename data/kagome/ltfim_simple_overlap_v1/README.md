# Jastrow/RBM overlap distributions and field scan

All computation, validation and notebook execution in this campaign run on
AutoDL. Local files are edited sources and retrieved artifacts. Changes are
uncommitted. Remote root: `/root/autodl-tmp/simple-guides-20261103-v1/`.

## Definition and completed results

The convention follows `notebook/exact_guide_overlap_distributions.ipynb`:
q = sum_i(s_i t_i)/N for independent copies of the same disorder. Curves are
probability masses at the 13 allowed L2 overlaps, not probability densities.
The XOR convolution was checked against explicit two-copy summation on small
systems, uniform-binomial and point-mass limits, and independent first/second
moment identities. Raw input hashes are checked before notebook execution.

`distributions.json` contains all 36 original full-basis fits, with the
minimum-energy seed marked. `sampled_distributions.json` contains all 32 L2
fits from the completed original 48-task sampled-VMC campaign; continuation
used more updates in that original campaign. Every displayed guide curve is
unprojected. Enumeration of frozen probabilities is restricted to L2.

`scan/` contains 72 field/disorder contexts, 432 fits: eight fields, nine
disorders, Jastrow and alpha=2 RBM, three seeds each. Each fit uses 800 Adam
updates plus 200 L-BFGS iterations with full-basis energy-only optimization.
Minimum-energy seed selection precedes exact-state evaluation. The eight
disorders 50000–50007 form the matched ensemble; difficult disorder 50043 is
kept separate. Individual distributions are averaged only after computing
same-disorder P(q). Disorder standard errors are descriptive, with D=8.

The difficult-disorder scan gives the following selected RBM results:

| Gamma | Exact negative-overlap weight | RBM P(q) TV | RBM connected-spectrum error |
|---:|---:|---:|---:|
| 0.50 | 0.08314 | 0.07034 | 23.81% |
| 0.55 | 0.13466 | 0.03057 | 15.22% |
| 0.60 | 0.19946 | 0.00600 | 3.19% |
| 0.65 | 0.26650 | 0.00218 | 0.94% |
| 0.70 | 0.32358 | 0.00161 | 0.61% |
| 0.75 | 0.36471 | 0.00091 | 0.32% |
| 0.85 | 0.40556 | 0.00045 | 0.17% |
| 1.00 | 0.41723 | 0.00006 | 0.02% |

Gamma=0.60–0.70 is a candidate RBM window: the negative-overlap shoulder remains
visible and the frozen guide improves strongly. All three RBM seeds on 50043
have TV below 0.01 from Gamma=0.60 onward. In the eight-disorder ensemble, all
24 RBM seeds pass descriptive TV<=0.05 from Gamma=0.60 onward. Jastrow remains
substantially less reliable in this range; on 50043 its TV reaches 0.243 at
Gamma=0.65. Neither broad P(q) nor training failure establishes a thermodynamic
phase. Negative-overlap weight alone is not an order parameter: it can also
increase as the distribution moves toward a symmetric paramagnet.

## Follow-up calculations

`sampled/` is a new 64-task matched-update VMC campaign on disorder 50043:
Gamma=0.50,0.60,0.70,0.75; L=2,4; two seeds; both ansatze; direct and decreasing
field schedules. Every fit uses 1200 updates, 256 chains, eight temperatures,
two sweeps per update, and separate frozen random/up/down diagnostics. Thus
update/sample budgets match, though wall-clock costs can differ. Exact L2
references are used only for post-fit evaluation. L4 has no exact reference.

`projector/` is a 96-run CT GFMC sensitivity campaign across all eight fields
for disorder 50043 and both selected guides. It compares 1024 and 4096 walkers
and projection times 32 and 64, with two repeats and 16 measurement cycles.
It measures connected spectra; it does not measure projected P(q). Two runs
per setting are sensitivity diagnostics, not a precision certificate.

Consult each campaign's status.json for completion; launch records retain the
commands and PIDs. Full model checkpoints remain remotely available. Raw
failures, invalid lengths and ancestry alarms must be retained. These follow-ups
are required before calling the candidate window numerically reliable, and
none of the L2 results qualifies a ground-state protocol for L4–10.

Notebook: `notebook/jastrow_rbm_overlap_distributions.ipynb`.
Standalone PNG/PDF figures: `figures/simple_overlap/`.
