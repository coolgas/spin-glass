# Shared low-field optimization experiments

The new campaign compares larger-batch Adam, shared stochastic reconfiguration
(SR; two seeds), SR with continuous fields plus anchor replay, and a completion
input ablation. Each run has
one parameter state for L=2,4,6,8. All four sizes contribute before every update.
No exact states enter training. The initialization is the corrected main
campaign's frozen step003600 checkpoint, identified by SHA-256 in each manifest.
The main Adam campaign continues independently to step006000.

## SR definition

For each Hamiltonian c, form per-configuration derivatives O=d log(psi)/d theta.
Center O and the physical local energy separately within each context. The
metric is S=(1/4) sum_c Cov(O,O)/N_c; the force is the physical energy-per-spin
gradient g=(1/4) sum_c 2 Cov(O,E_local)/N_c. Covariances use B-1 denominators.
The per-site metric prevents a large system's extensive score from automatically
dominating the geometry. The solve is (S+lambda I)d=g with lambda=0.01.

`sr.py` uses the Woodbury sample-space solve, exact for the specified sampled
metric, and checks the solve residual. Parameter gradients use `torch.func`
and math attention; no second attention derivatives are required. The nominal
rate is 0.05, capped so each context's predicted RMS log-amplitude change is at
most 0.05. This sampled linear cap is not an exact KL guarantee.

The initial float32-only solve stopped on a residual check with batch1024.
The production solver uses a float32 Cholesky preconditioner and float64
residual correction and direction reconstruction, with a full float64 fallback.
The residual must be below1e-7. The frozen v1 failure and v2 float64 benchmark
are preserved. The refined v3 twelve-update benchmark finished in53.57s versus
171.68s for full float64, with residuals around1e-11 and14.17GB peak tensors.

The completion input approximates the future-neighbor energy response for each
candidate cell as minus sqrt(1+B^2) in units of gamma, using fields from already
generated cells and the candidate. It omits future/future interactions, which
the network must learn. Only future nearest neighbors are materialized. A tanh
gate starts at zero, exactly preserving the initial guide and retaining a
nonzero derivative for learning the feature strength.

## Controlled settings

- 600 updates, seed601 Adam and SR, seed602 SR, seed601 SR with replay,
  and seed601 SR with the completion input on the discrete schedule.
- Batch1024 per size for gamma<=1, otherwise batch512; score derivative chunk64.
- Adam initial learning rate0.00015, cosine decay to0.000015.
- Discrete schedule rotates (gamma,h)=(0.5,0),(0.5,0.3),(2,0.3).
- Replay alternates anchors with log-uniform gamma in[0.4,2.2], h uniform in[0,0.4].
- Train IDs0,1,100–355; validation10000–10001. IDs40000–40007 stay reserved
  until development comparisons select a continuation. Main test30000+ remains
  associated with the independent baseline.
- Every100 updates: deterministic L2 energy/fidelity/connected susceptibility
  and sampled larger-size observables. Every20: resumable state including both
  RNGs, optimizer if applicable, history, source hashes and initialization hash.
- Compare update counts and wall time: SR's larger derivative cost must be
  visible. A small local-energy variance alone cannot establish ground-state
  accuracy. Low-energy spectral weights diagnose excited-state contamination.

## Implementation verification and measured throughput

Ten tests cover exact symmetry/normalization/physical gradients, per-state
scores versus ordinary autograd, dense versus sample-space shared SR, exact
energy descent, bitwise identical resumed all-size SR updates, mixed-precision
refinement and causal completion inputs. They pass locally and on AutoDL.
The benchmark measurements above include startup while the independent main
campaign was active. These are short benchmark numbers,
not guaranteed sustained campaign throughput.

Remote frozen campaign root:
`/root/autodl-tmp/transformer-af-ltfim-20261027-remedies-v3/`.
Runs: `adam601`, `sr601`, `sr602`, `sr_replay601`, `sr_completion601`. Executed comparison notebook:
`notebook/shared_alibi_low_field_remedies.ipynb`.

Run the monitor with `SSHPASS` loaded securely and `PYTHONPATH=code`:

```
python -B -m transformer_qmc.monitor_remedies \
  --remote /root/autodl-tmp/transformer-af-ltfim-20261027-remedies-v3 \
  --local data/kagome/transformer_af_low_field_remedies_v3 \
  --notebook notebook/shared_alibi_low_field_remedies.ipynb
```

The monitor downloads checkpoints atomically written by training, logs and
validation artifacts and executes the report. It fails visibly if a training
process exits early. A local file lock prevents duplicate monitor writers.
On completion it verifies each downloaded checkpoint against a hash computed
remotely, audits translations/parity across all sizes, computes low-energy
spectra and exact L2 projection trajectories, and runs the correlation/field
scan comparison before executing the final notebook. The independent main
monitor performs the same final diagnostics. Active remote source trees stay
frozen, preserving the training manifests' source identities.

## Starting-guide diagnostics

The executed `initial_correlation_comparison.json` includes full 12-spin
connected-correlation matrices at the anchors and at gamma=0.4,0.7,1,1.5 with
h=0.15. At gamma=0.5,h=0.3, validation IDs10000/10001 have starting fidelities
0.305317/0.568152 and off-diagonal connected-C RMS errors0.19206/0.12923.
At gamma=2,h=0.3 the fidelities are0.995800/0.997864.

`initial_exact_projection.json` applies normalized exp[-tau(H-E0)] to the
enumerated frozen guide for tau=0,0.5,1,2,4,8,16. Energy decreases and fidelity
increases at every recorded step; h=0 parity remains even. At tau=16 the
gamma=0.5,h=0.3 fidelities reach0.999879737/0.999527033; at h=0 they reach
0.999998915/0.999999265. This establishes only a 12-spin projection-time
reference. Finite-walker population, branching cadence and forward-time bias
require separate stochastic convergence studies.

The executed remedy notebook includes the initial and final correlation scans,
exact-projection curves, the largest validation update shared by all runs,
and validation curves against updates and recorded elapsed time. All figures
are also saved as PNG/PDF beside the JSON artifacts. Susceptibility comparisons
use absolute errors without Markdown pipe characters in column labels.

## Overnight continuation handoff (2026-10-27)

The detached supervisor is staged in a separate frozen source tree:
`/root/autodl-tmp/transformer-af-ltfim-20261027-overnight-v1/`.
It waits for the five 600-update remedies and the independent main run. It
then compares their final checkpoints plus the shared initialization on eight
development disorders10000–10007, including the h=0.15 field scan. This makes
10002–10007 development data; test40000–40007 remains untouched.

The selection policy is written before the comparison: rank by mean low-field
infidelity +0.25 mean off-diagonal connected-C RMS error +0.02 mean absolute
connected susceptibility error, preferring candidates whose worst high-field
fidelity is at least0.98. These are development-selection weights, not accuracy
certification thresholds. If every candidate fails the high-field gate, the
record explicitly preserves that failure.

Four new 1800-update branches follow: two independent seeds of the selected
family, conservative SR(rate0.025) from the selected checkpoint, and a different
candidate family. Each keeps one parameter set across all four sizes and uses
only sampled physical VMC training. Optimizer/RNG states start fresh with
seeds701–704; initialization checkpoints and hashes are recorded. The schedule
starts at context offset600. Validation is every300 updates; resumable saves
remain every20. Existing training sources/checkpoints are not overwritten.

Local retrieval/reporting uses `monitor_remedies --wait-for-launches`, writing
`notebook/shared_alibi_overnight_continuations.ipynb` after launches exist.
The supervisor survives SSH disconnects; neither it nor the monitors powers
off the instance. All15 current core/symmetry/SR/completion tests passed locally
and on the staged AutoDL source before launch.

Separately, `study_projector.py` runs a resumable local12-spin convergence
study using exact caching of the frozen **network**, not ground-state labels.
It uses four replicas, walker counts512/2048/8192, forward times2/8/16,
branch interval0.1 and a0.05 check at the largest population/forward time,
eight independent populations and12 forward cycles each. Every cycle is
retained; ancestry ESS and largest-lineage fractions are reported alongside
susceptibility errors. The initial study is stored in `projector_initial/`
and included in the remedy notebook.

## Completed independent main baseline

The6000-update seed501 main run completed. Its final checkpoint was downloaded,
verified against the remote SHA-256
`5f9fa33b04102ddff2e43179440514021090ed3aaf228ce67dce53c2139e2544`,
and passed the all-size translation/parity audit. The main notebook executed
all nine code cells without errors, including final spectral, correlation and
exact-projection diagnostics.

Relative to the step3600 remedy initialization, validation10000/10001 at
gamma=0.5,h=0.3 improved in fidelity from0.305317/0.568152 to0.561591/0.666248.
Connected-C RMS errors changed from0.192060/0.129231 to0.207331/0.108857:
the first disorder's correlations worsened despite improved fidelity. Final
energy errors per spin are0.021605/0.022244. At h=0, fidelities improved to
0.965823/0.900134, with connected-C RMS errors0.053751/0.166173. High-field
fidelities remain0.996696/0.997225.

Exact12-spin projection of the final guide to tau16 gives biased-low-field
fidelities0.999960623/0.999870929 and connected-C RMS errors
0.00187026/0.00187165. This is a deterministic finite-size reference, not a
finite-walker or larger-system convergence result. The overnight supervisor
includes this final baseline in its development-only candidate comparison.

## Eight-disorder development selection

The recorded comparison across IDs10000–10007 selected the independent main
seed501 final checkpoint, with score0.291550. It remains the best candidate
under the frozen composite metric and passed the high-field fidelity gate
(minimum0.991539). The next candidates were Adam remedy seed601 (0.321078),
completion SR seed601 (0.337052), SR seed602 (0.351644), SR seed601
(0.361593), replay SR seed601 (0.363593), and the step3600 initialization
(0.372679). All seven passed the0.98 high-field gate. The selected score terms
were mean low-field infidelity0.238681, connected-C RMS error0.159029 and
absolute connected-susceptibility error0.655594. Selection is development-only;
these eight disorders do not certify disorder generalization, and the reserved
40000–40007 set remains uninspected.

## Completed overnight continuations

All four branches completed1800/1800 updates and the detached supervisor reports
`continuations_completed`. All training processes exited normally; the final
live check found the GPU idle. The instance remains powered on. Each branch
still uses one identical parameter set for L=2,4,6,8.

At gamma=0.5,h=0.3, routine validation IDs10000/10001 give:

| Checkpoint | Fidelity10000 | Fidelity10001 | Mean fidelity | Mean change from own initialization | Recorded elapsed hours |
|---|---:|---:|---:|---:|---:|
| Main seed501 baseline | 0.561591 | 0.666248 | 0.613920 | — | — |
| Adam701 final | 0.442270 | 0.592317 | 0.517293 | -0.096626 | 3.238 |
| Adam702 final | 0.493401 | 0.591484 | 0.542442 | -0.071477 | 3.243 |
| Conservative SR703 final | 0.333178 | 0.668534 | 0.500856 | -0.113063 | 3.776 |
| Completion SR704 final | 0.348012 | 0.760805 | 0.554409 | -0.026469 | 3.848 |

Completion704 starts from the completion-remedy final, whose mean fidelity is
0.580878; the other three start from the main baseline. Every overnight final
decreases mean biased-low-field fidelity relative to its own initialization.
Completion704 improves one disorder but worsens the other. Its mean remains
below the main baseline. These two disorders are routine development evidence;
the predefined eight-disorder score determines final checkpoint selection.

The biased-low-field correlation and energy errors are:

| Checkpoint | C RMS10000 | C RMS10001 | Energy error/spin10000 | Energy error/spin10001 |
|---|---:|---:|---:|---:|
| Main seed501 baseline | 0.207331 | 0.108857 | 0.021605 | 0.022244 |
| Adam701 final | 0.183206 | 0.117689 | 0.024479 | 0.030050 |
| Adam702 final | 0.177139 | 0.119831 | 0.020950 | 0.027354 |
| Conservative SR703 final | 0.177995 | 0.105576 | 0.029892 | 0.020407 |
| Completion SR704 final | 0.189573 | 0.093832 | 0.029711 | 0.015673 |

Some correlation errors improve while fidelity decreases, so the continuation
results cannot be reduced to a single observable. The first16 eigenstates
retain between0.933 and0.975 of the biased-low-field final spectral weight on
these disorders, despite ground-state fidelities as low as0.333. The residual
error therefore includes substantial low-energy excited-state contamination.

At h=0, final fidelities10000/10001 are0.963955/0.854426 (Adam701),
0.950640/0.844585 (Adam702),0.936079/0.858703 (SR703), and
0.943143/0.824598 (completion704). At gamma=2,h=0.3, all eight routine
fidelities exceed0.997. The maximum SR solve residuals were6.4864e-9 (703)
and9.9474e-10 (704), both below the1e-7 requirement. Recorded elapsed time
includes preceding in-run validations and concurrent resource contention;
summing the four branch times does not measure GPU device-hours.

The monitor completed final retrieval, remote/local SHA-256 verification,
all-size translation/parity audits, spectra, exact projection and correlation
field scans for all four finals. The continuation notebook executed all nine
code cells without errors. Final checkpoints, every300-update validation
snapshot, histories and diagnostics are preserved locally. A report quoting
defect was fixed during the campaign; its monitor restart did not interrupt
the detached training processes.

The initial selection in `selection.json` remains the recorded launch decision.
The completed-final eight-disorder comparison is stored separately as
`final_development_comparison.json` and its ranking as `final_selection.json`.
It uses the same frozen score and high-field gate, comparing the four overnight
finals with the seven predecessor candidates. Intermediate snapshots are not
ranked in that comparison. Reserved IDs40000–40007 remain uninspected.

The completed comparison retains the main seed501 final as the selected guide:

| Candidate | Frozen score (lower is better) | Mean low-field fidelity | Mean C RMS error | Mean absolute chi error | Minimum high-field fidelity |
|---|---:|---:|---:|---:|---:|
| Main seed501 | 0.291550 | 0.761319 | 0.159029 | 0.655594 | 0.991539 |
| Adam702 final | 0.304952 | 0.745588 | 0.156600 | 0.569506 | 0.991666 |
| Adam701 final | 0.309836 | 0.742647 | 0.159967 | 0.624571 | 0.991311 |
| Completion SR704 final | 0.352575 | 0.704453 | 0.167809 | 0.753805 | 0.992639 |
| Conservative SR703 final | 0.363295 | 0.695139 | 0.171789 | 0.774369 | 0.991109 |

These means include all measured gamma<=1 fields on eight development
disorders, including h=0.15 scans, so they differ from the two-disorder
gamma=0.5,h=0.3 means above. All eleven candidates pass the high-field gate.
Adam702 improves mean correlation and susceptibility errors over the main
baseline, but its fidelity loss outweighs those gains under the predefined
score. Extra training therefore did not deliver a better selected final guide.
The selected identity remains SHA-256
`5f9fa33b04102ddff2e43179440514021090ed3aaf228ce67dce53c2139e2544`.

The initial-guide stochastic projector study is complete:40/40 settings and
320/320 independent populations. Larger populations substantially improve
ancestry, while long-forward-time low-field susceptibility
can remain biased and population-sensitive: for example, at ID10000,h=0.3,
tau-forward16, estimates are0.128±0.007 at2048 walkers and0.130±0.005 at8192,
versus exact0.11846 (with ancestry ESS falling to6.7 at2048 and about35 at
8192). At512 walkers the same estimate is0.152±0.014 and the minimum ancestry
ESS is about1.
The h=0 susceptibility estimates are much closer to exact at8192 walkers.
All cycles and ancestry diagnostics are retained in `projector_initial/`; the
executed remedy notebook now reports the completed study and its convergence
figures. At8192 walkers and forward time16, ID10000,h=0.3 gives
chi=0.12980±0.00457 versus exact0.11846, while ID10001,h=0.3 gives
0.99905±0.02372 versus exact1.05027. At h=0, the corresponding values are
2.62369±0.02643 versus2.62544 and2.52041±0.05956 versus2.50904. The finer
branching interval0.05 does not shift all disorders in the same direction:
for ID10000,h=0.3 it gives0.12777±0.00601, while for ID10001,h=0.3 it gives
1.07127±0.02131. This is a 12-spin frozen-guide qualification of population,
branching-cadence and forward-time behavior, not a larger-size convergence
result; energy agreement alone does not establish susceptibility convergence.
