# Gaussian bond size/field campaign

Completed successfully on AutoDL on 2026-10-26 at 04:54:50 UTC, after
5,565.21 seconds (92.75 minutes). All 192 independent populations were
downloaded, checked against remote SHA256 values and independently verified.
The run used PID 750389 with 16 worker processes. Remote directory:
`/root/autodl-tmp/gaussian-size-field-20261026/`.

The physical model, field grid, comparisons and uncertainty definitions are
recorded in `PROTOCOL.md`; `campaign.json` fixes every task and the source and
study hashes. Local and remote deterministic smoke checks passed at L=4 and 8.
The implementation notebook executed all 14 code cells in a fresh kernel.
An analytic fixture covering 192 archive slots also passed: 48 sample/setting
groups, 15 size/field curves, 18 convergence comparisons, uncertainty propagation
and both output plots. `git diff --check` passed.

The baseline has two disorder samples at each of L=4,6,8 and
Γ/σ=0.5,1,2,3,5, with four populations per sample. Three independent
convergence comparisons at Γ/σ=1,3 and sample0 test initial projection time,
forward time and walker count separately at each size. Full solver archives,
actual Gaussian bonds and deterministic seed metadata are retained.

`watch.py` monitors detached completion and downloads the campaign, including
its log. It verifies remote SHA256 values and invokes `verify.py`, which
independently recomputes moments, disorder means, Binder ratios, error estimates,
Welch contrasts and achieved-time/lineage diagnostics. `watch.log` records the
local watcher output. On successful completion expect `exit.json` with code 0,
`final.json`, `remote_sha256.json`, `local_verification.json`,
`size_field_screen.png` and `variance_rescaling.png`. On failure inspect
`failure.txt` or `watch_failure.txt`.

No transition has been located by this campaign. Two disorder samples are
an exploratory screen; production convergence and disorder precision need
assessment from the completed results. Existing single-sample production
exports remain available in the preceding overlap campaign.

## First downloaded batch

The first 28 populations were downloaded and their SHA256 digests checked
against population metadata. Susceptibility identities and the ancestry floor
were checked locally; see `partial_verification.json`. All 16 remote workers
were active, with no failure recorded at this check.

At L4, Γ/σ=0.5, 39 of 64 baseline cycles were truncated; at Γ/σ=1,
40 of 64 were truncated. All 32 cycles requesting forward time 1.6 at
L4/Γ=1 stopped early. The longer-projection and 20,000-walker comparisons
at this point also have truncated cycles. This is already evidence that the
high-field settings do not attain the requested forward times in this part
of the scan. These results remain exploratory, with no production
qualification or transition claim. Complete the bounded screen to assess
field and size dependence before choosing the next convergence study.

## Completed screen

The final independent verifier checked 192 populations, six distinct physical
disorder records (two at each size), statistics, metadata and lineage diagnostics.
There were 953 truncated cycles out of 1,536, including comparison settings.

Baseline truncated cycles, out of 64 at each size/field:

| Γ/σ | L4 | L6 | L8 |
|---:|---:|---:|---:|
| 0.5 | 39 | 64 | 64 |
| 1 | 40 | 64 | 64 |
| 2 | 3 | 56 | 64 |
| 3 | 0 | 24 | 64 |
| 5 | 0 | 0 | 10 |

At L4/sample0/Γ=1, doubling initial equilibration and measurement from
32+32 to 64+64 blocks changes χSG by −2.06490, with independent-population
Welch 95% interval [−3.25082, −0.87898]. This resolves dependence on the
initial projection schedule. The comparison still has 17/32 truncated cycles.
At L6/Γ=1 and L8/Γ=1, all three comparison settings have 32/32 truncated
cycles. At L8/Γ=3, the 20,000-walker setting still has 27/32 truncated cycles.
Thus the tested extensions do not resolve ancestry loss at these points.

L4/Γ=3 completes all baseline and comparison cycles. Its three susceptibility
contrasts include zero but have broad intervals; this is not an equivalence
test. L6/Γ=5 completes all baseline cycles. At L8/Γ=5, 10/64 baseline cycles
stop early, and sample0 has a resolved late-minus-early measurement energy
drift of −0.00041824 per spin, 95% interval [−0.00065088, −0.00018561].

`final.json` retains exploratory overlap, susceptibility and Binder curves,
all sample moments and systematic comparisons. `size_field_screen.png` and
`variance_rescaling.png` show those curves with their unqualified status.
The size dependence cannot be used to infer a transition while these
projection and ancestry effects remain unresolved. Further disorder averaging
alone will not remove this systematic problem.
