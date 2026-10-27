# Gaussian-bond size and field screen

This bounded campaign measures spin-glass susceptibility and the finite-size
RMS Edwards–Anderson overlap for independent zero-mean Gaussian nearest-neighbor
bonds on periodic Kagome lattices. Each L×L lattice has N=3L² spins. The solver
uses the fixed flat guide and frozen source from the completed issue-9 checks.

- Sizes: L=4,6,8. Dimensionless fields Γ/σ=0.5,1,2,3,5; σ=1 and h=0.
- Disorder samples: 0 and 1 at every size/field; identical stored bonds across
  fields for a given size/sample. Four independent populations per setting.
- Baseline: 10,000 walkers, two replicas, 32 equilibration and 32 measurement
  blocks of length 0.2; eight forward cycles requesting time 0.8.
- At Γ/σ=1 and 3, sample 0 at every size has three separate comparisons:
  64+64 initial blocks, forward time 1.6, or 20,000 walkers. Other settings
  match baseline. These isolate three convergence effects at six points.
- Total: 120 baseline populations plus 72 comparison populations = 192.
  At most 16 worker processes, one numerical-library thread each.
- Solver RNG entropy is [2026102604, task_id]; physical disorder entropy is
  [1000,L,sample_id,1]. The manifest fixes task identities before execution.
- Retain all solver fields and actual bonds/couplings. Kish floor 8 and unique
  ancestor floor 2 apply. Shortened cycles remain archived and flagged.

Compute per-population moments first, then equal-weight population averages
within each sample, then equal-weight disorder averages. Report
qEA²=[<q²>], qEA=sqrt([<q²>]), χSG=N[<q²>], [<q⁴>],
and U=0.5(3-[<q⁴>]/[<q²>]²). Retain sample Binder ratios too.
Use independent population means for projector error; cycles are grouped.
Disorder scatter includes projector noise. The variance-component estimate
subtracts mean population-estimation variance and clips at zero. Bootstrap
whole disorder samples using identical resampling across fields at fixed L.
With only two samples its intervals are exploratory and cannot establish a
transition or adequate disorder sampling.

Energy comparisons use the measurement trace, excluding forward-step energy.
Report paired late-minus-early measurement drift per spin and independent
Welch intervals for variant-minus-baseline susceptibility, moments and energy.
An interval containing zero does not establish equivalence. No susceptibility
accuracy threshold or automatic production qualification is imposed.
The successful Γ/σ=5 L6 sample0 check does not qualify other scan points.

This first campaign assesses numerical usability and brackets further work.
Inspect convergence, ancestry and disorder scatter before choosing a refined
field grid and expanding toward 32 disorder samples. A finite-size positive
qEA alone does not establish order: the uncoupled baseline is qEA=1/sqrt(N)
and χSG=1. Transition inference needs converged size-dependent observables.

For J=σz and h=0, H(σ,Γ)=σH(1,Γ/σ) exactly. Plot measured dimensionless
curves against Γ/σ, then rescale Γ=σ(Γ/σ) for σ=0.05,0.1,0.2,0.5,1,2.
These rescaled curves share data and are correlated. If a finite critical
ratio exists, Γc=σ(Γ/σ)c, proportional to sqrt(Var(J)).

Run `python scan.py --smoke-only`, then detached `python scan.py`.
`progress.json` records completed populations; `exit.json` records completion
or failure. `python verify.py` independently checks final downloaded archives.
