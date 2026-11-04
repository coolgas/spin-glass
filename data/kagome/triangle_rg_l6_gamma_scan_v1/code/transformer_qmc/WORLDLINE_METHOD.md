# Worldline QMC development

This is a candidate replacement for enumeration-based guide correction. It samples the TFIM directly with no wavefunction guide and no enumeration of spin configurations. It is not yet a validated production solver.

For H=−ΣJij ZiZj−hΣZi−ΓΣXi, the discrete-time partition function has positive weights

    W(s) ∝ exp[Δτ Σt(Σij Jij sit sjt + h Σi sit) + Kτ Σit sit si,t+1],
    Kτ = −log(tanh(Δτ Γ))/2,    Δτ=β/M.

Frustrated diagonal bonds do not create a sign problem. They can still create very long mixing times. Finite β approximates the ground state only after temperature convergence; finite Δτ has a Trotter error that must be extrapolated or bounded.

Each site update constructs ferromagnetic Fortuin–Kasteleyn bonds along its periodic time direction, with probability 1−exp(−2Kτ) between equal spins. Each resulting component receives a new sign with heat-bath probability (1+tanh(Σcomponent Δτ[h+Σj Jij sjt]))/2. Neighboring sites are fixed during this update. This is an exact conditional update for the discrete-time target, even for frustrated spatial couplings.

Replica exchange uses a geometric β ladder at fixed M. If F=Σt(ΣJss+hΣs) and T=Σit sit si,t+1, the swap log ratio is (Δτa−Δτb)(Fb−Fa)+(Kτa−Kτb)(Tb−Ta). Configuration-independent transfer-matrix prefactors cancel. Endpoint visits and hot–cold–hot round trips are saved, but they alone do not certify equilibration.

Four independent exchange ladders provide disjoint replica moment estimates for the connected-overlap spectrum. Moments are measured from a randomly selected equal-time slice and stored in contiguous time blocks. Neither slices nor sweeps are treated as independent uncertainty samples. Block bootstrap errors remain provisional until block-length and run-length checks pass.

Storage for worldlines is O(RNM); temporal updates and swap statistics cost O(RNM) per sweep at bounded spatial degree. Here R is the number of temperatures, N the number of spins, and M the number of time slices. Pair-moment measurement currently costs O(N²); four requested Fourier modes can later be measured directly if that becomes limiting. Increasing R, β, or run length may be necessary with size. Polynomial per-sweep cost is not a guarantee of polynomial equilibration time.

## Initial experiment

- L=2 contexts: (Γ,disorder)=(0.25,50025),(0.25,50059),(0.5,50043),(0.75,50004).
- Four independent ladders, initialized random/up/down/random.
- (β,Δτ)=(16,0.125),(32,0.125),(32,0.0625); 24 temperatures.
- 4,000 equilibration sweeps; 16,000 measurement sweeps; measure every fourth sweep; 32 saved blocks.
- L=4,6,10 timing pilots at Γ=0.75: 100 equilibration and400 measurement sweeps. These are cost measurements only.
- Exact references are used only by the offline L=2 assessor, never by the simulation.

The first batch diagnoses whether this route is viable. Before a larger-size physical scan, require exact-reference agreement on difficult L=2 cases, stable temperature/time-step limits, longer runs, multiple independent starts, and uncertainty estimates stable against block length. Failure should trigger algorithmic work on mixing, such as improved tempering or spatial/nonlocal updates, rather than a claim of convergence from stable means alone.
