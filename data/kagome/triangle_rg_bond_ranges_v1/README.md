# All-bond endpoint screen

Primary objective: roughly locate Gamma,h where physical P(q) changes while
triangle/superblock RG remains stable, as a prerequisite for validated scaling
of q_EA and physical connected correlation length.

Every archived base bond is multiplied by s=1,.5,.2,.1, giving K distributions
U(.5,1.5), U(.25,.75), U(.10,.30), U(.05,.15). There is no intertriangle-only
weighting (lambda=1). Within each size, all contexts share the same base draws,
disorder60000. The seed includes size, so L2 and L4 bonds differ.

One L4/N48 4-by-4 grid per bond range: rows Gamma=.05,.10,.25,.50; columns
h=0,.025,.075,.15. Schedules (m,chi)=(4,8),(6,8),(6,16), origin(0,0),
orientation0. Total: 192 L4 retained models and 64 exact L2 contexts with
m=4,6,8 checks. Two solver starts and 8192 independent physical replica
pairs per L4 model. Zero-field calculations use even global spin-flip parity;
an unrestricted pilot exposing near-degenerate state ambiguity is archived
separately. Cutoff degeneracies, start disagreement and sampling alarms are
retained and exclude provisional candidates.

The energy tolerance is .001/spin in units of the mean bond scale s, so a
smaller energy unit cannot trivially improve the screen. TV(m),TV(chi)<.05,
chi q2 change<.01, matching exact-L2 m6 errors, and no alarms are also required.
No matched full-spin L4 reference is supplied. Passing means candidate only,
not a certification of physical accuracy or a phase transition.

Scientific computations and plots run on AutoDL, one CPU worker, BLAS/OMP/MKL
threads=1, at `/root/autodl-tmp/triangle-rg-bond-ranges-20261104-v1/`.
`source_snapshot.tar.gz` preserves launch sources. Raw states from both starts,
couplings, seeds, isometries, spectra, physical replicas, correlations, failures
and source hashes accompany results. Existing campaigns remain independent.
