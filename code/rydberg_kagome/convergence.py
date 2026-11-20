"""Explicit convergence labels; completion is not physical convergence."""
import numpy as np


def compare(a,b,n,energy_tol=1e-7,qea_tol=1e-3):
    de=abs(a['energy']-b['energy'])/n
    dq=abs(a['observables']['qea']-b['observables']['qea'])
    return dict(energy_per_spin=de,qea_difference=dq,
                passed=bool(de<energy_tol and dq<qea_tol))


def sampling_check(hist,obs):
    return bool(abs(hist['qmean']-obs['qea'])<5*hist['qmean_se']+1e-4 and
                abs(hist['q2']-obs['q2'])<5*hist['q2_se']+1e-4)


def cdf_distance(p,q):
    return float(np.max(abs(np.cumsum(p)-np.cumsum(q))))
