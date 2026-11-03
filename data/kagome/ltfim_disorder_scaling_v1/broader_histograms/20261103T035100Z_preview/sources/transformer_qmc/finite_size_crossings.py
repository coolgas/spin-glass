"""Crossings and conditional drift extrapolation for qualified scaling data.

Bootstrap rows must share the same resample across all crossing pairs, preserving
the covariance from shared sizes. Exponent choices are sensitivity assumptions.
"""
import numpy as np


def crossing_brackets(fields, first, second):
    x=np.asarray(fields,dtype=float);a=np.asarray(first,dtype=float);b=np.asarray(second,dtype=float)
    if x.ndim!=1 or len(x)<2 or a.shape!=x.shape or b.shape!=x.shape:
        raise ValueError('Need two curves on the same field grid')
    if not np.isfinite([x,a,b]).all() or np.any(np.diff(x)<=0):
        raise ValueError('Need finite values and strictly increasing fields')
    delta=a-b;crossings=[]
    if np.any((delta[:-1]==0)&(delta[1:]==0)):
        raise ValueError('Coincident interval does not identify an isolated crossing')
    for i in range(len(x)):
        if delta[i]==0:
            crossings.append(dict(lower=float(x[i]),upper=float(x[i]),field=float(x[i])))
        elif i+1<len(x) and delta[i]*delta[i+1]<0:
            root=x[i]-delta[i]*(x[i+1]-x[i])/(delta[i+1]-delta[i])
            crossings.append(dict(lower=float(x[i]),upper=float(x[i+1]),field=float(root)))
    return crossings


def drift_sensitivity(size_pairs, crossing_samples, powers=(1.,2.,3.)):
    """Fit Gamma_cross=Gamma_c+A*(sqrt(L1*L2))**(-power).

    The symmetric effective length is a phenomenological drift convention.
    Compare fixed-ratio and compatible size families before interpreting it.
    At least three distinct effective lengths are needed, not three duplicated
    pairs. This is a conditional extrapolation, not a transition detector.
    """
    pairs=np.asarray(size_pairs,dtype=float);samples=np.asarray(crossing_samples,dtype=float)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)<3 or np.any(pairs<=0) or np.any(pairs[:,0]>=pairs[:,1]):
        raise ValueError('Need at least three positive ordered size pairs')
    lengths=np.sqrt(pairs.prod(axis=1))
    if len(np.unique(lengths))!=len(lengths):raise ValueError('Effective lengths must be distinct')
    if samples.ndim!=2 or samples.shape[1]!=len(pairs) or len(samples)<2 or not np.isfinite(samples).all():
        raise ValueError('Need finite jointly bootstrapped crossing samples')
    if not np.isfinite(pairs).all():raise ValueError('Invalid lengths')
    # Covariance is estimated from joint bootstrap replicates. No diagonal-only
    # replacement: overlapping crossing pairs share random inputs.
    covariance=np.cov(samples,rowvar=False)
    scale=max(float(np.trace(covariance)/len(pairs)),1e-20)
    inverse=np.linalg.pinv(covariance+np.eye(len(pairs))*scale*1e-8)
    results=[]
    for power in powers:
        if not np.isfinite(power) or power<=0:raise ValueError('Positive fixed drift powers required')
        matrix=np.column_stack([np.ones(len(pairs)),lengths**(-power)])
        fit=np.linalg.pinv(matrix.T@inverse@matrix)@matrix.T@inverse
        coefficients=samples@fit.T
        results.append(dict(power=float(power),critical_field_median=float(np.median(coefficients[:,0])),
            bootstrap_95_interval=np.quantile(coefficients[:,0],[.025,.975]).tolist(),
            drift_amplitude_median=float(np.median(coefficients[:,1])),
            residual_dof=len(pairs)-2,conditional_on_drift_model=True))
    return results
