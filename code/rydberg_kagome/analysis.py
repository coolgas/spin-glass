"""Paired disorder resampling; retain whole drive curves."""
import numpy as np


def paired_bootstrap(a,b,draws=2000,seed=0):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.shape[0]<2: raise ValueError("Matched disorder arrays required")
    rng=np.random.default_rng(seed); difference=a-b
    indices=rng.integers(0,len(a),size=(draws,len(a)))
    samples=difference[indices].mean(axis=1)
    return dict(mean=difference.mean(axis=0),interval95=np.quantile(samples,[.025,.975],axis=0))
