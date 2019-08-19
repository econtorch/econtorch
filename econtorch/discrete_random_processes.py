r"""
Provide Discretization methods for random processes
"""

import numpy as np
from scipy.stats import norm

class AR1Log:
    r"""
    Use Tauchen (1896) method to construct a discrete transition matrix
    for an AR1 process in logs of the form
        ln(z') = p ln(z) + e
    where
        - p is the autoregressive coefficient.
        - e is the error term that follows a truncated normal distribution
        of noise sigma.
        - N is the number of shocks
        - m is the maximum number of standard deviations to include in the matrix

    Returns a 2-dimentional transition matrix 
    """

    def __init__(self, p, sigma, N, m=3):
        minLnZ = -m*sigma/(np.sqrt(1-p**2))
        maxLnZ = m*sigma/(np.sqrt(1-p**2))
        # Array of possible shocks
        lnZ=np.linspace(minLnZ,maxLnZ,N)
        Z=np.exp(lnZ)
        self.states = Z
        # Creation of the transition matrix (Tauchen 1986 method)
        w=lnZ[1]-lnZ[0]
        zTrans=norm.cdf((lnZ[None,:]-p*lnZ[:,None]+w/2)/sigma) - norm.cdf((lnZ[None,:]-p*lnZ[:,None]-w/2)/sigma)
        # Compute zTrans[nzj,0] and last zTrans[nzj,Nk-1]
        zTrans[:,0]=norm.cdf((lnZ[0]-p*lnZ+w/2)/sigma)
        lastIndex=lnZ.size-1
        zTrans[:,lastIndex]=1-norm.cdf((lnZ[lastIndex]-p*lnZ-w/2)/sigma)
        self.transitions = zTrans

