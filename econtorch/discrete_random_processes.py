r"""
Provide Discretization methods for random processes
"""

import torch
from torch.distributions.normal import Normal
import numpy as np

from econtorch.base import State

class RandomProcess(State):
    def __init__(self):
        pass
    
    def get_next_states(self):
        self.ns = torch.empty(self.meshgrid.shape + self.values.shape)
        self.ns_transition = torch.empty(self.meshgrid.shape + self.values.shape)
        self.ns_indices = torch.empty(self.meshgrid.shape + self.values.shape)
        # Next states values
        self.ns[:] = self.values
        # Next states transition probabilities
        last_action_dim = self.meshgrid.dim()-1
        p = np.arange(0,self.ns.dim(),1)
        p[last_action_dim] = self.dim
        p[self.dim] = last_action_dim
        x_tmp = self.ns_transition.permute(list(p))
        x_tmp[:] = self.transitions
        self.ns_transition = x_tmp.permute(list(p))
        # Next states indices
        self.ns_indices[:] = torch.arange(0, len(self.values))


class Uniform(RandomProcess):
    r"""
    Uniform iid process
        - dim is the dimension number in the state space
        - a is the lower bound of the distribution
        - b is the upper bound of the distribution
        - N is the number of shocks
    """

    def __init__(self, dim, a, b, N):
        self.dim = dim
        self.a = a
        self.b = b
        self.N = N
        self.values = torch.linspace(a, b, N)
        self.size = self.values.size()
        self.transitions = torch.ones(N) / N

    def get_next_states(self):
        self.ns = torch.empty(self.meshgrid.shape + self.values.shape)
        self.ns[:] = self.values
        # Create the transition probability distribution (same shape)
        self.ns_transition = torch.ones(self.ns.shape) / self.N
        self.ns_indices = self.ns.clone()
        self.ns_indices[:] = torch.arange(0, len(self.values))


class MarkovBinomial(RandomProcess):
    r"""
    Markov binomial distribution where
        - dim is the dimension number in the state space
        - ss is a vector of length 2 representing the state space
            let us denote ss = [0,1]
        - alpha and beta are such that the transition matrix is

            P = ( alpha    1-alpha )
                ( 1-beta    beta  )

        or equivalently p00 = alpha and p11 = beta
    """

    def __init__(self, dim, ss, alpha, beta):
        self.dim = dim
        self.values = torch.Tensor(ss)
        self.size = self.values.size()
        if alpha>1 or beta>1:
            raise Exception('The transition probabilities should be lower than 1')
        self.transitions = torch.Tensor([[alpha, 1-alpha],[1-beta, beta]])

    #def get_next_states(self):
    #    ### Next shocks states
    #    ## States values
    #    # Add shape of the shock state and the states themselves
    #    x_t = torch.empty(self.meshgrid.shape + self.values.shape)
    #    x_t[:] = self.values
    #    # Create the transition probability distribution
    #    x_t_transition = x_t.clone() # Should have the same shape
    #    # We need to have z_tm1 and z_t as the two last dimensions
    #    # In order to correctly add the transition matrix
    #    # Permute the last action dimension with the z_tm1 dimension
    #    last_action_dim = self.meshgrid.dim()-1
    #    p = np.arange(0,x_t.dim(),1)
    #    p[last_action_dim] = self.dim
    #    p[self.dim] = last_action_dim
    #    x_tmp = x_t_transition.permute(list(p))
    #    x_tmp[:] = self.transitions
    #    x_t_transition = x_tmp.permute(list(p))
    #    self.ns = x_t
    #    self.ns_transition = x_t_transition
    #    self.ns_indices = self.ns.clone()
    #    self.ns_indices[:] = torch.arange(0, len(self.values))


class AR1Log(RandomProcess):
    r"""
    Use Tauchen (1896) method to construct a discrete transition matrix
    for an AR1 process in logs of the form
        ln(z') = p ln(z) + e
    where
        - dim is the dimension number in the state space
        - p is the autoregressive coefficient.
        - e is the error term that follows a truncated normal distribution
        of noise sigma.
        - N is the number of shocks
        - m is the maximum number of standard deviations to include in the matrix

    Returns a 2-dimentional transition matrix 
    """

    def __init__(self, dim, p, sigma, N, m=3):
        #self.dim = dim
        norm = Normal(0,1)
        minLnZ = -m*sigma/(np.sqrt(1-p**2))
        maxLnZ = m*sigma/(np.sqrt(1-p**2))
        # Array of possible shocks
        lnZ=torch.linspace(minLnZ,maxLnZ,N)
        Z=torch.exp(lnZ)
        self.values = Z
        self.size = self.values.size()
        # Creation of the transition matrix (Tauchen 1986 method)
        w=lnZ[1]-lnZ[0]
        zTrans=norm.cdf((lnZ[None,:]-p*lnZ[:,None]+w/2)/sigma) - norm.cdf((lnZ[None,:]-p*lnZ[:,None]-w/2)/sigma)
        # Compute zTrans[nzj,0] and last zTrans[nzj,Nk-1]
        zTrans[:,0]=norm.cdf((lnZ[0]-p*lnZ+w/2)/sigma)
        lastIndex=lnZ.shape[0]-1
        zTrans[:,lastIndex]=1-norm.cdf((lnZ[lastIndex]-p*lnZ-w/2)/sigma)
        self.transitions = zTrans
        # Compute the next state
        #self.next_states = Z

    #def get_next_states(self):
    #    r"""
    #    Return the possible states at time t given the states at time t-1
    #    and the dimension number of the state.
    #    """
    #    # z_tm1 should be of shape (states+actions)
    #    ### Next shocks states
    #    ## States values
    #    # Add shape of the shock state and the states themselves
    #    #z_t = torch.empty(z_tm1.shape + self.values.shape)
    #    z_t = torch.empty(self.meshgrid.shape + self.values.shape)
    #    z_t[:] = self.values
    #    # Create the transition probability distribution
    #    z_t_transition = z_t.clone() # Should have the same shape
    #    # We need to have z_tm1 and z_t as the two last dimensions
    #    # In order to correctly add the transition matrix
    #    # Permute the last action dimension with the z_tm1 dimension
    #    #last_action_dim = z_tm1.dim()-1
    #    #p = np.arange(0,z_t.dim(),1)
    #    last_action_dim = self.meshgrid.dim()-1
    #    p = np.arange(0,z_t.dim(),1)
    #    p[last_action_dim] = self.dim
    #    p[self.dim] = last_action_dim
    #    z_tmp = z_t_transition.permute(list(p))
    #    z_tmp[:] = self.transitions
    #    z_t_transition = z_tmp.permute(list(p))
    #    self.ns = z_t
    #    self.ns_transition = z_t_transition
    #    self.ns_indices = self.ns.clone()
    #    self.ns_indices[:] = torch.arange(0, len(self.values))
    #    #return [z_t, z_t_transition]

