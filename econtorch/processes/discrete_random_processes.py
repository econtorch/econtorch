r"""
Provide Discretization methods for random processes
"""

import torch
from torch.distributions.normal import Normal
import numpy as np

from econtorch.agents.discrete_agent import DiscreteState

class RandomProcess(DiscreteState):
    def __init__(self, values, transitions):
        super(RandomProcess, self).__init__(values)
        self.transitions = transitions

    def clone(self):
        return RandomProcess(self.values.clone(), self.transitions.clone())
    
    def get_next_states(self):
        r"""
        Compute the next states along with the transition probas for the
        meshgrid of the agent.
        This function assumes that self.transitions provide the transition
        matrix of the randomProcess.
        It does not support transitions that depend on the state. For this
        see the Belief class.
        """
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

    def __init__(self, a, b, N):
        self.a = a
        self.b = b
        self.N = N
        values = torch.linspace(a, b, N)
        transitions = torch.ones(N) / N
        super(Uniform, self).__init__(values, transitions)

    def clone(self):
        return Uniform(self.a, self.b, self.N)

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

    def __init__(self, ss, alpha, beta):
        values = torch.Tensor(ss)
        if alpha>1 or beta>1:
            raise Exception('The transition probabilities should be lower than 1')
        self.alpha = alpha
        self.beta = beta
        transitions = torch.Tensor([[alpha, 1-alpha],[1-beta, beta]])
        super(MarkovBinomial, self).__init__(values, transitions)

    def clone(self):
        return MarkovBinomial(self.values.clone(), self.alpha, self.beta)


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

    def __init__(self, p, sigma, N, m=3):
        self.p = p
        self.sigma = sigma
        self.N = N
        self.m = m
        norm = Normal(0,1)
        minLnZ = -m*sigma/(np.sqrt(1-p**2))
        maxLnZ = m*sigma/(np.sqrt(1-p**2))
        # Array of possible shocks
        lnZ=torch.linspace(minLnZ,maxLnZ,N)
        Z=torch.exp(lnZ)
        # Creation of the transition matrix (Tauchen 1986 method)
        w=lnZ[1]-lnZ[0]
        zTrans=norm.cdf((lnZ[None,:]-p*lnZ[:,None]+w/2)/sigma) - norm.cdf((lnZ[None,:]-p*lnZ[:,None]-w/2)/sigma)
        # Compute zTrans[nzj,0] and last zTrans[nzj,Nk-1]
        zTrans[:,0]=norm.cdf((lnZ[0]-p*lnZ+w/2)/sigma)
        lastIndex=lnZ.shape[0]-1
        zTrans[:,lastIndex]=1-norm.cdf((lnZ[lastIndex]-p*lnZ-w/2)/sigma)
        super(AR1Log, self).__init__(Z, zTrans)

    def clone(self):
        return AR1Log(self.p, self.sigma, self.N, self.m)


class Belief(DiscreteState):
    r"""
    Class to create and manipulat beliefs.
    A belief state is a set of priors regarding another state.
    Note that we cannot just model the belief as only one distribution
    of the state the belief is about because there would therefore only be
    one possible belief. We need to define
    """

    def __init__(self, state, N):
        r"""
        Initialize the belief state
        state: state linked to the belief state
        N: Number of possible values of probabilities
        """
        self.state = state
        self.N = N
        # Create all the possible priors
        n = torch.linspace(0,1,N)
        m = [n for x in range(0,state.length)]
        mg = torch.meshgrid(m) 
        mg_flat = [mg[i].reshape(-1).tolist() for i in range(0,state.length)]
        t = torch.tensor(mg_flat)
        keep = t.sum(dim=0)==1
        t_keep = [t[i][keep].tolist() for i in range(0,state.length)]
        priors = torch.tensor(t_keep)
        self.priors = priors.permute(1,0)
        # Create the State
        values = torch.arange(0, len(self.priors[:,0]), dtype=torch.float32)
        super(Belief, self).__init__(values)

        # TEMPORARY PROVIDE A DUMMY TRANSITION MATRIX
        self.transitions = torch.ones(self.length) / self.length

    def clone(self):
        return Belief(self.state.clone(), self.N)

    def compute_posterior(self):
        r"""
        Update the prior belief using the given likelihood.
        Note: The posterior is NOT next period prior. It is only the update
        of the prior given new information as given by the likelihood.
        """
        pass

    def get_next_states(self):
        # DUMMY
        RandomProcess.get_next_states(self)
        r"""
        Compute the next priors along with the transitions probas for the
        meshgrid of the agent.
        As of now, the transition between priors (from prior to posterior) is
        non-stochastic.
        """
        pass

    def set_likelihood(self, likelihood):
        r"""
        Set the likelihood function used to update the prior beliefs
        
        The likelihood tensor should be of shape:
            (states shape of agent + length of state linked to the beliefs)
        """
        correct_shape = torch.Size(self.meshgrid.shape + self.state.values.shape)
        if likelihood.shape == correct_shape:
            self.likelihood = likelihood
        else:
            raise Exception("The shape of the likelihood matrix is incorrect.") 
        # Compute the posterior
        self.compute_posterior()




