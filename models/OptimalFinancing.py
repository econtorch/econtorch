r"""
Optimal Financing Models
Based on "Dynamic Models and Structural Estimation in Corporate Finance"
by Streabulaev and Whited (2012)
"""

import torch
import numpy as np
from econtorch.base import _Agent



class Agent(_Agent):
    r"""
    The state vector is:
        [k, z]
        
    The action is a scalar: I

    The parameters vector is:
        [theta      Concavity of the production function
         delta      Capital depreciation rate
         r          Interest rate
         rho        Serial correlation of the AR1 shock process
         sigma]     Noise of the AR1 shock process

    """
    

    # Methods that need implementing for the class _Agent
    def __init__(self):
        # Global parameters
        nk = 100        # Grid size for capital
        k_min = 0
        k_max = 100
        nz = 10         # Grid size for the productivity shock
        z_min = 0
        z_max = 10
        # Model parameters
        self.theta = 0.7
        self.delta = 0.15
        self.r = 0.04
        self.phi0 = 0.01
        self.rho = 0.7
        self.sigma = 0.15
        # State space
        k = np.linspace(k_min, k_max, nk)
        z = np.linspace(z_min, z_max, nz)
        km, zm = np.meshgrid(k, z, indexing='ij')
        self.states = torch.tensor([km, zm])


    def reward(self, states, actions):
        # Cash flow
        return _cashFlow(states[0], actions[0], states[1])
        
    def value(self, states, actions):
        pass

    def policy(self, states):
        pass

    def action_value(state, action):
        pass


    # Methods specific to this problem
    # Profit function
    def _prof(self, k, z):
        return z*(k**self.theta)
    
    # Adjustment Costs
    def _adjCosts(self, I, k):
        return self.phi0*(I**2)/(2*k)
    
    # Cash Flow function
    def _cashFlow(self, k, I, z):
        return _prof(k,z)-I-_adjCosts(I,k)


