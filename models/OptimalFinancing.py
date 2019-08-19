r"""
Optimal Financing Models
Based on "Dynamic Models and Structural Estimation in Corporate Finance"
by Streabulaev and Whited (2012)
"""

import torch
import numpy as np
from econtorch.base import _Agent
from econtorch.discrete_ramdom_processes import AR1Log



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
        z_m = 3
        # Model parameters
        self.theta = 0.7
        self.delta = 0.15
        self.r = 0.04
        self.phi0 = 0.01
        self.rho = 0.7
        self.sigma = 0.15
        # State space
        k = np.linspace(k_min, k_max, nk)
        z_process = AR1Log(self.rho, self.sigma, nz, z_m)
        z = z_process.states
        # Action space
        I = np.linspace(-50, 50, 80)
        # Creation of the state and action spaces
        km, zm, Im = np.meshgrid(k, z, I, indexing='ij')
        self.states = torch.tensor([km, zm])
        self.actions = torch.tensor([Im])

    #def states(self):
    #    # Return the states space: shape [S]
    #    pass

    #def actions(self):
    #    # Return the actions space: shape [SxA]
    #    pass

    def reward(self):
        # Cash flow
        return self._cashFlow(self.states[0], self.actions[0], self.states[1])
        
    def value(self):
        pass

    def policy(self):
        pass

    def action_value(self):
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
        return self._prof(k,z)-I-self._adjCosts(I,k)


