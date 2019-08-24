r"""
Optimal Financing Models
Based on "Dynamic Models and Structural Estimation in Corporate Finance"
by Streabulaev and Whited (2012)
"""

import torch
import numpy as np
from econtorch.base import _Agent
from econtorch.discrete_random_processes import AR1Log



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
        nI = 80
        I_min = -50
        I_max = 50
        # Model parameters
        self.theta = 0.7
        self.delta = 0.15
        self.r = 0.04
        self.phi0 = 0.01
        self.rho = 0.7
        self.sigma = 0.15
        # State space
        k = torch.linspace(k_min, k_max, nk)
        self.z_process = AR1Log(self.rho, self.sigma, nz, z_m)
        z = self.z_process.states
        self.states = [k,z]
        self.states_shape = [nk, nz]
        # Action space
        I = torch.linspace(I_min, I_max, nI)
        self.actions = [I]
        self.actions_shape = [nI]
        # Creation of the state and action spaces
        #km, zm, Im = np.meshgrid(k, z, I, indexing='ij')
        #self.states = torch.tensor([k, z])
        #self.actions = torch.tensor([I])
        # Value (V) and Action-Value (Q) functions (array here)
        self.V = torch.zeros([nk, nz])
        self.Q = torch.zeros([nk, nz, nI])


    #def states(self):
    #    # Return the states space: shape [S]
    #    pass

    #def actions(self):
    #    # Return the actions space: shape [SxA]
    #    pass

    def reward(self, states, actions):
        # Cash flow
        k, z, I = torch.meshgrid(states + actions)
        return self._cashFlow(k, I, z)
        
    def value(self, states):
        # Provide the value for given states


        # Expand to have the same shape and allow for one last dimension
        # for creating the combined indices
        ## Do it for states and transitions
        k = next_states[0].view(self.states_shape + self.actions_shape + [1,1] + [1])
        z = next_states[1].view(self.states_shape + self.actions_shape + [1,10] + [1])
        k = k.expand_as(z)

        # Transitions
        k_trans = next_states_transitions[0].view(self.states_shape + self.actions_shape + [1,1] + [1])
        z_trans = next_states_transitions[1].view(self.states_shape + self.actions_shape + [1,10] + [1])
        k_trans = k_trans.expand_as(z)
        # Multiply the transitions matrices to obtain the joint distribution
        p_trans = k_trans * z_trans

        sh = z.shape
        n_idx = 8000

        # Get the indices of the closest values in the V grid
        k = k.reshape(1,-1) 
        k_grid = self.states[0].view(-1,1)
        diff = torch.abs(k - k_grid)
        k_idx = torch.argmin(diff, 0).view(sh)

        z = z.reshape(1,-1)
        z_grid = self.states[1].view(-1,1)
        diff = torch.abs(z - z_grid)
        z_idx = torch.argmin(diff, 0).view(sh)

        # Create the combined indices using the last dimension
        Idx = torch.cat([k_idx, z_idx], 5)
        # Find the number of values needed to evaluate
        n_idx = 800000

        # Get the values
        values = self.V[Idx.chunk(chunks=n_idx, dim=5)]

        # Remove the values that are outside the grid
        # K state
        k = k.reshape(sh)
        k_ingrid = (k > 0).float() * (k < 100).float()
        k_ingrid[k_ingrid==0] = np.nan
        values = values*k_ingrid
        # Z state
        z = z.reshape(sh)

        # Compute the expectation using the transition matrix
        values_p = values * p_trans
        values_p = values_p.reshape(100,10,80,10)
        exp_values = values_p.sum(3)

        


        pass

    def next_states(self, states, actions):
        # Create the mesgrids
        k_tm1, z_tm1, I_tm1 = torch.meshgrid(states + actions)
        ## Next capital states
        k_t = (1-self.delta) * k_tm1 + I_tm1
        k_t_transition = k_t.clone()
        k_t_transition[:] = 1
        ## Next productivity shock states
        # States values
        z_t = torch.empty(z_tm1.shape + self.states[1].shape)
        z_t[:] = self.z_process.states
        # Probability distribution
        z_t_transition = z_t.clone()
        z_tmp = z_t_transition.permute(0,2,1,3)
        z_tmp[:] = self.z_process.transitions
        z_t_transition = z_tmp.permute(0,2,1,3)
        # Create the next states and distributions
        next_states = [k_t, z_t]
        next_states_transitions = [k_t_transition, z_t_transition]
        return next_states, next_states_transitions

    def continuation_value(self, states, actions):
        s_tm1 = states
        a_tm1 = actions
        [s_t, p_s_t] = self.next_states(states, actions)
        cont_v = torch.empty(self.states_shape + self.actions_shape)


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


