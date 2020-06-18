r"""
Define classes for the environment: States/Actions
"""

import torch
from torch.distributions.categorical import Categorical
import numpy as np


class DiscreteState():

    def __init__(self, states):
        self.values = torch.Tensor(states)
        self.size = self.values.size() 
        self.length = len(self.values)
        self.indices = torch.arange(0, self.length)
        self.agent = None

    def clone(self):
        values = self.values.clone()
        clone = DiscreteState(values)
        return clone

    def _get_next_states_deterministic(self):
        r"""
        Return the next state when an action is linked to the state
        """
        self.ns = self.meshgrid
        self.ns_transition = torch.ones(self.ns.shape) # Deterministic
        self.ns_indices = self.ns.clone() 
        self.ns_indices[:] = torch.arange(0,len(self.values)).type(torch.int32)


class DiscreteAction():

    def __init__(self, actions):
        if isinstance(actions, DiscreteState):
            self.values = actions.values
            self.state = actions
            actions.action = self
            # Assign the generic get_next_states function to the state
            actions.get_next_states = actions._get_next_states_deterministic
        else:
            self.values = torch.Tensor(actions)
        self.size = self.values.size() 
        self.length = len(self.values)
        self.indices = torch.arange(0, self.length)


# class require states / actions
class DiscreteEnvironment():

# Environment class contains all states for all agents
    def __init__(self, params):
            # Capital Grid - k
        self.k = DiscreteState(torch.linspace(params['k_min'], params['k_max'],
            params['nk']))
        # State of the World - w
        self.q = params['q']
        self.w = MarkovBinomial([params['w0'],params['w1']],
                self.q, self.q)
        # Fraction kept by the manager - x
        self.x = Uniform(params['min_x'], params['max_x'],
                params['N_x']) 
        # Productivity shock
        self.eps = Uniform(params['min_eps'], params['max_eps'],
                params['N_eps']) 

        # Market belief about the state of nature w
        self.gw = Belief(self.w , params['ngw'])
