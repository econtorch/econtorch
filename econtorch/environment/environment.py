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
    def __init__(self, states):

        self.states = []
        self.states_values = []
        self.states_shape = torch.Size() 
        self.add_states(states)
    
    ########## Functions to manipulate the object ##########

    def add_state(self, s):
        if not(isinstance(s, DiscreteState)):
            s = DiscreteState(s)
        s.dim = len(self.states)
        self.states += [s]
        self.states_values += [s.values]
        self.states_shape += s.size
        #self.update_states_actions_shape()
        #self.update_meshgrids()

    def add_states(self, s_array):
        for s in s_array:
            self.add_state(s)
