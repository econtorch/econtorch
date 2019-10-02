# Define base classes for the econtorch package

#from abc import *   # For abstract classes 

import torch
#import econtorch.discrete_random_processes.RandomProcess as RandomProcess


class State():

    def __init__(self, states):
        self.values = torch.Tensor(states)
        self.size = self.values.size() 

    def _get_next_states_deterministic(self):
        r"""
        Return the next state when an action is linked to the state
        """
        self.ns = self.meshgrid
        self.ns_transition = torch.ones(self.ns.shape) # Deterministic
        self.ns_indices = self.ns.clone() 
        self.ns_indices[:] = torch.arange(0,len(self.values))


class Action():

    def __init__(self, actions):
        if isinstance(actions, State):
            self.values = actions.values
            self.state = actions
            actions.action = self
            # Assign the generic get_next_states function to the state
            actions.get_next_states = actions._get_next_states_deterministic
        else:
            self.values = torch.Tensor(actions)
        self.size = self.values.size() 


class Agent():
    r"""
    Class reprensenting an Agent.

    Attributes:
        states          List of state objects.
        states_values   List of 1D torch.Tensor with the states values
        states_shape    Shape of the state space (torch.Size)
        actions         List of action objects.
        actions_shape   Shape of the state space (torch.Size)
    """
    
    def __init__(self, states=[], actions=[],
            discount_rate=None):
        # Create the states and actions space
        self.states = []
        self.states_values = []
        self.states_shape = torch.Size() 
        self.actions = []
        self.actions_values = []
        self.actions_shape = torch.Size() 
        self.add_states(states)
        self.add_actions(actions)

        # Discount Rate 
        self.set_discount_rate(discount_rate)

    def add_state(self, s):
        if not(isinstance(s, State)):
            s = state(s)
        s.dim = len(self.states)
        self.states += [s]
        self.states_values += [s.values]
        self.states_shape += s.size
        self.update_states_actions_shape()
        self.update_meshgrids()

    def add_states(self, s_array):
        for s in s_array:
            self.add_state(s)

    def add_action(self, a):
        if not(isinstance(a, Action)):
            a = action(a)
        a.dim = len(self.actions)
        self.actions += [a]
        self.actions_values += [a.values]
        self.actions_shape += a.size
        self.update_states_actions_shape()
        self.update_meshgrids()

    def add_actions(self, values_array):
        for v in values_array:
            self.add_action(v)

    def update_states_actions_shape(self):
        self.states_actions_shape = self.states_shape + self.actions_shape

    def update_meshgrids(self):
        meshgrids = torch.meshgrid(self.states_values + self.actions_values)
        self.states_meshgrids = []
        self.actions_meshgrids = []
        for i in range(0, len(self.states)):
            self.states[i].meshgrid = meshgrids[i]
            self.states_meshgrids += [meshgrids[i]]
        for i in range(len(self.states), len(self.states)+len(self.actions)):
            self.actions[i-len(self.states)].meshgrid = meshgrids[i]
            self.actions_meshgrids += [meshgrids[i]]

    def set_discount_rate(self, discount_rate):
        self.discount_rate = discount_rate


