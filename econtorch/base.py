# Define base classes for the econtorch package

import torch
from abc import *   # For abstract classes 

class _Agent(ABC):
    r"""
    Class _Agent provide the base class for all the dynamic problems
    We note:
    - pi(a|s) the current policy
    - v_pi(s) the value of being in state s given pi
    - q_pi(s,a) the value of taking action a in state s given pi
    """ 
    
    def __init__(self):
        pass

    @abstractmethod
    def policy(state):
        r"""
        Return the current policy given the state
        pi(a|s) = probability that At=a is St=s
        """
        pass

    @abstractmethod
    def value(state):
        r"""
        Return the expected value of a state under the current policy
        v_pi(s)
        """
        pass
    
    @abstractmethod
    def reward(state, action):
        pass

    def action_value(state, action):
        r"""
        Return the expected value of taking action a
        in state s under the current policy
        q_pi(s,a)
        """
        pass


