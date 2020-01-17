r"""
The agents module provides different types of agents.
"""

from .discrete_agent import *
from .dqn_agent import *

__all__ = [
        'DiscreteAgent', 'DiscreteState', 'DiscreteAction'
        ]
