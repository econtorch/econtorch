# Define base classes for the econtorch package

import torch
from abc import *   # For abstract classes 

class _Agent(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def value(state, action):
        pass

    @abstractmethod
    def policy(state):
        pass

    @abstractmethod
    def reward(state, action):
        pass

