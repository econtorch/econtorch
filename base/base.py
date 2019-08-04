# Define base classes for the econtorch package

from abc import ABC      # For abstract methods

class Agent(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def value(state, control):
        pass

    @abstractmethod
    def policy(state):
        pass

