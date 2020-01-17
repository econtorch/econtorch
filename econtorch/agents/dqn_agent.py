r"""
Define a Q-Learning agent using a neural network to represent the Q-value.
Implements the DQN algorithm from
"Human level control through deep reinforcement learning"
  Mnih et al., 2015
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DqnAgent():

    def __init__(self, states=[], actions=[]):
        pass

