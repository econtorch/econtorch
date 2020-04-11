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

# Temporary for testing purposes (replicate the 'CartPole' example
# from pytorch tutorial)
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image


# Create the agent once algorithm works
class DqnAgent():

    def __init__(self, states=[], actions=[]):
        pass


# Pytorch tutorial for CartPole
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

env.reset()

# if gpu is to be used
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, inputs, outputs, layer_params=100):
        super(DQN, self).__init__()
        # Input layer
        self.input_layer = nn.Linear(inputs, layer_params)
        # Output layer
        self.output_layer = nn.Linear(layer_params, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return x


##### Training #####

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10



# Create the Firm class
class Firm(object):

    def __init__(self, params):
        # Global parameters
        self.d = params['d']
        self.theta = params['theta']
        self.r = params['r']
        self.beta = 1/(1+self.r)
        # Capital
        self.Kmin = params['k_min']         # Minimum capital stock
        self.Kmax = params['k_max']         # Maximum capital stock
        self.nk = params['nk']              # Nb of discrete capital stocks
        #self.Kinit = int(self.nk/2)         # Initial capital stock index
        # Productivity
        self.w0 = params['w0']      # Low productivity state
        self.w1 = params['w1']      # High productivity state
        self.q = params['q']        # Probability of staying in the same state
        #self.winit = 0
        # Investment
        ## Note: Need to shift the action space to be minimum 0
        ## to comply with the dqn agent
        #self.Imin_shifted = 0
        #self.Imax_shifted = self.Kmax - self.Kmin
        #self.Ishift = self.Imax_shifted / 2
        
        # Use next capital state as action

        # Define action_spec and observation_spec
        self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32,
                minimum=0, maximum=int(self.nk-1),
                name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
                shape=(2,), dtype=np.int32,
                minimum=[0,0], maximum=[int(self.nk-1), 1],
                name='observation')
        self._state = [np.random.randint(0, self.nk),
                np.random.randint(0, 2)]
        self._episode_ended = False
        self._age = 0

    def K(self,iK):
        iK = tf.cast(iK, dtype=tf.float32)
        return iK*(self.Kmax-self.Kmin)/(self.nk-1) + self.Kmin

    def w(self,iw):
        iw = tf.cast(iw, dtype=tf.float32)
        return iw*(self.w1-self.w0) + self.w0

    def I(self,iK,iI):
        iI = tf.cast(iI, dtype=tf.float32)
        return self.K(iI) - self.K(iK)*(1-self.d)

    def reward(self, action):
        # Cash flow
        K = self.K(self._state[0])
        w = self.w(self._state[1])
        I = self.I(self._state[0], action)
        return w*(K**self.theta) - (I**2)/2

    def next_state(self, action):
        self._state[0] = action
        if np.random.binomial(1, self.q)==0:
            self._state[1] = (self._state[1] + 1) % 2

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        #self._state = [self.Kinit, self.winit]
        self._state[0] = np.random.randint(0, self.nk)
        self._state[1] = np.random.randint(0, 2)
        self._episode_ended = False
        self._age = 0
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        # Compute the reward
        reward = self.reward(action)
        # Compute the next state
        self.next_state(action)
        self._age += 1
        # make sure episodes don't go forever
        if self._age >= 5:
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            # Return the transition
            return ts.transition(np.array(self._state, dtype=np.int32),
            reward=reward,
            discount=self.beta)

        #if self._episode_ended:
        #    reward = -1e10
        #    return ts.termination(np.array(self._state, dtype=np.int32), reward)
        #else:
        #    return ts.transition(
        #        np.array(self._state, dtype=np.int32),
        #        reward=reward,
        #        discount=self.beta)




# Get number of actions from gym action space
n_actions = env.action_space.n
dim_observations = env.observation_space.shape[0]

policy_net = DQN(dim_observations, n_actions).to(device)
target_net = DQN(dim_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = (EPS_END + (EPS_START - EPS_END) * 
        math.exp(-1. * steps_done / EPS_DECAY))
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

