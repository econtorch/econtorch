# Test File for the DiscreteAgent class

import torch
from econtorch.base import *
from econtorch.DiscreteAgent import DiscreteAgent
from econtorch.discrete_random_processes import AR1Log

# So far, uses Strebulaev White Optimal Financing Problem

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
theta = 0.7
delta = 0.15
r = 0.04
phi0 = 0.01
rho = 0.7
sigma = 0.15
discount_rate = 1/(1+r)

def prof(k, z):
    return z*(k**theta)

# Adjustment Costs
def adjCosts(I, k):
    return phi0*(I**2)/(2*k)

# Cash Flow function
def cashFlow(k, I, z):
    return prof(k,z)-I-adjCosts(I,k)

def reward(states, actions):
    # Cash flow
    k = states[0]
    z = states[1]
    k1 = actions[0]
    # Retrieve the Investment value from next period capital
    I = k1 - (1-delta)*k
    return cashFlow(k, I, z)


# State space
k = State(torch.linspace(k_min, k_max, nk))
z_process = AR1Log(1, rho, sigma, nz, z_m)
# Action space
k1 = Action(k)

agent = DiscreteAgent(states=[k,z_process], actions=[k1], discount_rate=discount_rate)
agent.reward = reward
agent.iterate_value_function(1)

#agent.plot_simulation(10, [50,1])
