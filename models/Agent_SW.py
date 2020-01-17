r"""
Optimal Financing Models
Based on "Dynamic Models and Structural Estimation in Corporate Finance"
by Streabulaev and Whited (2012)
"""

import torch
import numpy as np
from econtorch.base import *
from econtorch.DiscreteAgent import DiscreteAgent
from econtorch.discrete_random_processes import AR1Log

import seaborn as sns
import pandas as pd


class Firm(DiscreteAgent):
    r"""
    The state vector is:
        [k, z]
        
    The action is a scalar: I

    The parameters vector is:
        [theta      Concavity of the production function
         delta      Capital depreciation rate
         r          Interest rate
         rho        Serial correlation of the AR1 shock process
         sigma]     Noise of the AR1 shock process

    """
    

    # Methods that need implementing for the class _Agent
    def __init__(self):
        # Global parameters
        nk = 100        # Grid size for capital
        k_min = 0
        k_max = 1000
        nz = 10         # Grid size for the productivity shock
        z_m = 3
        # Model parameters
        self.theta = 0.7
        self.delta = 0.15
        self.r = 0.04
        self.phi0 = 0.01
        self.rho = 0.7
        self.sigma = 0.15
        # State space
        k = State(torch.linspace(k_min, k_max, nk))
        z_process = AR1Log(1, self.rho, self.sigma, nz, z_m)
        # Action space (choose next period k instead of investment I)
        k1 = Action(k)
        # Discount rate
        discount_rate = 1/(1+self.r)
        super(Agent_SW, self).__init__(states=[k,z_process], actions=[k1],
                discount_rate=discount_rate)


    def reward(self, states, actions):
        # Cash flow
        k = states[0]
        z = states[1]
        k1 = actions[0]
        # Retrieve the Investment value from next period capital
        I = k1 - (1-self.delta)*k
        return self._cashFlow(k, I, z)


    def plot_policy(self):
        # Find the steady state capital stock (without adjustment costs)
        kss=((self.r+self.delta)/self.theta)**(1/(self.theta-1))
        # Find the index of the capital steady state
        kss_idx = torch.argmin(torch.abs(self.states[0].values - kss))
        # Plot the policy function at the steady state
        # as a function of the shock state z
        pi_ss = self.pi[kss_idx]
        z = torch.log(self.states[1].values)
        data = pd.DataFrame()
        data['z'] = z
        data['I'] = pi_ss / kss
        sns.lineplot('z', 'I', data=data)

    def plot_simulation(self, N, initial_state):
        # Simulate
        (sim_states, sim_actions) = self.simulate(N, initial_state)

        data = pd.DataFrame()
        data['K'] = sim_states[0]
        data['z'] = sim_states[1]
        data['K1'] = sim_actions[0]
        data['I'] = data['K1'] - data['K'] * (1-self.delta)
        data['time'] = data.index
        sns.lineplot(x='time', y='value', hue='variable', style='variable',
                data=data[['time','K','z','I']].melt('time'))


    # Profit function
    def _prof(self, k, z):
        return z*(k**self.theta)
    
    # Adjustment Costs
    def _adjCosts(self, I, k):
        return self.phi0*(I**2)/(2*k)
    
    # Cash Flow function
    def _cashFlow(self, k, I, z):
        return self._prof(k,z)-I-self._adjCosts(I,k)


