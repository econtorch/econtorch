
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import numpy as np

from econtorch.base import *
from econtorch.DiscreteAgent import DiscreteAgent
from econtorch.discrete_random_processes import Uniform
from econtorch.discrete_random_processes import MarkovBinomial

import seaborn as sns
import pandas as pd


class Manager(DiscreteAgent):
    r"""
    Implements the Manager for the posterior beliefs problem.
    The states are:
        - K: Capital
        - w: State of the world (binary: high or low)
        - eps: Productivity shock
        - rho: Market belief of a high state of the world next period
    The actions are:
        - I: Investment
    Parameters include:
        - x: Fraction of the firm kept by the manager every period
             (1-x) is sold at an endogeneous price
        - d: Capital depreciation rate
        - beta: discount rate

    """

    def __init__(self, params=None):
        
        if params==None:
            # Model's parameters
            params = {}
            params['x'] = 1 # Simpler problem with x=1 so far
            params['d'] = .15
            params['theta'] = 0.7
            params['r'] = 0.04
            # Capital Grid - k
            params['k_min'] = 10
            params['k_max'] = 50
            params['nk'] = 200        
            # State of the World - w
            ## Proba[w_t-1 = w_t] = q
            params['q'] = .9
            params['w0'] = 1
            params['w1'] = 5
            # Productivity shock - eps
            ## Uniformly distributed between min_eps and max_eps
            params['min_eps'] = -10
            params['max_eps'] = 10
            params['N_eps'] = 5 # Number of shocks used in the discretization
            # Investors' Beliefs - rho
            params['nrho'] = 10 # We allow 10 possible different beliefs between 0 and 1

        # Global parameters
        self.x = params['x']
        self.d = params['d']
        self.theta = params['theta']
        self.r = params['r']
        self.beta = 1/(1+self.r)

        # Capital Grid - k
        k = State(torch.linspace(params['k_min'], params['k_max'],
            params['nk']))
        # State of the World - w
        self.q = params['q']
        w_process = MarkovBinomial(1, [params['w0'],params['w1']],
                self.q, self.q)
        # Productivity shock
        eps_process = Uniform(2, params['min_eps'], params['max_eps'],
                params['N_eps']) 
        # Market belief of high state - rho
        #params['nrho'] = 10
        #rho = State(torch.linspace(0, 1, nrho))
        #rho.get_next_states = self.update_market_belief
        # Investment - Choose next period capital state 
        k1 = Action(k)

        # Create the Investor agent
        #self.investor = Investor(k, rho, beta)
        # Create the DiscreteAgent
        super(Manager, self).__init__(states=[k,w_process,eps_process],
                actions=[k1], discount_rate=self.beta)

    def update_market_belief(self):
        pass

    def reward(self, states, actions):
        # Cash flow
        k = states[0]
        w = states[1]
        eps = states[2]
        k1 = actions[0]
        I = k1 - (1-self.d)*k
        return self.cashFlow(k, w, eps, I)

    def cashFlow(self, k, w, eps, I):
        return w*(k**self.theta) - (I**2)/2 + eps
        
    def plot_policy(self):
        # Find the steady state capital stock (without adjustment costs)
        kss=((self.r+self.d)/self.theta)**(1/(self.theta-1))
        # Find the index of the capital steady state
        kss_idx = torch.argmin(torch.abs(self.states[0].values - kss))
        # Plot the policy function at the steady state
        # as a function of the shock state z
        pi_ss = self.pi[kss_idx]
        #z = torch.log(self.states[1])
        data = pd.DataFrame()
        data['w'] = self.states[1]
        data['I'] = pi_ss / kss
        sns.lineplot('w', 'I', data=data)

    def plot_simulation(self, sim):
        # Simulate
        sim_states = sim[0]
        sim_actions = sim[1]

        data = pd.DataFrame()
        data['K'] = sim_states[0]
        data['w'] = sim_states[1]
        data['eps'] = sim_states[2]
        data['K1'] = sim_actions[0]
        data['I'] = data['K1'] - data['K'] * (1-self.d)
        data['time'] = data.index
        sns.lineplot(x='time', y='value', hue='variable', style='variable',
                data=data[['time','K','w','eps','I']].melt('time'))


class Investor(DiscreteAgent):
    r"""
    Implements the Investors for the posterior beliegs problem.
    
    Note, the investor does not choose any action.
    """
    
    def __init__(self, k, rho, beta):
        super(Agent_PostBeliefs, self).__init__(states=[k,rho],
                actions=[], discount_rate=beta)

    def reward(self, states, actions):
        k = states[0]
        rho = states[1]

class Beliefs(nn.Module):
    r"""
    Investors beliefs are represented by a Neural Network.

    The Network has the following inputs:
    e   Firm profits
    K   Firm capital
    I   Firm investment

    And the following outputs:
    w1  Good state of the world
    """

    def __init__(self):
        super(Beliefs, self).__init__()
        # Define only one perceptron for now
        self.perceptron = nn.Linear(3, 1)
        # Define the activation function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.perceptron(x)
        x = self.sigmoid(x)
        return x

    def criterion(self, output, label):
        return torch.sum((label - output)**2)

    def learn(self, inputs, labels):
        # Use a Stochastic Gradient Descent optimizer
        optimizer = optim.SGD(self.parameters(), lr=.01, momentum=.5)
        # Optimize the weights
        for epoch in range(5000):
            #inputs, labels = d
            inputs = Variable(torch.FloatTensor(inputs), requires_grad=True)
            labels = Variable(torch.FloatTensor(labels), requires_grad=True)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 99:
                print(loss)

    def data_test(self):
        return [[[1,1,1],[0]],[[2,2,2],[0]],[[3,3,3],[1]],[[4,4,4],[1]]]

    def inputs_test(self):
        d = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
        return Variable(torch.FloatTensor(d), requires_grad=True)

    def labels_test(self):
        d = [[0],[0],[1],[1]]
        return Variable(torch.FloatTensor(d), requires_grad=True)






