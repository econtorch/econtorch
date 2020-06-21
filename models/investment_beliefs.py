
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.binomial import Binomial

import numpy as np

from econtorch.agents.discrete_agent import *
from econtorch.agents.discrete_dqn_agent import *
from econtorch.processes.discrete_random_processes import Uniform
from econtorch.processes.discrete_random_processes import MarkovBinomial
from econtorch.environment import DiscreteEnvironment

# May remove it later
from econtorch.processes.discrete_random_processes import Belief

import pandas as pd
import seaborn as sns
#import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
#import matplotlib.pyplot as plt

def demo_params():
    # Global parameters
    params = {}
    params['d'] = .15       # Depreciation rate
    params['theta'] = 0.7   # Productivity of capital
    params['r'] = 0.04      # Interest rate (discount rate inferred)

    # Capital Grid - k
    params['k_min'] = 10 
    params['k_max'] = 250
    params['nk'] = 50       # Number of capital states
    # State of the World - w    (Productivity, Binomial process)
    params['q'] = .95        # Probability of staying in the same state
    params['w0'] = 1        # Low productivity state
    params['w1'] = 6        # High productivity state
    #params['w0'] = 3        # Low productivity state
    #params['w1'] = 3        # High productivity state
    # Productivity shock - eps
    ## Uniformly distributed between min_eps and max_eps
    #params['min_eps'] = -10
    #params['max_eps'] = 10
    params['min_eps'] = 0
    params['max_eps'] = 0
    params['N_eps'] = 5     # Number of states used in the discretization

    # Fraction kept by the manager every period - x     (Uniform distribution)
    params['min_x'] = .5
    params['max_x'] = .7
    params['N_x'] = 3       # Number of states used in the discretization
    # Investors' Beliefs - rho
    params['ngw'] = 10      # Number of beliefs to describe the distribution

    # Return the parameters
    return params


def demo_single_manager():
    params = demo_params()
    # Capital Grid - k
    k = DiscreteState(torch.linspace(params['k_min'], params['k_max'],
        params['nk']))
    # Productivity shock
    eps = Uniform(params['min_eps'], params['max_eps'],
            params['N_eps']) 

    # State of the World - w
    q = params['q']
    w = MarkovBinomial([params['w0'],params['w1']],
            q, q)
    # Fraction kept by the manager - x
    x = Uniform(params['min_x'], params['max_x'],
            params['N_x']) 

    # Market belief about the state of nature w
    gw = Belief(w , params['ngw'])

    # Create the environment
    env = DiscreteEnvironment(states=[w,k,eps,gw,x])
    params = demo_params()
    # Create the agent
    man = Single_Manager(params,env)
    # Solve the value function
    man.iterate_value_function(1)
    # Plot the results
    ## Simulate a path starting at init_state
    init_state = [1,50,4]
    N = 200
    
    sim = man.simulate(N, init_state)
    
    #import ipdb; ipdb.set_trace();
    
    man.plot_simulation(sim)

# demo for Manager and Environment class
def demo_manager_with_environment():
    params = demo_params()
    # Capital Grid - k
    k = DiscreteState(torch.linspace(params['k_min'], params['k_max'],
        params['nk']))
    # Productivity shock
    eps = Uniform(params['min_eps'], params['max_eps'],
            params['N_eps']) 

    # State of the World - w
    q = params['q']
    w = MarkovBinomial([params['w0'],params['w1']],
            q, q)
    # Fraction kept by the manager - x
    x = Uniform(params['min_x'], params['max_x'],
            params['N_x']) 

    # Market belief about the state of nature w
    gw = Belief(w , params['ngw'])

    import ipdb; ipdb.set_trace()
    # Create the environment
    env = DiscreteEnvironment(states=[w,k,eps,gw,x])

    print("environment states" + str(env.states))
    # observable states for this manager
    #obs_states = [env.w, env.k, env.x, env.gw, env.eps]
    obs_states = [w,k,gw]
   
    print("herez")
    man = Manager(env, params, obs_states)
    print("herea")
    man.iterate_value_function(1)
    print("hereb")

    init_state = [1, 50, 4]
    N = 200

    sim = man.simulate(N, init_state)
    man.plot_simulation(sim)


def demo_single_DQN_manager():

    init_state = [1,20,4]
    N = 500


    params = demo_params()
    # Create the agent
    man = Single_DQN_Manager(params)

    # Training and simulation

    #man.train_Q_grid_sarsa()
    #man.train_Q_grid_qlearning()
    man.train_agent()
    man.update_reward()
    man.update_next_states()
    Q_pi, Q_pi_indices = man.get_Q_policy()
    #Q_pi, Q_pi_indices = man.get_Q_policy_nn()
    sim_Q = man.simulate(N, init_state, Q_pi_indices)
    man.plot_simulation(sim_Q)


    # Solve the value function
    # Plot the results
    ## Simulate a path starting at init_state

    # Training and simulation with DP
    man.iterate_value_function(1)
    sim = man.simulate(N, init_state, man.pi_indices)
    man.plot_simulation(sim)
    


    
    #import ipdb; ipdb.set_trace();
    


def demo_manager():
    params = demo_params()
    params['nk'] = 100
    # Create the agent
    man = Single_Manager(params)
    # Solve the value function
    man.iterate_value_function(1)
    # Plot the results
    ## Simulate a path starting at init_state
    init_state = [1,20,4]
    N = 100
    
    sim = man.simulate(N, init_state)
    
    #import ipdb; ipdb.set_trace();
    
    man.plot_simulation(sim)



class Single_DQN_Manager(DiscreteDQNAgent):
    r""" Implements the Manager for the posterior beliefs problem.

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

    def __init__(self, params):
        
        # Global parameters
        self.d = params['d']
        self.theta = params['theta']
        self.r = params['r']
        self.beta = 1/(1+self.r)

        # Capital Grid - k
        self.k = DiscreteState(torch.linspace(params['k_min'], params['k_max'],
            params['nk']))
        # State of the World - w
        self.q = params['q']
        self.w = MarkovBinomial([params['w0'],params['w1']],
                self.q, self.q)
        # Productivity shock
        self.eps = Uniform(params['min_eps'], params['max_eps'],
                params['N_eps']) 

        # Investment - Choose next period capital state 
        self.k1 = DiscreteAction(self.k)

        # Create the DiscreteAgent 
        super(Single_DQN_Manager, self).__init__(states=[self.w, self.k, self.eps],
            actions=[self.k1], discount_rate=self.beta)

    def reward(self):
        # Cash flow
        w = self.w.meshgrid
        k = self.k.meshgrid
        eps = self.eps.meshgrid
        k1 = self.k1.meshgrid
        I = k1 - (1-self.d)*k
        # Need to integrate the cashflows over epsilon
        # The rewards needs to be independent of epsilon
        cf = self.cashFlow(k, w, eps, I)
        #cf_fin = self.integrate_current(cf, self.eps)
        #return cf_fin
        return cf

    def cashFlow(self, k, w, eps, I):
        return w*(k**self.theta) - (I**2)/2 + eps

    def next_states(self, states, actions):
        # Return the next states for a list N of states and actions
        # states: shape (N, states dim)
        # actions: shape (N, actions dim)
        N = len(states)
        # Current states
        wi = states[:,0]
        ki = states[:,1]
        epsi = states[:,2]
        # Action
        Ii = actions[:,0]
        # Next state: shape (N, states dim)
        ns = torch.zeros([N, len(self.states)])
        # Next productivity shock
        ## Proba of staying in the same state w
        d = Bernoulli(self.q)
        identical_ns = d.sample((N,))
        ns[:,0] = wi.float()*identical_ns + (1-wi.float())*(1-identical_ns)
        # Next capital state
        ns[:,1] = Ii
        # Next eps (doesn't really matter in fact, but need to input it)
        Neps = self.states[2].length
        d = Categorical(torch.ones(Neps)/Neps)
        ns[:,2] = d.sample((N,))
        # Return the next states
        return ns

    def rewards(self, states, actions):
        w = states[:,0]
        k = states[:,1]
        eps = states[:,2]
        k1 = actions[:,0]
        I = k1 - (1-self.d)*k
        cf = self.cashFlow(k, w, eps, I)
        return cf





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
        data['w'] = sim_states[0]
        data['K'] = sim_states[1]
        data['eps'] = sim_states[2]
        data['K1'] = sim_actions[0]
        data['I'] = data['K1'] - data['K'] * (1-self.d)
        data['time'] = data.index
        #sns.set_palette("Blues_d", 4)
        sns.lineplot(x='time', y='value', hue='variable', style='variable',
                data=data[['time','K','w','eps','I']].melt('time'))


class Single_Manager(DiscreteAgent):
    r""" Implements the Manager for the posterior beliefs problem.

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

    def __init__(self, params, env):
        
        # Global parameters
        self.d = params['d']
        self.theta = params['theta']
        self.r = params['r']
        self.beta = 1/(1+self.r)

        # Capital Grid - k
        self.k = DiscreteState(torch.linspace(params['k_min'], params['k_max'],
            params['nk']))
        # State of the World - w
        self.q = params['q']
        self.w = MarkovBinomial([params['w0'],params['w1']],
                self.q, self.q)
        # Productivity shock
        self.eps = Uniform(params['min_eps'], params['max_eps'],
                params['N_eps']) 

        # Investment - Choose next period capital state 
        self.k1 = DiscreteAction(self.k)

        # # Create the DiscreteAgent 
        super(Single_Manager, self).__init__(obs_states=[self.w, self.k, self.eps], environment=env,
            actions=[self.k1], discount_rate=self.beta)

    def reward(self):
        # Cash flow
        w = self.w.meshgrid
        k = self.k.meshgrid
        eps = self.eps.meshgrid
        k1 = self.k1.meshgrid
        I = k1 - (1-self.d)*k
        # Need to integrate the cashflows over epsilon
        # The rewards needs to be independent of epsilon
        cf = self.cashFlow(k, w, eps, I)
        cf_fin = self.integrate_current(cf, self.eps)
        return cf_fin

    def cashFlow(self, k, w, eps, I):
        return w*(k**self.theta) - (I**2)/4 + eps

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
        data['w'] = sim_states[0]
        data['K'] = sim_states[1]
        data['eps'] = sim_states[2]
        data['K1'] = sim_actions[0]
        data['I'] = data['K1'] - data['K'] * (1-self.d)
        data['time'] = data.index
        #sns.set_palette("Blues_d", 4)
        sns.lineplot(x='time', y='value', hue='variable', style='variable',
                data=data[['time','K','w','eps','I']].melt('time'))


class Manager(DiscreteAgent):
    r""" Implements the Manager for the posterior beliefs problem.

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

    # def __init__(self, params):
        
    #     # Global parameters
    #     self.d = params['d']
    #     self.theta = params['theta']
    #     self.r = params['r']
    #     self.beta = 1/(1+self.r)

    #     # Capital Grid - k
    #     self.k = DiscreteState(torch.linspace(params['k_min'], params['k_max'],
    #         params['nk']))
    #     # State of the World - w
    #     self.q = params['q']
    #     self.w = MarkovBinomial([params['w0'],params['w1']],
    #             self.q, self.q)
    #     # Fraction kept by the manager - x
    #     self.x = Uniform(params['min_x'], params['max_x'],
    #             params['N_x']) 
    #     # Productivity shock
    #     self.eps = Uniform(params['min_eps'], params['max_eps'],
    #             params['N_eps']) 

    #     # Market belief about the state of nature w
    #     self.gw = Belief(self.w , params['ngw'])

    #     # Investment - Choose next period capital state 
    #     self.k1 = DiscreteAction(self.k)

    #     # Create the DiscreteAgent 
    #     super(Manager, self).__init__(states=[self.w, self.k, self.x,
    #         self.gw, self.eps],
    #         actions=[self.k1], discount_rate=self.beta)


    #     # Create the Investor agent         
    #     self.investor = Investor(self)

        # # Add the beliefs function to the rho state of the manager:
        # # Needed to compute the next state beliefs
        # rho.beliefs = self.investor.beliefs
        # rho.get_next_states = self.next_beliefs

    def __init__(self, env, params, obs_states):
        # Global parameters
        self.d = params['d']
        self.theta = params['theta']
        self.r = params['r']
        self.beta = 1/(1+self.r)


        # Capital Grid - k
        self.k = env.states[1]
        # State of the World - w
        self.w = env.states[0]
    
        # # Productivity shock
        # self.eps = env.states[2]

        # Market belief about the state of nature w
        self.gw = env.states[3]
        self.obs_states = obs_states
        self.environment = env

        # Investment - Choose next period capital state 
        self.k1 = DiscreteAction(self.k)
        super(Manager, self).__init__(obs_states=obs_states, environment=env,
        actions=[self.k1], discount_rate=self.beta)


        # TODO
        # create different types of investors: perfect/imperfect
        # Create the Investor agent         
        #self.investor = Investor(self)

    def reward(self):
        # Cash flow
        w = self.w.meshgrid
        k = self.k.meshgrid
        # eps = self.eps.meshgrid
        k1 = self.k1.meshgrid
        I = k1 - (1-self.d)*k
        # Need to integrate the cashflows over epsilon
        # The rewards needs to be independent of epsilon
        cf = self.cashFlow(k, w, I)
        # cf_fin = self.integrate_current(cf, self.eps)
        return cf

    # def cashFlow(self, k, w, eps, I):
    #     return w*(k**self.theta) - (I**2)/2 + eps
    
    def cashFlow(self, k, w, I):
        return w*(k**self.theta) - (I**2)/2

    def update_continuation_value(self):
        # Overwrites the standard continuation value
        # For now just reproduce it
        s_int = self.obs_states
        cont = self.integrate_next(self.next_states_values,s_int)
        # Remove the non-stochastic states
        self.continuation_value = cont.squeeze()

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
        data['w'] = sim_states[0]
        data['K'] = sim_states[1]
        data['eps'] = sim_states[2]
        #data['gw'] = sim_states[3]
        data['K1'] = sim_actions[0]
        data['I'] = data['K1'] - data['K'] * (1-self.d)
        data['time'] = data.index
        sns.lineplot(x='time', y='value', hue='variable', style='variable',
                data=data[['time','K','w','eps','I']].melt('time'))


class Investor(DiscreteAgent):
    r"""
    Implements the Investors for the posterior beliefs problem.
    
    Note, the investor does not choose any action.
    """
    
    def __init__(self, manager):
        # link both objects
        self.manager = manager
        # Use the same state space for the manager and the investor
        self.w = manager.w.clone()
        self.k = manager.k.clone()
        self.x = manager.x.clone()
        self.gw = manager.gw.clone()
        # self.eps = manager.eps.clone()
        self.k1 = manager.k.clone()
        # Determine next states for k and k1
        self.k.get_next_states = self.k._get_next_states_deterministic
        self.k1.get_next_states = self.next_investment_state

        # Create the Investor
        # Note that the manager has no action and one more state
        # super(Investor, self).__init__(states=[self.w, self.k, self.x,
        #     self.gw, self.eps, self.k1],
        #     actions=[], discount_rate=manager.discount_rate)

    # # alternative constructor
    # def __init__(self, manager, notPerfect):
        
    #     # link both objects
    #     self.manager = manager
    #     # Use the same state space for the manager and the investor
    #     self.w = manager.w.clone()
    #     self.k = manager.k.clone()
    #     self.x = manager.x.clone()
    #     self.gw = manager.gw.clone()
    #     self.eps = manager.eps.clone()
    #     self.k1 = manager.k.clone()
    #     # Determine next states for k and k1
    #     self.k.get_next_states = self.k._get_next_states_deterministic
    #     self.k1.get_next_states = self.next_investment_state

    #     # Create the investor with imperfect information
    #     super(Investor, self).__init__(states=[self.w, self.k, self.x,
    #         self.gw, self.eps, self.k1],
    #         actions=[], discount_rate=manager.discount_rate)

    def reward(self):
        # Cash flow
        w = self.w.meshgrid
        k = self.k.meshgrid
        # eps = self.eps.meshgrid
        k1 = self.k1.meshgrid
        I = k1 - (1-self.manager.d)*k
        # Need to integrate the cashflows over epsilon
        cf = self.manager.cashFlow(k, w, eps, I)
        #cf_int = cf.sum(4)
        return cf

    def next_investment_state(self):
        # Compute the next investment state
        ## Use the optimal policy of the manager
        # Need to correct the shape to include k1 in the state space
        self.k1.ns = self.manager.pi.unsqueeze(-1).expand(self.manager.states_actions_shape)
        self.k1.ns_indices = self.manager.pi_indices.unsqueeze(-1).expand(self.manager.states_actions_shape)
        self.k1.ns_indices = self.k1.ns_indices.type(torch.float32)
        self.k1.ns_transition = torch.ones(self.k1.ns.shape) # Deterministic

    def update_beliefs_OLD(self):
        pass
        #     r"""
        #     Update the belief function belief(K,e)
        #     The updating needs to occur when the Manager's optimal investment
        #     policy changes.
        #     """
        #     # Create the training data
        #     # (Create a dataset of (e, w, k) using the Manager's optimal investment
        #     # policy.
        #     ## Create N observations for each (k, w)
        #     ## Total number of observation in the training set: nk x nw x N
        #     ## NOTE: Other possibility: Simulate a path and learn from the generated
        #     ## data (as if investors only learn from history and not from their
        #     ## knowledge of the Manager's optimization problem). That would limit
        #     ## the information available to investors.
        #     N = 10
        #     m = self.manager
        #     k, w, eps = torch.meshgrid(m.states[0].values,
        #             m.states[1].values, torch.zeros([N]))
        #     k_ind, w_ind, eps_ind = torch.meshgrid(m.states[0].indices,
        #             m.states[1].indices, torch.zeros([N], dtype=torch.int64))
        #     # Draw the noise eps
        #     ## Create a categorical distribution to simulate the indices
        #     ## NOTE: The noise MUST be iid (and the transition probabilities
        #     ## uni-dimensionals)
        #     eps_dist = Categorical(m.states[2].transitions)
        #     eps_ind = eps_dist.sample(eps.shape)
        #     eps = m.states[2].values[eps_ind]
        #     # Get the optimal investment
        #     I = m.get_pi([k_ind, w_ind, eps_ind])
        #     # Compute the profits e (reward)
        #     e = m.reward([k, w, eps], [I])
        #     # Reshape the data in a compatible format for training
        #     n = np.prod(k.shape)
        #     inputs = torch.stack([e.reshape(n), k.reshape(n)], dim=1)
        #     labels = w.reshape(n)
        #     # Train the neural network using the data.
        #     self.beliefs.learn(inputs, labels)


class Beliefs(nn.Module):
    pass
    #    r"""
    #    Investors beliefs are represented by a Neural Network.
    #
    #    The Network has the following inputs:
    #    e   Firm profits
    #    K   Firm capital
    #    I   Firm investment
    #
    #    And the following outputs:
    #    w1  Good state of the world
    #    """
    #
    #    def __init__(self):
    #        super(Beliefs, self).__init__()
    #        # Define only one perceptron for now
    #        self.perceptron = nn.Linear(2, 1)
    #        # Define the activation function
    #        self.sigmoid = nn.Sigmoid()
    #        
    #    def forward(self, x):
    #        x = self.perceptron(x)
    #        x = self.sigmoid(x)
    #        return x
    #
    #    def criterion(self, output, label):
    #        return torch.sum((label - output)**2)
    #
    #    def learn(self, inputs, labels):
    #        # Use a Stochastic Gradient Descent optimizer
    #        optimizer = optim.SGD(self.parameters(), lr=.01, momentum=.5)
    #        # Optimize the weights
    #        for epoch in range(5000):
    #            #inputs, labels = d
    #            inputs = Variable(torch.FloatTensor(inputs), requires_grad=True)
    #            labels = Variable(torch.FloatTensor(labels), requires_grad=True)
    #            optimizer.zero_grad()
    #            outputs = self(inputs)
    #            loss = self.criterion(outputs, labels)
    #            loss.backward()
    #            optimizer.step()
    #            if epoch % 100 == 99:
    #                print(loss)
    #
    #    def inputs_test(self):
    #        #d = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
    #        d = [[1,1],[2,2],[3,3],[4,4]]
    #        return Variable(torch.FloatTensor(d), requires_grad=True)
    #
    #    def labels_test(self):
    #        d = [[0],[0],[1],[1]]
    #        return Variable(torch.FloatTensor(d), requires_grad=True)
    





