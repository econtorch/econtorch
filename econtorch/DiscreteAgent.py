r"""
Discrete Agent
Discrete State Space and Discrete Action Space
"""

import torch
from torch.distributions.categorical import Categorical
from econtorch.base import *
from econtorch.discrete_random_processes import *
import numpy as np


class DiscreteAgent(Agent):
    
    def __init__(self, **kwargs):

        super(DiscreteAgent, self).__init__(**kwargs)

        # Value (V), Action-Value (Q) and Policy (pi) functions (array here)
        self.V = torch.zeros(self.states_shape)
        self.Q = torch.zeros(self.states_actions_shape)
        self.pi = torch.zeros(self.states_shape)

        # Update the next states and rewards
        self.update_reward()
        self.update_next_states()



########## Continuation Value ##########

    def update_reward(self):
        self.rew = self.reward(self.states_meshgrids, self.actions_meshgrids)
        self.rew[torch.isnan(self.rew)] = -np.inf


    def update_next_states(self):
        r"""
        For some states, we know the next states
            - Random Processes
            - States linked to an action
        for states that we do not know, the function must be provided.

        Note: we only need to execute this one time as the next states
        and their indices will always be the same.
        It needs to be updated if thee is a modification of the state
        or action space.
        """
        sa_sh = self.states_actions_shape
        sa_dim = len(sa_sh)

        # Compute the next state and the final shape
        stoch_sh = []
        for s in self.states:
            if hasattr(s, 'action') or isinstance(s, RandomProcess):
                s.get_next_states()
            else:
                # Find the corresponding indices and transition if not given
                s.get_next_states()
                # Find the indices (closest on the grid)
                ns = s.ns.view(1,-1)
                grid = s.values.view(-1,1)
                diff = torch.abs(ns - grid)
                s.ns_indices = torch.argmin(diff, 0)
                s.ns_indices = s.ns_indices.reshape(s.ns.shape)
                # Compute the transition matrix if needed - TODO
                s.ns_transition = torch.ones(s.ns.shape)

            # Compute the final shape
            if s.ns.shape == self.states_actions_shape:
                stoch_sh += [1]
            else:
                stoch_sh += list(s.ns.shape[len(sa_sh):])


        ## Expand the states and transitions as needed
        for i in range(0,len(stoch_sh)):
            if stoch_sh[i]==1:
                for s in self.states:
                    s.ns = torch.unsqueeze(s.ns, sa_dim+i)
                    s.ns_transition = torch.unsqueeze(s.ns_transition, sa_dim+i)
                    s.ns_indices = torch.unsqueeze(s.ns_indices, sa_dim+i)
            else:
                for s in self.states:
                    if i != s.dim:
                        s.ns = torch.unsqueeze(s.ns, sa_dim+i)
                        s.ns_transition = torch.unsqueeze(s.ns_transition, sa_dim+i)
                        s.ns_indices = torch.unsqueeze(s.ns_indices, sa_dim+i)
                        shape = list(s.ns.shape)
                        shape[sa_dim+i] = stoch_sh[i]
                        s.ns = s.ns.expand(shape)
                        s.ns_transition = s.ns_transition.expand(shape)
                        s.ns_indices = s.ns_indices.expand(shape)
        self.ns_shape = self.states_actions_shape + torch.Size(stoch_sh)

        # Multiply the transitions matrices to obtain the joint distribution
        self.joint_transition = torch.ones(self.ns_shape)
        for s in self.states:
            self.joint_transition = self.joint_transition * s.ns_transition

        # Compute the one dimensional indices
        n_idx = int(np.prod(self.ns_shape))
        self.oneD_indices = 0
        mul = 1
        for i in range(len(self.states)-1,-1,-1):
            s = self.states[i]
            s.ns_indices = s.ns_indices.reshape(n_idx)
            #Idx = Idx + (s.ns_indices * mul).float()
            self.oneD_indices = self.oneD_indices + (s.ns_indices * mul)
            mul = mul * np.prod(s.size)
            s.ns_indices = s.ns_indices.reshape(self.ns_shape)
        self.oneD_indices = self.oneD_indices.long()


        # Compute the constraints (values that are inside the grid)
        self.ingrid = torch.ones(self.ns_shape)
        for s in self.states:
            ingrid_tmp = torch.zeros(self.ns_shape)
            max_val = torch.max(s.values).item()
            min_val = torch.min(s.values).item()
            ingrid_tmp = (s.ns >= min_val).float() * (s.ns <= max_val).float()
            self.ingrid = self.ingrid * ingrid_tmp
        self.ingrid[self.ingrid==0] = -np.inf


    def update_next_states_values(self):
        self.V_flat = self.V.view(int(np.prod(self.states_shape)))
        values_flat = self.V_flat[self.oneD_indices]
        values = values_flat.reshape(self.ns_shape)
        # Apply constraints
        self.next_states_values = values * self.ingrid
        # Reshape V 
        #self.V = self.V.reshape(self.states_shape)


    def update_continuation_value(self):
        r"""
        Provide the expected continuation values for given states and actions.
        """
        # Multiply the transitions matrices to obtain the joint distribution
        p_trans = torch.ones(self.ns_shape)
        for s in self.states:
            p_trans = p_trans * s.ns_transition

        # Compute the expectation using the transition matrix
        values_p = self.next_states_values * self.joint_transition
        ## Sum on the stochastic dimensions
        sa_len = len(self.states_actions_shape)
        sas_len = len(self.ns_shape)
        self.continuation_value = values_p.sum(list(range(sa_len, sas_len)))


########## Optimal Policy ##########

    def policy_improvement(self):
        self.update_next_states_values()
        self.update_continuation_value()
        action_value = self.rew + self.discount_rate * self.continuation_value
        # Change NaNs to -Inf for the maximization
        #import ipdb; ipdb.set_trace();
        #action_value[torch.isnan(action_value)] = -np.inf
        action_value[action_value!=action_value] = -float('inf')

        # Below code only correct for one-dimension action space !!
        if len(self.actions) > 1:
            raise Exception('Only one dimensional action space is supported')
            # Below code is a work in progress for multi-dimensional case
            # Linearize the action tensor
            n_a = 1
            for a in self.actions:
                n_a = n_a * len(a)
            shape = self.states_shape + [n_a]
            action_value = action_value.reshape(shape)
            max_V, opt_pi_ind = action_value.max(dim=len(shape)-1)

        self.V, self.pi_indices = action_value.max(dim=len(self.states))
        # Replace -Inf to Nan if no action is possible
        # Probably a misspecification of the state-action space
        self.V[torch.isinf(self.V)] = np.nan
        # Find the optimal policy
        size_s = np.prod(self.states_shape)
        self.pi = self.actions_values[0][self.pi_indices.view(size_s)].view(self.states_shape)
        # Correct the optimal policy for Nans
        self.pi[torch.isnan(self.V)] = np.nan

    def policy_evaluation(self, criterion):
        r"""
        Compute the state-value function V under the current policy pi.
        """
        self.update_next_states_values()
        Ns = len(self.states)

        V_old = self.V
        V_diff = V_old - self.V + 1
        V_diff[torch.isnan(V_diff)] = 0.
        while (torch.norm(V_diff) > criterion):
            self.update_next_states_values()
            self.update_continuation_value()
            action_value = self.rew + self.discount_rate * self.continuation_value
            # Find the value function given the current policy
            self.V = torch.gather(action_value,
                    index=self.pi_indices.unsqueeze(dim=len(self.states)),
                    dim=Ns).squeeze(dim=Ns)
            V_diff = V_old - self.V
            V_old = self.V
            V_diff[torch.isnan(V_diff)] = 0.
            print(torch.norm(V_diff))


    
########## Value Function Iteration ##########

    def iterate_value_function(self, criterion):
        # Update next_states and rewards
        self.update_reward()
        self.update_next_states()
        V_old = self.V
        V_diff = V_old - self.V + 1
        V_diff[torch.isnan(V_diff)] = 0.
        while (torch.norm(V_diff) > criterion):
            # Update optimal policy
            self.policy_improvement()
            V_diff = V_old - self.V
            V_old = self.V
            V_diff[torch.isnan(V_diff)] = 0.
            print(torch.norm(V_diff))

########## Simulation Functions ##########

    def simulate(self, N, initial_state):
        r"""
        Simulate one realization of the world and of the optimal policy
        for N periods starting at the initial_state.

        initial_state       array of initial states values
        """
        Ns = len(self.states)
        Na = len(self.actions)
        sim_states = np.zeros([Ns,N])
        #sim_states_indices = np.zeros([len(self.states),N])
        sim_pi = np.zeros([Na,N]) 
        #sim_pi_indices = np.zeros([len(self.actions),N]) 
        #s = []
        s_ind = []
        #pi = []
        pi_ind = []

        # Get the indices of the initial states (closest)
        for i in range(0, Ns):
            s_ind += [torch.argmin(torch.abs(self.states[i].values
                - initial_state[i]))]

        # add the optimal policy (indices)
        # TODO: Adapt to multidimentional actions here !
        pi_ind += [self.pi_indices[tuple(s_ind)]]


        for i in range(0,N):
            # Log the current state
            for j in range(0, Ns):
                sim_states[j,i] = self.states[j].values[s_ind[j]]
            # Log the current action
            for j in range(0, Na):
                sim_pi[j,i] = self.actions[j].values[pi_ind[j]]


            # Get the probability distribution of the next states
            dist = self.joint_transition[s_ind+pi_ind]
            Nns = np.prod(dist.shape)
            dist_flat = dist.view(Nns)

            # Create a categorical distribution (flatten the distribution) 
            cat_dist = Categorical(dist_flat)
            
            # Draw a sample 
            ns_ind_flat = cat_dist.sample()

            # Get the next states indices 
            #states_flat = []
            s_ind_old = s_ind
            s_ind = []
            for s in self.states:
                s_flat = s.ns_indices[s_ind_old+pi_ind].view(Nns)
                s_ind += [s_flat[ns_ind_flat].long()]
                
            pi_ind = []
            pi_ind += [self.pi_indices[tuple(s_ind)]]

        return (sim_states, sim_pi)


