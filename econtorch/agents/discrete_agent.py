r"""
Define discrete States/Actions and a discrete agent.
"""

import torch
from torch.distributions.categorical import Categorical
import numpy as np

from econtorch.environment import DiscreteState
from econtorch.environment import DiscreteAction


class DiscreteAgent():
    r"""
    Class reprensenting an Agent.

    Attributes:
        states          List of state objects.
        states_values   List of 1D torch.Tensor with the states values
        states_shape    Shape of the state space (torch.Size)
        actions         List of action objects.
        actions_shape   Shape of the state space (torch.Size)
    """
    
    def __init__(self, obs_states, environment, actions=[], 
            discount_rate=None):
        # Create the states and actions space
        self.states = []
        self.states_values = []
        self.states_shape = torch.Size() 
        self.actions = []
        self.actions_values = []
        self.actions_shape = torch.Size() 
        self.add_states(obs_states)
        self.add_actions(actions)
        # Discount Rate 
        self.set_discount_rate(discount_rate)
        # Create the shapes and meshgrids
        self.update_states_actions_shape()
        self.update_meshgrids()
        # Value (V), Action-Value (Q) and Policy (pi) functions (arrays here)
        self.V = torch.zeros(self.states_shape)
        self.Q = torch.zeros(self.states_actions_shape)
        self.pi = torch.zeros(self.states_shape)
        self.pi_indices = torch.zeros(self.states_shape)
        self.environment = environment

########## Functions to manipulate the object ##########

    def add_state(self, s):
        if not(isinstance(s, DiscreteState)):
            s = DiscreteState(s)
        s.dim = len(self.states)
        self.states += [s]
        self.states_values += [s.values]
        self.states_shape += s.size
        #self.update_states_actions_shape()
        #self.update_meshgrids()

    def add_states(self, s_array):
        for s in s_array:
            self.add_state(s)

    def add_action(self, a):
        if not(isinstance(a, DiscreteAction)):
            a = action(a)
        a.dim = len(self.actions)
        self.actions += [a]
        self.actions_values += [a.values]
        self.actions_shape += a.size
        self.update_states_actions_shape()
        self.update_meshgrids()

    def add_actions(self, values_array):
        for v in values_array:
            self.add_action(v)

    def update_states_actions_shape(self):
        self.states_actions_shape = self.states_shape + self.actions_shape

    def update_meshgrids(self):
        meshgrids = torch.meshgrid(self.states_values + self.actions_values)
        self.states_meshgrids = []
        self.actions_meshgrids = []
        for s in self.states:
            s.meshgrid = meshgrids[s.dim]
            self.states_meshgrids += [s.meshgrid]
        for a in self.actions:
            a.meshgrid = meshgrids[len(self.states)+a.dim]
            self.actions_meshgrids += [a.meshgrid]

    def set_discount_rate(self, discount_rate):
        self.discount_rate = discount_rate

########## Integration functions ##########
    def integrate_current(self, tensor, state):
        r"""
        Integrates tensor along a given state
        Returns a vector of the same shape as tensor.
        Note: integrates over current state, not the next state.
        """
        # Reshape the distribution
        ## Put the state dimension at the end
        last_dim = len(self.states_actions_shape)-1
        perm = np.arange(0,last_dim+1,1)
        perm[last_dim] = state.dim
        perm[state.dim] = last_dim
        import ipdb; ipdb.set_trace()
        tensor = tensor.permute(list(perm))

        # TODO dimensions
        state.transitions.unsqueeze(-1)
        proba = state.transitions.expand(tensor.shape)
        tensor = tensor.permute(list(perm))
        proba = proba.permute(list(perm))
        tensor_p = tensor*proba
        tensor_int = tensor_p.sum(state.dim)
        tensor_final = tensor_int.unsqueeze(state.dim).expand(tensor.shape)
        return tensor_final

    def integrate_next(self, tensor, states):
        r"""
        Integrates the tensor along the given states
        Returns a vector of reduced shape (the state dimension is removed)
        Note: Integrates over the next states, not the current states.
        """
        # Get the joint proba distribution
        joint_transition = torch.ones(tensor.shape)
        for s in states:
            joint_transition = joint_transition * s.ns_transition
        tensor_p = tensor*joint_transition
        dims = [len(self.states_actions_shape)+s.dim for s in states]
        tensor_final = tensor_p.sum(dims)
        return(tensor_final)


########## Convenient Functions to get Values and Policies ##########

    def get_pi(self, indices):
        r"""
        Get the optimal policy for given states.

        Args:
            indices: list of the indices of the states, ordered

        warning::
            The shapes of the indices should be similar

        """
        # Checks
        if len(indices) != len(self.states):
            raise Exception("Incorrect number of indices")
        sh = indices[0].shape
        for ind in indices:
            if sh != ind.shape:
                raise Exception("One or more indices are of different shapes")
        # Compute the flatten indices
        n_idx = int(np.prod(sh))
        flatten_indices = 0
        mul = 1
        for i in range(len(indices)-1,-1,-1):
            ind = indices[i].reshape(n_idx)
            flatten_indices = flatten_indices + (ind * mul)
            mul = mul * np.prod(self.states[i].size)
        # Get the optimal policy
        policy = self.pi.reshape(np.prod(self.pi.shape))[flatten_indices].reshape(sh)
        return policy

########## Continuation Value ##########

    def update_reward(self):
        #self.rew = self.reward(self.states_meshgrids, self.actions_meshgrids)
        self.rew = self.reward()
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
            s.get_next_states()
            #TODO: Rewrite this part to allow for incomplete objects
            #if (hasattr(s, 'action') 
            #        or isinstance(s, RandomProcess) 
            #        or isinstance(s, Belief)):
            #    s.get_next_states()
            #else:
            #    # TODO: Rewrite 
            #    # Find the corresponding indices and transition if not given
            #    # Works only for non-stochastic transitions
            #    s.get_next_states()
            #    # Find the indices (closest on the grid)
            #    ns = s.ns.view(1,-1)
            #    grid = s.values.view(-1,1)
            #    diff = torch.abs(ns - grid)
            #    s.ns_indices = torch.argmin(diff, 0)
            #    s.ns_indices = s.ns_indices.reshape(s.ns.shape, dtype=torch.float32)
            #    # Compute the transition matrix if needed - TODO
            #    s.ns_transition = torch.ones(s.ns.shape)

            # Compute the final shape
            if s.ns.shape == self.states_actions_shape:
                stoch_sh += [1]
            else:
                stoch_sh += list(s.ns.shape[len(sa_sh):])


        print("before expand")
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

        print("after expand")
        for s in self.states:
            print("i" + str(s))
            self.joint_transition = self.joint_transition * s.ns_transition

        # Compute the one dimensional indices
        n_idx = int(np.prod(self.ns_shape))
        self.oneD_indices = 0
        mul = 1
        print("before dim")
        for i in range(len(self.states)-1,-1,-1):
            print("i" + str(i))
            s = self.states[i]
            s.ns_indices = s.ns_indices.reshape(n_idx)
            #Idx = Idx + (s.ns_indices * mul).float()
            self.oneD_indices = self.oneD_indices + (s.ns_indices * mul)
            mul = mul * np.prod(s.size)
            s.ns_indices = s.ns_indices.reshape(self.ns_shape)
        self.oneD_indices = self.oneD_indices.long()

        print("before constraints")
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
        # # Multiply the transitions matrices to obtain the joint distribution
        # p_trans = torch.ones(self.ns_shape)
        # for s in self.states:
        #     p_trans = p_trans * s.ns_transition

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
        V_diff = V_old - self.V + criterion
        V_diff[torch.isnan(V_diff)] = 0.
        while (torch.norm(V_diff) > criterion):
            self.update_next_states_values()
            self.update_continuation_value()
            action_value = self.rew + self.discount_rate * self.continuation_value
            # Find the value function given the current policy
            if len(self.actions) > 0:
                # If the agent actual has a policy
                self.V = torch.gather(action_value,
                        index=self.pi_indices.unsqueeze(dim=len(self.states)),
                        dim=Ns).squeeze(dim=Ns)
            else:
                # Otherwise, just get the value given the next state
                self.V = action_value
            V_diff = V_old - self.V
            V_old = self.V
            V_diff[torch.isnan(V_diff)] = 0.
            print(torch.norm(V_diff))
    
########## Value Function Iteration ##########

    def iterate_value_function(self, criterion):
        # Update next_states and rewards
        print("reward start")
        self.update_reward()
        print("next states start")
        self.update_next_states()
        print("before while")
        V_old = self.V
        V_diff = V_old - self.V + criterion
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

