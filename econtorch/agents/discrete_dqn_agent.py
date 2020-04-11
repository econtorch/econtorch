r"""
Define a discrete DQn agent.
"""

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import numpy as np
#import econtorch.discrete_random_processes.RandomProcess as RandomProcess

from econtorch.environment import DiscreteState
from econtorch.environment import DiscreteAction


class DiscreteDQNAgent():
    r"""
    Class reprensenting an Agent.

    Attributes:
        states          List of state objects.
        states_values   List of 1D torch.Tensor with the states values
        states_shape    Shape of the state space (torch.Size)
        actions         List of action objects.
        actions_shape   Shape of the state space (torch.Size)
    """
    
    def __init__(self, states=[], actions=[],
            discount_rate=None):
        # Create the states and actions space
        self.states = []
        self.states_values = []
        self.states_shape = torch.Size() 
        self.actions = []
        self.actions_values = []
        self.actions_shape = torch.Size() 
        self.add_states(states)
        self.add_actions(actions)
        # Discount Rate 
        self.set_discount_rate(discount_rate)
        # Create the shapes and meshgrids
        self.update_states_actions_shape()
        self.update_meshgrids()
        # Create the Q network
        self.Q = _Q_network(self)
        self.Qa = _Q_network(self)
        self.Qb = _Q_network(self)
        # Create a Q grid for testing purposes
        #self.Q_grid = -1e-10*torch.ones(self.states_actions_shape)
        self.Q_grid = torch.zeros(self.states_actions_shape)
        self.Qa_grid = torch.zeros(self.states_actions_shape)
        self.Qb_grid = torch.zeros(self.states_actions_shape)

        self.V = torch.zeros(self.states_shape)
        self.pi = torch.zeros(self.states_shape)
        self.pi_indices = torch.zeros(self.states_shape)

########## Functions to manipulate the object ##########

    def add_state(self, s):
        if not(isinstance(s, DiscreteState)):
            s = DiscreteState(s)
        if s.agent is not None: # Clone the state if already assigned
            s = s.clone()
        s.dim = len(self.states)
        s.agent = self
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
        tensor = tensor.permute(list(perm))
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
        We can figure out the next states for the following states types:
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
        self.update_reward()
        self.update_next_states()
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

    def get_experience_greedy(self, epsilon):
        # Implement an epsilon-greedy policy
        # Create the experience states
        inds = []
        for s in self.states:
            inds.append(s.indices)
        inds_mesh = torch.meshgrid(inds)
        inds_flat = []
        for i in range(len(self.states)):
            inds_flat.append(torch.flatten(inds_mesh[i]))
        states_ind = torch.stack(inds_flat,1)
        # Get the current states values
        states = torch.zeros(states_ind.shape)
        for i in range(len(self.states)):
            states[:,i] = self.states[i].values[states_ind[:,i].long()]
        # Choose action using epsilon-greedy policy
        actions_ind = self.get_actions_epsilon_greedy_nn(states, epsilon)
        # Get the actions values
        actions = self.actions_values[0][actions_ind[:,0]].unsqueeze(-1)
        # Get the next_states
        ns_ind = self.next_states(states_ind, actions_ind).long()
        # Get the next states values
        next_states = torch.zeros(ns_ind.shape)
        for i in range(len(self.states)):
            next_states[:,i] = self.states[i].values[ns_ind[:,i].long()]
        # Get the rewards
        rewards = self.rewards(states, actions)
        # Return the experience (indices)
        return [states, actions, rewards, next_states]

    def get_Q_states(self, states):
        Na = self.actions[0].length
        states_all = states.repeat_interleave(repeats=Na, dim=0)
        actions_all = self.actions[0].values.repeat(len(states)).unsqueeze(-1)
        sa = torch.cat((states_all, actions_all), -1)
        # Choose action using epsilon-greedy policy
        ## Find the action following current Q policy
        return self.Q(sa).view((len(states), Na)).detach()

    def get_Q_max_indices(self, states):
        Q_states = self.get_Q_states(states)
        return Q_states.max(1).indices

    def get_Q_max_values(self, states):
        Q_states = self.get_Q_states(states)
        return Q_states.max(1).values

    def get_actions_epsilon_greedy_nn(self, states, epsilon):
        # Add all the possible actions
        # Na = self.actions[0].length
        # states_all = states.repeat_interleave(repeats=Na, dim=0)
        # actions_all = self.actions[0].values.repeat(len(states)).unsqueeze(-1)
        # sa = torch.cat((states_all, actions_all), -1)
        # # Choose action using epsilon-greedy policy
        # ## Find the action following current Q policy
        # actions_ind = self.Q(sa).view((len(states), Na)).detach().max(1).indices
        actions_ind = self.get_Q_max_indices(states)
        # Select the random action (proba epsilon)
        Na = self.actions[0].length
        d = Bernoulli(epsilon)
        random_ind = d.sample((states.shape[0],))
        da = Categorical(torch.ones(self.actions[0].indices.shape)/Na)
        random_actions_ind = da.sample((random_ind.sum().int().item(),))
        actions_ind[random_ind.bool()] = random_actions_ind
        return actions_ind.unsqueeze(-1).long()

    def iterate_Q_function(self, experience, criterion, optimizer, alpha):
        # The experience is a list of states and actions tensors
        # The shape should be (N,states_dim) and (N,actions_dim)
        states, actions, rewards, next_states  = experience
        # Get the current Q values
        sa = torch.cat((states, actions), -1)
        #Q_current = self.Q(states).gather(1,actions)
        Q_current = self.Q(sa)
        # Get the max of next state Q values
        max_Q_ns = self.get_Q_max_values(next_states).unsqueeze(-1)
        # # Get the next states Q values (shape (states,action))
        # Q_ns = self.Q(next_states).detach() # Don't backpropagate
        # ## Get the max next state Q values
        # max_Q_ns = Q_ns.max(1).values.view(actions.shape)
        # Compute the target Q value
        #Q_target = (1-alpha)*Q_current + alpha*(rewards.unsqueeze(-1) + self.discount_rate * max_Q_ns)
        Q_target = rewards.unsqueeze(-1) + self.discount_rate * max_Q_ns
        # Update (train) the neural network
        loss = criterion(Q_current, Q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # DEBUG #
        # # Current layers weights
        # params = list(self.Q.parameters())
        # w_old = params[0]

        # ## Compute the loss
        # # Output (current): Q_current
        # # Target (new Q): Q_target
        # loss = criterion(Q_current, Q_target)
        # # Backward propagation
        # optimizer.zero_grad()
        # params = list(self.Q.parameters())
        # grad_zero = params[0].grad

        # loss.backward()

        # ## DEBUG ##

        # # Gradients 
        # params = list(self.Q.parameters())
        # grad = params[0].grad

        # optimizer.step()

        # params = list(self.Q.parameters())
        # w_new = params[0]

        # # Check new loss
        # Q_new = self.Q(sa)
        # loss_new = criterion(Q_new, Q_target)

    def train_agent(self):
        ##### Define the hyperparameters #####
        # Define the learning rate
        alpha = 1e-2
        epsilon = 1e-3
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        #optimizer = optim.SGD(self.Q.parameters(), lr=1e-4)
        optimizer = optim.Adam(self.Q.parameters(), lr=1e-3)

        # Only needed for the simulations
        #self.update_reward()
        #self.update_next_states()
        ##### Training #####
        # Iterate the Q function
        for i in range(10000):
            # Get experience
            #if i % 100 == 0:
            Q_old = self.get_Q_grid_nn()
            Q_pi_old, Q_pi_ind_old = self.get_Q_policy_nn()
            #experience = self.get_experience()
            experience = self.get_experience_greedy(epsilon=epsilon)
            self.iterate_Q_function(experience, criterion, optimizer, alpha)
            # Compute distance
            #if i % 100 == 0:
            dist = torch.sum((self.get_Q_grid_nn() - Q_old)**2).item()
            Q_pi, Q_pi_ind = self.get_Q_policy_nn()
            dist_pi = torch.sum((Q_pi - Q_pi_old)**2).item()
            print("Dist Q: {:.1e} Dist pi: {:.1e} Epsilon: {:.1e}".format(dist, dist_pi, epsilon))

    def learn_Q(self):
        inds = []
        vals = []
        for s in self.states:
            inds.append(s.indices)
            vals.append(s.values)
        # Temp
        inds.append(self.actions[0].indices)
        vals.append(self.actions[0].values)

        inds_mesh = torch.meshgrid(inds)
        vals_mesh = torch.meshgrid(vals)
        inds_flat = []
        vals_flat = []
        for i in range(len(self.states)):
            inds_flat.append(torch.flatten(inds_mesh[i]))
            vals_flat.append(torch.flatten(vals_mesh[i]))
        # Temp
        inds_flat.append(torch.flatten(inds_mesh[len(self.states)]))
        vals_flat.append(torch.flatten(vals_mesh[len(self.states)]))


        states_ind = torch.stack(inds_flat,1)
        states = torch.stack(vals_flat,1)

        # Temp
        sa_inds = torch.stack(inds_flat,1)
        sa = torch.stack(vals_flat,1)

        Q_target = self.Q_grid[inds_flat]

        # Temp
        Q_target = self.Q_grid[inds_flat].unsqueeze(-1)

        criterion = nn.MSELoss()
        #criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(self.Q.parameters(), lr=1e-2)

        # Get policy
        #Q_pi, Q_pi_indices = self.get_Q_policy_nn()


        for i in range(1000):
            #Q_current = self.Q(states)
            Q_current = self.Q(sa) # Temp
            loss = criterion(Q_current, Q_target)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Current layers weights
        params = list(self.Q.parameters())
        w_old = params[0]

        ## Compute the loss
        # Output (current): Q_current
        # Target (new Q): Q_target
        loss = criterion(Q_current, Q_target)
        # Backward propagation
        optimizer.zero_grad()
        params = list(self.Q.parameters())
        grad_zero = params[0].grad

        loss.backward()

        ## DEBUG ##

        # Gradients 
        params = list(self.Q.parameters())
        grad = params[0].grad

        optimizer.step()

        params = list(self.Q.parameters())
        w_new = params[0]

        # Check new loss
        Q_new = self.Q(states)
        loss_new = criterion(Q_new, Q_target)


##### Sarsa with NN (NOT READY) #####


    def iterate_Q_function_sarsa(self, experience, criterion, optimizer, alpha):
        # The experience is a list of states and actions tensors
        # The shape should be (N,states_dim) and (N,actions_dim)
        states, actions, rewards, next_states  = experience
        # Get the current Q values
        Q_current = self.Q(states).gather(1,actions)
        # Get the next states Q values (shape (states,action))
        Q_ns = self.Q(next_states).detach() # Don't backpropagate
        ## Get the max next state Q values
        max_Q_ns = Q_ns.max(1).values.view(actions.shape)
        # Compute the target Q value
        Q_target = (1-alpha)*Q_current + alpha*(rewards.unsqueeze(-1) + self.discount_rate * max_Q_ns)
        #Q_target = r + self.discount_rate * max_Q_ns
        # Update (train) the neural network
        ## Compute the loss
        # Output (current): Q_current
        # Target (new Q): Q_target
        loss = criterion(Q_current, Q_target)
        # Backward propagation
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    def train_agent_sarsa(self):
        ##### Define the hyperparameters #####
        # Define the learning rate
        alpha = 1e-2
        epsilon = 1e-3
        # Define the loss function and optimizer
        #criterion = nn.MSELoss()
        criterion = nn.MSELoss()
        optimizer = optim.SDG(self.Q.parameters(), lr=.01)

        # Only needed for the simulations
        self.update_reward()
        self.update_next_states()
        ##### Training #####
        # Iterate the Q function
        for i in range(10000):
            # Get experience
            Q_old = self.get_Q_grid_nn()
            #experience = self.get_experience()
            experience = self.get_experience_greedy(epsilon=epsilon)
            self.iterate_Q_function_sarsa(experience, criterion, optimizer, alpha)
            # Compute distance
            if i % 100 == 0:
                dist = torch.sum((self.get_Q_grid_nn() - Q_old)**2).item()
                print(str(i)+" Dist: "+str(dist)+" Epsilon: "+str(epsilon))



##### Common function NN #####


    def get_Q_policy_nn(self):
        inds = []
        for i in range(len(self.states)):
            inds.append(self.states[i].indices)
        inds_mesh = torch.meshgrid(inds)
        inds_flat = []
        for i in range(len(self.states)):
            inds_flat.append(torch.flatten(inds_mesh[i]))
        # Find the action following current Q policy
        actions_ind = self.get_Q_grid_nn()[inds_flat].max(1).indices
        actions = self.actions[0].values[actions_ind].view(self.states_shape)
        return actions, actions_ind.view(self.states_shape)

    def get_experience(self):
        # Get the full experience for now (get all possible (state, action))
        # Experience is a succession of [state, action, reward, next_state]
        ## Get all the indices
        inds = []
        for i in range(len(self.states)):
            inds.append(self.states[i].indices)
        for i in range(len(self.actions)):
            inds.append(self.actions[i].indices)
        # Create meshgrids
        inds_mesh = torch.meshgrid(inds)
        # Flatten the grids
        inds_flat = []
        for i in range(len(inds_mesh)):
            inds_flat.append(torch.flatten(inds_mesh[i]))
        # Create the experience states
        Ns = len(self.states)
        states_ind = torch.stack(inds_flat[:Ns],1)
        # Get the current states values
        states = torch.zeros(states_ind.shape)
        for i in range(len(self.states)):
            states[:,i] = self.states[i].values[states_ind[:,i].long()]
        # Create the experience actions
        actions = torch.stack(inds_flat[Ns:],1)
        # Get the rewards
        rewards = self.rew[inds_flat].view(actions.shape)
        # Get the next_states
        ns_ind = self.next_states(states_ind, actions)
        # Get the next states values
        next_states = torch.zeros(ns_ind.shape)
        for i in range(len(self.states)):
            next_states[:,i] = self.states[i].values[ns_ind[:,i].long()]
        # Return the experience
        return [states, actions, rewards, next_states]

    def get_Q_grid_nn(self):
        vals = []
        for s in self.states:
            vals.append(s.values)
        # Temp
        vals.append(self.actions[0].values)
        vals_mesh = torch.meshgrid(vals)

        vals_flat = []
        for i in range(len(self.states)):
            vals_flat.append(torch.flatten(vals_mesh[i]))
        vals_flat.append(torch.flatten(vals_mesh[len(self.states)]))

        states_val = torch.stack(vals_flat, -1)

        Q = self.Q(states_val).detach().view(vals_mesh[0].shape)
        return Q

    def get_argmax_Q_grid(self, Qg, states_ind):
        q_inds = list(states_ind.transpose(1,0))
        return Qg[q_inds].max(1).indices

    def iterate_Q_grid_qlearning(self, experience, alpha):

        states_ind, actions, rewards, ns_ind = experience
        # Get the current Q value indices
        q_inds = list(states_ind.transpose(1,0))
        q_inds.append(actions[:,0])

        # Randomly deide to update Qa or Qb (double Q-learning)
        if np.random.binomial(1,.5):
        #if self.update_Qa:
            # Update Qa
            # Get optimal next action from Qa but uses Qb's value
            Q_update = self.Qa_grid
            Q_eval = self.Qb_grid
            self.update_Qa = False
        else:
            # Update Qb
            # Get optimal next action from Qb but uses Qa's value
            Q_update = self.Qb_grid
            Q_eval = self.Qa_grid
            self.update_Qa = True

        #import ipdb; ipdb.set_trace();

        # Get current Q values
        Q_current = Q_update[q_inds]
        # Get the next state best action
        nsa_ind = self.get_argmax_Q_grid(Q_update, states_ind)
        q_nsa_inds = list(ns_ind.transpose(1,0)) + [nsa_ind]
        # Evaluate the state-action
        max_Q_ns = Q_eval[q_nsa_inds]
        # Compute the target Q value
        Q_target = (1-alpha)*Q_current + alpha*(rewards + self.discount_rate * max_Q_ns)

        Q_update[q_inds] = Q_target
        ## Update the Q grid
        #if self.update_Qa:
        #    self.Qa_grid[q_inds] = Q_target
        #    #self.Qb_grid[q_inds] = Q_target
        #    self.update_Qa = False
        #else:
        #    #self.Qa_grid[q_inds] = Q_target
        #    self.Qb_grid[q_inds] = Q_target
        #    self.update_Qa = True

    def train_Q_grid_qlearning(self, decay=1e-3):
        # Implement Q-learning algorithm
        alpha = 1e-5
        # Update next_states and rewards
        #self.update_reward()
        #self.update_next_states()
        # Iterate the Q function
        epsilon = 1e-3
        self.update_Qa = True
        for i in range(10000):
            self.update_Q_grid()
            #print("Epsilon: "+str(epsilon))
            # Get current Q
            if i % 100 == 0:
                Q_old = self.Q_grid.clone()
                Q_pi_old, Q_pi_ind_old = self.get_Q_policy()
            # Get experience
            experience = self.get_experience_indices_greedy(1000, epsilon)
            self.iterate_Q_grid_qlearning(experience, alpha)
            epsilon *= (1-decay)
            self.update_Q_grid()
            # Compute distance
            if i % 100 == 0:
                dist = torch.sum((self.Q_grid - Q_old)**2).item()
                Q_pi, Q_pi_ind = self.get_Q_policy()
                dist_pi = torch.sum((Q_pi - Q_pi_old)**2).item()
                print("Dist Q: {:.1e} Dist pi: {:.1e} Epsilon: {:.1e}".format(dist, dist_pi, epsilon))

    def get_Q_policy(self):
        inds = []
        for i in range(len(self.states)):
            inds.append(self.states[i].indices)
        inds_mesh = torch.meshgrid(inds)
        inds_flat = []
        for i in range(len(self.states)):
            inds_flat.append(torch.flatten(inds_mesh[i]))
        # Find the action following current Q policy
        actions_ind = self.Q_grid[inds_flat].max(1).indices
        actions = self.actions[0].values[actions_ind].view(self.states_shape)
        return actions, actions_ind.view(self.states_shape)

    def get_experience_indices(self):
        ##### NOT USED #####
        # Get the full experience for now (get all possible (state, action))
        # Experience is a succession of [state, action, reward, next_state]
        ## Get all the indices
        inds = []
        for i in range(len(self.states)):
            inds.append(self.states[i].indices)
        for i in range(len(self.actions)):
            inds.append(self.actions[i].indices)
        # Create meshgrids
        inds_mesh = torch.meshgrid(inds)
        # Flatten the grids
        inds_flat = []
        for i in range(len(inds_mesh)):
            inds_flat.append(torch.flatten(inds_mesh[i]))
        # Create the experience states
        Ns = len(self.states)
        states_ind = torch.stack(inds_flat[:Ns],1)
        # Create the experience actions
        actions = torch.stack(inds_flat[Ns:],1)
        # Get the rewards
        rewards = self.rew[inds_flat]
        # Get the next_states
        ns_ind = self.next_states(states_ind, actions)
        # Return the experience
        return [states_ind, actions, rewards, ns_ind]

    def get_experience_indices_greedy(self, N, epsilon):
        # Create the experience states
        ##### Draw N states #####
        inds_flat = []
        vals_flat = []
        for s in self.states:
            d = Categorical(torch.ones(s.length)/s.length)
            s_ind = d.sample((N,))
            s_val = s.values[s_ind]
            inds_flat.append(s_ind)
            vals_flat.append(s_val)
        ##### Use all possible states #####
        #inds = []
        #vals = []
        #for s in self.states:
        #    inds.append(s.indices)
        #    vals.append(s.values)
        #inds_mesh = torch.meshgrid(inds)
        #vals_mesh = torch.meshgrid(vals)
        #inds_flat = []
        #vals_flat = []
        #for i in range(len(self.states)):
        #    inds_flat.append(torch.flatten(inds_mesh[i]))
        #    vals_flat.append(torch.flatten(vals_mesh[i]))
        ##### Construct the states #####
        states_ind = torch.stack(inds_flat,1)
        states = torch.stack(vals_flat,1)
        # Implement an epsilon-greedy policy
        # Choose action using epsilon-greedy policy
        actions_ind = self.get_actions_epsilon_greedy(states_ind, epsilon)
        actions = self.actions[0].values[actions_ind[:,0]].unsqueeze(-1)
        # Get the rewards
        #inds_flat.append(actions[:,0])
        #rewards = self.rew[inds_flat]
        rewards = self.rewards(states, actions)
        # Get the next_states
        ns_ind = self.next_states(states_ind, actions_ind).long()
        # Return the experience (indices)
        return [states_ind, actions_ind, rewards, ns_ind]

    def get_actions_epsilon_greedy(self, states_ind, epsilon):
        # Choose action using epsilon-greedy policy
        ## Find the action following current Q policy
        actions = self.Q_grid[list(states_ind.transpose(1, 0))].max(1).indices
        # Select the random action (proba epsilon)
        d = Bernoulli(epsilon)
        random_ind = d.sample((states_ind.shape[0],))
        Na = self.actions[0].length
        da = Categorical(torch.ones(self.actions[0].indices.shape)/Na)
        random_actions_ind = da.sample((random_ind.sum().int().item(),))
        actions[random_ind.bool()] = random_actions_ind
        return actions.unsqueeze(-1).long()

    def update_Q_grid(self):
        # Create a sum or average Q grid for double Q_learning
        # Used for action choice and for final Q function
        self.Q_grid = (self.Qa_grid + self.Qb_grid) / 2

    def iterate_Q_grid_sarsa(self, experience, alpha, epsilon):
        states_ind, actions, rewards, ns_ind = experience
        # Get the current Q value
        q_inds = list(states_ind.transpose(1,0))
        q_inds.append(actions[:,0])
        Q_current = self.Q_grid[q_inds]
        # Get the next Q_value following an epsilon greedy policy
        actions_ns = self.get_actions_epsilon_greedy(ns_ind, epsilon)
        q_ns_inds = list(ns_ind.transpose(1,0))
        q_ns_inds.append(actions_ns[:,0])
        Q_ns = self.Q_grid[q_ns_inds]

        # Compute the new Q value
        Q_new = (1-alpha)*Q_current + alpha*(rewards + self.discount_rate*Q_ns)
        # Update V
        self.Q_grid[q_inds] = Q_new
        
    def train_Q_grid_sarsa(self, decay=1e-4):
        # Implement Sarsa algorithm
        alpha = 1e-2 
        # Update next_states and rewards
        self.update_reward()
        self.update_next_states()
        # Iterate the Q function
        epsilon = 1e-3
        dist = 10
        for i in range(10000):
            #print("Epsilon: "+str(epsilon))
            # Get current Q
            Q_old = self.Q_grid.clone()
            # Get experience
            experience = self.get_experience_indices_greedy(epsilon=epsilon)
            self.iterate_Q_grid_sarsa(experience, alpha, epsilon=epsilon)
            #epsilon *= (1-decay)
            # Compute distance
            if i % 100 == 0:
                dist = torch.sum((self.Q_grid - Q_old)**2).item()
                print(str(i)+" Dist: "+str(dist)+" Epsilon: "+str(epsilon))

        

########## Simulation Functions ##########

    def simulate(self, N, initial_state, pi_indices):
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
        pi_ind += [pi_indices[tuple(s_ind)]]


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
            pi_ind += [pi_indices[tuple(s_ind)]]

        return (sim_states, sim_pi)

class _Q_network(nn.Module):
    r"""
    Class representing the value function as a Neural Network.
    """

    def __init__(self, agent):
        super(_Q_network, self).__init__()
        # Input: number of state dimensions
        # Output: number of actions
        # Determine the input dimension
        n_inputs = len(agent.states)
        # Determine the output dimension
        if len(agent.actions) > 1:
            raise Error("Multidimension action space is not yet supported")
        else:
            n_outputs = agent.actions[0].length
        # # Temp
        n_inputs = len(agent.states) + 1 + 1
        n_outputs = 1
        self.agent = agent
        self.a_vals = self.agent.actions[0].values
        # For now, uses arbitrary mid-layer size
        layer_params = 5
        self.input_layer = nn.Linear(n_inputs, layer_params)
        # Mid Layer
        self.m0 = nn.Linear(layer_params, layer_params)
        self.m1 = nn.Linear(layer_params, layer_params)
        self.m2 = nn.Linear(layer_params, layer_params)
        self.m3 = nn.Linear(layer_params, layer_params)
        self.m4 = nn.Linear(layer_params, layer_params)
        #self.m5 = nn.Linear(layer_params, layer_params)
        #self.m6 = nn.Linear(layer_params, layer_params)
        #self.m7 = nn.Linear(layer_params, layer_params)
        #self.m8 = nn.Linear(layer_params, layer_params)
        # Output layer
        self.output_layer = nn.Linear(layer_params, n_outputs)
        # Initialization
        self.input_layer.weight.data.normal_(1e-10, 0.1)   # initialization
        self.m0.weight.data.normal_(0, 0.1)   # initialization
        self.m1.weight.data.normal_(0, 0.1)   # initialization
        self.m2.weight.data.normal_(0, 0.1)   # initialization
        self.m3.weight.data.normal_(0, 0.1)   # initialization
        self.m4.weight.data.normal_(0, 0.1)   # initialization
        self.output_layer.weight.data.normal_(1e-10, 0.1)   # initialization

    def normalize_input(self, x):
        for i in range(len(self.agent.states)):
            m = self.agent.states[i].values.min().item()
            d = self.agent.states[i].values.max().item() - m
            if d==0: # All same values
                x[:i] = 0
            else:
                x[:,i] = (1/d)*(x[:,i]-m)
        m = self.agent.actions[0].values.min().item()
        d = self.agent.actions[0].values.max().item() - m
        x[:,-1] = (1/d)*(x[:,-1]-m)
        return x


    def forward(self, x):
        # Normalize the inputs
        #y = self.normalize_input(x)
        #y = x

        # # Add the reward (x should then be trying to learn the continuation value)
        states = x[:,:-1]
        actions = x[:,-1].unsqueeze(-1)
        rewards = self.agent.rewards(states, actions).unsqueeze(-1)

        # Normalize inputs
        #x = self.normalize_input(x)

        y = torch.cat((x, rewards), -1)


        y = F.relu(self.input_layer(y))
        y = F.relu(self.m0(y))
        y = F.relu(self.m1(y))
        y = F.relu(self.m2(y))
        y = F.relu(self.m3(y))
        y = F.relu(self.m4(y))
        #x = F.relu(self.m5(x))
        #x = F.relu(self.m6(x))
        #x = F.relu(self.m7(x))
        #x = F.relu(self.m8(x))
        #y = torch.cat((y, rewards), -1)
        y = self.output_layer(y)
        return y







