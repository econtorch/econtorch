# Test File for the DiscreteAgent class

import models.Agent_PostBeliefs as pb


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


# Create the agent
agent = pb.Manager(params)

# Solve the value function
agent.iterate_value_function(1)

# Plot the results

## Simulate a path starting at init_state
init_state = [20,1,0]
N = 100

sim = agent.simulate(N, init_state)

import ipdb; ipdb.set_trace();

agent.plot_simulation(sim)

