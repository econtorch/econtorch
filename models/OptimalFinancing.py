r"""
Optimal Financing Models
Based on "Dynamic Models and Structural Estimation in Corporate Finance"
by Streabulaev and Whited (2012)
"""

from econtorch.base import _Agent


class Agent(_Agent):
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
    def __init__(self, theta, delta, r, rho, sigma):
        pass

    def reward(state, action):
        # Cash Flow function
        
    def 


    # Methods specific to this problem
    # Profit function
    def _prof(state,params):
        k=state['k']
        z=state['z']
        theta=params['theta']
        return z*(k**theta)
    
    # Adjustment Costs
    def _adjCosts(state,controls,params):
        k=state['k']
        I=controls['I']
        phi0=params['phi0']
        phi1=params['phi1']
        fixedCosts=0
        if I!=0:
            fixedCosts=phi1*k
        return phi0*(I**2)/(2*k)+fixedCosts
    
    # Cash Flow function
    def _cashFlow(state,controls,params):
        I=controls['I']
        return prof(state,params)-I-adjCosts(state,controls,params)


