import numpy as np
from scipy.stats import norm
import casadi as ca

class Market:

    seed = 1133

    def generate_spotprice(self, N):
        '''
        Returns an array of spot prices for each quarter hour. Number of quarter hours amounts to N. 
        N = 96 -> List of spot prices for a whole 24h period. 
        N = 94 -> 2 first QH are cut off. Assuming that the mpc ends on a whole 

        This is in kr/Kw. In order to work with mwh, you must factor that in :)
        '''


        sigma=0.10

        hours = int(np.ceil(N/4))
        timesteps = np.linspace(0, hours-1, hours)
        timesteps = np.repeat(timesteps, 4)
        
        prices_mean = 0.7 - 0.2*np.cos(np.pi * timesteps/6) - 0.15*np.cos(np.pi * timesteps/12) 


        # Generate 24 random values from a normal distribution
        np.random.seed(self.seed)  # You can choose any integer value for the seed
        spot_prices = prices_mean + np.repeat(np.random.normal(loc=0, scale=sigma, size=hours), 4)

        # Ensure that no values are below 0
        spot_prices = np.clip(spot_prices, a_min=0, a_max=None)
        
        # Repeat each value 4 times to get a total of 96 elements
        return spot_prices[(hours*4)-N:]
    

    def Pr_a_dn(self, Bc_dn):
        
        mu_dn = 50
        sigma_dn = 10
        
        Bc_dn_norm = (Bc_dn - mu_dn)/sigma_dn

        # return norm.cdf(-Bc_dn_norm)
        return (1.0 + ca.erf(-Bc_dn_norm / ca.sqrt(2.0))) / 2.0

    def Pr_a_up(self, Bc_up):
        
        mu_up = 50
        sigma_up = 10
        
        Bc_up_norm = (Bc_up - mu_up)/sigma_up

        # return norm.cdf(-Bc_up_norm)
        return (1.0 + ca.erf(-Bc_up_norm / ca.sqrt(2.0))) / 2.0


