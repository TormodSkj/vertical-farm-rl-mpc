import numpy as np
from market import Market, Bid
from model import *
from config import Config
from controller import Controller
from globals import *
import time
from utils import *


class Simulator():
    
    model: PlantModel
    market: Market
    config: Config
    mpc_controller: Controller

    N: float
    T: float
    dt: float

    x_mpc: np.array
    u_mpc: np.array
    bids_mpc: np.array


    def __init__(self, timehorizon, plantmodel, market, config, mpc_controller):
        self.T = timehorizon
        self.N = timehorizon * QUARTER_HOURS_PER_DAY
        self.model = plantmodel  
        self.market = market
        self.config = config
        self.mpc_controller = mpc_controller
        self.dt = self.mpc_controller.dt
        self.t = np.linspace(0, self.T, self.N)

        self.bids_mpc = np.zeros((4, self.N))
        


    def simulate_random_activation(self, controller: Controller, m: int):
        '''
        Simulate m number of cases where different activation demands and clearing prices are chosen randomly
        Results in a series of fresh-weights. 
        This is to see the spread in growth from activation uncertainty
        '''

        # Generate activation demands
        # Generate clearing prices
        # 
        # Filter out accepted bids
        # Simulate the plant now with only the accepted bids
        market = controller.market
        N = controller.N
        dt = controller.dt
        seed = None     # Set to None for random 

        u_base = controller.runs['runs']['Baseline']['timeseries']['u']

        assert 'Bidding' in controller.runs['runs'], 'Unable to perform random bid activations due to lack of bidding data'
        
        # Get bidding data
        bidding_vol_up = controller.runs['runs']['Bidding']['timeseries']['P_up']
        bidding_vol_dn = controller.runs['runs']['Bidding']['timeseries']['P_dn']
        bidding_price_up = controller.runs['runs']['Bidding']['timeseries']['C_up']
        bidding_price_dn = controller.runs['runs']['Bidding']['timeseries']['C_dn']

        # Preallocation
        freshweights = np.zeros((m, N))

        for case in range(m):

            activation_demands = controller.market.generate_activation_demands(N, seed)
            # clearing_prices_up = generate_samples_from_cdf(controller.market.mfrr_prices_up[:N], N, seed)  #TODO REMOVE check if we get better performance if we use january-data  
            # clearing_prices_dn = generate_samples_from_cdf(controller.market.mfrr_prices_dn[:N], N, seed)   #TODO REMOVE

            mu_up = conditional_expectation(controller.p_spot, market.price_means, market.price_cov)[0]
            mu_dn = conditional_expectation(controller.p_spot, market.price_means, market.price_cov)[1]

            clearing_prices_up = np.random.normal(loc=mu_up, scale=market.sigma_up)
            clearing_prices_dn = np.random.normal(loc=mu_dn, scale=market.sigma_dn)

            u = u_base + 1000*(np.where(np.logical_and(activation_demands == -1, bidding_price_dn < clearing_prices_dn), bidding_vol_dn, 0)\
                             - np.where(np.logical_and(activation_demands == 1, bidding_price_up < clearing_prices_up), bidding_vol_up, 0))/controller.model.C_conv_PPFD

            X = np.zeros((controller.model.nx, N+1))
            X[:,0] = controller.model.x_init.flatten()
            
            for k in range(N):
                #Forward euler
                # X[:,k+1] = X[:,k] + dt*np.array(controller.model.derivative(X[:,k], np.array([u[k]]))).reshape(1, -1)
                X[:,k+1] = np.array(controller.model.casadi_function()(X[:,k], np.array([u[k]]))).reshape(1, -1)

            fw = controller.model.freshweight(X)
        
            freshweights[case,:] = np.array(fw[1:]).flatten()

        return freshweights


    






    def Simulate_mpc(self):

        # TODO let's get to work

        X = np.zeros((self.model.nx, self.N+1))
        X[:,0] = self.model.x_init

        U = np.zeros((self.model.nu, self.N))

        # Init bid list with two empty bids
        Bids = [Bid(), Bid()]

        bidding_z_opt = self.mpc_controller.bidding_z_init
        baseline_z_opt = self.mpc_controller.baseline_z_init
        self.mpc_controller.surpress_output = True

        for k in range(self.N-1):
        # for k in range(4):     # TODO remove 

            iter_starttime = time.time()

            # Idk why i do this, but i have a feeling it's right B-)
            TH = self.N - k
            self.mpc_controller.N = TH

            # Init controller with current state
            self.mpc_controller.set_bids(Bids[k], Bids[k+1])
            self.mpc_controller.x_init = X[:,k]
            

            # Decide next bids
            self.mpc_controller.optimize_baseline()
            self.mpc_controller.optimize_bidding()

            next_bid = self.mpc_controller.B_bid[:,0].flatten()
            Bids.append(Bid(next_bid[0], next_bid[1], next_bid[2], next_bid[3]))
            u0 = np.array(self.mpc_controller.u_bid)[0]
            
            bidding_z_opt = np.array(self.mpc_controller.sol_bid['x'])
            baseline_z_opt = np.array(self.mpc_controller.sol_base['x'])
            
            # Give optimizer a more optimal starting point next iteration
            self.mpc_controller.bidding_z_init = np.vstack((bidding_z_opt[2:TH], bidding_z_opt[TH+4:]))     # Skip first state and first bid 
            self.mpc_controller.baseline_z_init = np.vstack((baseline_z_opt[2:TH], baseline_z_opt[TH+1:]))  # Skip first state and first u

            # Grow the plant
            U[:,k] = u0
            X[:,k+1] = X[:,k] + self.dt*np.array(self.model.derivative(X[:,k], u0)).flatten()

            # Simulate market response
            self.mpc_controller.A_up = 1
            self.mpc_controller.A_down = 0

            # Move spot price one step forwards in time
            self.mpc_controller.p_spot = self.mpc_controller.p_spot[1:]

            # Status update
            iter_time = time.time()-iter_starttime
            minutes, seconds = divmod(iter_time, 60)
            print(f"Completed iteration {k} of {self.N-2} in: {int(minutes)} minutes and {seconds:.2f} seconds.")


        self.x_mpc = X
        self.u_mpc = U

        for i in range(self.N):
            self.bids_mpc[:, i] = Bids[i].as_array()



        return 0
            



            





        