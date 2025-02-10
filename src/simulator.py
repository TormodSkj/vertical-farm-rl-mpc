import numpy as np
from market import Market
from bid import Bid
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
    controller: Controller

    N: float
    T: float
    dt: float
    T_TH: float
    N_TH: int
    T_iter: float
    N_iter: int

    x_mpc: np.array
    u_mpc: np.array
    bids_mpc: np.array


    def __init__(self, timehorizon, plantmodel, market: Market, config: Config, controller: Controller, time_horizon = 3, time_iteration = 1):
        self.T = timehorizon
        self.N = timehorizon * QUARTER_HOURS_PER_DAY
        self.model = plantmodel  
        self.market = market
        self.config = config
        self.controller = controller
        self.dt = self.controller.dt
        self.t = np.linspace(0, self.T, self.N)
        self.T_TH = time_horizon
        self.N_TH = int(np.ceil(self.T_TH * SECONDS_PER_QUARTER_HOUR * QUARTER_HOURS_PER_DAY/self.dt))
        self.T_iter = time_iteration
        self.N_iter = int(np.ceil(self.T_iter * SECONDS_PER_QUARTER_HOUR * QUARTER_HOURS_PER_DAY/self.dt))

        self.bids_mpc = np.zeros((4, self.N))
        

    def simulate_random_activation(self, controller: Controller, refrun_id: str, m: int):
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
        seed = None     # Set to None for new random outcome each time
        F = controller.model.casadi_function_fe()

        
        assert refrun_id in controller.optimization_results['runs'], 'Unable to perform random bid activations due to lack of bidding data'
        
        # Extract run data
        refrun              = controller.optimization_results['runs'][refrun_id]
        u_nom               = refrun['timeseries']['u_nom']
        bidding_vol_up      = refrun['timeseries']['P_up']
        bidding_vol_dn      = refrun['timeseries']['P_dn']
        bidding_price_up    = refrun['timeseries']['C_up']
        bidding_price_dn    = refrun['timeseries']['C_dn']

        # Preallocation
        freshweights = np.zeros((m, N))

        for case in range(m):

            activation_demands = controller.market.generate_activation_demands(N, seed)

            mu_up = conditional_expectation(controller.spot_prices, market.price_means, market.price_covs)[0]
            mu_dn = conditional_expectation(controller.spot_prices, market.price_means, market.price_covs)[1]

            clearing_prices_up = np.random.normal(loc=mu_up, scale=market.sigma_up)
            clearing_prices_dn = np.random.normal(loc=mu_dn, scale=market.sigma_dn)

            u = u_nom + 1000*(np.where(np.logical_and(activation_demands == -1, bidding_price_dn < clearing_prices_dn), bidding_vol_dn, 0)\
                            - np.where(np.logical_and(activation_demands == 1, bidding_price_up < clearing_prices_up), bidding_vol_up, 0))/controller.model.C_conv_PPFD

            X = np.zeros((controller.model.nx, N+1))
            X[:,0] = controller.model.x_init.flatten()
            
            for k in range(N):
                X[:,k+1] = np.array(F(X[:,k], np.array([u[k]]))).reshape(1, -1)

            fw = controller.model.freshweight(X)
        
            freshweights[case,:] = np.array(fw[1:]).flatten()

        return freshweights


    def apply_mfrr_clearing_prices(self, run_id, refrun_id):
        '''
        '''
        # Generate activation demands
        # Generate clearing prices
        
        # Filter out accepted bids
        # Simulate the plant now with only the accepted bids

        start_time = time.time()

        controller = self.controller
        market = controller.market
        N = controller.N
        dt = controller.dt
        date = market.date
        spot_prices = controller.spot_prices
        F = controller.model.casadi_function_fe()

        clearing_prices_up, clearing_prices_down = market.get_clearing_prices(date)
        assert len(clearing_prices_up)==N and len(clearing_prices_down)==N, f'Clearing price arrays have inconsistent lengths with simulation duration. N = {self.N}, len(clearing prices up) = {len(clearing_prices_up)}, len(clearing prices down) = {len(clearing_prices_down)}'
        
        activation_demands_up, activation_demands_down = market.mfrr_demands_up, market.mfrr_demands_down
        assert len(activation_demands_down)==N and len(activation_demands_up)==N, f'Activation demand arrays have inconsistent lengths with simulation duration. N = {self.N}, len(demands up) = {len(activation_demands_up)}, len(demands down) = {len(activation_demands_down)}'

        assert refrun_id in controller.optimization_results['runs'], f"{run_id}| Error: {refrun_id} not in run data"
        refrun = controller.optimization_results['runs'][refrun_id]

        # Get bidding data
        U_nom               = refrun['timeseries']['u_nom']
        bidding_vol_up      = refrun['timeseries']['P_up']
        bidding_vol_down    = refrun['timeseries']['P_dn']
        bidding_price_up    = refrun['timeseries']['C_up']
        bidding_price_down  = refrun['timeseries']['C_dn']
        B = np.vstack((bidding_vol_up, bidding_vol_down, bidding_price_up, bidding_price_down))
        
        # Evaluate activations
        activation_up   = np.where(np.logical_and(activation_demands_up > 0, bidding_price_up <= clearing_prices_up), 1, 0)
        activation_down = np.where(np.logical_and(activation_demands_down > 0, bidding_price_down <= clearing_prices_down), 1, 0)
        A = np.vstack((activation_up, activation_down))

        u = U_nom + 1000*(np.where(activation_down == 1, bidding_vol_down, 0)\
                        - np.where(activation_up == 1, bidding_vol_up, 0))/controller.model.C_conv_PPFD

        X = np.zeros((controller.model.nx, N+1))
        X[:,0] = controller.model.x_init.flatten()
        
        for k in range(N):
            X[:,k+1] = np.array(F(X[:,k], np.array([u[k]]))).reshape(1, -1)


        f = 0.25*(self.model.C_conv_PPFD * np.sum(np.multiply(spot_prices,U_nom)) \
            + np.sum(np.where(activation_down == 1, np.multiply((1000*spot_prices - controller.market.C_eur2nok * clearing_prices_down),  bidding_vol_down), 0)) \
            - np.sum(np.where(activation_up == 1,   np.multiply((1000*spot_prices + controller.market.C_eur2nok * clearing_prices_up),    bidding_vol_up), 0)))


        Eps = max(0, controller.model.Final_fw_sht - self.model.freshweight(X[:,-1]))

        sol = {}
        sol['eps'] = Eps
        sol['f'] = f
        sol['elapsed_time'] = time.time() - start_time
    
        controller.save_run(run_id, sol, X, u, A, B, U_nom, refrun_id=refrun_id)
        
        return 0


#'''