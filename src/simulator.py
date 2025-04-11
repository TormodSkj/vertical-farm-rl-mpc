import numpy as np
from market import Market
from bid import Bid
from model import *
from config import Config
from settings import Settings
from controller import Controller
from globals import *
import time
from utils import *


class Simulator():
    
    settings:   Settings
    model:      PlantModel
    market:     Market
    config:     Config
    controller: Controller

    N: float
    T: float
    dt: float

    x_mpc: np.array
    u_mpc: np.array
    bids_mpc: np.array


    def __init__(self, settings: Settings, plantmodel: PlantModel, market: Market, config: Config, controller: Controller):
        
        self.settings = settings
        self.sim_settings = settings.get_settings_group('general')
        
        self.T          = self.sim_settings['SIMULATION_LENGTH']
        self.N          = self.sim_settings['SIM_N_TIMESTEPS']
        self.dt         = self.sim_settings['SIM_TIMEDELTA']
        
        self.model      = plantmodel  
        self.market     = market
        self.config     = config
        self.controller = controller

        self.t          = np.linspace(0, self.T, self.N)
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
        F = controller.model.casadi_function()

        
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

            activation_demands = controller.market.generate_activation_demands('Activation Market', N, seed)

            mu_up = conditional_expectation(controller.spot_prices, market.price_means, market.price_covs)[0]
            mu_dn = conditional_expectation(controller.spot_prices, market.price_means, market.price_covs)[1]

            clearing_prices_up = np.random.normal(loc=mu_up, scale=market.sigma_AM_up)
            clearing_prices_dn = np.random.normal(loc=mu_dn, scale=market.sigma_AM_down)

            u = u_nom + 1000*(np.where(np.logical_and(activation_demands == -1, bidding_price_dn < clearing_prices_dn), bidding_vol_dn, 0)\
                            - np.where(np.logical_and(activation_demands == 1, bidding_price_up < clearing_prices_up), bidding_vol_up, 0))/controller.model.C_conv_PPFD

            X = np.zeros((controller.model.nx, N+1))
            X[:,0] = controller.model.x_init.flatten()
            
            for k in range(N):
                X[:,k+1] = np.array(F(X[:,k], np.array([u[k]]))).reshape(1, -1)

            fw = controller.model.freshweight(X)
        
            freshweights[case,:] = np.array(fw[1:]).flatten()

        return freshweights


    def apply_mfrr_clearing_prices(self, run_id, refrun_id, plot_run=False):
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
        F = controller.F

        refrun = controller.optimization_results['runs'][refrun_id]
        # assert refrun_id in controller.optimization_results['runs'], f"{run_id}| Error: {refrun_id} not in run data"
        # refrun = controller.optimization_results['runs'][refrun_id]
        U_nom = refrun['timeseries']['u_nom'].reshape(1,-1)
        U = U_nom   # Will be overwritten if there are AM bids

        # for market_type, market_data in run['Markets'].items():
        #     balancing_market = self.market.get_balancing_market(market_type)

        #     clearing_prices_up, clearing_prices_down = balancing_market.get_clearing_prices()
        #     assert len(clearing_prices_up)==N and len(clearing_prices_down)==N, f'Clearing price arrays have inconsistent lengths with simulation duration. N = {self.N}, len(clearing prices up) = {len(clearing_prices_up)}, len(clearing prices down) = {len(clearing_prices_down)}'
            
        #     activation_demands_up, activation_demands_down = balancing_market.get_activations()
        #     assert len(activation_demands_down)==N and len(activation_demands_up)==N, f'Activation demand arrays have inconsistent lengths with simulation duration. N = {self.N}, len(demands up) = {len(activation_demands_up)}, len(demands down) = {len(activation_demands_down)}'

        #     # Get bidding data
        #     bidding_vol_up      = market_data['Bids']['Up']['Volume'].reshape(1,-1)
        #     bidding_vol_down    = market_data['Bids']['Down']['Volume'].reshape(1,-1)
        #     bidding_price_up    = market_data['Bids']['Up']['Price'].reshape(1,-1)
        #     bidding_price_down  = market_data['Bids']['Down']['Price'].reshape(1,-1)

        #     activated_volumes, earnings = balancing_market.subject_bids_to_market_data()
        #     activated_volumes_up        = activated_volumes[0,:]
        #     activated_volumes_down      = activated_volumes[1,:]

        #     P_tilde = activated_volumes_down - activated_volumes_up


        market_data = refrun['markets']
        
        market_earnings = {}

        for market_type in market_data:

            balancing_market = self.market.get_balancing_market(market_type)

            bid_volumes_up    = market_data[market_type]['Bids']['Up']['Volume'].reshape(1,-1)
            bid_volumes_down  = market_data[market_type]['Bids']['Down']['Volume'].reshape(1,-1)
            bid_prices_up     = market_data[market_type]['Bids']['Up']['Price'].reshape(1,-1)
            bid_prices_down   = market_data[market_type]['Bids']['Down']['Volume'].reshape(1,-1)
            
            bid_volumes = np.vstack((bid_volumes_up,
                                     bid_volumes_down))
            bid_prices  = np.vstack((bid_prices_up,
                                     bid_prices_down))

            activations, activated_volumes, balancing_market_earnings = balancing_market.subject_bids_to_market_data(self.market.MTU_start, bid_volumes, bid_prices)
            activated_volumes_up   = activated_volumes[0,:]
            activated_volumes_down = activated_volumes[1,:]
            market_earnings[market_type] = np.sum(balancing_market_earnings)

            activations_up   = activations[0,:]
            activations_down = activations[1,:]

            bid_activations_up, bid_activations_down = balancing_market.get_activations()
            market_data[market_type]['Activations'] = {
                'Up'    : activations_up,
                'Down'  : activations_down
            }

            if balancing_market.market_type == 'Activation Market':
                P_tilde = activated_volumes_down - activated_volumes_up
                U_tilde = 1000/controller.model.C_conv_PPFD * P_tilde
                U = U_nom + U_tilde

        # B = np.vstack((bidding_vol_up, bidding_vol_down, bidding_price_up, bidding_price_down))
        
        # # Evaluate activations
        # activation_up   = np.where(np.logical_and(activation_demands_up > 0, bidding_price_up <= clearing_prices_up), 1, 0)
        # activation_down = np.where(np.logical_and(activation_demands_down > 0, bidding_price_down <= clearing_prices_down), 1, 0)
        # A = np.vstack((activation_up, activation_down))


        X = np.zeros((controller.model.nx, N+1))
        X[:,0] = controller.model.x_init.flatten()
        
        for k in range(N):
            X[:,k+1] = np.array(F(X[:,k], np.array([U[:,k]]))).flatten()


        f = 0.25*self.model.C_conv_PPFD/1000 * np.sum(np.multiply(spot_prices, U))\
                  - sum([market_earnings[market_type] for market_type in market_earnings])
        # \
        #     + np.sum(np.where(activation_down == 1, np.multiply((spot_prices - clearing_prices_down),  bidding_vol_down), 0)) \
        #     - np.sum(np.where(activation_up   == 1, np.multiply((spot_prices + clearing_prices_up),    bidding_vol_up), 0)))


        Eps = max(0, controller.model.Final_fw_sht - self.model.freshweight(X[:,-1]))

        sol = {}
        sol['eps'] = Eps
        sol['f'] = f
        sol['elapsed_time'] = time.time() - start_time
    
        dependencies = ()
        refrun_dependencies = tuple(controller.optimization_results['runs'][refrun_id]['dependencies'])
        controller.store_run(run_id, dependencies, sol, X, U, refrun_id = refrun_id, market_data=market_data, plot_run=plot_run)
        # controller.store_run(run_id, refrun_dependencies + dependencies, sol, X, u.reshape(1,-1), A, B, U_nom.reshape(1,-1), refrun_id=refrun_id, balancing_market=self.market.AM, plot_run=plot_run)
        
        return 0


#'''