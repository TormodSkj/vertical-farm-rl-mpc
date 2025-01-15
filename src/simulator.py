import numpy as np
from market import Market, Bid
from model import *
from config import Config
from controller import Controller
from globals import *
import time
from utils import *


class Simulator():
    
    model: MpcPlantModel
    market: Market
    config: Config
    mpc_controller: Controller

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


    def __init__(self, timehorizon, plantmodel: MpcPlantModel, market: Market, config: Config, mpc_controller: Controller, time_horizon = 3, time_iteration = 1):
        self.T = timehorizon
        self.N = timehorizon * QUARTER_HOURS_PER_DAY
        self.model = plantmodel  
        self.market = market
        self.config = config
        self.mpc_controller = mpc_controller
        self.dt = self.mpc_controller.dt
        self.t = np.linspace(0, self.T, self.N)
        self.T_TH = time_horizon
        self.N_TH = int(np.ceil(self.T_TH * SECONDS_PER_QUARTER_HOUR * QUARTER_HOURS_PER_DAY/self.dt))
        self.T_iter = time_iteration
        self.N_iter = int(np.ceil(self.T_iter * SECONDS_PER_QUARTER_HOUR * QUARTER_HOURS_PER_DAY/self.dt))

        self.bids_mpc = np.zeros((4, self.N))

        self.mpc_controller.rigid_baseline()
        ref_x = mpc_controller.runs['runs']['Rigid']['timeseries']['x']
        self.reference_weight = self.model.freshweight(ref_x)
        

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

            mu_up = conditional_expectation(controller.spot_prices, market.price_means, market.price_cov)[0]
            mu_dn = conditional_expectation(controller.spot_prices, market.price_means, market.price_cov)[1]

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

            iter_starttime = time.time()

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
            self.mpc_controller.spot_prices = self.mpc_controller.spot_prices[1:]

            # Status update
            iter_time = time.time()-iter_starttime
            minutes, seconds = divmod(iter_time, 60)
            print(f"Completed iteration {k} of {self.N-2} in: {int(minutes)} minutes and {seconds:.2f} seconds.")


        self.x_mpc = X
        self.u_mpc = U

        for i in range(self.N):
            self.bids_mpc[:, i] = Bids[i].as_array()



        return 0
    


    def solve_mpc(self):

        N = self.N              # Number of time steps for the whole optimization problem
        N_TH = self.N_TH        # Number of time steps for internal open-loop solver
        spot_prices = self.mpc_controller.spot_prices
        nx, nu = self.model.nx, self.model.nu
        F = self.model.casadi_function_fe()

        # Set up optimizers
        opti_base,  X_base, U_base, _,      Eps_base    = self.setup_optimizer(nx, nu, N_TH)
        opti_bid,   X_bid,  _ ,     B_bid,  Eps_bid     = self.setup_optimizer(nx, nu, N_TH)

        # Set up state vectors
        X = ca.DM.zeros(nx, N+1)
        X[:,0] = self.mpc_controller.x_init
        U = ca.DM.zeros(nu, N)
        B = ca.DM.zeros(4, N)


        # for k in range(N):
        k = 0
        while k < N:
            
            N_horizon = min(N-k, N_TH)

            # Update baseline optimizer
            opti_base = self.update_optimizer_baseline(opti_base.copy(), N_TH = N_horizon, spot_prices = spot_prices[k:k+N_horizon], 
                                                       X    = X_base,   x0 = X[:,k], 
                                                       U    = U_base, 
                                                       Eps  = Eps_base, 
                                                       ref_weight = self.reference_weight[k+N_horizon])

            sol_base = opti_base.solve()
            u_opt = sol_base.value(U_base)

            # opti_bid = self.update_optimizer_bidding(opti=opti_bid.copy(), N_TH = N_horizon, spot_prices = spot_prices[k:k+N_horizon],
            #                                          X      = X_bid,    x0 = X[:,k], 
            #                                          B      = B_bid, 
            #                                          Eps    = Eps_bid, 
            #                                          U_base = u_opt, 
            #                                          ref_weight = self.reference_weight[k+N_horizon])
            # sol_bid = opti_bid.solve()
            # B_opt = sol_bid.value(B_bid)
            # Solve baseline
            # Solve bidding

            # Get u
            # U[:,k] = self.model.get_u(U_base, B, spot_prices[k,k+N_horizon], self.market)

            U[:,k:k+min(self.N_iter, N_horizon)] = u_opt[0:min(self.N_iter, N_horizon)]

            # Store inputs
            # U[:,k:k+min(self.N_iter, N_horizon)] = self.model.get_u(U_base      = u_opt[0:min(self.N_iter, N_horizon)], 
            #                                                         B           = B_opt[:,0:min(self.N_iter, N_horizon)],
            #                                                         spot_prices = spot_prices[k:k+N_horizon],
            #                                                         market      = self.market)

            # B[:,k:k+min(self.N_iter, N_horizon)] = B_opt[:,0:min(self.N_iter, N_horizon)]
            

            # Integrate states
            for i in range(min(self.N_iter, N_horizon)):
                X[:,k+1+i] = F(X[:,k+i], U[:,k+i])

            k += self.N_iter

        self.x_mpc = np.array(X[:,1:])
        self.u_mpc = np.array(U)


    def setup_optimizer(self, nx, nu, N_horizon):
        opti = ca.Opti()

        X = opti.variable(nx, N_horizon+1)
        U = opti.variable(nu, N_horizon)
        B = opti.variable(4*nu, N_horizon)
        Eps = opti.variable(1, 1)

        x0 = opti.parameter(nx, 1)
        spot_prices = opti.parameter(1, N_horizon)

        J = 0

        opti.minimize(J)
        opts = {}
        opti.solver('ipopt', opts)
        return opti, X, U, B, Eps, x0, spot_prices
                


    def set_constraints(self, opti, N_TH, X, x0, U, Eps, ref_weight, B=None, U_base=None):

        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_process_constraints(g_eq, g_ineq, N_TH, self.dt, X, x0, U, Eps, ref_weight)
        
        lb_X, ub_X = self.model.get_state_bounds(N_TH)
        lb_U, ub_U = self.model.get_input_bounds(N_TH)

    
        # if type(B) != type(None) and type(U_base) != type(None):
        if B is not None and U_base is not None:
            g_eq, g_ineq = self.model.get_bidding_constraints(g_eq, g_ineq, N_TH, B, U_base)

        # opti.subject_to()
        [opti.subject_to(equality_constraint == 0) for equality_constraint in g_eq]
        [opti.subject_to(inequality_constraint >= 0) for inequality_constraint in g_ineq]

        return opti


    def update_optimizer_baseline(self, opti, N_TH, spot_prices, X,x0, U, Eps, ref_weight):

        J = self.model.baseline_obj_function(N_TH, spot_prices, X, U) + self.model.terminal_cost(Eps)
        opti.minimize(J)
        
        opti = self.set_constraints(opti=opti, N_TH=N_TH, X=X, x0=x0, U=U, Eps=Eps, ref_weight=ref_weight)
        return opti

    def update_optimizer_bidding(self, opti, N_TH, spot_prices, X, x0, B, Eps, U_base, ref_weight):

        J = self.model.bidding_obj_function(self.N_TH, spot_prices, X, B, U_base, self.market) + self.model.terminal_cost(Eps)
        opti.minimize(J)

        U = self.model.get_u(U_base=U_base, B=B, spot_prices=spot_prices, market=self.market)
        
        opti = self.set_constraints(opti=opti, N_TH=N_TH, X=X, x0=x0, U=U, Eps=Eps, ref_weight=ref_weight, B=B, U_base = U_base)
        return opti


            





        