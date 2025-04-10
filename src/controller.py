import casadi as ca
import numpy as np
from market import Market, BalancingMarket
from model import *
from config import Config
from settings import Settings
from bid import Bid
from utils import *
import time
import os
import json
from globals import *
from tabulate import tabulate
from datetime import datetime
from tqdm import tqdm
from typing import List

class Controller():
    """
    Generate optimal inputs sequences given a model and a set of constraints.

    Mainly, this is used to generate light schedules for plant models, 
    but it may be used for entirely different process models.

    All generated input sequences along with predicted state trajectories are stored
    in the `optimization_results` dict. 
    """

    surpress_output: bool
    import_file: str

    model: PlantModel
    market: Market
    config: Config
    settings: Settings

    N:  float
    T:  float
    dt: float
    t:  np.array

    specs: dict

    mpc_T_horizon: float
    mpc_N_horizon: int
    mpc_step_time: float
    mpc_step_N: int

    spot_prices:    np.array
    x_init:         np.array

    bids:       List[Bid]
    A_up:       List[bool]
    A_down:     List[bool]

    optimization_results: dict
    search_cache:   bool
    warm_start:     bool
    calculate_fw:   bool
    # u_base:         np.array
    # x_base:         np.array

    def __init__(self, settings: Settings, plantmodel, market, config):
        '''
        Get settings from settings-object

        Gather specs from other plantmodel, market.
        Setup optimization_results dict
        Generate fixed light schedule for reference
        '''

        self.settings = settings
        self.controller_settings = settings.get_settings_group('options', 'general', 'controller', 'market', 'plantmodel')

        self.sim_name           = self.controller_settings['SIM_NAME']
        self.surpress_output    = self.controller_settings['SURPRESS_OUTPUT']
        self.warm_start         = self.controller_settings['WARM_START']
        self.calculate_fw       = self.controller_settings['CALCULATE_FW']
        self.T                  = self.controller_settings['SIMULATION_LENGTH'] 
        self.N                  = self.controller_settings['SIM_N_TIMESTEPS'] 
        self.dt                 = self.controller_settings['SIM_TIMEDELTA'] 

        self.search_cache       = self.controller_settings['SEARCH_SIM_CACHE']
        self.import_file        = self.controller_settings['IMPORT_FILE']

        self.mpc_settings       = self.settings.get_settings_group('mpc')
        self.mpc_T_horizon      = self.mpc_settings['MPC_TIMEHORIZON']
        mpc_steplength          = self.mpc_settings['MPC_STEPLENGTH']
        
        self.mpc_N_horizon = int(np.ceil(self.mpc_T_horizon * QUARTER_HOURS_PER_DAY))
        self.mpc_T_step = max(mpc_steplength, self.dt/(SECONDS_PER_QUARTER_HOUR * QUARTER_HOURS_PER_DAY))
        self.mpc_N_step = int(np.ceil(self.mpc_T_step * QUARTER_HOURS_PER_DAY))
        assert self.mpc_N_horizon >= self.mpc_N_step, f'MPC horizon ({self.mpc_N_horizon}) must be equal to or longer than the steplength ({self.mpc_N_step}).'

        self.model = plantmodel
        self.market = market
        self.config = config

        self.settings_data = {
            'general'    : settings.get_settings_group('general'),
            'controller' : settings.get_settings_group('controller'),
            'model'      : settings.get_settings_group('plantmodel'),
            'market'     : settings.get_settings_group('market')
        }

        self.optimization_results = {
            'name'              : self.sim_name,
            'timestamp'         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'specs'             : self.settings_data,
            'runs'              : {}
            }

        self.t = np.linspace(0, self.T, self.N)
        self.spot_prices = self.market.get_spotprice()

        self.bids = []
        for i in range(self.market.n_given_bids):
            self.bids.append(Bid())

        self.A_up, self.A_down = [],[]
        for i in range(self.market.n_given_activations):
            self.A_up.append(0)
            self.A_down.append(0)


        self.F = self.model.casadi_function(self.dt)

        # Set initial state
        self.x_init = self.model.x_init

        # Generate freshweight for the mpc bidding controller to use as reference trajectory
        self.fixed_light_schedule()   
        


    def optimize_mfrr(self, run_id, refrun_id = 'fixed', plot_run = False):

        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')

        if not self.load_from_json(run_id, refrun_id, dependencies): 
            # Identical run located. Using its solution instead
            return 0
        
        if not self.surpress_output: print(f'{run_id} | Generating bidding strategy')
        

        # Just check if there is a basline before proceeding
        assert refrun_id in self.optimization_results['runs'], f"{run_id} | Error: {refrun_id} has not been generated"   
        refrun = self.optimization_results['runs'][refrun_id]

        N = self.N
        T = self.T
        dt = self.dt
        spot_prices = self.spot_prices
        market = self.market

        U_nom = refrun['timeseries']['u'].reshape((1, N))

        # State and control dimensions
        nx = self.model.nx                      # Dimension of state x (x1, x2)
        nu = self.model.nu                      # Dimension of control u (scalar)
        neps = self.model.neps                  # DImension of slack variables

        # Create decision variables for the optimization problem
        X = ca.MX.sym('X', nx, N+1)             # States over time (2x(N+1) vector)
        B = ca.MX.sym('B', 4, N)                # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        Eps = ca.MX.sym('Eps', neps, 1)            # Slack variable for feasibility

        U = self.model.get_u(N, U_nom, B[:2,:], B[2:4,:], spot_prices, market.AM)           # Express U in terms of bidding outcomes

        # Initialize cost function and constraints
        J = self.model.AM_bidding_obj_function(N, self.spot_prices, B_volumes = B[:2,:], B_prices = B[2:4,:], U_nom = U_nom, balancing_market = self.market.AM)\
                       + self.model.terminal_cost(self, X, U, Eps)


        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, X, U, Eps)
        
        # format constraints
        n_eq    = ca.vertcat(*g_eq).size()[0]
        n_ineq  = ca.vertcat(*g_ineq).size()[0]
        g       = g_eq + g_ineq
        lbg     = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)                         # \ Eq-constraints = 0
        ubg     = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)   # / Ineq-constraints >= 0


        # Extract state and bidding bounds
        lbx,    ubx     = self.model.get_state_bounds(self)
        lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices = self.model.get_bidding_bounds(N, U_nom)
        lb_eps, ub_eps  = np.zeros((neps, 1)), np.inf * np.ones((neps, 1))

        # Flatten decision variables and bounds
        Z   = ca.vertcat(ca.reshape(X,   -1, 1), ca.reshape(B,    -1, 1),                                           ca.reshape(Eps,    -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(ca.vertcat(lb_B_volumes, lb_B_prices), -1, 1),   ca.reshape(lb_eps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ca.vertcat(ub_B_volumes, ub_B_prices), -1, 1),   ca.reshape(ub_eps, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        z0 = ca.DM.zeros((N + 1)*nx + N*4 + neps)  
        if self.warm_start: 
            z0[:nx*(N+1)]                   = refrun['timeseries']['x'].flatten()
            z0[nx*(N+1):nx*(N+1)+2*N]       = ub_B_volumes.reshape((2*N,1))             # Bid volumes
            z0[nx*(N+1)+2*N:nx*(N+1)+3*N]   = self.market.AM.expected_clearing_prices_up            # Bid prices up
            z0[nx*(N+1)+3*N:nx*(N+1)+4*N]   = self.market.AM.expected_clearing_prices_down          # Bid prices down
        
        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution

        x = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        B = np.array(sol['x'][(nx*(N+1)):(nx*(N+1) + 4*N)].reshape((4, N)))
        u = np.array(self.model.get_u(N, U_nom, B[:2,:], B[2:4,:], spot_prices, market.AM)).reshape(1,-1)
        eps   = float(sol['x'][-neps][0])

        
        end_time = time.time()
        sol['elapsed_time'] = end_time - start_time
        sol['eps'] = eps
        
        self.store_run(run_id, dependencies, sol, x, u, B=B, U_nom=refrun['timeseries']['u'], refrun_id = refrun_id, balancing_market=self.market.AM, plot_run=plot_run)

        if not self.surpress_output: print(f'{run_id} | Optimized mFRR bidding strategy')
        
        self.save_to_json()
        return 0
        

    def co_optimize_CM_AM(self, run_id = 'probabilistic_co_opt_CM_AM', plot_run = False):
        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')
        if not (self.load_from_json(f"{run_id}_nom", None, dependencies) or self.load_from_json(f"{run_id}", f"{run_id}_nom", dependencies)) : 
            # Identical run located. Using its solution instead
            return 0
        
        if not self.surpress_output: print(f'{run_id} | Generating theoretically optimal Capacity Market bids')
        
        AM = self.market.AM
        CM = self.market.CM

        N = self.N
        T = self.T
        dt = self.dt
        spot_prices = self.spot_prices

        # State and control dimensions
        nx = self.model.nx                      # Dimension of state x (x1, x2)
        nu = self.model.nu                      # Dimension of control u (scalar)
        neps = self.model.neps

        # Create decision variables for the optimization problem
        nom_X       = ca.MX.sym('nom_X', nx, N+1)         # States over time (2x(N+1) vector)
        nom_U       = ca.MX.sym('nom_U', nu, N)
        nom_Eps     = ca.MX.sym('nom_Eps', neps, 1)       # Slack variable for feasibility
        
        CM_X           = ca.MX.sym('CM_X', nx, N+1)             # States over time (2x(N+1) vector)
        CM_B_volumes   = ca.MX.sym('CM_B_volumes', 2, N)        # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        CM_B_prices    = ca.MX.sym('CM_B_volumes', 2, N)        # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        CM_Eps         = ca.MX.sym('CM_Eps', neps, 1)           # Slack variable for feasibility

        AM_X           = ca.MX.sym('AM_X', nx, N+1)             # States over time (2x(N+1) vector)
        AM_B_volumes   = ca.MX.sym('AM_B_volumes', 2, N)        # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        AM_B_prices    = ca.MX.sym('AM_B_volumes', 2, N)        # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        AM_Eps         = ca.MX.sym('AM_Eps', neps, 1)           # Slack variable for feasibility
        
        CM_U = self.model.get_u_CM(N, nom_U, CM_B_volumes, CM_B_prices, AM_B_volumes, AM_B_prices, spot_prices, CM, AM)
        AM_U = self.model.get_u(N, nom_U, AM_B_volumes, AM_B_prices, spot_prices, AM)
        

        J = 0
        J += self.model.spotopt_obj_function(N, spot_prices, AM_U)
        J += self.model.CM_bidding_obj_function(N, spot_prices, CM_B_volumes, CM_B_prices, CM)
        J += self.model.AM_bidding_obj_function(N, spot_prices, AM_B_volumes, AM_B_prices, nom_U, AM)
        J += self.model.terminal_cost(self, nom_X, nom_U, nom_Eps) \
            + self.model.terminal_cost(self, CM_X, CM_U, CM_Eps) \
            + self.model.terminal_cost(self, AM_X, AM_U, AM_Eps)

        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, nom_X, nom_U, nom_Eps)
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, CM_X, CM_U, CM_Eps)
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, AM_X, AM_U, AM_Eps)
        g_eq, g_ineq = self.model.get_bidding_constraints(g_eq, g_ineq, N, nom_U, CM_B_volumes)
        g_eq, g_ineq = self.model.get_bidding_constraints(g_eq, g_ineq, N, nom_U, AM_B_volumes, B_volumes_lower_bound=CM_B_volumes)
        
        # Enforce hourly bid volumes and prices in capacity market
        for i in range(int(N/4)):
            for j in range(3):
                g_eq.append(CM_B_volumes[0, i+j] - CM_B_volumes[0, i+j+1])
                g_eq.append(CM_B_volumes[1, i+j] - CM_B_volumes[1, i+j+1])
                g_eq.append(CM_B_prices[0, i+j]  - CM_B_prices[0, i+j+1])
                g_eq.append(CM_B_prices[1, i+j]  - CM_B_prices[1, i+j+1])

        # format constraints
        n_eq    = ca.vertcat(*g_eq).size()[0]
        n_ineq  = ca.vertcat(*g_ineq).size()[0]
        g       = g_eq + g_ineq
        lbg     = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)                         # \ Eq-constraints = 0
        ubg     = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)   # / Ineq-constraints >= 0


        # Extract state and bidding bounds
        lbx, ubx = self.model.get_state_bounds(self)
        lbu, ubu = self.model.get_input_bounds(self)
        lb_B_volumes = np.zeros((2, N))
        ub_B_volumes = self.model.P_cap_max * np.ones((2, N))
        lb_B_prices = np.zeros((2, N))
        ub_B_prices = 1000 * np.ones((2, N))
        lb_eps, ub_eps = np.zeros((neps,1)), np.inf * np.ones((neps,1))

        # Flatten decision variables and bounds
        Z   = ca.vertcat(ca.reshape(nom_X, -1, 1), ca.reshape(CM_X, -1, 1), ca.reshape(AM_X, -1, 1), ca.reshape(nom_U, -1, 1), ca.reshape(CM_B_volumes, -1, 1), ca.reshape(CM_B_prices, -1, 1), ca.reshape(AM_B_volumes, -1, 1), ca.reshape(AM_B_prices, -1, 1), ca.reshape(nom_Eps, -1, 1), ca.reshape(CM_Eps, -1, 1), ca.reshape(AM_Eps, -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx,   -1, 1), ca.reshape(lbx,  -1, 1), ca.reshape(lbx,  -1, 1), ca.reshape(lbu,   -1, 1), ca.reshape(lb_B_volumes, -1, 1), ca.reshape(lb_B_prices, -1, 1), ca.reshape(lb_B_volumes, -1, 1), ca.reshape(lb_B_prices, -1, 1), ca.reshape(lb_eps,  -1, 1), ca.reshape(lb_eps, -1, 1), ca.reshape(lb_eps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx,   -1, 1), ca.reshape(ubx,  -1, 1), ca.reshape(ubx,  -1, 1), ca.reshape(ubu,   -1, 1), ca.reshape(ub_B_volumes, -1, 1), ca.reshape(ub_B_prices, -1, 1), ca.reshape(ub_B_volumes, -1, 1), ca.reshape(ub_B_prices, -1, 1), ca.reshape(ub_eps,  -1, 1), ca.reshape(ub_eps, -1, 1), ca.reshape(ub_eps, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        N_vars = (N+1)*nx*3 + N*nu + N*4*2 + neps*3
        z0 = ca.DM.zeros(N_vars)
        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution

        nom_X_idx       = 0
        CM_X_idx        = nom_X_idx         + nx*(N+1)
        AM_X_idx        = CM_X_idx          + nx*(N+1)
        nom_U_idx       = AM_X_idx          + nx*(N+1)
        CM_B_vols_idx   = nom_U_idx         + nu*N
        CM_B_prices_idx = CM_B_vols_idx     + 2*N
        AM_B_vols_idx   = CM_B_prices_idx   + 2*N
        AM_B_prices_idx = AM_B_vols_idx     + 2*N
        nom_eps_idx     = AM_B_prices_idx   + 2*N
        CM_eps_idx      = nom_eps_idx       + neps
        AM_eps_idx      = CM_eps_idx        + neps

        nom_x       = np.array(sol['x'][:CM_X_idx].reshape((nx, N+1)))
        CM_x        = np.array(sol['x'][CM_X_idx:AM_X_idx].reshape((nx, N+1)))
        AM_x        = np.array(sol['x'][AM_X_idx:nom_U_idx].reshape((nx, N+1)))
        nom_u       = np.array(sol['x'][nom_U_idx:CM_B_vols_idx].reshape((nu, N)))
        CM_B_vols   = np.array(sol['x'][CM_B_vols_idx:CM_B_prices_idx].reshape((2, N)))
        CM_B_prices = np.array(sol['x'][CM_B_prices_idx:AM_B_vols_idx].reshape((2, N)))
        AM_B_vols   = np.array(sol['x'][AM_B_vols_idx:AM_B_prices_idx].reshape((2, N)))
        AM_B_prices = np.array(sol['x'][AM_B_prices_idx:nom_eps_idx].reshape((2, N)))
        nom_eps     = float(sol['x'][nom_eps_idx:CM_eps_idx][0])
        CM_eps      = float(sol['x'][CM_eps_idx:AM_eps_idx][0])
        AM_eps      = float(sol['x'][AM_eps_idx:][0])

        
        CM_u        = np.array(self.model.get_u_CM(N, nom_u, CM_B_vols, CM_B_prices, AM_B_vols, AM_B_prices, spot_prices, CM, AM)).reshape((1,-1))
        AM_u        = np.array(self.model.get_u(N, nom_u, AM_B_vols, AM_B_prices, spot_prices, AM)).reshape((1,-1))

        CM_B = np.vstack((CM_B_vols, CM_B_prices))
        AM_B = np.vstack((AM_B_vols, AM_B_prices))
        
        end_time = time.time()
        sol['elapsed_time'] = end_time - start_time
        
        nom_sol, CM_sol, AM_sol = sol.copy(), sol.copy(), sol.copy()
        nom_sol['eps']  = nom_eps
        CM_sol['eps']   = CM_eps
        AM_sol['eps']   = AM_eps
        
        self.store_run(f"{run_id}_nom", dependencies, nom_sol, nom_x, nom_u, refrun_id = 'None', plot_run = plot_run)
        self.store_run(f"{run_id}_CM",  dependencies, CM_sol,  CM_x, CM_u, B=CM_B, U_nom=nom_u, refrun_id = f"{run_id}_nom", balancing_market=self.market.CM, plot_run = plot_run)
        self.store_run(f"{run_id}_AM",  dependencies, AM_sol,  AM_x, AM_u, B=AM_B, U_nom=nom_u, refrun_id = f"{run_id}_CM",  balancing_market=self.market.AM, plot_run = plot_run)

        if not self.surpress_output: print(f'{run_id} | Generated theoretically optimal bid plan')

        self.save_to_json()
        return 0
    

    def optimize_spotprice(self, run_id: str, refrun_id = 'fixed', plot_run = False):
        '''
        Optimizes the light schedule based on spot price. Used as reference for mFRR optimization (Called baseline in MARI terms)
        '''
        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')

        if not self.load_from_json(run_id, refrun_id, dependencies): 
            # Identical run located. Using its solution instead
            return 0
        
        if not self.surpress_output: print(f'{run_id} | Optimizing light schedule based on spot price')
        
        # assert refrun_id in self.optimization_results['runs'], f"{run_id} | Error: {refrun_id} has not been generated"   
        # refrun = self.optimization_results['runs'][refrun_id]

        N = self.N
        T = self.T

        # State and control dimensions
        nx = self.model.nx                              # Dimension of state x (x1, x2)
        nu = self.model.nu                              # Dimension of control u (scalar)
        neps = self.model.neps

        # Create decision variables for the optimization problem
        X = ca.MX.sym('X', nx, N+1)                     # States over time ((N+1)x1 vector)
        U = ca.MX.sym('U', nu, N)                       # Controls over time (Nx1 vector)
        Eps = ca.MX.sym('Eps', neps, 1)                    # Slack variable for feasibility

        J = self.model.spotopt_obj_function(N, self.spot_prices, U)\
                     + self.model.terminal_cost(self, X, U, Eps)#\
                    # + self.model.fluctuating_light_cost(self, U)         # Cost function

        # Get bounds
        lbx, ubx = self.model.get_state_bounds(self)
        lbu, ubu = self.model.get_input_bounds(self)
        lb_eps, ub_eps = np.zeros((neps, 1)), np.inf * np.ones((neps, 1))

        # Get constraints
        g_eq, g_ineq = [],[]
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, X, U, Eps)
        g = g_eq + g_ineq 


        # Flatten decision variables and bounds
        Z =   ca.vertcat(ca.reshape(X, -1, 1),   ca.reshape(U, -1, 1),   ca.reshape(Eps,    -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(lbu, -1, 1), ca.reshape(lb_eps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ubu, -1, 1), ca.reshape(ub_eps, -1, 1))

        # Format constraints
        n_eq    = ca.vertcat(*g_eq).size()[0]
        n_ineq  = ca.vertcat(*g_ineq).size()[0]
        lbg     = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)
        ubg     = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)

        # Nonlinear problem definition
        nlp     = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts    = {'ipopt.print_level': 0, 'print_time': 0}
        solver  = ca.nlpsol('solver', 'ipopt', nlp, opts)

        z0 = ca.DM.zeros((N + 1)*nx + N*nu + neps)
        # if self.warm_start and refrun != 'None':
        #     z0[:nx*(N+1)]                   = refrun['timeseries']['x'].flatten()
        #     z0[nx*(N+1):(nx*(N+1)+N*nu)]    = refrun['timeseries']['u'].flatten()

        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution
        f     = float(sol['f'])
        x     = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        u     = np.array(sol['x'][(nx*(N+1)):(nx*(N+1)+N*nu)].reshape((nu, N)))[0,:].reshape(1,-1)
        eps   = float(sol['x'][-neps:][0])

        end_time     = time.time()
        elapsed_time = end_time - start_time

        sol['elapsed_time'] = elapsed_time
        sol['eps'] = eps

        self.store_run(run_id, dependencies, sol, x, u, refrun_id = refrun_id, plot_run = plot_run)

        if not self.surpress_output: print(f'{run_id} | Optimized light schedule for spot price')
        
        self.save_to_json()
        return 0



    def optimize_AM_mpc(self, run_id, target_run_id = 'fixed', plot_run = False):

        start_time = time.time()
        
        dependencies = ('general', 'controller', 'mpc', 'plantmodel', 'market')

        if not self.load_from_json(run_id, target_run_id, dependencies): 
            # Identical run located. Using its solution instead
            return 0

        if not self.surpress_output: print(f'{run_id} | Running MPC')
        
        N = self.N                      # Number of time steps for the whole optimization problem
        N_TH = self.mpc_N_horizon       # Number of time steps for internal open-loop solver
        N_iter = self.mpc_N_step        # Number of time steps between each open-loop solution
        spot_prices = self.spot_prices
        nx, nu, neps = self.model.nx, self.model.nu, self.model.neps
        market = self.market
        F = self.F

        clearing_prices_up, clearing_prices_down = market.AM.get_clearing_prices(self.market.date)
        assert len(clearing_prices_up)==N and len(clearing_prices_down)==N, f'Clearing price arrays have inconsistent lengths with simulation duration. N = {self.N}, len(clearing prices up) = {len(clearing_prices_up)}, len(clearing prices down) = {len(clearing_prices_down)}'
        
        activation_demands_up, activation_demands_down = market.mfrr_demands_up, market.mfrr_demands_down
        assert len(activation_demands_down)==N and len(activation_demands_up)==N, f'Activation demand arrays have inconsistent lengths with simulation duration. N = {self.N}, len(demands up) = {len(activation_demands_up)}, len(demands down) = {len(activation_demands_down)}'

        
        target_run = self.optimization_results['runs'][target_run_id]
        target_X = target_run['timeseries']['x']
        target_U = target_run['timeseries']['u']
        target_weight = self.model.freshweight(target_X)

        # Set up optimizers
        opti_base,  opt_vars_base = self.setup_optimizer(nx, nu, neps, N_TH, 'spot_opt')
        opti_bid,   opt_vars_bid  = self.setup_optimizer(nx, nu, neps, N_TH, 'mfrr_opt')

        # Set up constraints using parameters
        opti_base = self.set_constraints(opti_base,  N_TH, opt_vars_base)
        opti_bid  = self.set_constraints(opti_bid,   N_TH, opt_vars_bid)

        # Set up state vectors
        X = ca.DM.zeros(nx, N+1)
        X[:,0] = self.x_init
        past_X = target_X[:,:QUARTER_HOURS_PER_DAY].copy()
        U = ca.DM.zeros(nu, N)
        U_nom = ca.DM.zeros(nu, N)
        B = ca.DM.zeros(4, N)
        Eps = 0

        k = 0
        with tqdm(total=N, desc=f"{run_id}: Running MPC") as pbar:
            while k < N:
                
                N_horizon = min(N-k, N_TH)
                extract_solution_slice = slice(0, min(N_iter, N_horizon))
                start_iter = k
                end_iter = k+N_horizon
                iter_slice = slice(start_iter, end_iter)

                # Update baseline optimizer
                # Apply current parameters and set initial guess
                opti_base_copy = self.update_optimizer_baseline(
                                opti_base.copy(), k = k, N = N, N_TH = N_horizon, opt_vars = opt_vars_base, spot_prices = spot_prices[iter_slice], 
                                x0          = X[:,k], 
                                init_X      = target_X[:,start_iter:end_iter+1],
                                past_X      = past_X,
                                init_U      = target_U[:,iter_slice],
                                ref_weight  = target_weight[end_iter]
                )

                # Solve baseline
                sol_base = opti_base_copy.solve()
                u_opt_base = sol_base.value(opt_vars_base['U']).reshape(1,-1)

                # Apply current parameters and set initial guess
                opti_bid_copy = self.update_optimizer_bidding(
                                opti_bid.copy(), k = k, N = N, N_TH = N_horizon, opt_vars = opt_vars_bid, spot_prices = spot_prices[iter_slice],
                                x0          = X[:,start_iter], 
                                past_X      = past_X,
                                U_nom       = u_opt_base[:,:N_horizon], 
                                ref_weight  = target_weight[end_iter]
                )

                # Solve bidding
                sol_bid         = opti_bid_copy.solve()
                x_opt_bid       = sol_bid.value(opt_vars_bid['X'])
                B_volumes_opt   = sol_bid.value(opt_vars_bid['B_volumes'])[:,:N_horizon]
                B_prices_opt    = sol_bid.value(opt_vars_bid['B_prices'])[:,:N_horizon]

                # Apply bid activations
                # Evaluate activations
                activation_up   = np.where(np.logical_and(activation_demands_up[iter_slice] > 0,    B_prices_opt[0,:] <= clearing_prices_up[iter_slice]),   1, 0)
                activation_down = np.where(np.logical_and(activation_demands_down[iter_slice] > 0,  B_prices_opt[1,:] <= clearing_prices_down[iter_slice]), 1, 0)


                # Store data
                store_data_slice = slice(k, k+min(N_iter, N_horizon))
                
                # Store inputs
                U_nom[:,store_data_slice] = u_opt_base[:,extract_solution_slice]
                
                u_tilde = 1000/self.model.C_conv_PPFD * (np.where(activation_down == 1, B_volumes_opt[1,:], 0)\
                                - np.where(activation_up == 1, B_volumes_opt[0,:], 0))
                u = (np.array(U_nom[:,iter_slice]).flatten() + u_tilde)[extract_solution_slice]
                U[:,store_data_slice] = u
                
                # Store bid data
                B[:2,store_data_slice]    = B_volumes_opt[:,extract_solution_slice]
                B[2:4,store_data_slice]   = B_prices_opt[:,extract_solution_slice]


                # Integrate states
                # X[:,k:k+1+min(N_iter, N_horizon)] = x_opt_bid[:,:1+min(N_iter, N_horizon)]
                for i in range(k, k+min(N_iter, N_horizon)):
                    X[:,i+1] = np.array(F(X[:,i], np.array([U[:,i]]))).reshape(1, -1)

                past_X[:,-min(QUARTER_HOURS_PER_DAY, min(N_iter, N_horizon)):] = X[:,k:k+min(QUARTER_HOURS_PER_DAY, min(N_iter, N_horizon))]

                Eps = max(0, target_weight[-1] - self.model.freshweight(X[:,-1]))

                k += N_iter

                pbar.update(min(N_iter, N_horizon))

        end_time = time.time()
        sol = {}
        sol['eps'] = Eps
        sol['f'] = self.model.AM_bidding_obj_function(N, spot_prices, B[:2, :], B[2:4, :], U_nom, self.market)
        sol['elapsed_time'] = end_time - start_time
        
        self.store_run(run_id, dependencies, sol, np.array(X), np.array(U).reshape((1,-1)), B=np.array(B), U_nom=np.array(U_nom).reshape((1,-1)), refrun_id = target_run_id, balancing_market=self.market.AM, plot_run=plot_run)

        if not self.surpress_output: print(f'{run_id} | Optimized mFRR bidding strategy using MPC')
        self.save_to_json()
        return 0


    def setup_optimizer(self, nx, nu, neps, N_horizon, opti_type: str):
        '''Creates opti variables. Creates opt_vars dictionaries containing opti symbolic optimization variables'''

        opti = ca.Opti()
        opts = {'ipopt.print_level':0, 'print_time':0}
        opti.solver('ipopt', opts)

        X = opti.variable(nx, N_horizon+1)
        Eps = opti.variable(neps, 1)

        x0          = opti.parameter(nx, 1)         # Starting weight
        ref_weight  = opti.parameter(1, 1)          # End weight (To be substituted)
        spot_prices = opti.parameter(1, N_horizon)  # Spot prices for optimization window

        opti.set_value(spot_prices, ca.DM.ones(spot_prices.shape))

        opt_vars = {'opti_type': opti_type,
                    'N_horizon': N_horizon,
                    'X':    X,
                    'x0':   x0,
                    'Eps':  Eps,
                    'spot_prices' : spot_prices,
                    'ref_weight'  : ref_weight
                    }
        

        if opti_type=='spot_opt':
            U = opti.variable(nu, N_horizon)

            # Register opti_variable to opt_vars
            opt_vars['U'] = U
            return opti, opt_vars
        elif opti_type=='mfrr_opt':
            B_prices    = opti.variable(2*nu, N_horizon)
            B_volumes   = opti.variable(2*nu, N_horizon)
            U_nom       = opti.parameter(1, N_horizon)

            # Initialize parameters with 0-values
            opti.set_value(U_nom,       ca.DM.zeros(U_nom.shape))
            # opti.set_value(B_volumes,   ca.DM.zeros(B_volumes.shape))

            # Register opti_variables and opti_params to opt_vars
            opt_vars['B_prices']    = B_prices
            opt_vars['B_volumes']   = B_volumes
            opt_vars['U_nom']       = U_nom
            return opti, opt_vars
        else:
            assert False, f'SETUP MPC | invalid opti type: {opti_type}'
                
    def set_constraints(self, opti: ca.Opti, N_TH, opt_vars: dict):
        
        # Extract symbolic optimization variables and parameters
        X, Eps, x0, spot_prices, ref_weight = [opt_vars[key] for key in ['X', 'Eps', 'x0', 'spot_prices', 'ref_weight']]    
        
        
        g_eq, g_ineq = [], []

        if 'B_prices' in opt_vars and 'U_nom' in opt_vars and 'U' not in opt_vars:
            B_prices, B_volumes, U_nom = [opt_vars[key] for key in ['B_prices', 'B_volumes', 'U_nom']]

            U = self.model.get_u(N_TH, U_nom=U_nom, B_volumes=B_volumes, B_prices=B_prices, spot_prices=spot_prices, balancing_market=self.market.AM).reshape((1,-1))
            g_eq, g_ineq = self.model.get_bidding_constraints(g_eq, g_ineq, N_TH, U_nom, B_prices = B_prices, B_volumes = B_volumes)

        elif 'B' not in opt_vars and 'U_nom' not in opt_vars and 'U' in opt_vars:
            U = opt_vars['U']

        else:
            assert False, 'Content in opt_vars is inconsistent'

        g_eq, g_ineq = self.model.get_static_process_constraints(g_eq, g_ineq, N_TH, self.dt, X, x0, U, Eps)

        # [opti.subject_to(equality_constraint == 0) for equality_constraint in g_eq]
        # [opti.subject_to(inequality_constraint >= 0) for inequality_constraint in g_ineq]

        with tqdm(total=len(g_eq), desc=f"    MPC {opt_vars['opti_type']}: Adding equality constraints") as pbar:
            for equality_constraint in g_eq:
                opti.subject_to(equality_constraint == 0)
                pbar.update(1)
        
        with tqdm(total=len(g_ineq), desc=f"    MPC {opt_vars['opti_type']}: Adding inequality constraints") as pbar:
            for inequality_constraint in g_ineq:
                opti.subject_to(inequality_constraint >= 0)
                pbar.update(1)

        return opti


    def update_optimizer_baseline(self, opti: ca.Opti, k, N, N_TH, opt_vars, spot_prices, x0, init_X, past_X, init_U, ref_weight):

        X, U, Eps = [opt_vars[key] for key in ['X', 'U', 'Eps']]    # Extract symbolic optimization variables and parameters

        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_dynamic_process_constraints(g_eq, g_ineq, N_TH, X, Eps, ref_weight, past_X = past_X)
        [opti.subject_to(equality_constraint    == 0) for equality_constraint   in g_eq]
        [opti.subject_to(inequality_constraint  >= 0) for inequality_constraint in g_ineq]

        J = self.model.spotopt_obj_function(N_TH, spot_prices, U) + self.model.terminal_cost(self, X, U, Eps)

        # if (k + N_TH >= N): J += self.model.terminal_cost(self, X, U, Eps)
        # else:               J += self.model.running_cost(self.market, k, N, Eps)
        
        opti.minimize(J)

        opti.set_initial(X[:,:N_TH+1], init_X)
        opti.set_initial(U[:,:N_TH], init_U)

        opti.set_value(opt_vars['x0'], x0)
        opti.set_value(opt_vars['ref_weight'], ref_weight)
        opti.set_value(opt_vars['spot_prices'][:,:N_TH], spot_prices)

        return opti

    def update_optimizer_bidding(self, opti: ca.Opti, k, N, N_TH, opt_vars, spot_prices, x0, past_X, U_nom, ref_weight):

        X, B_volumes, B_prices, Eps = [opt_vars[key] for key in ['X','B_volumes', 'B_prices', 'Eps']]    # Extract symbolic optimization variables and parameters

        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_dynamic_process_constraints(g_eq, g_ineq, N_TH, X, Eps, ref_weight, past_X = past_X)
        [opti.subject_to(equality_constraint == 0)   for equality_constraint in g_eq]
        [opti.subject_to(inequality_constraint >= 0) for inequality_constraint in g_ineq]

        U = self.model.get_u(N_TH, U_nom, B_volumes, B_prices, spot_prices, self.market.AM)
        J = self.model.AM_bidding_obj_function(N_TH, spot_prices, B_volumes, B_prices, U_nom, self.market) + self.model.terminal_cost(self, X, U, Eps)
        
        # if (k + N_TH >= N): J += self.model.terminal_cost(self, X, U, Eps)
        # else:               J += self.model.running_cost(self.market, k, N, Eps)
        
        opti.minimize(J)

        _, ub_B_volumes, _, _  = self.model.get_bidding_bounds(N_TH, U_nom)
        B_max_volumes = ub_B_volumes

        # update parameters
        opti.set_value(opt_vars['x0'], x0)
        opti.set_value(opt_vars['U_nom'][:,:N_TH], U_nom)
        opti.set_value(opt_vars['ref_weight'], ref_weight)
        opti.set_value(opt_vars['spot_prices'][:,:N_TH], spot_prices)
        # opti.set_value(opt_vars['B_volumes'][:,:N_TH], B_max_volumes)

        # Set initial guesses


        # Expected value of clearing prices given spot prices
        clearing_price_mu_up    = conditional_expectation(spot_prices, self.market.AM.price_stats[self.bidding_zone]['Up']['means'],    self.market.AM.price_stats[self.bidding_zone]['Up']['cov'])
        clearing_price_mu_down  = conditional_expectation(spot_prices, self.market.AM.price_stats[self.bidding_zone]['Down']['means'],  self.market.AM.price_stats[self.bidding_zone]['Down']['cov'])

        # clearing_price_mu_up = 10*clearing_price_mu[0]
        # clearing_price_mu_dn = 10*clearing_price_mu[1]

        # Set initial optimal bidding guess to be maximum possible volume and exactly at clearing price
        B_prices_initial_guess = np.vstack((clearing_price_mu_up, clearing_price_mu_down))
        U_initial_guess = np.array(self.model.get_u(N_TH, U_nom, B_prices_initial_guess, B_max_volumes, spot_prices, self.market.AM)).flatten()
        X_initial_guess = ca.DM.zeros(self.model.nx, N_TH+1)
        X_initial_guess[:,0] = x0
        F = self.F
        for k in range(len(U_initial_guess)):
            X_initial_guess[:,k+1] = F(X_initial_guess[:,k], U_initial_guess[k])

        opti.set_initial(X[:,:N_TH+1],      X_initial_guess)
        opti.set_initial(B_prices[:,:N_TH], B_prices_initial_guess)

        return opti


    def fixed_light_schedule(self, run_id = 'fixed'):
        '''
        Genertating a basic on-off schedule. Default is 16h at 200 PPFD, and 8h at 0 PPFD
        '''
        start_time = time.time()

        N = self.N
        F = self.F

        # 18 hours on, 6 hours off in 15 minute intervals
        intervals_per_hour = 4   # Time steps per hour
        hours_on = self.model.PHOTOPERIOD
        hours_off = 24-hours_on

        RIGID_INTY = self.model.LIGHT_INTY

        # Daily schedule
        day_schedule = np.array([RIGID_INTY] * (hours_on * intervals_per_hour) + [0] * (hours_off * intervals_per_hour))
        # Repeat daily schedule
        full_schedule = np.tile(day_schedule, int(np.ceil(N / len(day_schedule))))[:N]


        u = full_schedule.reshape(1,-1)

        X       = np.zeros((self.model.nx, N+1))
        X[:,0]  = self.x_init.flatten()

        for k in range(N):
            #Forward euler
            X[:,k+1] = np.array(F(X[:,k], np.array([u[:,k]]))).reshape(1, -1)

        end_time = time.time()
        elapsed_time = end_time - start_time

        sol ={}
        sol['elapsed_time'] = elapsed_time
        sol['f'] = self.model.spotopt_obj_function(N, self.spot_prices, u)
        sol['eps'] = 0
        
        
        dependencies = ('general', 'controller', 'plantmodel', 'market')
        self.store_run(run_id, dependencies, sol, X, u, refrun_id = 'None', plot_run=False)
        if self.calculate_fw: self.model.Final_fw_sht = float(self.model.freshweight(X[:,-1]))

        if not self.surpress_output: print(f'{run_id} | Generated fixed light schedule: {hours_on}h/{hours_off}h at {RIGID_INTY} PPFD')
        return 0
    


    def store_run(self, run_id, dependencies, sol, x, u, market_data: dict = {}, 
                  U_nom = None, refrun_id = 'None', plot_run = False):
        '''
        Takes in run specifics and stores them as well as metrics in the `optimization_results` dictionary.
        If bids are specified, a balancingmarket must be given as well.

        Structure:

        run_id:
            - reference run: 
            - Markets:
                - [CM/AM]:
                    - Bids:
                        - [Up/Down]:
                            - Volume:
                            - Price:
                    - Activations:
                        - [Up/Down]:
            - timeseries:
                - t
                - x
                - u
                - u_nom
            - dependencies:
            - hash:

        '''

        # assert not (B is not None and balancing_market is None), 'Must specify a balancing market'
        if U_nom is None: U_nom = np.zeros((1,self.N))
        
        timeseries_data = {
            't'     : self.t,
            'x'     : x,
            'u'     : u,
            'u_nom' : U_nom
        }
        
        f       = float(sol['f'])
        eps     = float(sol['eps'])

        metrics_data = {
            'elapsed_time'  : sol['elapsed_time'],
            'f'             : f,
            'eps'           : eps
        }
        
        metrics_data = self.model.get_metrics(metrics_data, self, run_id, x, u)

        metrics_data['Costs'] = float(self.model.spotopt_obj_function(self.N, self.spot_prices, u))
        total = metrics_data['Costs']

        for market_type in market_data:

            balancing_market = self.market.get_balancing_market(market_type)

            bid_volumes_up    = market_data[market_type]['Bids']['Up']['Volume'].reshape(1,-1)
            bid_volumes_down  = market_data[market_type]['Bids']['Down']['Volume'].reshape(1,-1)
            bid_prices_up     = market_data[market_type]['Bids']['Up']['Price'].reshape(1,-1)
            bid_prices_down   = market_data[market_type]['Bids']['Down']['Volume'].reshape(1,-1)
            
            if  market_data[market_type]['Activations'] is None:
                bid_activations_up   = balancing_market.activation_prob_up(self.spot_prices,   bid_prices_up)
                bid_activations_down = balancing_market.activation_prob_down(self.spot_prices, bid_prices_down)
            else:
                bid_activations_up   =  market_data[market_type]['Activations']['Up'].reshape(1,-1)
                bid_activations_down =  market_data[market_type]['Activations']['Down'].reshape(1,-1)


            clearing_prices_up      = balancing_market.clearing_prices_up
            clearing_prices_down    = balancing_market.clearing_prices_down
        
            total_earnings   = 1/4 * np.sum(np.multiply(clearing_prices_up,    np.where(bid_activations_up,   bid_volumes_up,   0))) \
                             + 1/4 * np.sum(np.multiply(clearing_prices_down,  np.where(bid_activations_down, bid_volumes_down, 0)))
            
            metrics_data[f'{market_type} Earnings'] = float(total_earnings)


            activation_th   = 0.01
            volume_th       = 0.001 * self.model.P_cap_max

            prob_activations_up     = np.array(balancing_market.activation_prob_up(self.spot_prices, bid_prices_up)).reshape(1,-1)
            prob_activations_down   = np.array(balancing_market.activation_prob_down(self.spot_prices, bid_prices_down)).reshape(1,-1)

            up_bids     = np.where(np.logical_and(prob_activations_up > activation_th,   bid_volumes_up > volume_th))
            down_bids   = np.where(np.logical_and(prob_activations_down > activation_th, bid_volumes_down > volume_th))

            filtered_bid_volumes_up         = bid_volumes_up[up_bids]
            filtered_bid_volumes_down       = bid_volumes_down[down_bids]
            filtered_bid_prices_up          = bid_prices_up[up_bids]
            filtered_bid_prices_down        = bid_prices_down[down_bids]
            filtered_bid_activations_up     = bid_activations_up[up_bids]
            filtered_bid_activations_down   = bid_activations_down[down_bids]


            bidding_data = {
                'Up-regulation'     : {
                    'Bids submitted'            : len(filtered_bid_activations_up),
                    'Avg bid size'              : np.average(filtered_bid_volumes_up),
                    'Avg bid price'             : np.average(filtered_bid_prices_up),
                    'Avg activation rate'       : np.average(bid_activations_up)*100,
                    'Consumption impact'        : np.sum(np.multiply(bid_volumes_up, bid_activations_up))
                },
                'Down-regulation'   : {
                    'Bids submitted'            : len(filtered_bid_activations_down),
                    'Avg bid size'              : np.average(filtered_bid_volumes_down),
                    'Avg bid price'             : np.average(filtered_bid_prices_down),
                    'Avg activation rate'       : np.average(bid_activations_down)*100,
                    'Consumption impact'        : np.sum(np.multiply(bid_volumes_down, bid_activations_down))
                }
            }     

            market_data[market_type]['Earnings']        = total_earnings
            total                                       -= total_earnings
            market_data[market_type]['bidding result']  = bidding_data 

        metrics_data['Total'] = total
        fixed_schedule_cost = self.optimization_results['runs'].get('fixed', {}).get('metrics',{}).get('Costs',total)
        metrics_data['Cost Reduction'] = f"{100 * (fixed_schedule_cost - total)/fixed_schedule_cost:.2f}%"
        
        settings_dict = self.settings.get_settings_group(*dependencies)
        if refrun_id != 'None': 
            settings_dict.update({'refrun': refrun_id, 
                                  'refrun hash': self.optimization_results['runs'][refrun_id]['hash']})

        run_hash = generate_hash(settings_dict)
        # self.settings.add_setting('hash', {f'{run_id}_hash': run_hash})

        run_data = {
            'reference_run' : refrun_id,
            'plot_run'      : plot_run,
            'metrics'       : metrics_data,
            'markets'       : market_data,
            'timeseries'    : timeseries_data,
            'dependencies'  : list(dependencies),
            'hash'          : run_hash
            }

        # Storing runs in dictionaries
        self.optimization_results['runs'][run_id] = run_data 


    def save_to_json(self):
        """
        Save all runs and their data to a JSON file.
        """

        # Add spot price to data
        self.optimization_results['spotprice'] = self.spot_prices

        # Filepath
        sim_name = self.sim_name
        sim_save_path = os.path.join(self.config.simulations_path, f"{sim_name}.json")

        # Ensure the target json file exists
        os.makedirs(self.config.simulations_path, exist_ok=True)

        # Convert the entire runs dictionary
        runs_dict = convert_np_arrays_to_lists(self.optimization_results)

        # Save the data for all runs
        with open(sim_save_path, "w") as json_file:
            json.dump(runs_dict, json_file, indent=4)


    def load_from_json(self, run_id, refrun_id, dependencies):
        """
        Load completed runs from saved JSON files and populate `completed_runs`.
        Compares the settings_profile used previously in order to determine if old result is still valid

        Returns 0 if match is made.
        Returns 1 if match is not made
        """
        if not self.search_cache: return 1

        settings_dict = self.settings.get_settings_group(*dependencies)
        if refrun_id is not None: 
            settings_dict.update({'refrun': refrun_id, 
                                  'refrun hash': self.optimization_results['runs'][refrun_id]['hash']})
 
        hash = generate_hash(settings_dict)

        for file_name in os.listdir(self.config.simulations_path):
            if not file_name.endswith(".json"): continue

            file_path = os.path.join(self.config.simulations_path, file_name)
            with open(file_path, "r") as json_file:
                loaded_data = json.load(json_file)
            
            if run_id not in loaded_data.get('runs'):
                continue
            elif 'dependencies' not in loaded_data.get('runs')[run_id]:
                continue
            elif 'hash' not in loaded_data.get('runs')[run_id]:
                continue

            # Extract the hash from the JSON content
            specs_hash = loaded_data.get('runs')[run_id]['hash']
            
            if specs_hash is None or not specs_hash == hash:
                continue

            # Store the run data keyed by the extracted hash
            conv_loaded_data = convert_lists_to_np_arrays(loaded_data)

            self.optimization_results['runs'][run_id] = conv_loaded_data['runs'][run_id]
            # if run_name == 'Bidding': self.runs['bidding result'] = conv_loaded_data['bidding result']

            if not self.surpress_output: print(f"{run_id} | Loaded run from simulation \'{loaded_data['name']}\' dated {loaded_data['timestamp']}")
            return 0
                
        if not self.surpress_output: print(f'{run_id} | No matching run found')
        return 1

    def import_light_schedule(self, run_id, plot_run = False):
        '''
        Imports a previously made light schedule from a json file
        '''
        start_time = time.time()
        N = self.N
        F = self.F

        # Open and load the JSON file
        import_path = os.path.join(self.config.data_path, self.import_file)
        with open(import_path, "r") as json_file:
            json_data = json.load(json_file)
        
        if type(json_data) == dict:
            light_schedule = np.array(json_data['Light intensity']).flatten().reshape((1,-1))
        elif type(json_data) == list:
            light_schedule = np.array(json_data).flatten().reshape((1,-1))
        else:
            assert False, f'No light schedule located for file {self.import_file}'

        # Transform from hourly to quarter hourly basis
        # Scale from percentage based schedule to light intensity
        # u_base = self.model.PPFD_max/100*np.repeat(light_schedule, 4)     
        u_base = self.model.PPFD_max/100*light_schedule 

        assert u_base.shape[1] >= self.N, f"{run_id} | Imported light schedule too short. Len: {len(u_base)}, N: {N}"

        x0 = self.x_init
        X = np.zeros((self.model.nx, N+1))
        X[:,0] = x0.reshape(1,-1)
        for k in range(N):
            #Forward euler
            X[:,k+1] = np.array(F(X[:,k], np.array([u_base[:,k]]))).reshape(1, -1)

        sol ={}
        x = X
        u = u_base[:,:N]

        end_time = time.time()
        elapsed_time = end_time - start_time

        sol['elapsed_time'] = elapsed_time
        sol['f'] = self.model.spotopt_obj_function(N, self.spot_prices, u)
        sol['eps'] = 0
        
        dependencies = ('general', 'controller', 'plantmodel', 'market')
        self.store_run(run_id, dependencies, sol, x, u, plot_run = plot_run)

        if not self.surpress_output: print(f'{run_id} | Successfully imported light schedule')
        return 0


    def generate_true_optimum_CM(self, run_id = 'optimal_CM', plot_run = False):

        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')
        if not (self.load_from_json(f"{run_id}_nom", None, dependencies) or self.load_from_json(f"{run_id}", f"{run_id}_nom", dependencies)) : 
            # Identical run located. Using its solution instead
            return 0
        
        if not self.surpress_output: print(f'{run_id} | Generating theoretically optimal Capacity Market bids')
        
        clearing_prices_up, clearing_prices_down = self.market.CM.get_clearing_prices()
        reservations_up, reservations_down       = self.market.CM.get_activations()
        reservations = np.vstack((reservations_up,
                                  reservations_down))

        N = self.N
        T = self.T
        dt = self.dt
        spot_prices = self.spot_prices

        # State and control dimensions
        nx = self.model.nx                      # Dimension of state x (x1, x2)
        nu = self.model.nu                      # Dimension of control u (scalar)
        neps = self.model.neps

        # Create decision variables for the optimization problem
        X_nom       = ca.MX.sym('X_nom', nx, N+1)         # States over time (2x(N+1) vector)
        U_nom       = ca.MX.sym('U_nom', nu, N)
        Eps_nom     = ca.MX.sym('Eps_nom', neps, 1)       # Slack variable for feasibility
        
        X           = ca.MX.sym('X', nx, N+1)             # States over time (2x(N+1) vector)
        B_volumes   = ca.MX.sym('B_volumes', 2, N)        # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        Eps         = ca.MX.sym('Eps', neps, 1)           # Slack variable for feasibility
        
        reservations_up     = reservations_up
        reservations_down   = reservations_down
        bid_volumes_up      = B_volumes[0,:]
        bid_volumes_down    = B_volumes[1,:]
       
        def get_u(N, U_nom, B_volumes, reservations):
            bid_volumes_up      = B_volumes[0,:]
            bid_volumes_down    = B_volumes[1,:]
            reservations_up      = reservations[0,:]
            reservations_down    = reservations[1,:]
       
            U = np.array([])
            for k in range(N):
                u_tilde = 1000/self.model.C_conv_PPFD*(bid_volumes_down[k]*reservations_down[k] - bid_volumes_up[k]*reservations_up[k])
                U = np.append(U, U_nom[:,k] + u_tilde)

            return ca.vertcat(*U).reshape((1,-1))
        
        U = get_u(N, U_nom, B_volumes, reservations)
        # # expected_prices_up, expected_prices_down = self.market.expected_AM_prices_up, self.market.expected_AM_prices_down

        L_nom, L = 0, 0
        for k in range(0, N): #from k = 2, to N-1. 
            L_nom += spot_prices[k] * self.model.C_conv_PPFD/1000 * U_nom[:,k]
            L     += spot_prices[k] * self.model.C_conv_PPFD/1000 * U_nom[:,k] \
                  + (spot_prices[k] - clearing_prices_down[k]) * bid_volumes_down[k] * reservations_down[k]\
                  - (spot_prices[k] + clearing_prices_up[k])   * bid_volumes_up[k]   * reservations_up[k]

        w1, w2 = 0, 1
        J = w1 * (L_nom/4 ) + self.model.terminal_cost(self, X_nom, U_nom, Eps_nom) + \
            w2 * (L/4)      + self.model.terminal_cost(self, X, U, Eps)

        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, X_nom, U_nom, Eps_nom)
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, X, U, Eps)
        g_eq, g_ineq = self.model.get_bidding_constraints(g_eq, g_ineq, N, U_nom, B_volumes)
        
        # format constraints
        n_eq    = ca.vertcat(*g_eq).size()[0]
        n_ineq  = ca.vertcat(*g_ineq).size()[0]
        g       = g_eq + g_ineq
        lbg     = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)                         # \ Eq-constraints = 0
        ubg     = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)   # / Ineq-constraints >= 0


        # Extract state and bidding bounds
        lbx, ubx = self.model.get_state_bounds(self)
        lbu, ubu = self.model.get_input_bounds(self)
        lb_B = np.zeros((2, N))
        ub_B = self.model.P_cap_max * np.ones((2, N))
        lb_eps, ub_eps = np.zeros((neps,1)), np.inf * np.ones((neps,1))

        # Flatten decision variables and bounds
        Z   = ca.vertcat(ca.reshape(X,   -1, 1), ca.reshape(X_nom,  -1, 1), ca.reshape(U_nom,   -1, 1), ca.reshape(B_volumes, -1, 1), ca.reshape(Eps,    -1, 1), ca.reshape(Eps_nom, -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(lbx,    -1, 1), ca.reshape(lbu,     -1, 1), ca.reshape(lb_B,      -1, 1), ca.reshape(lb_eps, -1, 1), ca.reshape(lb_eps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ubx,    -1, 1), ca.reshape(ubu,     -1, 1), ca.reshape(ub_B,      -1, 1), ca.reshape(ub_eps, -1, 1), ca.reshape(ub_eps, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        N_vars = (N+1)*nx*2 + N*nu + N*2 + neps*2
        z0 = ca.DM.zeros(N_vars)  
        # if self.warm_start: 
        #     z0[:nx*(N+1)]                   = refrun['timeseries']['x'].flatten()
        #     z0[nx*(N+1):nx*(N+1)+2*N]       = ub_B[:2,:].reshape(2*N,1)
        
        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution


        X_nom_idx   = nx*(N+1)
        U_idx       = 2*X_nom_idx
        B_idx       = U_idx + nu*N
        eps_idx     = B_idx + 2*N
        eps_nom_idx = eps_idx + neps

        x           = np.array(sol['x'][:X_nom_idx].reshape((nx, N+1)))
        x_nom       = np.array(sol['x'][X_nom_idx:U_idx].reshape((nx, N+1)))
        u_nom       = np.array(sol['x'][U_idx:B_idx].reshape((nu, N)))
        B_volumes   = np.array(sol['x'][B_idx:eps_idx].reshape((2, N)))
        eps         = float(sol['x'][eps_idx:eps_nom_idx][0])
        eps_nom     = float(sol['x'][eps_nom_idx:][0])
        u           = np.array(get_u(N, u_nom, B_volumes, reservations)).reshape((1,-1))

        A = np.vstack((reservations_up, reservations_down))
        B = np.vstack((B_volumes, clearing_prices_up, clearing_prices_down))
        
        end_time = time.time()
        sol['elapsed_time'] = end_time - start_time
        sol['eps'] = eps
        
        market_data = {
            'CM': build_market_participation(B_volumes, np.vstack((clearing_prices_up, clearing_prices_down)), reservations)
        }
        
        # self.store_run(f"{run_id}_nom", dependencies, sol, x_nom, u_nom, refrun_id = 'None', plot_run=plot_run)
        self.store_run(run_id, dependencies, sol, x, u, refrun_id = f"None", market_data=market_data, plot_run=plot_run)

        if not self.surpress_output: print(f'{run_id} | Generated theoretically optimal bid plan')

        self.save_to_json()
        return 0

    def generate_true_optimum_AM(self, run_id = 'optimal', refrun_id = 'fixed', plot_run=False):

        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')
        if not self.load_from_json(run_id, refrun_id, dependencies): 
            # Identical run located. Using its solution instead
            return 0
        
        if not self.surpress_output: print(f'{run_id} | Generating theoretically optimal bidding strategy')
        
        # Just check if there is a basline before proceeding
        assert refrun_id in self.optimization_results['runs'], f"{run_id} | Error: {refrun_id} has not been generated"   
        refrun = self.optimization_results['runs'][refrun_id]

        activations_up, activations_down = self.market.AM.get_activations()
        clearing_prices_up, clearing_prices_down = self.market.AM.get_clearing_prices()
        U_nom = refrun['timeseries']['u']

        N = self.N
        T = self.T
        dt = self.dt
        spot_prices = self.spot_prices

        # State and control dimensions
        nx = self.model.nx                      # Dimension of state x (x1, x2)
        nu = self.model.nu                      # Dimension of control u (scalar)
        neps = self.model.neps

        # Create decision variables for the optimization problem
        X = ca.MX.sym('X', nx, N+1)                 # States over time (2x(N+1) vector)
        B_volumes = ca.MX.sym('B_volumes', 2, N)    # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        Eps = ca.MX.sym('Eps', neps, 1)                # Slack variable for feasibility
        
        activations_up      = activations_up
        activations_down    = activations_down
        bid_volumes_up      = B_volumes[0,:]
        bid_volumes_down    = B_volumes[1,:]
       
        def get_u(N, U_nom, B_volumes, Activations):
            bid_volumes_up      = B_volumes[0,:]
            bid_volumes_down    = B_volumes[1,:]
            activations_up      = Activations[0,:]
            activations_down    = Activations[1,:]
       
            U = np.array([])
            for k in range(N):
                u_tilde = 1000/self.model.C_conv_PPFD*(bid_volumes_down[k]*activations_down[k] - bid_volumes_up[k]*activations_up[k])
                U = np.append(U, U_nom[:,k] + u_tilde)

            return ca.vertcat(*U).reshape((1,-1))
        
        U = get_u(N, U_nom, B_volumes, np.vstack((activations_up, activations_down)))
        # expected_prices_up, expected_prices_down = self.market.expected_AM_prices_up, self.market.expected_AM_prices_down

        L = 0
        for k in range(0, N): #from k = 2, to N-1. 
            L += spot_prices[k] * self.model.C_conv_PPFD/1000 * U_nom[:,k] \
                  + (spot_prices[k] - clearing_prices_down[k]) * bid_volumes_down[k] * activations_down[k]\
                  - (spot_prices[k] + clearing_prices_up[k])   * bid_volumes_up[k]   * activations_up[k]

        J = L/4 + self.model.terminal_cost(self, X, U, Eps)


        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, X, U, Eps)
        
        # format constraints
        n_eq    = ca.vertcat(*g_eq).size()[0]
        n_ineq  = ca.vertcat(*g_ineq).size()[0]
        g       = g_eq + g_ineq
        lbg     = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)                         # \ Eq-constraints = 0
        ubg     = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)   # / Ineq-constraints >= 0


        # Extract state and bidding bounds
        lbx, ubx = self.model.get_state_bounds(self)
        lb_B = np.zeros((2, N))
        ub_B = np.vstack((self.model.C_conv_PPFD * U_nom/1000,                        # Bid vol up
                          self.model.C_conv_PPFD * (self.model.PPFD_max - U_nom)/1000))    # Bid vol down
        lb_eps, ub_eps = np.zeros((neps,1)), np.inf * np.ones((neps,1))

        # Flatten decision variables and bounds
        Z   = ca.vertcat(ca.reshape(X,   -1, 1), ca.reshape(B_volumes, -1, 1),  ca.reshape(Eps, -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(lb_B, -1, 1),       ca.reshape(lb_eps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ub_B, -1, 1),       ca.reshape(ub_eps, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        z0 = ca.DM.zeros((N + 1)*nx + N*2 + neps)  
        if self.warm_start: 
            z0[:nx*(N+1)]                   = refrun['timeseries']['x'].flatten()
            z0[nx*(N+1):nx*(N+1)+2*N]       = ub_B[:2,:].reshape(2*N,1)
        
        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution

        x = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        B_volumes = np.array(sol['x'][(nx*(N+1)):(nx*(N+1) + 2*N)].reshape((2, N)))
        A = np.vstack((activations_up, activations_down))
        B = np.vstack((B_volumes, clearing_prices_up, clearing_prices_down))
        u = np.array(get_u(N, U_nom.reshape(1,-1), B[:2,:], A)).reshape((1,-1))
        eps   = float(sol['x'][-neps][0])
        
        end_time = time.time()
        sol['elapsed_time'] = end_time - start_time
        sol['eps'] = eps
        
        self.store_run(run_id, dependencies, sol, x, u, A=A, B=B, U_nom=refrun['timeseries']['u'].reshape((1,-1)), refrun_id = refrun_id, balancing_market=self.market.AM, plot_run=plot_run)

        if not self.surpress_output: print(f'{run_id} | Generated theoretically optimal bid plan')

        self.save_to_json()
        return 0



    def generate_true_optimum_BL_CM_AM(self, run_id = 'co_opt_BL_CM_AM', plot_run = False):

        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')
        if not (self.load_from_json(f"{run_id}_nom", None, dependencies) or self.load_from_json(f"{run_id}", f"{run_id}_nom", dependencies)) : 
            # Identical run located. Using its solution instead
            return 0
        
        if not self.surpress_output: print(f'{run_id} | Generating theoretically optimal Capacity Market bids')
        
        CM_clearing_prices_up, CM_clearing_prices_down  = self.market.CM.get_clearing_prices()
        CM_activations_up, CM_activations_down          = self.market.CM.get_activations()
        CM_activations = np.vstack((CM_activations_up,
                                    CM_activations_down))
        
        AM_clearing_prices_up, AM_clearing_prices_down  = self.market.AM.get_clearing_prices()
        AM_activations_up,     AM_activations_down      = self.market.AM.get_activations()
        AM_activations = np.vstack((AM_activations_up,
                                    AM_activations_down))

        N = self.N
        T = self.T
        dt = self.dt
        spot_prices = self.spot_prices

        # State and control dimensions
        nx = self.model.nx                      # Dimension of state x (x1, x2)
        nu = self.model.nu                      # Dimension of control u (scalar)
        neps = self.model.neps

        # Create decision variables for the optimization problem
        nom_X       = ca.MX.sym('nom_X', nx, N+1)         # States over time (2x(N+1) vector)
        nom_U       = ca.MX.sym('nom_U', nu, N)
        nom_Eps     = ca.MX.sym('nom_Eps', neps, 1)       # Slack variable for feasibility
        
        CM_X           = ca.MX.sym('CM_X', nx, N+1)             # States over time (2x(N+1) vector)
        CM_B_volumes   = ca.MX.sym('CM_B_volumes', 2, N)        # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        CM_Eps         = ca.MX.sym('CM_Eps', neps, 1)           # Slack variable for feasibility
        
        AM_X           = ca.MX.sym('AM_X', nx, N+1)             # States over time (2x(N+1) vector)
        AM_B_volumes   = ca.MX.sym('AM_B_volumes', 2, N)        # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        AM_Eps         = ca.MX.sym('AM_Eps', neps, 1)           # Slack variable for feasibility
        
        CM_bid_volumes_up     = CM_B_volumes[0,:]
        CM_bid_volumes_down   = CM_B_volumes[1,:]
        AM_bid_volumes_up     = AM_B_volumes[0,:]
        AM_bid_volumes_down   = AM_B_volumes[1,:]
       
        def get_u(N, U_nom, B_volumes, activations):
            bid_volumes_up      = B_volumes[0,:]
            bid_volumes_down    = B_volumes[1,:]
            activations_up      = activations[0,:]
            activations_down    = activations[1,:]
       
            U = np.array([])
            for k in range(N):
                u_tilde = 1000/self.model.C_conv_PPFD*(bid_volumes_down[k]*activations_down[k] - bid_volumes_up[k]*activations_up[k])
                U = np.append(U, U_nom[:,k] + u_tilde)

            return ca.vertcat(*U).reshape((1,-1))
        
        CM_U = get_u(N, nom_U, CM_B_volumes, CM_activations)
        AM_U = get_u(N, nom_U, AM_B_volumes, AM_activations)
        # # expected_prices_up, expected_prices_down = self.market.expected_AM_prices_up, self.market.expected_AM_prices_down

        L = 0
        for k in range(0, N): #from k = 2, to N-1. 
            L   += spot_prices[k] * self.model.C_conv_PPFD/1000 * nom_U[:,k] \
                 - CM_clearing_prices_down[k] * CM_bid_volumes_down[k] * CM_activations_down[k] \
                 - CM_clearing_prices_up[k]   * CM_bid_volumes_up[k]   * CM_activations_up[k] \
                 + (spot_prices[k] - AM_clearing_prices_down[k]) * AM_bid_volumes_down[k] * AM_activations_down[k] \
                 - (spot_prices[k] + AM_clearing_prices_up[k])   * AM_bid_volumes_up[k]   * AM_activations_up[k]

        J = L/4 + self.model.terminal_cost(self, nom_X, nom_U, nom_Eps) \
                + self.model.terminal_cost(self, CM_X, CM_U, CM_Eps) \
                + self.model.terminal_cost(self, AM_X, AM_U, AM_Eps)

        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, nom_X, nom_U, nom_Eps)
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, CM_X, CM_U, CM_Eps)
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, AM_X, AM_U, AM_Eps)
        g_eq, g_ineq = self.model.get_bidding_constraints(g_eq, g_ineq, N, nom_U, CM_B_volumes)
        g_eq, g_ineq = self.model.get_bidding_constraints(g_eq, g_ineq, N, nom_U, AM_B_volumes, B_volumes_lower_bound=CM_B_volumes)
        
        # Enforce hourly bid volumes in capacity market
        for i in range(int(N/4)):
            for j in range(3):
                g_eq.append(CM_B_volumes[0, i+j] - CM_B_volumes[0, i+j+1])
                g_eq.append(CM_B_volumes[1, i+j] - CM_B_volumes[1, i+j+1])

        # format constraints
        n_eq    = ca.vertcat(*g_eq).size()[0]
        n_ineq  = ca.vertcat(*g_ineq).size()[0]
        g       = g_eq + g_ineq
        lbg     = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)                         # \ Eq-constraints = 0
        ubg     = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)   # / Ineq-constraints >= 0


        # Extract state and bidding bounds
        lbx, ubx = self.model.get_state_bounds(self)
        lbu, ubu = self.model.get_input_bounds(self)
        lb_B = np.zeros((2, N))
        ub_B = self.model.P_cap_max * np.ones((2, N))
        lb_eps, ub_eps = np.zeros((neps,1)), np.inf * np.ones((neps,1))

        # Flatten decision variables and bounds
        Z   = ca.vertcat(ca.reshape(nom_X, -1, 1), ca.reshape(CM_X, -1, 1), ca.reshape(AM_X, -1, 1), ca.reshape(nom_U, -1, 1), ca.reshape(CM_B_volumes, -1, 1), ca.reshape(AM_B_volumes, -1, 1), ca.reshape(nom_Eps, -1, 1), ca.reshape(CM_Eps, -1, 1), ca.reshape(AM_Eps, -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx,   -1, 1), ca.reshape(lbx,  -1, 1), ca.reshape(lbx,  -1, 1), ca.reshape(lbu,   -1, 1), ca.reshape(lb_B,         -1, 1), ca.reshape(lb_B,         -1, 1), ca.reshape(lb_eps,  -1, 1), ca.reshape(lb_eps, -1, 1), ca.reshape(lb_eps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx,   -1, 1), ca.reshape(ubx,  -1, 1), ca.reshape(ubx,  -1, 1), ca.reshape(ubu,   -1, 1), ca.reshape(ub_B,         -1, 1), ca.reshape(ub_B,         -1, 1), ca.reshape(ub_eps,  -1, 1), ca.reshape(ub_eps, -1, 1), ca.reshape(ub_eps, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        N_vars = (N+1)*nx*3 + N*nu + N*2*2 + neps*3
        z0 = ca.DM.zeros(N_vars)
        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution

        nom_X_idx   = 0
        CM_X_idx    = nom_X_idx     + nx*(N+1)
        AM_X_idx    = CM_X_idx      + nx*(N+1)
        nom_U_idx   = AM_X_idx      + nx*(N+1)
        CM_B_idx    = nom_U_idx     + nu*N
        AM_B_idx    = CM_B_idx      + 2*N
        nom_eps_idx = AM_B_idx      + 2*N
        CM_eps_idx  = nom_eps_idx   + neps
        AM_eps_idx  = CM_eps_idx    + neps

        nom_x     = np.array(sol['x'][:CM_X_idx].reshape((nx, N+1)))
        CM_x      = np.array(sol['x'][CM_X_idx:AM_X_idx].reshape((nx, N+1)))
        AM_x      = np.array(sol['x'][AM_X_idx:nom_U_idx].reshape((nx, N+1)))
        nom_u     = np.array(sol['x'][nom_U_idx:CM_B_idx].reshape((nu, N)))
        CM_B_vols = np.array(sol['x'][CM_B_idx:AM_B_idx].reshape((2, N)))
        AM_B_vols = np.array(sol['x'][AM_B_idx:nom_eps_idx].reshape((2, N)))
        nom_eps   = float(sol['x'][nom_eps_idx:CM_eps_idx][0])
        CM_eps    = float(sol['x'][CM_eps_idx:AM_eps_idx][0])
        AM_eps    = float(sol['x'][AM_eps_idx:][0])
        CM_u      = np.array(get_u(N, nom_u, CM_B_vols, CM_activations)).reshape((1,-1))
        AM_u      = np.array(get_u(N, nom_u, AM_B_vols, AM_activations)).reshape((1,-1))

        CM_A = np.vstack((CM_activations_up, CM_activations_down))
        CM_B = np.vstack((CM_B_vols, CM_clearing_prices_up, CM_clearing_prices_down))
        AM_A = np.vstack((AM_activations_up, AM_activations_down))
        AM_B = np.vstack((AM_B_vols, AM_clearing_prices_up, AM_clearing_prices_down))
        
        end_time = time.time()
        sol['elapsed_time'] = end_time - start_time
        
        nom_sol, CM_sol, AM_sol = sol.copy(), sol.copy(), sol.copy()
        nom_sol['eps']  = nom_eps
        CM_sol['eps']   = CM_eps
        AM_sol['eps']   = AM_eps
        
        market_participation = {
            'CM'    : build_market_participation(CM_B_vols, np.vstack((CM_clearing_prices_up, CM_clearing_prices_down)), CM_A),
            'AM'    : build_market_participation(AM_B_vols, np.vstack((AM_clearing_prices_up, AM_clearing_prices_down)), AM_A)
        }

        self.store_run(f"{run_id}", dependencies, AM_sol, AM_x, AM_u, U_nom=nom_u, market_data = market_participation, refrun_id = f"spot_opt", plot_run = plot_run)

        if not self.surpress_output: print(f'{run_id} | Generated theoretically optimal bid plan')

        self.save_to_json()
        return 0



    def generate_true_optimum_CM_AM(self, run_id = 'co_opt_CM_AM', refrun_id = 'Fixed', plot_run = False):

        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')
        if not (self.load_from_json(f"{run_id}_nom", None, dependencies) or self.load_from_json(f"{run_id}", f"{run_id}_nom", dependencies)) : 
            # Identical run located. Using its solution instead
            return 0
        
        if not self.surpress_output: print(f'{run_id} | Generating theoretically optimal Capacity Market bids')

        assert refrun_id in self.optimization_results['runs'], f"{run_id} | Error: {refrun_id} has not been generated"   
        refrun = self.optimization_results['runs'][refrun_id]

        
        CM_clearing_prices_up, CM_clearing_prices_down  = self.market.CM.get_clearing_prices()
        CM_activations_up, CM_activations_down          = self.market.CM.get_activations()
        CM_activations = np.vstack((CM_activations_up,
                                    CM_activations_down))
        
        AM_clearing_prices_up, AM_clearing_prices_down  = self.market.AM.get_clearing_prices()
        AM_activations_up,     AM_activations_down      = self.market.AM.get_activations()
        AM_activations = np.vstack((AM_activations_up,
                                    AM_activations_down))

        N = self.N
        T = self.T
        dt = self.dt
        spot_prices = self.spot_prices

        # State and control dimensions
        nx = self.model.nx                      # Dimension of state x (x1, x2)
        nu = self.model.nu                      # Dimension of control u (scalar)
        neps = self.model.neps

        # Create decision variables for the optimization problem
        # nom_X       = ca.MX.sym('nom_X', nx, N+1)         # States over time (2x(N+1) vector)
        # nom_U       = ca.MX.sym('nom_U', nu, N)
        # nom_Eps     = ca.MX.sym('nom_Eps', neps, 1)       # Slack variable for feasibility

        nom_U = refrun['timeseries']['u']
        
        CM_X           = ca.MX.sym('CM_X', nx, N+1)             # States over time (2x(N+1) vector)
        CM_B_volumes   = ca.MX.sym('CM_B_volumes', 2, N)        # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        CM_Eps         = ca.MX.sym('CM_Eps', neps, 1)           # Slack variable for feasibility
        
        AM_X           = ca.MX.sym('AM_X', nx, N+1)             # States over time (2x(N+1) vector)
        AM_B_volumes   = ca.MX.sym('AM_B_volumes', 2, N)        # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        AM_Eps         = ca.MX.sym('AM_Eps', neps, 1)           # Slack variable for feasibility
        
        CM_bid_volumes_up     = CM_B_volumes[0,:]
        CM_bid_volumes_down   = CM_B_volumes[1,:]
        AM_bid_volumes_up     = AM_B_volumes[0,:]
        AM_bid_volumes_down   = AM_B_volumes[1,:]
       
        def get_u(N, U_nom, B_volumes, activations):
            bid_volumes_up      = B_volumes[0,:]
            bid_volumes_down    = B_volumes[1,:]
            activations_up      = activations[0,:]
            activations_down    = activations[1,:]
       
            U = np.array([])
            for k in range(N):
                u_tilde = 1000/self.model.C_conv_PPFD*(bid_volumes_down[k]*activations_down[k] - bid_volumes_up[k]*activations_up[k])
                U = np.append(U, U_nom[:,k] + u_tilde)

            return ca.vertcat(*U).reshape((1,-1))
        
        CM_U = get_u(N, nom_U, CM_B_volumes, CM_activations)
        AM_U = get_u(N, nom_U, AM_B_volumes, AM_activations)
        # # expected_prices_up, expected_prices_down = self.market.expected_AM_prices_up, self.market.expected_AM_prices_down

        L = 0
        for k in range(0, N): #from k = 2, to N-1. 
            L   += spot_prices[k] * self.model.C_conv_PPFD/1000 * nom_U[:,k] \
                 - CM_clearing_prices_down[k] * CM_bid_volumes_down[k] * CM_activations_down[k] \
                 - CM_clearing_prices_up[k]   * CM_bid_volumes_up[k]   * CM_activations_up[k] \
                 + (spot_prices[k] - AM_clearing_prices_down[k]) * AM_bid_volumes_down[k] * AM_activations_down[k] \
                 - (spot_prices[k] + AM_clearing_prices_up[k])   * AM_bid_volumes_up[k]   * AM_activations_up[k]

        J = L/4 \
                + self.model.terminal_cost(self, CM_X, CM_U, CM_Eps) \
                + self.model.terminal_cost(self, AM_X, AM_U, AM_Eps)

        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, CM_X, CM_U, CM_Eps)
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, AM_X, AM_U, AM_Eps)
        g_eq, g_ineq = self.model.get_bidding_constraints(g_eq, g_ineq, N, nom_U, CM_B_volumes)
        g_eq, g_ineq = self.model.get_bidding_constraints(g_eq, g_ineq, N, nom_U, AM_B_volumes, B_volumes_lower_bound=CM_B_volumes)
        
        # Enforce hourly bid volumes in capacity market
        for i in range(int(N/4)):
            for j in range(3):
                g_eq.append(CM_B_volumes[0, i+j] - CM_B_volumes[0, i+j+1])
                g_eq.append(CM_B_volumes[1, i+j] - CM_B_volumes[1, i+j+1])

        # format constraints
        n_eq    = ca.vertcat(*g_eq).size()[0]
        n_ineq  = ca.vertcat(*g_ineq).size()[0]
        g       = g_eq + g_ineq
        lbg     = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)                         # \ Eq-constraints = 0
        ubg     = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)   # / Ineq-constraints >= 0


        # Extract state and bidding bounds
        lbx, ubx = self.model.get_state_bounds(self)
        lbu, ubu = self.model.get_input_bounds(self)
        lb_B = np.zeros((2, N))
        ub_B = self.model.P_cap_max * np.ones((2, N))
        lb_eps, ub_eps = np.zeros((neps,1)), np.inf * np.ones((neps,1))

        # Flatten decision variables and bounds
        Z   = ca.vertcat(ca.reshape(CM_X, -1, 1), ca.reshape(AM_X, -1, 1), ca.reshape(CM_B_volumes, -1, 1), ca.reshape(AM_B_volumes, -1, 1), ca.reshape(CM_Eps, -1, 1), ca.reshape(AM_Eps, -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx,  -1, 1), ca.reshape(lbx,  -1, 1), ca.reshape(lb_B,         -1, 1), ca.reshape(lb_B,         -1, 1), ca.reshape(lb_eps, -1, 1), ca.reshape(lb_eps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx,  -1, 1), ca.reshape(ubx,  -1, 1), ca.reshape(ub_B,         -1, 1), ca.reshape(ub_B,         -1, 1), ca.reshape(ub_eps, -1, 1), ca.reshape(ub_eps, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        N_vars = (N+1)*nx*2 + N*2*2 + neps*2
        z0 = ca.DM.zeros(N_vars)
        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution

        CM_X_idx    = 0
        AM_X_idx    = CM_X_idx      + nx*(N+1)
        CM_B_idx    = AM_X_idx      + nx*(N+1)
        AM_B_idx    = CM_B_idx      + 2*N
        CM_eps_idx  = AM_B_idx      + 2*N
        AM_eps_idx  = CM_eps_idx    + neps

        CM_x      = np.array(sol['x'][CM_X_idx:AM_X_idx].reshape((nx, N+1)))
        AM_x      = np.array(sol['x'][AM_X_idx:CM_B_idx].reshape((nx, N+1)))
        CM_B_vols = np.array(sol['x'][CM_B_idx:AM_B_idx].reshape((2, N)))
        AM_B_vols = np.array(sol['x'][AM_B_idx:CM_eps_idx].reshape((2, N)))
        CM_eps    = float(sol['x'][CM_eps_idx:AM_eps_idx][0])
        AM_eps    = float(sol['x'][AM_eps_idx:][0])
        CM_u      = np.array(get_u(N, nom_U, CM_B_vols, CM_activations)).reshape((1,-1))
        AM_u      = np.array(get_u(N, nom_U, AM_B_vols, AM_activations)).reshape((1,-1))

        CM_A = np.vstack((CM_activations_up, CM_activations_down))
        CM_B = np.vstack((CM_B_vols, CM_clearing_prices_up, CM_clearing_prices_down))
        AM_A = np.vstack((AM_activations_up, AM_activations_down))
        AM_B = np.vstack((AM_B_vols, AM_clearing_prices_up, AM_clearing_prices_down))
        
        end_time = time.time()
        sol['elapsed_time'] = end_time - start_time
        
        nom_sol, CM_sol, AM_sol = sol.copy(), sol.copy(), sol.copy()
        CM_sol['eps']   = CM_eps
        AM_sol['eps']   = AM_eps
        
        market_participation = {
            'CM'    : build_market_participation(CM_B_vols, np.vstack((CM_clearing_prices_up, CM_clearing_prices_down)), CM_A),
            'AM'    : build_market_participation(AM_B_vols, np.vstack((AM_clearing_prices_up, AM_clearing_prices_down)), AM_A)
        }

        self.store_run(f"{run_id}", dependencies, AM_sol, AM_x, AM_u, U_nom=nom_U, market_data = market_participation, refrun_id = f"spot_opt", plot_run = plot_run)

        if not self.surpress_output: print(f'{run_id} | Generated theoretically optimal bid plan')

        self.save_to_json()
        return 0



    def export_intensity_to_json(self, run_id: str):

        u = np.array(self.optimization_results['runs'][run_id]['timeseries']['u']).flatten()

        u_scaled = 100 * u / self.model.PPFD_max

        intensity_schedule_dict = {'Light intensity': u_scaled}
        # Filepath
        sim_name = self.sim_name
        inty_save_path = os.path.join(self.config.output_path, f"{sim_name}_{run_id}_inty_schedule_{self.market.date}.json")

        # Ensure the target json file exists
        os.makedirs(self.config.simulations_path, exist_ok=True)

        # Convert the entire runs dictionary
        intensity_schedule = convert_np_arrays_to_lists(intensity_schedule_dict)

        # Save the data for all runs
        with open(inty_save_path, "w") as json_file:
            json.dump(intensity_schedule, json_file, indent=4)


        


    def status_report(self):
        '''
        Extract and print metrics from the optimization. Outputs metrics in tables. 
        '''

        costs       = [self.optimization_results['runs'][run]['metrics']['Costs']       for run in self.optimization_results['runs']]
        earnings    = [self.optimization_results['runs'][run]['metrics']['Earnings']    for run in self.optimization_results['runs']]
        totals      = [self.optimization_results['runs'][run]['metrics']['Total']       for run in self.optimization_results['runs']]
        cost_reduction_percent = [(totals[0] - totals[i])/totals[0] * 100 for i in range(len(totals))]

        cost_data = [
            ['Costs'] + costs,
            ['Earnings'] + earnings,
            ['Totals'] + totals,
            ['Cost reduction (%)'] + cost_reduction_percent,
        ]

        cost_table = generate_table(cost_data, header=[run for run in self.optimization_results['runs']])
        print(f'COST DATA: \n{cost_table}\n')


        metrics_table = get_metrics_table(self.optimization_results['runs'])
        print(f'METRICS DATA: \n{metrics_table}\n')


        # Print bidding metrics
        for run in self.optimization_results['runs']:
            if 'bidding result' not in self.optimization_results['runs'][run]:
                continue

            bidding_result_up = self.optimization_results['runs'][run]['bidding result']['Up-regulation']
            bidding_result_dn = self.optimization_results['runs'][run]['bidding result']['Down-regulation']

            balancing_market = self.market.get_balancing_market(self.optimization_results['runs'][run]['Attributes']['Balancing_market'])
            
            bidding_data = [
                ['Avg bid size',                        bidding_result_up['Avg bid size'],                              bidding_result_dn['Avg bid size'],                              "MW"], 
                ['Avg bid price',                       bidding_result_up['Avg bid price'],                             bidding_result_dn['Avg bid price'],                             "€/MW"], 
                ['Avg activation rate',                 bidding_result_up['Avg activation rate'],                       bidding_result_dn['Avg activation rate'],                       "%"], 
                ['Chance of activation given demand',   bidding_result_up['Avg activation rate']/balancing_market.demand_prob_up(), bidding_result_dn['Avg activation rate']/balancing_market.demand_prob_down(), "%"],
                ['Impact on consumption',               bidding_result_up['Consumption impact'],                        bidding_result_dn['Consumption impact'],                        "MW"],
                ['Submitted bids',                      bidding_result_up['Bids submitted'],                            bidding_result_dn['Bids submitted'],                            "-"]
            ]
            bidding_header = ['', 'Up-regulation', 'Down-regulation', 'Unit']

            print(f'BIDDING REPORT {run}: \n{generate_table(bidding_data, header = bidding_header)}\n')


        for _, balancing_market in self.market.balancing_markets.items():
            # Print market metrics
            market_data = [
                # ['Mean Expected clearing price', np.mean(self.market.expected_AM_prices_up), np.mean(self.market.expected_AM_prices_down)],
                ['Mean Recorded clearing price', np.mean(balancing_market.clearing_prices_up), np.mean(balancing_market.clearing_prices_down)],
                ['Mean Recorded activated clearing price', np.mean(balancing_market.clearing_prices_up[np.where(balancing_market.activations_up >0)]), np.mean(balancing_market.clearing_prices_down[np.where(balancing_market.activations_down >0)])],
                # ['Mean Expected / Recorded clearing price delta',  np.mean(self.market.expected_AM_prices_up - balancing_market.get_clearing_prices()[0]), np.mean(self.market.expected_AM_prices_down - balancing_market.get_clearing_prices()[1])],
                # ['Clearing price standard deviation', self.market.sigma_AM_up, self.market.sigma_AM_down], 
                ['Expected activation occurence rate', balancing_market.demand_prob_up(), balancing_market.demand_prob_down()],
                ['Recorded activation occurence rate', np.mean(balancing_market.activations_up), np.mean(balancing_market.activations_down)]
            ]
            market_header = ['', 'Up-regulation', 'Down-regulation']

            print(f'MARKET REPORT: \n{generate_table(market_data, header = market_header)}\n')


        # Print solve times
        for run in self.optimization_results['runs']:
            minutes, seconds = divmod(self.optimization_results['runs'][run]['metrics']['elapsed_time'], 60)
            print(f"{run} solved in: {int(minutes)} minutes and {seconds:.2f} seconds. ")
        

