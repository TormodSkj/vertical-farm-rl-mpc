import casadi as ca
import numpy as np
from market import Market
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
    """The controller handles open-loop optimization given a model and a set of constraints."""

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
        timehorizon, 
        mpc_timehorizon, mpc_steplength,
        warm_start = False, calculate_fw = False):
        '''
        self.settings = settings
        self.controller_settings = settings.get_settings_group('general', 'controller', 'market', 'plantmodel')

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
        

        self.specs = { 
            'time horizon'          : self.T,
            'N'                     : self.N,
            'mpc time horizon'      : self.mpc_T_horizon,
            'mpc step time'         : self.mpc_T_step,
            'calculate freshweight' : self.calculate_fw,
            'warm start'            : self.warm_start
        }
        specs_data = {
            'controller' : self.specs,
            'model'      : self.model.specs,
            'market'     : self.market.specs
        }

        self.optimization_results = {
            'name'              : self.config.sim_name,
            'timestamp'         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'specs'             : specs_data,
            'runs'              : {}
            }
        self.hash = generate_hash(self.optimization_results['specs'])

        self.t = np.linspace(0, self.T, self.N)
        self.spot_prices = self.market.get_spotprice()

        self.bids = []
        for i in range(self.market.n_given_bids):
            self.bids.append(Bid())

        self.A_up, self.A_down = [],[]
        for i in range(self.market.n_given_activations):
            self.A_up.append(0)
            self.A_down.append(0)


        # Set initial state
        self.x_init = self.model.x_init

        # Generate freshweight for the mpc bidding controller to use as reference trajectory
        self.fixed_light_schedule()   
        
 
        
    def set_bids(self, Bid_0, Bid_1):
        self.Bid_0 = Bid_0
        self.Bid_1 = Bid_1



    def optimize_mfrr(self, run_id, refrun_id = 'fixed'):

        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')

        if not self.load_from_json(run_id, dependencies): 
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

        U = self.model.get_u(N, U_nom, B[:2,:], B[2:4,:], spot_prices, market)           # Express U in terms of bidding outcomes

        # Initialize cost function and constraints
        J = self.model.bidding_obj_function(N, self.spot_prices, X, B_volumes = B[:2,:], B_prices = B[2:4,:], U_nom = U_nom, market = self.market)\
                       + self.model.terminal_cost(self, X, U, Eps)#\
                       #+ self.model.fluctuating_light_cost(self, U) # Cost function


        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, X, U, Eps)
        # g_eq, g_ineq = self.model.get_static_process_constraints(g_eq, g_ineq, N, dt, X, self.x_init, U, Eps)
        # g_eq, g_ineq = self.model.get_dynamic_process_constraints(g_eq, g_ineq, N, X, Eps, self.model.Final_fw_sht)
        # g_eq, g_ineq = self.model.get_initial_bid_constraints(self, g_eq, g_ineq, B)

        
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
            z0[:nx*(N+1)]                   = refrun['timeseries']['x'].flatten()   # TODO check if works as intended
            z0[nx*(N+1):nx*(N+1)+2*N]       = ub_B_volumes.reshape((2*N,1))         # Bid volumes
            z0[nx*(N+1)+2*N:nx*(N+1)+3*N]   = self.market.expected_prices_up        # Bid prices up
            z0[nx*(N+1)+3*N:nx*(N+1)+4*N]   = self.market.expected_prices_down        # Bid prices down
        
        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution

        x = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        B = np.array(sol['x'][(nx*(N+1)):(nx*(N+1) + 4*N)].reshape((4, N)))
        u = np.array(self.model.get_u(N, U_nom, B[:2,:], B[2:4,:], spot_prices, market)).reshape(1,-1)
        eps   = float(sol['x'][-neps][0])

        
        end_time = time.time()
        sol['elapsed_time'] = end_time - start_time
        sol['eps'] = eps
        
        self.store_run(run_id, dependencies, sol, x, u, B=B, U_nom=refrun['timeseries']['u'], refrun_id = refrun_id)

        if not self.surpress_output: print(f'{run_id} | Optimized mFRR bidding strategy')
        
        self.save_to_json()
        return 0
        

    def optimize_spotprice(self, run_id: str, refrun_id = 'fixed'):
        '''
        Optimizes the light schedule based on spot price. Used as reference for mFRR optimization (Called baseline in MARI terms)
        '''
        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')

        if not self.load_from_json(run_id, dependencies): 
            # Identical run located. Using its solution instead
            return 0
        
        if not self.surpress_output: print(f'{run_id} | Optimizing light schedule based on spot price')
        
        assert refrun_id in self.optimization_results['runs'], f"{run_id} | Error: {refrun_id} has not been generated"   
        refrun = self.optimization_results['runs'][refrun_id]

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

        J = self.model.spotopt_obj_function(N, self.spot_prices, X, U)\
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
        if self.warm_start:
            z0[:nx*(N+1)]                   = refrun['timeseries']['x'].flatten()
            z0[nx*(N+1):(nx*(N+1)+N*nu)]    = refrun['timeseries']['u'].flatten()

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

        self.store_run(run_id, dependencies, sol, x, u, refrun_id = refrun_id)

        if not self.surpress_output: print(f'{run_id} | Optimized light schedule for spot price')
        
        self.save_to_json()
        return 0



    def optimize_mfrr_mpc(self, run_id, target_run_id = 'fixed'):

        start_time = time.time()
        
        dependencies = ('general', 'controller', 'mpc', 'plantmodel', 'market')

        if not self.load_from_json(run_id, dependencies): 
            # Identical run located. Using its solution instead
            return 0

        if not self.surpress_output: print(f'{run_id} | Generating bidding strategy using mpc')
        
        N = self.N                      # Number of time steps for the whole optimization problem
        N_TH = self.mpc_N_horizon       # Number of time steps for internal open-loop solver
        N_iter = self.mpc_N_step        # Number of time steps between each open-loop solution
        spot_prices = self.spot_prices
        nx, nu, neps = self.model.nx, self.model.nu, self.model.neps
        market = self.market
        F = self.model.casadi_function_fe()

        clearing_prices_up, clearing_prices_down = market.get_clearing_prices(self.market.date)
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

                #TODO temp solution
                Eps = max(0, target_weight[-1] - self.model.freshweight(X[:,-1]))

                k += N_iter

                pbar.update(min(N_iter, N_horizon))

        end_time = time.time()
        sol = {}
        sol['eps'] = Eps
        sol['f'] = self.model.bidding_obj_function(N, spot_prices, X, B[:2, :], B[2:4, :], U_nom, self.market)
        sol['elapsed_time'] = end_time - start_time
        
        self.store_run(run_id, dependencies, sol, np.array(X), np.array(U).reshape((1,-1)), B=np.array(B), U_nom=np.array(U_nom).reshape((1,-1)), refrun_id = target_run_id)

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

            U = self.model.get_u(N_TH, U_nom=U_nom, B_volumes=B_volumes, B_prices=B_prices, spot_prices=spot_prices, market=self.market).reshape((1,-1))
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

        J = self.model.spotopt_obj_function(N_TH, spot_prices, X, U) + self.model.terminal_cost(self, X, U, Eps)

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

        U = self.model.get_u(N_TH, U_nom, B_volumes, B_prices, spot_prices, self.market)
        J = self.model.bidding_obj_function(N_TH, spot_prices, X, B_volumes, B_prices, U_nom, self.market) + self.model.terminal_cost(self, X, U, Eps)
        
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
        clearing_price_mu = conditional_expectation(spot_prices, self.market.price_means, self.market.price_covs)
        clearing_price_mu_up = clearing_price_mu[0]
        clearing_price_mu_dn = clearing_price_mu[1]

        # clearing_price_mu_up = 10*clearing_price_mu[0]
        # clearing_price_mu_dn = 10*clearing_price_mu[1]

        # Set initial optimal bidding guess to be maximum possible volume and exactly at clearing price
        B_prices_initial_guess = np.vstack((clearing_price_mu_up, clearing_price_mu_dn))
        U_initial_guess = np.array(self.model.get_u(N_TH, U_nom, B_prices_initial_guess, B_max_volumes, spot_prices, self.market)).flatten()
        X_initial_guess = ca.DM.zeros(self.model.nx, N_TH+1)
        X_initial_guess[:,0] = x0
        F = self.model.casadi_function_fe()
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
        F = self.model.casadi_function_rk()

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
        sol['f'] = self.model.spotopt_obj_function(N, self.spot_prices, X, u)
        sol['eps'] = 0
        
        
        dependencies = ('general', 'controller', 'plantmodel', 'market')
        self.store_run(run_id, dependencies, sol, X, u, refrun_id = 'None')
        if self.calculate_fw: self.model.Final_fw_sht = float(self.model.freshweight(X[:,-1]))

        if not self.surpress_output: print(f'{run_id} | Generated fixed light schedule: {hours_on}h/{hours_off}h at {RIGID_INTY} PPFD')
        return 0





    def store_run(self, run_id, dependencies, sol, x, u, A = None, B = None, U_nom = None, refrun_id = 'None'):

        timeseries_data = {
            't'     : self.t,
            'x'     : x,
            'u'     : u
        }

        if U_nom is not None:
            timeseries_data['u_nom'] = U_nom

        
        f       = float(sol['f'])
        eps     = float(sol['eps'])


        metrics_data = {
            'elapsed_time'  : sol['elapsed_time'],
            'f'             : f,
            'eps'           : eps
        }

        metrics_data = self.model.get_metrics(self, run_id, metrics_data, x, u, B)

        run_data = {
            'reference_run' : refrun_id,
            'metrics'       : metrics_data
            }

        if B is None:
            costs = float(self.model.spotopt_obj_function(self.N, self.spot_prices, x, u))
            metrics_data['Costs'] = costs
            metrics_data['Earnings'] = 0
            metrics_data['Total'] = costs - 0
        else:

            bid_volumes_up      = B[0,:].reshape(1,-1)
            bid_volumes_down    = B[1,:].reshape(1,-1)
            bid_prices_up       = B[2,:].reshape(1,-1)
            bid_prices_down     = B[3,:].reshape(1,-1)

            if A is None:
                prob_activations_up     = np.array(self.market.activation_prob_up(self.spot_prices, bid_prices_up)).reshape(1,-1)
                prob_activations_down   = np.array(self.market.activation_prob_down(self.spot_prices, bid_prices_down)).reshape(1,-1)
                bid_activations_up      = prob_activations_up
                bid_activations_down    = prob_activations_down
            else:
                bid_activations_up      = A[0,:].reshape(1,-1)
                bid_activations_down    = A[1,:].reshape(1,-1)
                timeseries_data['A_up'] = bid_activations_up
                timeseries_data['A_dn'] = bid_activations_down
                prob_activations_up     = np.array(self.market.activation_prob_up(self.spot_prices, bid_prices_up)).reshape(1,-1)
                prob_activations_down   = np.array(self.market.activation_prob_down(self.spot_prices, bid_prices_down)).reshape(1,-1)

            timeseries_data['P_up'] = bid_volumes_up
            timeseries_data['P_dn'] = bid_volumes_down
            timeseries_data['C_up'] = bid_prices_up
            timeseries_data['C_dn'] = bid_prices_down


            expected_prices_up, expected_prices_down = self.market.expected_prices_up, self.market.expected_prices_down
            bidding_earnings_up     = self.market.C_eur2nok * 1/4 * np.multiply(np.multiply(bid_activations_up,     bid_volumes_up),    expected_prices_up)
            bidding_earnings_down   = self.market.C_eur2nok * 1/4 * np.multiply(np.multiply(bid_activations_down,   bid_volumes_down),  expected_prices_down)
            bidding_earnings    = np.sum(bidding_earnings_up) + np.sum(bidding_earnings_down)

            bidding_costs = self.model.spotopt_obj_function(self.N, self.spot_prices, x, u)
            bidding_total = bidding_costs - bidding_earnings

            metrics_data['Costs']       = float(bidding_costs)
            metrics_data['Earnings']    = float(bidding_earnings)
            metrics_data['Total']       = float(bidding_total)

            # b_a_up  = np.array(self.market.activation_prob_up(self.spot_prices, b_c_up))
            # b_a_dn  = np.array(self.market.activation_prob_dn(self.spot_prices,b_c_dn))

            activation_th   = 0.01
            volume_th       = 0.001

            up_bids = np.where(np.logical_and(prob_activations_up > activation_th, bid_volumes_up > volume_th))
            down_bids = np.where(np.logical_and(prob_activations_down > activation_th, bid_volumes_down > volume_th))

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

            run_data['bidding result'] = bidding_data       

        run_data['timeseries']      = timeseries_data
        run_data['dependencies']    = list(dependencies)
        run_data['hash']            = generate_hash(self.settings.get_settings_group(*dependencies))

        # Storing runs in dictionaries
        self.optimization_results['runs'][run_id] = run_data 


    def save_to_json(self):
        """
        Save all runs and their data to a JSON file.
        """

        # Add spot price to data
        self.optimization_results['spotprice'] = self.spot_prices

        # Filepath
        sim_name = self.config.sim_name
        sim_save_path = os.path.join(self.config.sim_path, f"{sim_name}.json")

        # Ensure the target json file exists
        os.makedirs(self.config.sim_path, exist_ok=True)

        # Convert the entire runs dictionary
        runs_dict = convert_np_arrays_to_lists(self.optimization_results)

        # Save the data for all runs
        with open(sim_save_path, "w") as json_file:
            json.dump(runs_dict, json_file, indent=4)


    def load_from_json(self, run_id, dependencies):
        """
        Load completed runs from saved JSON files and populate `completed_runs`.
        Compares the settings_profile used previously in order to determine if old result is still valid

        Returns 0 if match is made.
        Returns 1 if match is not made
        """
        if not self.search_cache: return 1

        hash = generate_hash(self.settings.get_settings_group(*dependencies))

        for file_name in os.listdir(self.config.sim_path):
            if not file_name.endswith(".json"): continue

            file_path = os.path.join(self.config.sim_path, file_name)
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

            if not self.surpress_output: print(f"Loaded run {run_id} from simulation \'{loaded_data['name']}\' dated {loaded_data['timestamp']}")
            return 0
                
        if not self.surpress_output: print(f'{run_id} | No matching run found')
        return 1

    def import_light_schedule(self, sim_id):
        '''
        Imports a previously made light schedule from a json file
        '''
        start_time = time.time()
        N = self.N
        F = self.model.casadi_function_rk()

        # Open and load the JSON file
        import_path = os.path.join(self.config.data_path, self.import_file)
        with open(import_path, "r") as json_file:
            light_schedule = json.load(json_file)
        

        # Transform from hourly to quarter hourly basis
        # Scale from percentage based schedule to light intensity
        u_base = self.model.PPFD_max/100*np.repeat(light_schedule, 4)     

        assert len(u_base) >= self.N, f"{sim_id} | Imported light schedule too short. Len: {len(u_base)}, N: {N}"

        x0 = self.x_init
        X = np.zeros((self.model.nx, N+1))
        X[:,0] = x0.reshape(1,-1)
        for k in range(N):
            #Forward euler
            X[:,k+1] = np.array(F(X[:,k], np.array([u_base[k]]))).reshape(1, -1)

        sol ={}
        x = X
        u = u_base[:N]

        end_time = time.time()
        elapsed_time = end_time - start_time

        sol['elapsed_time'] = elapsed_time
        sol['f'] = self.model.spotopt_obj_function(N, self.spot_prices, x, u)
        sol['x'] = np.hstack((x.flatten(), u, 0))
        sol['eps'] = 0
        
        dependencies = ('general', 'controller', 'plantmodel', 'market')
        self.store_run('Imported', dependencies, sol, x, u)

        if not self.surpress_output: print(f'{sim_id} | Successfully imported light schedule')
        return 0


    def generate_optimal_bidding_strategy(self, run_id = 'optimal', refrun_id = 'fixed'):

        start_time = time.time()

        dependencies = ('general', 'controller', 'plantmodel', 'market')
        if not self.load_from_json(run_id, dependencies): 
            # Identical run located. Using its solution instead
            return 0
        
        if not self.surpress_output: print(f'{run_id} | Generating theoretically optimal bidding strategy')
        
        # Just check if there is a basline before proceeding
        assert refrun_id in self.optimization_results['runs'], f"{run_id} | Error: {refrun_id} has not been generated"   
        refrun = self.optimization_results['runs'][refrun_id]

        activations_up, activations_down = self.market.get_activation_demands()
        clearing_prices_up, clearing_prices_down = self.market.get_clearing_prices()
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
        bid_prices_up       = clearing_prices_up
        bid_prices_down     = clearing_prices_down

       
        U = np.array([])
        for k in range(N):
            u_tilde = 1000*(bid_volumes_down[k]*activations_down[k]\
                             - bid_volumes_up[k]*activations_up[k])/self.model.C_conv_PPFD
            U = np.append(U, U_nom[:,k] + u_tilde)

        U = ca.vertcat(*U)

        expected_prices_up, expected_prices_down = self.market.expected_prices_up, self.market.expected_prices_down

        L = 0
        for k in range(0, N): #from k = 2, to N-1. 
            L += spot_prices[k] * self.model.C_conv_PPFD * U_nom[:,k] \
                  + (1000*spot_prices[k] - self.market.C_eur2nok * expected_prices_down[:,k]) * bid_volumes_down[k] * activations_down[k]\
                  - (1000*spot_prices[k] + self.market.C_eur2nok * expected_prices_up[:,k]) * bid_volumes_up[k] * activations_up[k]

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
        B = np.vstack((B_volumes, clearing_prices_up, clearing_prices_down))
        u = np.array(self.model.get_u(N, U_nom.reshape(1,-1), B[:2,:], B[2:4,:], spot_prices, self.market)).reshape((1,-1))
        eps   = float(sol['x'][-neps][0])
        A = np.vstack((activations_up, activations_down))
        
        end_time = time.time()
        sol['elapsed_time'] = end_time - start_time
        sol['eps'] = eps
        
        self.store_run(run_id, dependencies, sol, x, u, B=B, U_nom=refrun['timeseries']['u'].reshape((1,-1)), refrun_id = refrun_id)

        if not self.surpress_output: print(f'{run_id} | Generated theoretically optimal bid plan')
        return 0




    def export_intensity_to_json(self, run_id: str):

        u = self.optimization_results['runs'][run_id]['timeseries']['u']

        u_scaled = 100 * u / self.model.PPFD_max

        intensity_schedule_dict = {'Light intensity': u_scaled}
        # Filepath
        sim_name = self.config.sim_name
        inty_save_path = os.path.join(self.config.output_path, f"{sim_name}_{run_id}_inty_schedule_{self.market.date}.json")

        # Ensure the target json file exists
        os.makedirs(self.config.sim_path, exist_ok=True)

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
            ['Total percentage cost reduction'] + cost_reduction_percent,
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
            
            bidding_data = [
                ['Avg bid size',                        bidding_result_up['Avg bid size'],                              bidding_result_dn['Avg bid size'],                              "MW"], 
                ['Avg bid price',                       bidding_result_up['Avg bid price'],                             bidding_result_dn['Avg bid price'],                             "€/MW"], 
                ['Avg activation rate',                 bidding_result_up['Avg activation rate'],                       bidding_result_dn['Avg activation rate'],                       "%"], 
                ['Chance of activation given demand',   bidding_result_up['Avg activation rate']/self.market.demand_prob_up(), bidding_result_dn['Avg activation rate']/self.market.demand_prob_down(), "%"],
                ['Impact on consumption',               bidding_result_up['Consumption impact'],                        bidding_result_dn['Consumption impact'],                        "MW"],
                ['Submitted bids',                      bidding_result_up['Bids submitted'],                            bidding_result_dn['Bids submitted'],                            "-"]
            ]
            bidding_header = ['', 'Up-regulation', 'Down-regulation', 'Unit']

            print(f'BIDDING REPORT {run}: \n{generate_table(bidding_data, header = bidding_header)}\n')


        # Print market metrics
        market_data = [
            ['Mean Expected clearing price', np.mean(self.market.expected_prices_up), np.mean(self.market.expected_prices_down)],
            ['Mean Recorded clearing price', np.mean(self.market.get_clearing_prices()[0]), np.mean(self.market.get_clearing_prices()[1])],
            ['Mean Recorded activated clearing price', np.mean(self.market.get_clearing_prices()[0][np.where(self.market.mfrr_demands_up >0)]), np.mean(self.market.get_clearing_prices()[1][np.where(self.market.mfrr_demands_down >0)])],
            ['Mean Expected / Recorded clearing price delta',  np.mean(self.market.expected_prices_up - self.market.get_clearing_prices()[0]), np.mean(self.market.expected_prices_down - self.market.get_clearing_prices()[1])],
            ['Clearing price standard deviation', self.market.sigma_up, self.market.sigma_dn], 
            ['Expected activation occurence rate', self.market.demand_prob_up(), self.market.demand_prob_down()],
            ['Recorded activation occurence rate', np.mean(self.market.mfrr_demands_up), np.mean(self.market.mfrr_demands_down)]
        ]
        market_header = ['', 'Up-regulation', 'Down-regulation']

        print(f'MARKET REPORT: \n{generate_table(market_data, header = market_header)}\n')


        # Print solve times
        for run in self.optimization_results['runs']:
            minutes, seconds = divmod(self.optimization_results['runs'][run]['metrics']['elapsed_time'], 60)
            print(f"{run} solved in: {int(minutes)} minutes and {seconds:.2f} seconds. ")
        

