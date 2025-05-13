import numpy as np
import casadi as ca
from globals import *
from model import PlantModel
from market import Market
from tqdm import tqdm
import pandas as pd


class MPCSimulation():
    
    # settings:   Settings
    model:      PlantModel
    market:     Market
    # controller: Controller

    # N: float
    # T: float
    # dt: float

    def __init__(self, controller, run_id, target_run_id):

        self.run_id = run_id

        self.controller = controller
        self.T      = controller.T
        self.N      = controller.N                      # Number of time steps for the whole optimization problem
        self.N_TH   = controller.mpc_N_horizon       # Number of time steps for internal open-loop solver
        self.N_iter = controller.mpc_N_step        # Number of time steps between each open-loop solution
        self.market = controller.market
        self.model  = controller.model
        self.spot_prices = controller.spot_prices
        self.nx, self.nu, self.neps = self.model.nx, self.model.nu, self.model.neps
        self.CM     = self.market.CM
        self.AM     = self.market.AM
        self.F      = controller.F
        self.surpress_output = controller.surpress_output


        self.terminate_simulation = False
        self.mtu_start = self.market.MTU_start

        self.simulation_days            = list(range(int(np.ceil(self.T))))
        self.quarter_hourly_intervals   = list(range(0, QUARTER_HOURS_PER_DAY, self.N_iter))


        self.target_run_id   = target_run_id
        self.target_run      = self.controller.optimization_results['runs'][target_run_id]
        self.target_X        = self.target_run['timeseries']['x']
        self.target_U        = self.target_run['timeseries']['u']
        self.target_weight   = self.model.freshweight(self.target_X)


        return
    

    def setup_optimizers(self):


        # Set up optimizers
        self.opti_CM, self.opt_vars_CM = setup_optimizer(self.nx, self.nu, self.neps, self.N_TH, 'CM') # Handles daily optimization
        self.opti_AM, self.opt_vars_AM = setup_optimizer(self.nx, self.nu, self.neps, self.N_TH, 'AM') # Handles quarter-hourly optimization

        # Set up constraints using parameters
        self.opti_CM  = set_constraints(self.controller, self.opti_CM, self.N_TH, self.opt_vars_CM)
        self.opti_AM  = set_constraints(self.controller, self.opti_AM, self.N_TH, self.opt_vars_AM)


        return


    def setup_statevectors(self):

        self.X_log, self.past_X, self.U_log, self.U_nom_log, self.CM_bids_log, self.AM_bids_log, self.Eps_log = setup_simulation_statevectors(self.controller, self.model, self.target_run_id)

        return
        


    def run_mpc(self):

        with tqdm(total=self.N, desc=f"{self.run_id}: Running MPC") as pbar:

            self.pbar = pbar

            for day in self.simulation_days:
                for qh in self.quarter_hourly_intervals:

                    self.day, self.qh = day, qh

                    self.get_iteration()

                    if qh == 0:     # Start of every day only

                        # At 00:00 every day
                        # Bid on CM for the COMING day
                        self.solve_CM_bids()

                        # # Store solution
                        self.store_CM_solution()

                        if self.terminate_simulation: break
                        # End if qh == 0
                        

                    self.extract_CM_bid_result()

                    self.solve_AM_bids()

                    self.store_AM_solution()

                    self.integrate_model()

                    pbar.update(min(self.N_iter, QUARTER_HOURS_PER_DAY))
                    if self.terminate_simulation: break
                    # End for qh in day

                if self.terminate_simulation: break
                # End for day

        return 0
    


    def get_iteration(self):

        day = self.day
        qh  = self.qh

        self.k              = day*QUARTER_HOURS_PER_DAY + qh
        self.current_MTU    = self.mtu_start + pd.Timedelta(minutes = 15*self.k)
        self.N_horizon      = min(self.N-self.k, self.N_TH)
        self.start_iter     = self.k
        self.end_iter       = self.k + self.N_horizon
        self.CM_iter_slice  = slice(self.start_iter, self.end_iter)

        self.AM_N_horizon    = min(self.N_TH - qh, self.N_horizon)
        self.AM_iter_slice   = slice(self.start_iter, self.start_iter + self.AM_N_horizon)



    def solve_CM_bids(self):
        market          = self.market
        model           = self.model
        opt_vars_CM     = self.opt_vars_CM
        N_horizon       = self.N_horizon
        

        opti_CM_copy = update_optimizer_CM_bids(self, market, model,
                        self.opti_CM.copy(), MTU = self.current_MTU, N_TH = N_horizon, opt_vars = opt_vars_CM, spot_prices = self.spot_prices[self.CM_iter_slice],
                        x0          = self.X_log[:,self.start_iter], 
                        past_X      = self.past_X,
                        ref_weight  = self.target_weight[self.end_iter]
        )

        # Solve CM bidding
        self.pbar.set_postfix(status=f"Solving CM, MTU: {self.current_MTU}") 

        try: 
            sol_CM              = opti_CM_copy.solve()
        except Exception as e:
            print(f"\nCM Solver failed, using last known values.\n{e}")
            terminate_simulation = True
            sol_CM              = opti_CM_copy.debug
            if not self.surpress_output: check_violated_constraints(opti_CM_copy)
        
        
        self.x_opt_CM            = sol_CM.value(opt_vars_CM['X'])
        self.u_nom_opt_CM        = sol_CM.value(opt_vars_CM['U_nom']).reshape((1, -1))
        self.CM_bid_volumes_opt  = sol_CM.value(opt_vars_CM['B_volumes'])[:,:N_horizon]
        self.CM_bid_prices_opt   = sol_CM.value(opt_vars_CM['B_prices'])[:,:N_horizon]
        self.Eps_opt_CM          = sol_CM.value(opt_vars_CM['Eps'])
        self.Eps_nom_opt_CM      = sol_CM.value(opt_vars_CM['Eps_nom'])

        return


    def store_CM_solution(self):
        
        CM = self.CM
        N_horizon = self.N_horizon
        
        
        # Store solution

        # Apply CM bids to CM market
        # Extract required AM bid volumes 
        self.CM_activations, self.CM_activated_volumes, self.CM_earnings = CM.subject_bids_to_market_data(self.current_MTU, self.CM_bid_volumes_opt, self.CM_bid_prices_opt)

        # Store data

        CM_extract_solution_slice   = slice(0,              N_horizon)
        CM_store_data_slice         = slice(self.k, self.k + N_horizon)
        
        # Store inputs
        self.U_nom_log[:,CM_store_data_slice] = self.u_nom_opt_CM[:,CM_extract_solution_slice]         

        # Store CM bid data
        self.CM_bids_log[:2, CM_store_data_slice] = self.CM_bid_volumes_opt[:,CM_extract_solution_slice]
        self.CM_bids_log[2:4,CM_store_data_slice] = self.CM_bid_prices_opt[:, CM_extract_solution_slice]


    def extract_CM_bid_result(self):

        AM = self.AM
        qh = self.qh

        self.AM_B_volumes_min = self.CM_activated_volumes[:,qh:]
        self.max_prices_reserved    = np.vstack(AM.get_estimated_clearing_prices(start_date=self.current_MTU, n_data = self.AM_N_horizon))
        self.AM_B_prices_max        = np.where(self.CM_activations[:,qh:], self.max_prices_reserved, AM.bid_price_limit) # TODO Define a proper upper bound for the bidding price

        return



    def solve_AM_bids(self):

        market          = self.market
        model           = self.model
        opt_vars_AM     = self.opt_vars_AM
        AM_N_horizon    = self.AM_N_horizon
        AM_iter_slice   = self.AM_iter_slice

        # Update AM bid optimizer
        opti_AM_copy = update_optimizer_AM_bids(self, market, model,
                        self.opti_AM.copy(), MTU = self.current_MTU, N_TH = AM_N_horizon, opt_vars = opt_vars_AM, spot_prices = self.spot_prices[AM_iter_slice],
                        x0          = self.X_log[:,self.start_iter], 
                        past_X      = self.past_X,
                        U_nom       = self.U_nom_log[:, AM_iter_slice], 
                        ref_weight  = self.target_weight[self.end_iter],
                        B_volumes_min = self.AM_B_volumes_min,
                        B_prices_max  = self.AM_B_prices_max
        )

        # Solve AM bidding
        self.pbar.set_postfix(status=f"Solving AM, MTU: {self.current_MTU}") 
        try: 
            sol_AM                      = opti_AM_copy.solve()
        except Exception as e:
            print(f"\nAM Solver failed, using last known values.\n{e}")
            self.terminate_simulation   = True
            sol_AM                      = opti_AM_copy.debug
            if not self.surpress_output: check_violated_constraints(opti_AM_copy)
    
        self.x_opt_AM               = sol_AM.value(opt_vars_AM['X'])
        self.AM_bid_volumes_opt     = sol_AM.value(opt_vars_AM['B_volumes'])[:,:AM_N_horizon]
        self.AM_bid_prices_opt      = sol_AM.value(opt_vars_AM['B_prices'])[:,:AM_N_horizon]
        self.Eps_opt_AM             = sol_AM.value(opt_vars_AM['Eps'])
                                
        return


    def store_AM_solution(self):

        market          = self.market
        AM_N_horizon    = self.AM_N_horizon
        N_iter          = self.N_iter
        k               = self.k

        # Apply bid activations
        # Evaluate activations
        self.AM_activations, self.AM_activated_volumes, self.AM_earnings = market.AM.subject_bids_to_market_data(self.current_MTU, self.AM_bid_volumes_opt, self.AM_bid_prices_opt)

        self.AM_extract_solution_slice = slice(0,     min(N_iter, AM_N_horizon)) if not self.terminate_simulation else slice(0,     AM_N_horizon)
        self.AM_store_data_slice       = slice(k, k + min(N_iter, AM_N_horizon)) if not self.terminate_simulation else slice(k, k + AM_N_horizon)

        self.AM_activated_volumes_up    = self.AM_activated_volumes[0,:]
        self.AM_activated_volumes_down  = self.AM_activated_volumes[1,:]

        # Store AM bid data
        self.AM_bids_log[0:2, self.AM_store_data_slice] = self.AM_bid_volumes_opt[:, self.AM_extract_solution_slice]
        self.AM_bids_log[2:4, self.AM_store_data_slice] = self.AM_bid_prices_opt[:,  self.AM_extract_solution_slice]

        return




    def integrate_model(self):

        k               = self.k
        AM_N_horizon    = self.AM_N_horizon
        N_iter          = self.N_iter
        X_log           = self.X_log
        U_nom_log       = self.U_nom_log
        U_log           = self.U_log


        u_tilde = 1000/self.model.C_conv_PPFD * (self.AM_activated_volumes_down - self.AM_activated_volumes_up)
        u = (np.array(U_nom_log[:,self.AM_iter_slice]).flatten() + u_tilde)[self.AM_extract_solution_slice]
        U_log[:,self.AM_store_data_slice] = u

        iteration_range = range(k, k+min(N_iter, AM_N_horizon)) if not self.terminate_simulation else range(k, k + AM_N_horizon)
        for i in iteration_range:
            self.X_log[:,i+1] = np.array(self.F(X_log[:,i], np.array([U_log[:,i]]))).reshape(1, -1)

        self.past_X[:,-min(QUARTER_HOURS_PER_DAY, min(N_iter, AM_N_horizon)):] = X_log[:,k:k+min(QUARTER_HOURS_PER_DAY, min(N_iter, AM_N_horizon))]

        # TODO fix eps
        self.Eps_log = np.array([[float(max(0, self.target_weight[-1] - self.model.freshweight(X_log[:,-1])))], [0], [0]])


        return






def setup_simulation_statevectors(controller, model: PlantModel, target_run_id):

    target_run  = controller.optimization_results['runs'][target_run_id]
    target_X    = target_run['timeseries']['x']
    N  = controller.N
    nx = model.nx
    nu = model.nu

    # Simulation state vectors
    X = ca.DM.zeros(nx, N+1)
    X[:,0] = controller.x_init
    past_X = target_X[:,:QUARTER_HOURS_PER_DAY].copy()      # Used for backwards DLI calculation
    U = ca.DM.zeros(nu, N)                                  # System input after subjected to bid activations
    U_nom = ca.DM.zeros(nu, N)                              # Baseline system input
    CM_bid_log = ca.DM.zeros(4, N)                          # Capacity Market Bids
    AM_bid_log = ca.DM.zeros(4, N)                          # Activation Market Bids
    Eps = ca.DM.zeros(model.neps, 1)                        # Slack variables

    return X, past_X, U, U_nom, CM_bid_log, AM_bid_log, Eps




def setup_optimizer(nx, nu, neps, N_horizon, opti_type: str):
    '''Creates opti variables. Creates opt_vars dictionaries containing opti symbolic optimization variables'''

    opti = ca.Opti()
    opts = {'ipopt.print_level':0, 'print_time':0}
    opti.solver('ipopt', opts)

    X           = opti.variable(nx, N_horizon+1)
    Eps         = opti.variable(neps, 1)
    B_prices    = opti.variable(2*nu, N_horizon)
    B_volumes   = opti.variable(2*nu, N_horizon)

    x0          = opti.parameter(nx, 1)         # Starting weight
    ref_weight  = opti.parameter(1, 1)          # End weight (To be substituted)
    spot_prices = opti.parameter(1, N_horizon)  # Spot prices for optimization window
    Est_prices  = opti.parameter(2*nu, N_horizon)


    # opti.set_value(spot_prices, np.mean(self.spot_prices)* ca.DM.ones(spot_prices.shape))
    opti.set_value(spot_prices, 10000 * ca.DM.ones(spot_prices.shape))
    opti.set_value(Est_prices,  1000*ca.DM.ones(Est_prices.shape))

    opt_vars = {'opti_type'     : opti_type,
                'N_horizon'     : N_horizon,
                'X'             : X,
                'x0'            : x0,
                'Eps'           : Eps,
                'spot_prices'   : spot_prices,
                'ref_weight'    : ref_weight,
                'B_volumes'     : B_volumes,
                'B_prices'      : B_prices,
                'Est_prices'    : Est_prices
                }
    

    if opti_type=='CM':
        # Add U and nominal U as variables
        X_nom       = opti.variable(nx, N_horizon+1)
        U_nom       = opti.variable(nu, N_horizon)
        Eps_nom     = opti.variable(neps, 1)

        # Register opti_variables to opt_vars
        opt_vars['U_nom']       = U_nom
        opt_vars['X_nom']       = X_nom
        opt_vars['Eps_nom']     = Eps_nom
        return opti, opt_vars
    
    elif opti_type=='AM':
        
        # Nominal U as parameter
        U_nom       = opti.parameter(nu, N_horizon)
        Req_volumes = opti.parameter(2*nu, N_horizon)
        Max_prices  = opti.parameter(2*nu, N_horizon)

        # Initialize parameters with 0-values
        opti.set_value(U_nom,       ca.DM.zeros(U_nom.shape))
        opti.set_value(Req_volumes, ca.DM.zeros(Req_volumes.shape))
        opti.set_value(Max_prices,  1000*ca.DM.ones(Max_prices.shape))

        # Register opti_variables and opti_params to opt_vars
        opt_vars['U_nom']       = U_nom
        opt_vars['Req_volumes'] = Req_volumes   # Required volumes (from CM participation)
        opt_vars['Max_prices']  = Max_prices    # Max bid prices   (for AM bids required by CM reservations)
        return opti, opt_vars
        
    else:
        assert False, f'SETUP MPC | invalid opti type: {opti_type}'



def set_constraints(controller, opti: ca.Opti, N_TH, opt_vars: dict):
    
    # Extract symbolic optimization variables and parameters
    X           = opt_vars['X']
    Eps         = opt_vars['Eps']
    x0          = opt_vars['x0']
    spot_prices = opt_vars['spot_prices']
    ref_weight  = opt_vars['ref_weight']
    B_volumes   = opt_vars['B_volumes']
    B_prices    = opt_vars['B_prices']
    Est_prices  = opt_vars['Est_prices']
            
    g_eq, g_ineq = [], []

    if opt_vars['opti_type'] == 'AM':
        U_nom       = opt_vars['U_nom']
        Req_volumes = opt_vars['Req_volumes']
        Max_prices  = opt_vars['Max_prices']
        U = controller.model.get_u(N_TH, U_nom=U_nom, B_volumes=B_volumes, B_prices=B_prices, spot_prices=spot_prices, balancing_market=controller.market.AM, clearing_prices = Est_prices).reshape((1,-1))
        g_eq, g_ineq = controller.model.get_static_process_constraints(g_eq, g_ineq, N_TH, controller.dt, X, x0, U, Eps)
        g_eq, g_ineq = controller.model.get_bidding_constraints(g_eq, g_ineq, N_TH, U_nom, controller.market.AM, B_prices = B_prices, B_volumes=B_volumes, 
                                                            B_volumes_lower_bound=Req_volumes, B_prices_upper_bound=Max_prices)
    
    elif opt_vars['opti_type'] == 'CM':
        U_nom   = opt_vars['U_nom']
        X_nom   = opt_vars['X_nom']
        Eps_nom = opt_vars['Eps_nom']
        U = controller.model.get_u_CM(N_TH, U_nom=U_nom, CM_B_volumes=B_volumes, CM_B_prices=B_prices, spot_prices=spot_prices, CM=controller.market.CM, AM=controller.market.AM, CM_clearing_prices=Est_prices).reshape((1,-1))
        g_eq, g_ineq = controller.model.get_static_process_constraints(g_eq, g_ineq, N_TH, controller.dt, X_nom, x0, U_nom, Eps_nom)
        g_eq, g_ineq = controller.model.get_static_process_constraints(g_eq, g_ineq, N_TH, controller.dt, X, x0, U, Eps)
        g_eq, g_ineq = controller.model.get_bidding_constraints(g_eq, g_ineq, N_TH, U_nom, controller.market.CM, B_volumes = B_volumes, B_prices = B_prices)
    
    else:
        assert False, f'Inconsistent opti_type: {opt_vars["opti_type"]}'


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




def update_optimizer_CM_bids(controller, market: Market, model: PlantModel, opti: ca.Opti, MTU, N_TH, opt_vars, spot_prices, x0, past_X, ref_weight): #, init_U = None, init_X = None):

    X           = opt_vars['X']
    X_nom       = opt_vars['X_nom']
    U_nom       = opt_vars['U_nom']
    Eps         = opt_vars['Eps']
    Eps_nom     = opt_vars['Eps_nom']
    B_volumes   = opt_vars['B_volumes']
    B_prices    = opt_vars['B_prices']
    Est_prices  = opt_vars['Est_prices']

    CM_prices_up, CM_prices_down = market.CM.get_estimated_clearing_prices(start_date = MTU, n_data = N_TH)
    CM_prices = np.vstack((CM_prices_up, CM_prices_down))

    g_eq, g_ineq = [], []
    g_eq, g_ineq = model.get_dynamic_process_constraints(g_eq, g_ineq, N_TH, X,     Eps,     ref_weight, past_X = past_X)
    g_eq, g_ineq = model.get_dynamic_process_constraints(g_eq, g_ineq, N_TH, X_nom, Eps_nom, ref_weight, past_X = past_X)
    for equality_constraint   in g_eq:   opti.subject_to(equality_constraint   == 0)   
    for inequality_constraint in g_ineq: opti.subject_to(inequality_constraint >= 0) 

    U = model.get_u_CM(N_TH, U_nom, B_volumes, B_prices, spot_prices, CM = market.CM, AM = market.AM, CM_clearing_prices=Est_prices)
    # U = self.model.get_u(N_TH, U_nom, B_volumes, B_prices, spot_prices, self.market.CM)
    J = model.elcost_obj_function(N_TH, spot_prices, U) \
        + model.CM_bidding_obj_function(N_TH, spot_prices, B_volumes, B_prices, market.CM, MTU_start = MTU)\
        + model.terminal_cost(controller, X, U, Eps)\
        + model.terminal_cost(controller, X_nom, U_nom, Eps_nom)
    
    opti.minimize(J)

    # Specify initial guesses

    U_nom_initguess = model.PPFD_max * np.ones(opt_vars['U_nom'][:,:N_TH].shape) / 2
    lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices = model.get_bidding_bounds(N_TH, U_nom_initguess, market.CM)
    B_volumes_initguess = ub_B_volumes
    B_prices_initguess = lb_B_prices

    opti.set_initial(opt_vars['U_nom'][:,:N_TH],     U_nom_initguess)
    opti.set_initial(opt_vars['B_volumes'][:,:N_TH], B_volumes_initguess)
    opti.set_initial(opt_vars['B_prices'][:,:N_TH],  B_prices_initguess)

    U_initguess = model.get_u_CM(N_TH, U_nom_initguess, B_volumes_initguess, B_prices_initguess, spot_prices, CM = market.CM, AM = market.AM, CM_clearing_prices = CM_prices)
    X_initguess = model.simulate_growth(x0, U_initguess)
    X_nom_initguess = model.simulate_growth(x0, U_nom_initguess)

    opti.set_initial(opt_vars['X'][:,:N_TH+1],     X_initguess)
    opti.set_initial(opt_vars['X_nom'][:,:N_TH+1], X_nom_initguess)


    # Specify parameter values

    opti.set_value(opt_vars['x0'], x0)
    opti.set_value(opt_vars['ref_weight'], ref_weight)
    opti.set_value(opt_vars['spot_prices'][:,:N_TH], spot_prices)

    opti.set_value(opt_vars['Est_prices'][:,:N_TH], CM_prices)

    return opti

def update_optimizer_AM_bids(controller, market: Market, model: PlantModel, opti: ca.Opti, MTU, N_TH, opt_vars, spot_prices, x0, past_X, U_nom, ref_weight, B_volumes_min, B_prices_max):

    X           = opt_vars['X']
    Eps         = opt_vars['Eps']
    B_volumes   = opt_vars['B_volumes']
    B_prices    = opt_vars['B_prices']
    Est_prices  = opt_vars['Est_prices']

    g_eq, g_ineq = [], []
    g_eq, g_ineq = model.get_dynamic_process_constraints(g_eq, g_ineq, N_TH, X, Eps, ref_weight, past_X = past_X)
    for equality_constraint   in g_eq:   opti.subject_to(equality_constraint   == 0)   
    for inequality_constraint in g_ineq: opti.subject_to(inequality_constraint >= 0) 

    U = model.get_u(N_TH, U_nom, B_volumes, B_prices, spot_prices, market.AM, clearing_prices=Est_prices)
    J = model.elcost_obj_function(N_TH, spot_prices, U)\
        + model.AM_bidding_obj_function(N_TH, spot_prices, B_volumes, B_prices, market.AM, MTU_start=MTU)\
        + model.terminal_cost(controller, X, U, Eps)
            
    opti.minimize(J)

    # update parameters
    opti.set_value(opt_vars['x0'],                      x0)
    opti.set_value(opt_vars['U_nom'][:,:N_TH],          U_nom)
    opti.set_value(opt_vars['Req_volumes'][:,:N_TH],    B_volumes_min)
    opti.set_value(opt_vars['Max_prices'][:,:N_TH],     B_prices_max)
    opti.set_value(opt_vars['ref_weight'],              ref_weight)
    opti.set_value(opt_vars['spot_prices'][:,:N_TH],    spot_prices)
    
    AM_prices_up, AM_prices_down = market.AM.get_estimated_clearing_prices(start_date = MTU, n_data = N_TH)
    AM_prices = np.vstack((AM_prices_up, AM_prices_down))
    opti.set_value(opt_vars['Est_prices'][:,:N_TH], AM_prices)
    # opti.set_value(opt_vars['B_volumes'][:,:N_TH], B_max_volumes)


    # Set initial guesses

    # U_nom_initguess = self.model.PPFD_max * np.ones(opt_vars['U_nom'][:,:N_TH].shape) / 2
    lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices = model.get_bidding_bounds(N_TH, U_nom, market.AM)
    B_volumes_initguess = ub_B_volumes
    B_prices_initguess = lb_B_prices

    # opti.set_initial(opt_vars['U_nom'][:,:N_TH],     U_nom_initguess)
    opti.set_initial(opt_vars['B_volumes'][:,:N_TH], B_volumes_initguess)
    opti.set_initial(opt_vars['B_prices'][:,:N_TH],  B_prices_initguess)

    U_initguess = model.get_u(N_TH, U_nom, B_volumes_initguess, B_prices_initguess, spot_prices, balancing_market = market.AM, clearing_prices = AM_prices)
    X_initguess = model.simulate_growth(x0, U_initguess)

    opti.set_initial(opt_vars['X'][:,:N_TH+1],     X_initguess)

    # Set initial guesses

    return opti




def check_violated_constraints(opti, tol=1e-6):
    """
    Checks for violated constraints in a failed CasADi Opti solve.

    Parameters:
        opti : casadi.Opti
            The Opti object after a failed solve attempt (with debug info).
        tol : float
            Tolerance for considering a constraint violated.

    Returns:
        List of tuples: [(i, residual, constraint_expr)] for violated constraints
    """
    violations = []
    
    constraints = opti.g
    lb = opti.lbg
    ub = opti.ubg
    values = opti.debug.value(constraints)

    constraints_split = ca.vertsplit(constraints)

    for i, constr in enumerate(constraints_split):
        val = float(values[i])
        lb_i = float(lb[i])
        ub_i = float(ub[i])

        if val < lb_i - tol or val > ub_i + tol:
            residual = max(val - ub_i, lb_i - val)
            violations.append({
                "index": i,
                "value": val,
                "lower_bound": lb_i,
                "upper_bound": ub_i,
                "residual": residual,
                "expr": constr,
            })

    for v in violations:
        print(f"[Constraint #{v['index']}] Violation by {v['residual']:.2e}")
        print(f"  Expr:        {v['expr']}")
        print(f"  Value:       {v['value']:.4f}")
        print(f"  Lower Bound: {v['lower_bound']}")
        print(f"  Upper Bound: {v['upper_bound']}\n")

    return