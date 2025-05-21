import numpy as np
import casadi as ca
from globals import *
from model import PlantModel
from market import Market
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from balancingmarket import BalancingMarket
from matplotlib.patches import Patch



class MPCSimulation():
    
    # settings:   Settings
    model:      PlantModel
    market:     Market
    CM:         BalancingMarket
    AM:         BalancingMarket
    # controller: Controller

    # N: float
    # T: float
    # dt: float

    def __init__(self, controller, run_id, target_run_id):

        self.run_id = run_id

        mpc_settings = controller.mpc_settings

        self.controller = controller
        self.T      = controller.T
        self.N      = controller.N                                  # Number of time steps for the whole optimization problem
        self.N_TH   = min(controller.mpc_N_horizon, controller.N)   # Number of time steps for internal open-loop solver
        self.N_iter = controller.mpc_N_step                         # Number of time steps between each open-loop solution
        self.market = controller.market
        self.model  = controller.model
        self.spot_prices = controller.spot_prices
        self.nx, self.nu, self.neps = self.model.nx, self.model.nu, self.model.neps
        self.CM     = self.market.CM
        self.AM     = self.market.AM
        self.CM_N_bids = max(QUARTER_HOURS_PER_DAY, min(mpc_settings['CM_N_BIDS'], self.N_TH))
        self.AM_N_bids = max(self.N_iter, min(mpc_settings['AM_N_BIDS'], self.N_TH))
        
        self.CM_N_initial_bids = 96
        self.CM_N_bids_to_submit = 96
        self.AM_N_initial_bids = 3
        self.AM_N_bids_to_submit = self.N_iter

        self.F      = controller.F
        self.surpress_output = controller.surpress_output


        self.terminate_simulation = False
        self.mtu_start = self.market.MTU_start
        self.start_MTU = self.mtu_start

        self.simulation_days            = list(range(int(np.ceil(self.T))))
        self.quarter_hourly_intervals   = list(range(0, QUARTER_HOURS_PER_DAY, self.N_iter))


        self.target_run_id   = target_run_id
        self.target_run      = self.controller.optimization_results['runs'][target_run_id]
        self.target_X        = self.target_run['timeseries']['x']
        self.target_U        = self.target_run['timeseries']['u']
        self.target_weight   = self.model.freshweight(self.target_X)


        self.optimization_schedule_df = pd.DataFrame(columns=['day', 'qh', 'optimizer'] + [k+1 for k in range(self.N)], index=range(1, 2 + int(self.T) + int(np.ceil(self.N/self.N_iter)))).fillna(0)
        self.n_schedule_entries = 0

        return
    

    def setup_optimizers(self):


        # Set up optimizers
        self.opti_CM, self.opt_vars_CM = setup_optimizer(self.controller, self.nx, self.nu, self.neps, self.N_TH, self.CM_N_bids, 'CM') # Handles daily optimization
        self.opti_AM, self.opt_vars_AM = setup_optimizer(self.controller, self.nx, self.nu, self.neps, self.N_TH, self.AM_N_bids, 'AM') # Handles quarter-hourly optimization

        # Set up constraints using parameters
        self.opti_CM  = set_constraints(self.controller, self.market, self.model, self.opti_CM, self.N_TH, self.CM_N_bids, self.opt_vars_CM)
        self.opti_AM  = set_constraints(self.controller, self.market, self.model, self.opti_AM, self.N_TH, self.AM_N_bids, self.opt_vars_AM)


        return


    def setup_statevectors(self):

        target_run  = self.controller.optimization_results['runs'][self.target_run_id]
        target_X    = target_run['timeseries']['x']
        N  = self.N
        nx = self.model.nx
        nu = self.model.nu

        # Simulation state vectors
        X               = ca.DM.zeros(nx, N+1)
        X[:,0]          = self.controller.x_init

        self.X_log              = X             
        self.past_X             = target_X[:,:QUARTER_HOURS_PER_DAY].copy()     # Used for backwards DLI calculation            
        self.U_log              = ca.DM.zeros(nu, N)                            # System input after subjected to bid activations
        self.U_nom_log          = ca.DM.zeros(nu, N)                            # Baseline system input
        self.CM_bids_log        = ca.DM.zeros(4, N)                             # Capacity Market Bids
        self.AM_bids_log        = ca.DM.zeros(4, N)                             # Activation Market Bids
        self.Eps_log            = ca.DM.zeros(self.model.neps, 1)               # Slack variables
        
        self.CM_activations_log = ca.DM.zeros(2, N)
        self.AM_activations_log = ca.DM.zeros(2, N)

        self.CM_bid_submissions = ca.DM.zeros(4,0)
        self.AM_bid_submissions = ca.DM.zeros(4,0)

        self.CM_bid_results = ca.DM.zeros(4,0)          # [:2,:]: Activations, [2:4,:]: Volumes
        self.AM_bid_results = ca.DM.zeros(4,0)          # [:2,:]: Activations, [2:4,:]: Volumes

        return
        


    def run_mpc(self):

        with tqdm(total=self.N, desc=f"{self.run_id}: Running MPC") as pbar:

            self.pbar = pbar

            self.day = 0   # Day ahead
            self.qh = 0     # 23:00
            self.get_iteration()

            self.solve_initial_CM_bids()
            self.store_CM_solution()

            self.extract_CM_bid_result()
            self.solve_initial_AM_bids()
            self.store_AM_solution()
            self.update_AM_bid_results()


            for day in self.simulation_days:
                for qh in self.quarter_hourly_intervals:

                    self.day, self.qh = day, qh

                    self.get_iteration()

                    if qh == 0:     # Start of every day only

                        # At 00:00 every day
                        # Bid on CM for the COMING day
                        self.solve_CM_bids()
                        self.store_CM_solution()

                        if self.terminate_simulation: 
                            break
                        # End if qh == 0
                            
                    self.extract_CM_bid_result()
                    self.update_AM_bid_results()
                    self.integrate_model()

                    self.solve_AM_bids()
                    self.store_AM_solution()


                    
                    if self.terminate_simulation: 
                        break
                    else: 
                        pbar.update(min(self.N_iter, QUARTER_HOURS_PER_DAY))
                    # End of qh in day

                if self.terminate_simulation: 
                        break
                # End of day


        self.day +=1
        self.qh = 0

        self.get_iteration()
        self.update_AM_bid_results()
        self.integrate_model()

        return 0
        



    def solve_initial_CM_bids(self):
        
        '''
        IN PROGRESS
        '''        

        market          = self.market
        model           = self.model
        opt_vars_CM     = self.opt_vars_CM
        N_horizon       = self.N_horizon        
        self.CM_N_horizon = N_horizon

        opti_CM_copy = update_optimizer_CM_bids(self.controller, market, model,
                        self.opti_CM.copy(), MTU = self.current_MTU, N_TH = N_horizon, N_bids = self.CM_N_bids, opt_vars = opt_vars_CM, spot_prices = self.spot_prices[self.CM_iter_slice],
                        x0          = self.X_log[:,self.start_iter], 
                        past_X      = self.past_X,
                        ref_weight  = self.target_weight[self.end_iter]
        )

        # Solve CM bidding
        self.pbar.set_postfix(status=f"Solving Initial CM, MTU: {self.current_MTU}") 

        try: 
            sol_CM              = opti_CM_copy.solve()
        except Exception as e:
            print(f"\nCM Solver failed, using last known values.\n{e}")
            self.terminate_simulation = True
            sol_CM              = opti_CM_copy.debug
            if not self.surpress_output: check_violated_constraints(opti_CM_copy, opt_vars_CM)
        
        
        self.x_opt_CM            = sol_CM.value(opt_vars_CM['X'])
        self.u_nom_opt_CM        = sol_CM.value(opt_vars_CM['U_nom']).reshape((1, -1))
        self.CM_bid_volumes_opt  = sol_CM.value(opt_vars_CM['B_volumes'])[:,:N_horizon]
        self.CM_bid_prices_opt   = sol_CM.value(opt_vars_CM['B_prices'])[:,:N_horizon]
        self.Eps_opt_CM          = sol_CM.value(opt_vars_CM['Eps'])
        self.Eps_nom_opt_CM      = sol_CM.value(opt_vars_CM['Eps_nom'])


        extracted_bids = ca.vertcat(self.CM_bid_volumes_opt[:,:self.CM_N_initial_bids], self.CM_bid_prices_opt[:,:self.CM_N_initial_bids])
        self.CM_bid_submissions = ca.horzcat(self.CM_bid_submissions, ca.vertcat(extracted_bids))

        self.update_mpc_schedule(self.day, self.qh, 'Init CM', self.start_iter, self.end_iter, 0, self.CM_N_initial_bids)

        return
    

    def solve_initial_AM_bids(self):
        '''
        IN PROGRESS
        '''

        market          = self.market
        model           = self.model
        opt_vars_AM     = self.opt_vars_AM
        AM_N_horizon    = self.AM_N_horizon
        AM_iter_slice   = self.AM_iter_slice

        # Update AM bid optimizer
        opti_AM_copy = update_optimizer_AM_bids(self.controller, market, model,
                        self.opti_AM.copy(), MTU = self.current_MTU, N_TH = AM_N_horizon, N_bids = self.AM_N_bids, opt_vars = opt_vars_AM, spot_prices = self.spot_prices[AM_iter_slice],
                        x0          = self.X_log[:,self.start_iter], 
                        past_X      = self.past_X,
                        U_nom       = self.U_nom_log[:, AM_iter_slice], 
                        ref_weight  = self.target_weight[self.end_iter],
                        B_volumes_min = self.AM_B_volumes_min,
                        B_prices_max  = self.AM_B_prices_max
                        # submitted_bids = np.zeros((4,0))
        )

        # Solve AM bidding
        self.pbar.set_postfix(status=f"Solving Initial AM, MTU: {self.current_MTU}") 
        try: 
            sol_AM                      = opti_AM_copy.solve()
        except Exception as e:
            print(f"\nAM Solver failed, using last known values.\n{e}")
            self.terminate_simulation   = True
            sol_AM                      = opti_AM_copy.debug
            if not self.surpress_output: check_violated_constraints(opti_AM_copy, opt_vars_AM)
    
        self.x_opt_AM               = sol_AM.value(opt_vars_AM['X'])
        self.AM_bid_volumes_opt     = sol_AM.value(opt_vars_AM['B_volumes'])[:,:AM_N_horizon]
        self.AM_bid_prices_opt      = sol_AM.value(opt_vars_AM['B_prices'])[:,:AM_N_horizon]
        self.Eps_opt_AM             = sol_AM.value(opt_vars_AM['Eps'])


        extracted_bids = ca.vertcat(self.AM_bid_volumes_opt[:,:self.AM_N_initial_bids], self.AM_bid_prices_opt[:,:self.AM_N_initial_bids])
        self.AM_bid_submissions = ca.horzcat(self.AM_bid_submissions, ca.vertcat(extracted_bids))

        self.update_mpc_schedule(self.day, self.qh, 'Init AM', self.start_iter, self.end_iter, 0, self.AM_N_initial_bids)
        return



    def get_iteration(self):

        day = self.day
        qh  = self.qh

        self.k              = day*QUARTER_HOURS_PER_DAY + qh
        self.current_MTU    = self.mtu_start + pd.Timedelta(minutes = 15*self.k)
        self.N_horizon      = min(self.N-self.k, self.N_TH)
        self.start_iter     = self.k
        # self.end_iter       = self.k + self.N_horizon
        # self.CM_iter_slice  = slice(self.start_iter, self.end_iter)

        self.current_iter            = self.k                   # Iteration point when optimization starts
        self.end_iter                = self.k + self.N_horizon  # Last iteration point of the optimization window


        # CM
        self.CM_start_iter          = self.CM_bid_submissions.shape[1]          # First iteration point of the optimization window
        self.CM_end_iter            = min(self.N, self.CM_start_iter + self.N_TH)
        self.CM_N_horizon           = self.CM_end_iter - self.CM_start_iter
        self.CM_iter_slice          = slice(self.CM_start_iter, self.CM_end_iter)
        self.CM_optimizer_start_MTU = self.start_MTU + pd.DateOffset(minutes=15*self.CM_start_iter)
        
        # AM
        # self.AM_start_iter      = self.current_iter + self.N_iter             # First iteration point of the optimization window
        self.AM_start_iter          = self.AM_bid_submissions.shape[1]              # First iteration point of the optimization window
        self.AM_N_horizon           = self.end_iter - self.AM_start_iter
        self.AM_iter_slice          = slice(self.AM_start_iter, self.end_iter)
        self.AM_optimizer_start_MTU = self.start_MTU + pd.DateOffset(minutes=15*self.AM_start_iter)
        self.AM_bids_slice          = slice(self.AM_start_iter, self.AM_start_iter + min(self.AM_N_bids, self.AM_N_horizon))

    def solve_CM_bids(self):

        if self.CM_N_horizon == 0: 
            self.update_mpc_schedule(self.day, self.qh, 'CM', self.CM_start_iter, self.CM_end_iter, 0, self.CM_N_bids_to_submit)
            return

        market          = self.market
        model           = self.model
        opt_vars_CM     = self.opt_vars_CM

        # N_horizon       = self.N_horizon
        # self.CM_N_horizon    = self.end_iter - self.CM_start_iter
        # self.CM_iter_slice   = slice(self.CM_start_iter, self.end_iter)
        
        # if start_iter == end_iter:
        #     self.update_mpc_schedule(self.day, self.qh, 'CM', start_iter, end_iter, 96, N_bids_to_submit)


        # TODO update u_CM_AM to account for clearing prices
        # TODO update u_CM_AM to account for activations as well
        # u_hat = self.model.get_u_hat(96, self.U_nom_log[:,self.current_iter:self.CM_start_iter], self.CM_bids_log[:2,self.current_iter:self.CM_start_iter], self.CM_bids_log[2:4,self.current_iter:self.CM_start_iter],
        #                                self.AM_bids_log[:2,self.current_iter:self.CM_start_iter], self.AM_bids_log[2:4,self.current_iter:self.CM_start_iter], self.spot_prices[self.current_iter:self.CM_start_iter],
        #                                self.CM, self.AM)

        est_N = self.CM_start_iter - self.current_iter
        est_slice = slice(self.current_iter, self.CM_start_iter)

        AM_est_prices_up, AM_est_prices_down = market.AM.get_estimated_clearing_prices(start_date = self.current_MTU, n_data = est_N)
        CM_est_prices_up, CM_est_prices_down = market.AM.get_estimated_clearing_prices(start_date = self.current_MTU, n_data = est_N)

        u_hat = self.model.get_u_hat(96, self.U_nom_log[:,est_slice],
                                       self.CM_bids_log[:2,est_slice], self.CM_bids_log[2:4,est_slice],
                                       self.AM_bids_log[:2,est_slice], self.AM_bids_log[2:4,est_slice], 
                                       self.CM_bid_results[:2,est_slice], self.AM_bid_results[:2,est_slice], self.spot_prices[est_slice],
                                       np.vstack((CM_est_prices_up, CM_est_prices_down)), np.vstack((AM_est_prices_up, AM_est_prices_down)), 
                                       self.CM, self.AM)
        
        x_hat = self.model.simulate_growth(self.X_log[:,self.start_iter], u_hat)   # = self.X_log[:,start_iter]
        x0_hat = x_hat[:,-1]
        past_X = ca.horzcat(self.past_X, x0_hat)

        opti_CM_copy = update_optimizer_CM_bids(self.controller, market, model,
                        self.opti_CM.copy(), MTU = self.CM_optimizer_start_MTU, N_TH = self.CM_N_horizon, N_bids = self.CM_N_bids, opt_vars = opt_vars_CM, spot_prices = self.spot_prices[self.CM_iter_slice],
                        x0          = x0_hat, 
                        past_X      = past_X,
                        ref_weight  = self.target_weight[self.CM_end_iter]
        )

        # Solve CM bidding
        self.pbar.set_postfix(status=f"Solving CM, MTU: {self.current_MTU}") 

        try: 
            sol_CM              = opti_CM_copy.solve()
        except Exception as e:
            print(f"\nCM Solver failed, using last known values.\n{e}")
            self.terminate_simulation = True
            sol_CM              = opti_CM_copy.debug
            if not self.surpress_output: check_violated_constraints(opti_CM_copy, opt_vars_CM)
        
        
        self.x_opt_CM            = sol_CM.value(opt_vars_CM['X'])
        self.u_nom_opt_CM        = sol_CM.value(opt_vars_CM['U_nom']).reshape((1, -1))
        self.CM_bid_volumes_opt  = sol_CM.value(opt_vars_CM['B_volumes'])[:,:self.CM_N_horizon]
        self.CM_bid_prices_opt   = sol_CM.value(opt_vars_CM['B_prices'])[:,:self.CM_N_horizon]
        self.Eps_opt_CM          = sol_CM.value(opt_vars_CM['Eps'])
        self.Eps_nom_opt_CM      = sol_CM.value(opt_vars_CM['Eps_nom'])


        extracted_bids = ca.vertcat(self.CM_bid_volumes_opt[:,:self.CM_N_bids_to_submit], self.CM_bid_prices_opt[:,:self.CM_N_bids_to_submit])
        self.CM_bid_submissions = ca.horzcat(self.CM_bid_submissions, ca.vertcat(extracted_bids))

        self.update_mpc_schedule(self.day, self.qh, 'CM', self.CM_start_iter, self.CM_end_iter, 0, self.CM_N_bids_to_submit)

        return


    def store_CM_solution(self):
        
        # Store data
        CM_extract_solution_slice   = slice(0, self.CM_N_horizon)
        CM_store_data_slice         = slice(self.CM_start_iter, self.CM_end_iter)
        CM_extract_bids_slice       = slice(0, min(self.CM_N_bids, self.CM_N_horizon))
        CM_store_bids_slice         = slice(self.CM_start_iter, self.CM_start_iter + min(self.CM_N_bids, self.CM_N_horizon))
        
        # Store inputs
        self.U_nom_log[:,CM_store_data_slice] = self.u_nom_opt_CM[:,CM_extract_solution_slice]         

        # Store CM bid data
        self.CM_bids_log[:2,CM_store_bids_slice] = self.CM_bid_volumes_opt[:,CM_extract_bids_slice]
        self.CM_bids_log[2:4,CM_store_bids_slice] = self.CM_bid_prices_opt[:, CM_extract_bids_slice]


    def extract_CM_bid_result(self):

        # self.CM_activations, self.CM_activated_volumes, self.CM_earnings = CM.subject_bids_to_market_data(self.current_MTU, self.CM_bid_volumes_opt, self.CM_bid_prices_opt)

        # Get market result
        CM_submitted_volumes = self.CM_bid_submissions[:2,:]
        CM_submitted_prices = self.CM_bid_submissions[2:4,:]
        
        CM_activations, CM_volumes, _ = self.CM.subject_bids_to_market_data(self.start_MTU, CM_submitted_volumes, CM_submitted_prices)
        self.CM_activations, self.CM_activated_volumes = CM_activations, CM_volumes

        CM_extraction_slice = slice(0,(self.day+1)*QUARTER_HOURS_PER_DAY) if self.qh < QUARTER_HOURS_PER_HOUR * (9 + 10/60) else slice(0,(self.day+2)*QUARTER_HOURS_PER_DAY)
        self.CM_bid_results = np.vstack((CM_activations, CM_volumes))[:,CM_extraction_slice]

        # define bounds for proper AM participation
        self.AM_B_volumes_min       = self.CM_bid_results[2:4,self.AM_bids_slice]
        self.N_CM_results           = self.AM_B_volumes_min.shape[1]
        self.max_prices_reserved    = np.vstack(self.AM.get_estimated_clearing_prices(start_date=self.current_MTU, n_data = min([self.AM_N_bids, self.AM_N_horizon, self.N_CM_results])))
        # self.AM_B_prices_max        = np.where(self.CM_activations[:,qh:], self.max_prices_reserved, AM.bid_price_limit) # TODO Define a proper upper bound for the bidding price
        self.AM_B_prices_max        = np.where(self.CM_bid_results[:2,self.AM_bids_slice], self.max_prices_reserved, self.AM.bid_price_limit) # TODO Define a proper upper bound for the bidding price

        return


    def update_AM_bid_results(self):

        # Get market result
        AM_submitted_volumes = self.AM_bid_submissions[:2,:]
        AM_submitted_prices = self.AM_bid_submissions[2:4,:]

        AM_activations, AM_volumes, _ = self.AM.subject_bids_to_market_data(self.start_MTU, AM_submitted_volumes, AM_submitted_prices)
        self.AM_bid_results = np.vstack((AM_activations[:,:self.current_iter], 
                                         AM_volumes[:,:self.current_iter]))

        self.AM_activated_volumes_up    = self.AM_bid_results[2,:]
        self.AM_activated_volumes_down  = self.AM_bid_results[3,:]

        return



    def solve_AM_bids(self):

        if self.AM_N_horizon == 0: return

        market          = self.market
        model           = self.model
        opt_vars_AM     = self.opt_vars_AM
        # AM_N_horizon    = self.AM_N_horizon
        # AM_iter_slice   = self.AM_iter_slice
        # AM_N_old_bids   = self.N_iter - 1 # TODO check

        # TODO update u_CM_AM to account for clearing prices
        # u_hat = self.model.get_u_hat(self.N_iter, self.U_nom_log[:,self.current_iter:self.AM_start_iter], self.CM_bids_log[:2,self.current_iter:self.AM_start_iter], self.CM_bids_log[2:4,self.current_iter:self.AM_start_iter],
        #                                self.AM_bids_log[:2,self.current_iter:self.AM_start_iter], self.AM_bids_log[2:4,self.current_iter:self.AM_start_iter], self.spot_prices[self.current_iter:self.AM_start_iter],
        #                                self.CM, self.AM)

        est_N = self.AM_start_iter - self.current_iter
        est_slice = slice(self.current_iter, self.AM_start_iter)

        AM_est_prices_up, AM_est_prices_down = market.AM.get_estimated_clearing_prices(start_date = self.current_MTU, n_data = est_N)
        CM_est_prices_up, CM_est_prices_down = market.AM.get_estimated_clearing_prices(start_date = self.current_MTU, n_data = est_N)

        u_hat = self.model.get_u_hat(est_N, self.U_nom_log[:,est_slice], self.CM_bids_log[:2,est_slice], self.CM_bids_log[2:4,est_slice],
                                       self.AM_bids_log[:2,est_slice], self.AM_bids_log[2:4,est_slice], 
                                       self.CM_bid_results[:2, est_slice], self.AM_bid_results[:2, est_slice], self.spot_prices[est_slice],
                                       np.vstack((CM_est_prices_up, CM_est_prices_down)), np.vstack((AM_est_prices_up, AM_est_prices_down)), 
                                       self.CM, self.AM)
        
        x_hat = self.model.simulate_growth(self.X_log[:,self.start_iter], u_hat)   # = self.X_log[:,start_iter]
        x0_hat = x_hat[:,-1]
        past_X = ca.horzcat(self.past_X, x0_hat)

        # Update AM bid optimizer
        opti_AM_copy = update_optimizer_AM_bids(self.controller, market, model,
                        self.opti_AM.copy(), MTU = self.AM_optimizer_start_MTU, N_TH = self.AM_N_horizon, N_bids = self.AM_N_bids, opt_vars = opt_vars_AM, spot_prices = self.spot_prices[self.AM_iter_slice],
                        x0          = x0_hat,
                        past_X      = past_X,
                        U_nom       = self.U_nom_log[:, self.AM_iter_slice], 
                        ref_weight  = self.target_weight[self.end_iter],
                        B_volumes_min = self.AM_B_volumes_min,
                        B_prices_max  = self.AM_B_prices_max
                        # submitted_bids = self.AM_bid_submissions[:,self.start_iter:]
        )

        # Solve AM bidding
        self.pbar.set_postfix(status=f"Solving AM, MTU: {self.current_MTU}") 
        try: 
            sol_AM                      = opti_AM_copy.solve()
        except Exception as e:
            print(f"\nAM Solver failed, using last known values.\n{e}")
            self.terminate_simulation   = True
            sol_AM                      = opti_AM_copy.debug
            if not self.surpress_output: check_violated_constraints(opti_AM_copy, opt_vars_AM)
    
        self.x_opt_AM               = sol_AM.value(opt_vars_AM['X'])
        self.AM_bid_volumes_opt     = sol_AM.value(opt_vars_AM['B_volumes'])[:,:self.AM_N_horizon]
        self.AM_bid_prices_opt      = sol_AM.value(opt_vars_AM['B_prices'])[:,:self.AM_N_horizon]
        self.Eps_opt_AM             = sol_AM.value(opt_vars_AM['Eps'])


        extracted_bids = ca.vertcat(self.AM_bid_volumes_opt[:,:self.AM_N_bids_to_submit], self.AM_bid_prices_opt[:,:self.AM_N_bids_to_submit])
        self.AM_bid_submissions = ca.horzcat(self.AM_bid_submissions, ca.vertcat(extracted_bids))

        # N_new_activations = self.AM_bid_results.shape[1] - self.AM_bid_submissions.shape[1]
        N_prev_submitted_bids = 2
        # N_new_activations = 1 if self.k == 0 else max(1, self.N_iter - N_prev_submitted_bids)
        self.update_mpc_schedule(self.day, self.qh, 'AM', self.AM_start_iter, self.end_iter, N_prev_submitted_bids, self.AM_N_bids_to_submit)
                       
        return


    def store_AM_solution(self):

        # market          = self.market
        # AM_N_horizon    = self.AM_N_horizon
        # N_iter          = self.N_iter
        # k               = self.k

        # Apply bid activations
        # Evaluate activations
        # self.AM_activations, self.AM_activated_volumes, self.AM_earnings = market.AM.subject_bids_to_market_data(self.current_MTU, self.AM_bid_volumes_opt, self.AM_bid_prices_opt)

        # AM_activations, AM_volumes, _ = self.AM.subject_bids_to_market_data(self.current_MTU, self.AM_bid_volumes_opt, self.AM_bid_prices_opt)
        # AM_bid_results = np.vstack((AM_activations, AM_volumes))

        # self.AM_extract_solution_slice = slice(0,     min(N_iter, AM_N_horizon)) if not self.terminate_simulation else slice(0,     AM_N_horizon)
        # self.AM_store_data_slice       = slice(k, k + min(N_iter, AM_N_horizon)) if not self.terminate_simulation else slice(k, k + AM_N_horizon)
        self.AM_extract_solution_slice  = slice(0, self.AM_N_horizon)                                 #if not self.terminate_simulation else slice(0, self.AM_N_horizon)                 
        self.AM_store_data_slice        = slice(self.AM_start_iter, self.end_iter) #if not self.terminate_simulation else slice(self.AM_start_iter, self.end_iter)
        self.AM_extract_bids_slice      = slice(0, min(self.AM_N_horizon, self.AM_N_bids))                                 #if not self.terminate_simulation else slice(0, self.AM_N_horizon)                 
        self.AM_store_bids_slice        = slice(self.AM_start_iter, self.AM_start_iter + min(self.AM_N_bids, self.AM_N_horizon))
        # self.AM_activated_volumes_up    = AM_bid_results[2,:]
        # self.AM_activated_volumes_down  = AM_bid_results[3,:]

        # Store AM bid data
        self.AM_bids_log[0:2, self.AM_store_bids_slice] = self.AM_bid_volumes_opt[:, self.AM_extract_bids_slice]
        self.AM_bids_log[2:4, self.AM_store_bids_slice] = self.AM_bid_prices_opt[:,  self.AM_extract_bids_slice]

        return
    

    def update_mpc_schedule(self, day, qh, optimizer_type, start_iter, end_iter, n_prev_subm_bids, n_new_bids):

        i = self.n_schedule_entries + 1

        self.optimization_schedule_df.at[i, 'day'] = day
        self.optimization_schedule_df.at[i, 'qh'] = qh
        self.optimization_schedule_df.at[i, 'optimizer'] = optimizer_type

        for k in range(start_iter+1 - n_prev_subm_bids, end_iter+1):
            
            if k <= start_iter:
                # Optimizer is aware of an already submitted unresolved bid: 
                self.optimization_schedule_df.at[i, k] = 3

            elif k <= start_iter + n_new_bids:
                # Optimizer is submitting bids for the current time slot: 
                self.optimization_schedule_df.at[i, k] = 2

            else:
                # Time slot is within the optimizer's optimization window:
                self.optimization_schedule_df.at[i, k] = 1

        self.n_schedule_entries += 1


    def integrate_model(self):

        k               = self.k
        AM_N_horizon    = self.AM_N_horizon
        N_iter          = self.N_iter
        X_log           = self.X_log
        U_nom_log       = self.U_nom_log
        U_log           = self.U_log


        # Store U
        u_tilde = 1000/self.model.C_conv_PPFD * (self.AM_activated_volumes_down - self.AM_activated_volumes_up)
        u = (np.array(U_nom_log[:,:self.current_iter]).flatten() + u_tilde).reshape((1,-1))
        U_log[:,:self.current_iter] = u

        # Iterate state and store X
        iteration_range = range(min(self.N, self.current_iter+1)) #if not self.terminate_simulation else range(AM_N_horizon)
        for i in iteration_range:
            self.X_log[:,i+1] = np.array(self.F(X_log[:,i], np.array([U_log[:,i]]))).reshape(1, -1)

        self.past_X[:,QUARTER_HOURS_PER_DAY-min(QUARTER_HOURS_PER_DAY, min(N_iter, AM_N_horizon)):QUARTER_HOURS_PER_DAY] = X_log[:,self.current_iter:self.current_iter+min(QUARTER_HOURS_PER_DAY, min(N_iter, AM_N_horizon))]


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

    CM_activations_log = ca.DM.zeros(2, N)
    AM_activations_log = ca.DM.zeros(2, N)

    return X, past_X, U, U_nom, CM_bid_log, AM_bid_log, Eps, CM_activations_log, AM_activations_log




def setup_optimizer(controller, nx, nu, neps, N_horizon, N_bids, opti_type: str):
    '''Creates opti variables. Creates opt_vars dictionaries containing opti symbolic optimization variables'''

    opti = ca.Opti()
    opts = {'ipopt.print_level':0, 'print_time':0}
    opti.solver('ipopt', opts)

    X           = opti.variable(nx, N_horizon+1)
    Eps         = opti.variable(neps, 1)
    B_prices    = opti.variable(2*nu, N_bids)
    B_volumes   = opti.variable(2*nu, N_bids)

    x0              = opti.parameter(nx, 1)         # Starting weight
    ref_weight      = opti.parameter(1, 1)          # End weight (To be substituted)
    spot_prices     = opti.parameter(1, N_horizon)  # Spot prices for optimization window
    CM_est_prices   = opti.parameter(2*nu, N_horizon)
    AM_est_prices   = opti.parameter(2*nu, N_horizon)

    # Set initial values:
    opti.set_initial(Eps, ca.DM.zeros(Eps.shape))


    # opti.set_value(spot_prices, np.mean(self.spot_prices)* ca.DM.ones(spot_prices.shape))
    opti.set_value(spot_prices,     10000 * ca.DM.ones(spot_prices.shape))
    opti.set_value(CM_est_prices,   1000 * ca.DM.ones(CM_est_prices.shape))
    opti.set_value(AM_est_prices,   1000 * ca.DM.ones(AM_est_prices.shape))

    opt_vars = {'opti_type'     : opti_type,
                'N_horizon'     : N_horizon,
                'X'             : X,
                'x0'            : x0,
                'Eps'           : Eps,
                'spot_prices'   : spot_prices,
                'ref_weight'    : ref_weight,
                'B_volumes'     : B_volumes,
                'B_prices'      : B_prices,
                'CM_est_prices' : CM_est_prices,
                'AM_est_prices' : AM_est_prices
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

        opti.set_initial(opt_vars['Eps_nom'], ca.DM.zeros(opt_vars['Eps_nom'].shape))

        return opti, opt_vars
    
    elif opti_type=='AM':
        
        # Nominal U as parameter
        U_nom       = opti.parameter(nu, N_horizon)
        Req_volumes = opti.parameter(2*nu, N_bids)
        Max_prices  = opti.parameter(2*nu, N_bids)

        # Initialize parameters with 0-values
        opti.set_value(U_nom,       ca.DM.zeros(U_nom.shape))
        opti.set_value(Req_volumes, ca.DM.zeros(Req_volumes.shape))
        opti.set_value(Max_prices,  controller.market.AM.bid_price_limit*ca.DM.ones(Max_prices.shape))

        # Register opti_variables and opti_params to opt_vars
        opt_vars['U_nom']       = U_nom
        opt_vars['Req_volumes'] = Req_volumes   # Required volumes (from CM participation)
        opt_vars['Max_prices']  = Max_prices    # Max bid prices   (for AM bids required by CM reservations)
        return opti, opt_vars
        
    else:
        assert False, f'SETUP MPC | invalid opti type: {opti_type}'



def set_constraints(controller, market: Market, model: PlantModel, opti: ca.Opti, N_TH, N_bids, opt_vars: dict):
    
    # Extract symbolic optimization variables and parameters
    X           = opt_vars['X']
    # U           = opt_vars['U']
    Eps         = opt_vars['Eps']
    x0          = opt_vars['x0']
    spot_prices = opt_vars['spot_prices']
    ref_weight  = opt_vars['ref_weight']
    B_volumes   = opt_vars['B_volumes']
    B_prices    = opt_vars['B_prices']
    CM_est_prices  = opt_vars['CM_est_prices']
    AM_est_prices  = opt_vars['AM_est_prices']
            
    g_eq, g_bounded = [], []

    if opt_vars['opti_type'] == 'AM':
        U_nom       = opt_vars['U_nom']
        Req_volumes = opt_vars['Req_volumes']
        Max_prices  = opt_vars['Max_prices']
        U = model.get_u(N_TH, N_bids, U_nom=U_nom, B_volumes=B_volumes, B_prices=B_prices, spot_prices=spot_prices, balancing_market=market.AM, clearing_prices = AM_est_prices).reshape((1,-1))
        g_eq, g_bounded = model.get_static_process_constraints(g_eq, g_bounded, N_TH, controller.dt, X, x0, U, Eps)
        g_bounded = model.get_bidding_constraints(g_bounded, N_bids, U_nom, market.AM, B_prices = B_prices, B_volumes=B_volumes, 
                                                                    B_volumes_lower_bound=Req_volumes, B_prices_upper_bound=Max_prices)
        
        g_bounded = model.get_static_variable_bounds(g_bounded, N_TH, controller.dt, X, x0, U, Eps)


    elif opt_vars['opti_type'] == 'CM':
        U_nom   = opt_vars['U_nom']
        X_nom   = opt_vars['X_nom']
        Eps_nom = opt_vars['Eps_nom']
        U = model.get_u_CM(N_TH, N_bids, U_nom=U_nom, CM_B_volumes=B_volumes, CM_B_prices=B_prices, spot_prices=spot_prices, CM=market.CM, AM=market.AM, CM_clearing_prices=CM_est_prices).reshape((1,-1))
        g_eq, g_bounded = model.get_static_process_constraints(g_eq, g_bounded, N_TH, controller.dt, X_nom, x0, U_nom, Eps_nom)
        g_eq, g_bounded = model.get_static_process_constraints(g_eq, g_bounded, N_TH, controller.dt, X, x0, U, Eps)
        g_bounded = model.get_bidding_constraints(g_bounded, N_bids, U_nom, market.CM, B_volumes = B_volumes, B_prices = B_prices)
    
        g_bounded = model.get_static_variable_bounds(g_bounded, N_TH, controller.dt, X_nom, x0, U_nom, Eps_nom)
        g_bounded = model.get_static_variable_bounds(g_bounded, N_TH, controller.dt, X, x0, U, Eps)

    else:
        assert False, f'Inconsistent opti_type: {opt_vars["opti_type"]}'


    # [opti.subject_to(equality_constraint == 0) for equality_constraint in g_eq]
    # [opti.subject_to(inequality_constraint >= 0) for inequality_constraint in g_ineq]

    g_labels = []

    with tqdm(total=len(g_eq), desc=f"    MPC {opt_vars['opti_type']}: Adding equality constraints") as pbar:
        for equality_constraint, label in g_eq:
            opti.subject_to(equality_constraint == 0)
            g_labels.append(label)
            pbar.update(1)
    
    # with tqdm(total=len(g_ineq), desc=f"    MPC {opt_vars['opti_type']}: Adding inequality constraints") as pbar:
    #     for inequality_constraint in g_ineq:
    #         opti.subject_to(inequality_constraint >= 0)
    #         pbar.update(1)

    with tqdm(total=len(g_bounded), desc=f"    MPC {opt_vars['opti_type']}: Adding variable bounds") as pbar:
        for lb, expr, ub, label in g_bounded:
            # opti.subject_to(lb <= expr <= ub)
            opti.subject_to(opti.bounded(lb,expr,ub))
            g_labels.append(label)
            pbar.update(1)

    return opti




def update_optimizer_CM_bids(controller, market: Market, model: PlantModel, opti: ca.Opti, MTU, N_TH, N_bids, opt_vars, spot_prices, x0, past_X, ref_weight):

    X           = opt_vars['X']
    X_nom       = opt_vars['X_nom']
    U_nom       = opt_vars['U_nom']
    Eps         = opt_vars['Eps']
    Eps_nom     = opt_vars['Eps_nom']
    B_volumes   = opt_vars['B_volumes']
    B_prices    = opt_vars['B_prices']
    Est_prices  = opt_vars['CM_est_prices']

    N_bids = min(N_bids, N_TH)

    CM_prices_up, CM_prices_down = market.CM.get_estimated_clearing_prices(start_date = MTU, n_data = N_TH)
    CM_prices = np.vstack((CM_prices_up, CM_prices_down))

    g_eq, g_ineq = [], []
    g_eq, g_ineq = model.get_dynamic_process_constraints(g_eq, g_ineq, N_TH, X,     Eps,     ref_weight, past_X = past_X)
    g_eq, g_ineq = model.get_dynamic_process_constraints(g_eq, g_ineq, N_TH, X_nom, Eps_nom, ref_weight, past_X = past_X)
    # g_eq, g_ineq = model.get_dynamic_bidding_constraints(g_eq, g_ineq, N_TH, B_volumes, B_prices, submitted_bids)
    for equality_constraint, label   in g_eq:   
        opti.subject_to(equality_constraint   == 0)
        # opti.g_labels.append(label)
    for inequality_constraint, label in g_ineq: 
        opti.subject_to(inequality_constraint >= 0)
        # opti.g_labels.append(label)

    U = model.get_u_CM(N_TH, N_bids, U_nom, B_volumes, B_prices, spot_prices, CM = market.CM, AM = market.AM, CM_clearing_prices=Est_prices)
    
    # U = self.model.get_u(N_TH, U_nom, B_volumes, B_prices, spot_prices, self.market.CM)
    J = model.elcost_obj_function(N_TH, spot_prices, U) \
        + model.CM_bidding_obj_function(N_bids, spot_prices, B_volumes, B_prices, market.CM, MTU_start = MTU)\
        + model.terminal_cost(controller, X, U, Eps)\
        + model.terminal_cost(controller, X_nom, U_nom, Eps_nom)
    
    opti.minimize(J)

    # Specify initial guesses

    U_nom_initguess = model.PPFD_max * np.ones(opt_vars['U_nom'][:,:N_TH].shape) / 2
    lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices = model.get_bidding_bounds(N_TH, U_nom_initguess, market.CM)
    B_volumes_initguess = ub_B_volumes
    B_prices_initguess = lb_B_prices

    opti.set_initial(opt_vars['U_nom'][:,:N_TH],        U_nom_initguess)
    opti.set_initial(opt_vars['B_volumes'][:,:N_bids],  B_volumes_initguess[:,:N_bids])
    opti.set_initial(opt_vars['B_prices'][:,:N_bids],   B_prices_initguess[:,:N_bids])

    U_initguess = model.get_u_CM(N_TH, N_bids, U_nom_initguess, B_volumes_initguess, B_prices_initguess, spot_prices, CM = market.CM, AM = market.AM, CM_clearing_prices = CM_prices)
    X_initguess = model.simulate_growth(x0, U_initguess)
    X_nom_initguess = model.simulate_growth(x0, U_nom_initguess)

    opti.set_initial(opt_vars['X'][:,:N_TH+1],     X_initguess)
    opti.set_initial(opt_vars['X_nom'][:,:N_TH+1], X_nom_initguess)


    # Specify parameter values

    opti.set_value(opt_vars['x0'], x0)
    opti.set_value(opt_vars['ref_weight'], ref_weight)
    opti.set_value(opt_vars['spot_prices'][:,:N_TH].reshape((-1,1)), spot_prices)

    opti.set_value(opt_vars['CM_est_prices'][:,:N_TH], CM_prices)

    return opti

def update_optimizer_AM_bids(controller, market: Market, model: PlantModel, opti: ca.Opti, MTU, N_TH, N_bids, opt_vars, spot_prices, x0, past_X, U_nom, ref_weight, B_volumes_min, B_prices_max):

    X           = opt_vars['X']
    Eps         = opt_vars['Eps']
    B_volumes   = opt_vars['B_volumes']
    B_prices    = opt_vars['B_prices']
    Est_prices  = opt_vars['AM_est_prices']

    N_bids = min(N_bids, N_TH)

    g_eq, g_ineq = [], []
    g_eq, g_ineq = model.get_dynamic_process_constraints(g_eq, g_ineq, N_TH, X, Eps, ref_weight, past_X = past_X)
    # g_eq, g_ineq = model.get_dynamic_bidding_constraints(g_eq, g_ineq, N_TH, B_volumes, B_prices, submitted_bids)
    for equality_constraint, label in g_eq:   
        opti.subject_to(equality_constraint   == 0) 
        # opti.g_labels.append(label)  
    for inequality_constraint, label in g_ineq: 
        opti.subject_to(inequality_constraint >= 0) 
        # opti.g_labels.append(label)

    U = model.get_u(N_TH, N_bids, U_nom, B_volumes, B_prices, spot_prices, market.AM, clearing_prices=Est_prices)
    J = model.elcost_obj_function(N_TH, spot_prices, U)\
        + model.AM_bidding_obj_function(N_bids, spot_prices, B_volumes, B_prices, market.AM, MTU_start=MTU)\
        + model.terminal_cost(controller, X, U, Eps)
            
    opti.minimize(J)

    # update parameters
    opti.set_value(opt_vars['x0'],                     x0)
    opti.set_value(opt_vars['U_nom'][:,:N_TH],         U_nom)
    opti.set_value(opt_vars['Req_volumes'][:,:B_volumes_min.shape[1]], B_volumes_min)
    opti.set_value(opt_vars['Max_prices'][:,:B_volumes_min.shape[1]],  B_prices_max)
    opti.set_value(opt_vars['ref_weight'],             ref_weight)
    opti.set_value(opt_vars['spot_prices'][:,:N_TH],   spot_prices)
    
    AM_prices_up, AM_prices_down = market.AM.get_estimated_clearing_prices(start_date = MTU, n_data = N_TH)
    AM_prices = np.vstack((AM_prices_up, AM_prices_down))
    opti.set_value(opt_vars['AM_est_prices'][:,:N_TH], AM_prices)


    # Set initial guesses
    lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices = model.get_bidding_bounds(N_TH, U_nom, market.AM)
    B_volumes_initguess = ub_B_volumes
    B_prices_initguess = lb_B_prices

    opti.set_initial(opt_vars['B_volumes'][:,:N_bids], B_volumes_initguess[:,:N_bids])
    opti.set_initial(opt_vars['B_prices'][:,:N_bids],  B_prices_initguess[:,:N_bids])

    U_initguess = model.get_u(N_TH, N_bids, U_nom, B_volumes_initguess, B_prices_initguess, spot_prices, balancing_market = market.AM, clearing_prices = AM_prices)
    X_initguess = model.simulate_growth(x0, U_initguess)

    opti.set_initial(opt_vars['X'][:,:N_TH+1],     X_initguess)

    # Set initial guesses

    return opti




def check_violated_constraints(opti, opt_vars, tol=1e-6):
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

    for opt_var_name, opt_var in opt_vars.items(): print(f"{opt_var_name:<15} {opt_var}")

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
        # print(f"  Label:       {opti.g_labels[v['index']]}")
        print(f"  Value:       {v['value']:.4f}")
        print(f"  Lower Bound: {v['lower_bound']}")
        print(f"  Upper Bound: {v['upper_bound']}\n")

    return


def plot_optimization_schedule(schedule_df):

    schedule_df = schedule_df.fillna(0)

    # Extract just the optimization matrix (time slots only)
    slot_cols = [col for col in schedule_df.columns if isinstance(col, int)]
    opt_matrix = schedule_df[slot_cols].astype(int)

    # Create a color map
    cmap = mcolors.ListedColormap(["white", "#a6cee3", "#1f78b4", "#b2df8a"])  # 0, 1, 2, 3
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Set up figure
    plt.figure(figsize=(max(10, len(slot_cols) * 0.5), max(4, len(opt_matrix) * 0.4)))
    sns.heatmap(
        opt_matrix,
        cmap=cmap,
        norm=norm,
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=slot_cols,
        yticklabels=[
            f"{int(schedule_df.loc[i, 'day'])}-{int(schedule_df.loc[i, 'qh'])} {schedule_df.loc[i, 'optimizer']}"
            for i in schedule_df.index
        ]
    )

    # Labels and title
    plt.xlabel("Time Slot")
    plt.ylabel("Optimization Step (Day-QH Optimizer)")
    plt.title("MPC Optimization Schedule")


    legend_labels = {
        0: "Not relevant",
        1: "In optimization window",
        2: "Submitting bid",
        3: "Previously submitted (unresolved)",
    }
    legend_patches = [Patch(facecolor=cmap(i), edgecolor='black', label=legend_labels[i]) for i in legend_labels]

    plt.legend(
        handles=legend_patches,
        title="Legend",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )
    plt.tight_layout()
    plt.show()