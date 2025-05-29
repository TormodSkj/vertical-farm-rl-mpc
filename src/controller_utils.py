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
from utils import * 


class MPCSimulation():
    
    model:      PlantModel
    market:     Market
    CM:         BalancingMarket
    AM:         BalancingMarket

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

        self.CM_N_bids = int(np.clip(mpc_settings['CM_N_BIDS'], QUARTER_HOURS_PER_DAY,  self.N_TH))
        self.AM_N_bids = int(np.clip(mpc_settings['AM_N_BIDS'], self.N_iter,            self.N_TH))
        self.check_feasibility = mpc_settings['CHECK_FEASIBILITY']
        
        self.CM_N_initial_bids      = QUARTER_HOURS_PER_DAY
        self.CM_N_bids_to_submit    = QUARTER_HOURS_PER_DAY
        self.AM_N_initial_bids      = 3
        self.AM_N_bids_to_submit    = self.N_iter

        self.F      = controller.F
        self.surpress_output = controller.surpress_output


        self.terminate_simulation = False
        self.error_msg = ''
        self.mtu_start = self.market.MTU_start
        self.start_MTU = self.mtu_start

        self.simulation_days            = list(range(int(np.ceil(self.T))))
        self.quarter_hourly_intervals   = list(range(0, QUARTER_HOURS_PER_DAY, self.N_iter))


        self.target_run_id   = target_run_id
        self.target_run      = self.controller.optimization_results['runs'][target_run_id]
        self.target_X        = self.target_run['timeseries']['x']
        self.target_U        = self.target_run['timeseries']['u']
        self.target_weight   = self.model.freshweight(self.target_X)


        self.optimization_schedule_df = pd.DataFrame(columns=['day', 'qh', 'optimizer'] + [k for k in range(self.N)], index=range(2 + int(self.T) + int(np.ceil(self.N/self.N_iter))), dtype=float).fillna(0)
        self.optimization_schedule_df['optimizer'] = self.optimization_schedule_df['optimizer'].astype(str)
        self.n_schedule_entries = 0

        self.salvations = []

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
        self.past_X[2,:]        = self.past_X[2,:] - self.past_X[2,-1]            
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

            self.day, self.qh = 0,0 
           
            self.get_iteration()

            self.solve_initial_CM_bids()
            self.store_CM_solution()

            self.extract_CM_bid_result()
            self.solve_initial_AM_bids()
            self.store_AM_solution()
            self.update_AM_bid_results()


            for day in self.simulation_days:
                if self.terminate_simulation: 
                    break
                for qh in self.quarter_hourly_intervals:

                    self.day, self.qh = day, qh

                    self.get_iteration()

                    if qh == 0:     # Start of every day only

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
            self.terminate_simulation = True
            sol_CM              = opti_CM_copy.debug
            if not self.surpress_output: sol_CM.show_infeasibilities()
            print(f"\nCM Solver failed, using last known values.\n{e}")
            self.error_msg = f"Init CM failed at iteration {self.current_iter}: {e}"
        
        
        self.x_opt_CM            = sol_CM.value(opt_vars_CM['X'])
        self.u_nom_opt_CM        = sol_CM.value(opt_vars_CM['U_nom']).reshape((1, -1))
        self.CM_bid_volumes_opt  = sol_CM.value(opt_vars_CM['B_volumes'])[:,:N_horizon]
        self.CM_bid_prices_opt   = sol_CM.value(opt_vars_CM['B_prices'])[:,:N_horizon]
        self.Eps_opt_CM          = sol_CM.value(opt_vars_CM['Eps'])
        self.Eps_nom_opt_CM      = sol_CM.value(opt_vars_CM['Eps_nom'])


        extracted_bids = ca.vertcat(self.CM_bid_volumes_opt[:,:self.CM_N_initial_bids], 
                                    self.CM_bid_prices_opt[:,:self.CM_N_initial_bids])
        self.CM_bid_submissions = ca.horzcat(self.CM_bid_submissions, ca.vertcat(extracted_bids))

        self.update_mpc_schedule(self.day, self.qh, 'Init CM', self.start_iter, self.end_iter, 0, self.CM_N_initial_bids, self.CM_N_bids)

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
        )

        # Solve AM bidding
        self.pbar.set_postfix(status=f"Solving Initial AM, MTU: {self.current_MTU}") 
        try: 
            sol_AM                      = opti_AM_copy.solve()
        except Exception as e:
            self.terminate_simulation   = True
            sol_AM                      = opti_AM_copy.debug
            if not self.surpress_output: sol_AM.show_infeasibilities()
            print(f"\nAM Solver failed, using last known values.\n{e}")
            self.error_msg = f"Init AM failed at iteration {self.current_iter}: {e}"
    
        self.x_opt_AM               = sol_AM.value(opt_vars_AM['X'])
        self.AM_bid_volumes_opt     = sol_AM.value(opt_vars_AM['B_volumes'])[:,:AM_N_horizon]
        self.AM_bid_prices_opt      = sol_AM.value(opt_vars_AM['B_prices'])[:,:AM_N_horizon]
        self.Eps_opt_AM             = sol_AM.value(opt_vars_AM['Eps'])


        extracted_bids = ca.vertcat(self.AM_bid_volumes_opt[:,:self.AM_N_initial_bids], self.AM_bid_prices_opt[:,:self.AM_N_initial_bids])
        self.AM_bid_submissions = ca.horzcat(self.AM_bid_submissions, ca.vertcat(extracted_bids))

        self.update_mpc_schedule(self.day, self.qh, 'Init AM', self.start_iter, self.end_iter, 0, self.AM_N_initial_bids, self.AM_N_bids)
        return



    def get_iteration(self):

        day = self.day
        qh  = self.qh

        self.k              = day*QUARTER_HOURS_PER_DAY + qh
        self.current_MTU    = self.mtu_start + pd.Timedelta(minutes = 15*self.k)
        self.N_horizon      = min(self.N-self.k, self.N_TH)
        self.start_iter     = self.k

        self.current_iter            = self.k                   # Iteration point when optimization starts
        self.end_iter                = self.k + self.N_horizon  # Last iteration point of the optimization window


        # CM
        self.CM_start_iter          = self.CM_bid_submissions.shape[1]          # First iteration point of the optimization window
        self.CM_end_iter            = min(self.N, self.CM_start_iter + self.N_TH)
        self.CM_N_horizon           = self.CM_end_iter - self.CM_start_iter
        self.CM_iter_slice          = slice(self.CM_start_iter, self.CM_end_iter)
        self.CM_optimizer_start_MTU = self.start_MTU + pd.DateOffset(minutes=15*self.CM_start_iter)
        
        # AM
        self.AM_start_iter          = self.AM_bid_submissions.shape[1]              # First iteration point of the optimization window
        self.AM_end_iter            = min(self.N, self.AM_start_iter + self.N_TH)
        self.AM_N_horizon           = self.AM_end_iter - self.AM_start_iter
        self.AM_iter_slice          = slice(self.AM_start_iter, self.AM_end_iter)
        self.AM_optimizer_start_MTU = self.start_MTU + pd.DateOffset(minutes=15*self.AM_start_iter)
        self.AM_bids_slice          = slice(self.AM_start_iter, self.AM_start_iter + min(self.AM_N_bids, self.AM_N_horizon))

    def solve_CM_bids(self):

        if self.CM_N_horizon == 0: 
            self.update_mpc_schedule(self.day, self.qh, 'CM', self.CM_start_iter, self.CM_end_iter, 0, self.CM_N_bids_to_submit, self.CM_N_bids)
            return

        market          = self.market
        model           = self.model
        opt_vars_CM     = self.opt_vars_CM

        est_start_iter  = self.AM_bid_results.shape[1]
        est_end_iter    = self.CM_start_iter
        est_slice   = slice(est_start_iter, est_end_iter)
        est_N       = est_end_iter - est_start_iter

        AM_est_prices_up, AM_est_prices_down = market.AM.get_estimated_clearing_prices(start_date = self.current_MTU, n_data = est_N)
        CM_est_prices_up, CM_est_prices_down = market.AM.get_estimated_clearing_prices(start_date = self.current_MTU, n_data = est_N)

        u_hat = self.model.get_u_hat(est_N, self.U_nom_log[:,est_slice],
                                     self.CM_bids_log[:2,est_slice], self.CM_bids_log[2:4,est_slice],
                                     self.AM_bids_log[:2,est_slice], self.AM_bids_log[2:4,est_slice], 
                                     self.CM_bid_results[:2,est_slice], self.AM_bid_results[:2,est_slice], self.spot_prices[est_slice],
                                     np.vstack((CM_est_prices_up, CM_est_prices_down)), np.vstack((AM_est_prices_up, AM_est_prices_down)), 
                                     self.CM, self.AM)
        
        x_hat = self.model.simulate_growth(self.X_log[:,est_start_iter], u_hat) 
        x0_hat = x_hat[:,-1]
        past_X = ca.horzcat(self.past_X, x_hat)[:,est_N:est_N+self.past_X.shape[1]]

        opti_CM_copy = update_optimizer_CM_bids(self.controller, market, model,
                        self.opti_CM.copy(), MTU = self.CM_optimizer_start_MTU, N_TH = self.CM_N_horizon, N_bids = self.CM_N_bids, opt_vars = opt_vars_CM, spot_prices = self.spot_prices[self.CM_iter_slice],
                        x0          = x0_hat, 
                        past_X      = past_X,
                        ref_weight  = self.target_weight[self.CM_end_iter]
        )

        if self.check_feasibility:
            check_if_feasible(opti_CM_copy)
            check_initguess_manually(opti_CM_copy)

        # Solve CM bidding
        self.pbar.set_postfix(status=f"Solving CM, MTU: {self.current_MTU}") 

        try: 
            sol_CM              = opti_CM_copy.solve()
        except Exception as e:
            self.terminate_simulation = True
            sol_CM              = opti_CM_copy.debug
            if not self.surpress_output: sol_CM.show_infeasibilities()
            print(f"\nCM Solver failed, using last known values.\n{e}")
            self.error_msg = f"CM failed at iteration {self.current_iter}: {e}"
        
        
        self.x_opt_CM            = sol_CM.value(opt_vars_CM['X'])
        self.u_nom_opt_CM        = sol_CM.value(opt_vars_CM['U_nom']).reshape((1, -1))
        self.CM_bid_volumes_opt  = sol_CM.value(opt_vars_CM['B_volumes'])[:,:self.CM_N_horizon]
        self.CM_bid_prices_opt   = sol_CM.value(opt_vars_CM['B_prices'])[:,:self.CM_N_horizon]
        self.Eps_opt_CM          = sol_CM.value(opt_vars_CM['Eps'])
        self.Eps_nom_opt_CM      = sol_CM.value(opt_vars_CM['Eps_nom'])


        extracted_bids = ca.vertcat(self.CM_bid_volumes_opt[:,:self.CM_N_bids_to_submit], 
                                    self.CM_bid_prices_opt[:,:self.CM_N_bids_to_submit])
        self.CM_bid_submissions = ca.horzcat(self.CM_bid_submissions, ca.vertcat(extracted_bids))

        self.update_mpc_schedule(self.day, self.qh, 'CM', self.CM_start_iter, self.CM_end_iter, 0, self.CM_N_bids_to_submit, self.CM_N_bids)

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
        self.CM_bids_log[:2,CM_store_bids_slice]    = self.CM_bid_volumes_opt[:,CM_extract_bids_slice]
        self.CM_bids_log[2:4,CM_store_bids_slice]   = self.CM_bid_prices_opt[:, CM_extract_bids_slice]


    def extract_CM_bid_result(self):


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
        self.AM_bid_results = np.vstack((AM_activations[:,:self.current_iter+1], 
                                          AM_volumes[:,:self.current_iter+1]))
        
        self.AM_activated_volumes_up    = self.AM_bid_results[2,:]
        self.AM_activated_volumes_down  = self.AM_bid_results[3,:]

        return



    def solve_AM_bids(self):

        if self.AM_N_horizon == 0: return

        market          = self.market
        model           = self.model
        opt_vars_AM     = self.opt_vars_AM

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
        past_X = ca.horzcat(self.past_X, x_hat)[:,est_N:est_N+self.past_X.shape[1]]

        # Update AM bid optimizer
        opti_AM_copy = update_optimizer_AM_bids(self.controller, market, model,
                        self.opti_AM.copy(), MTU = self.AM_optimizer_start_MTU, N_TH = self.AM_N_horizon, N_bids = self.AM_N_bids, opt_vars = opt_vars_AM, spot_prices = self.spot_prices[self.AM_iter_slice],
                        x0          = x0_hat,
                        past_X      = past_X,
                        U_nom       = self.U_nom_log[:, self.AM_iter_slice], 
                        ref_weight  = self.target_weight[self.end_iter],
                        B_volumes_min = self.AM_B_volumes_min,
                        B_prices_max  = self.AM_B_prices_max
        )

        if self.check_feasibility:
            check_if_feasible(opti_AM_copy)
            check_initguess_manually(opti_AM_copy)


        # Solve AM bidding
        self.pbar.set_postfix(status=f"Solving AM, MTU: {self.current_MTU}") 
        try: 
            ran_successfully = True
            sol_AM = opti_AM_copy.solve()
        except Exception as e:
            ran_successfully = False
            if check_violated_constraints(opti_AM_copy, opt_vars_AM, print_opt_vars=False) == 0:
                if not self.surpress_output: print(f"\nAM Solver terminated at a feasible point. Salvaging bids.")
                sol_AM = opti_AM_copy.debug
                self.salvations.append(self.AM_start_iter)
            
            else:
                self.terminate_simulation = True
                sol_AM = opti_AM_copy.debug
                if not self.surpress_output: sol_AM.show_infeasibilities()
                print(f"\nAM Solver failed, using last known values.\n{e}")
                self.error_msg = f"AM failed at iteration {self.current_iter}: {e}"
    
        self.x_opt_AM               = sol_AM.value(opt_vars_AM['X'])
        self.AM_bid_volumes_opt     = sol_AM.value(opt_vars_AM['B_volumes'])[:,:self.AM_N_horizon]
        self.AM_bid_prices_opt      = sol_AM.value(opt_vars_AM['B_prices'])[:,:self.AM_N_horizon]
        self.Eps_opt_AM             = sol_AM.value(opt_vars_AM['Eps'])


        extracted_bids = ca.vertcat(self.AM_bid_volumes_opt[:,:self.AM_N_bids_to_submit], 
                                    self.AM_bid_prices_opt[:,:self.AM_N_bids_to_submit])
        self.AM_bid_submissions = ca.horzcat(self.AM_bid_submissions, ca.vertcat(extracted_bids))

        N_prev_submitted_bids = 2
        self.update_mpc_schedule(self.day, self.qh, 'AM', self.AM_start_iter, self.AM_end_iter, N_prev_submitted_bids, self.AM_N_bids_to_submit, self.AM_N_bids, ran_successfully)
                       
        return


    def store_AM_solution(self):

        self.AM_extract_solution_slice  = slice(0, self.AM_N_horizon)
        self.AM_store_data_slice        = slice(self.AM_start_iter, self.end_iter)
        self.AM_extract_bids_slice      = slice(0, min(self.AM_N_horizon, self.AM_N_bids))             
        self.AM_store_bids_slice        = slice(self.AM_start_iter, self.AM_start_iter + min(self.AM_N_bids, self.AM_N_horizon))

        # Store AM bid data
        self.AM_bids_log[0:2, self.AM_store_bids_slice] = self.AM_bid_volumes_opt[:, self.AM_extract_bids_slice]
        self.AM_bids_log[2:4, self.AM_store_bids_slice] = self.AM_bid_prices_opt[:,  self.AM_extract_bids_slice]

        return
    

    def update_mpc_schedule(self, day, qh, optimizer_type, start_iter, end_iter, n_prev_subm_bids, n_new_bids, n_planned_bids, ran_successfully = True):

        i = self.n_schedule_entries

        self.optimization_schedule_df.at[i, 'day'] = day
        self.optimization_schedule_df.at[i, 'qh'] = qh
        self.optimization_schedule_df.at[i, 'optimizer'] = optimizer_type

        if 'CM' in optimizer_type:
            prefix = 10
        elif 'AM' in optimizer_type:
            prefix = 20
        else:
            assert False, f'Unrecognized optmizer type passed to update_mpc_schedule. Expected either CM, AM, Init CM or Init AM. Instead got {optimizer_type}'

        # if not ran_successfully: prefix += 100

        for k in range(start_iter - n_prev_subm_bids, end_iter):
            
            if k < start_iter:
                # Optimizer is aware of an already submitted unresolved bid: 
                self.optimization_schedule_df.at[i, k] = prefix + 3

            elif k < start_iter + n_new_bids:
                # Optimizer is submitting bids for the current time slot: 
                self.optimization_schedule_df.at[i, k] = prefix + 1

            elif k < start_iter + n_planned_bids:
                # Optimizer is planning, but not submitting bids for the current time slot: 
                self.optimization_schedule_df.at[i, k] = prefix + 2

            else:
                # Time slot is within the optimizer's optimization window:
                self.optimization_schedule_df.at[i, k] = prefix if ran_successfully else -1

        self.n_schedule_entries += 1


    def integrate_model(self):

        k               = self.k
        AM_N_horizon    = self.AM_N_horizon
        N_iter          = self.N_iter
        X_log           = self.X_log
        U_nom_log       = self.U_nom_log
        U_log           = self.U_log


        # Store U
        u_tilde = 1000/self.model.C_conv_PPFD * (self.AM_activated_volumes_down - self.AM_activated_volumes_up).reshape((1,-1))
        u = (np.array(U_nom_log[:,:u_tilde.shape[1]]).flatten() + u_tilde).reshape((1,-1))
        U_log[:,:u.shape[1]] = u    


        # Iterate state and store X
        iteration_range = range(min(self.N, self.current_iter+1)) if not self.terminate_simulation else range(u.shape[1])
        for i in iteration_range:
            self.X_log[:,i+1] = np.array(self.F(X_log[:,i], np.array([U_log[:,i]]))).reshape(1, -1)

        if not self.terminate_simulation:

            self.past_X = ca.horzcat(self.past_X[:,-QUARTER_HOURS_PER_DAY:], self.X_log)[:,min(self.N, self.current_iter+1):min(self.N, self.current_iter+1)+QUARTER_HOURS_PER_DAY]

        self.Eps_log = self.model.get_eps(self.end_iter, self.target_weight[self.end_iter], self.X_log, self.past_X)

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
    opts = {'ipopt': {'print_level': controller.opti_print_level, 'sb': 'yes', 'check_derivatives_for_naninf': 'yes'},
            'print_time': False}
    opti.solver('ipopt', opts)

    # if opti_type == 'CM': N_bid_vars = 

    X           = opti.variable(nx, N_horizon+1)
    Eps         = opti.variable(neps, 1)
    B_prices    = opti.variable(2*nu, N_bids)
    B_volumes   = opti.variable(2*nu, N_bids)

    x0              = opti.parameter(nx, 1)         # Starting weight
    ref_weight      = opti.parameter(1, 1)          # End weight (To be substituted)
    spot_prices     = opti.parameter(1, N_horizon)  # Spot prices for optimization window
    CM_est_prices   = opti.parameter(2*nu, N_horizon)
    AM_est_prices   = opti.parameter(2*nu, N_horizon)

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

        opti.set_initial(opt_vars['Eps_nom'], ca.DM([1, 5, 5]))

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
    Eps         = opt_vars['Eps']
    x0          = opt_vars['x0']
    spot_prices = opt_vars['spot_prices']
    ref_weight  = opt_vars['ref_weight']
    B_volumes   = opt_vars['B_volumes']
    B_prices    = opt_vars['B_prices']
    CM_est_prices  = opt_vars['CM_est_prices']
    AM_est_prices  = opt_vars['AM_est_prices']
            

    if opt_vars['opti_type'] == 'AM':
        U_nom       = opt_vars['U_nom']
        Req_volumes = opt_vars['Req_volumes']
        Max_prices  = opt_vars['Max_prices']
        U = model.get_u(N_TH, N_bids, U_nom=U_nom, B_volumes=B_volumes, B_prices=B_prices, spot_prices=spot_prices, balancing_market=market.AM, clearing_prices = AM_est_prices).reshape((1,-1))
        model.apply_static_process_constraints(opti, N_TH, controller.dt, X, x0, U, Eps)
        model.apply_bidding_constraints(opti, N_bids, U_nom, market.AM, B_prices = B_prices, B_volumes=B_volumes, 
                                        B_volumes_lower_bound=Req_volumes, B_prices_upper_bound=Max_prices)
        
        model.apply_static_variable_bounds(opti, N_TH, controller.dt, X, x0, U, Eps)


    elif opt_vars['opti_type'] == 'CM':
        U_nom   = opt_vars['U_nom']
        X_nom   = opt_vars['X_nom']
        Eps_nom = opt_vars['Eps_nom']

        U = model.get_u_CM(N_TH, N_bids, U_nom=U_nom, CM_B_volumes=B_volumes, CM_B_prices=B_prices, spot_prices=spot_prices, CM=market.CM, AM=market.AM, CM_clearing_prices=CM_est_prices).reshape((1,-1))
        model.apply_static_process_constraints(opti, N_TH, controller.dt, X_nom, x0, U_nom, Eps_nom)
        model.apply_static_process_constraints(opti, N_TH, controller.dt, X, x0, U, Eps)
        model.apply_bidding_constraints(opti, N_bids, U_nom, market.CM, B_volumes = B_volumes, B_prices = B_prices)
    
        model.apply_static_variable_bounds(opti, N_TH, controller.dt, X_nom, x0, U_nom, Eps_nom)
        model.apply_static_variable_bounds(opti, N_TH, controller.dt, X, x0, U, Eps)

        #TODO Add constraint to ensure same bid for every hour. Or change optimizer. Last option might be better, as that might speed things up
        model.get_hourly_bids_contraint(opti, N_bids, B_volumes, B_prices)

    else:
        assert False, f'Inconsistent opti_type: {opt_vars["opti_type"]}'

    return opti




def update_optimizer_CM_bids(controller, market: Market, model: PlantModel, opti: ca.Opti, MTU, N_TH, N_bids, opt_vars, spot_prices, x0, past_X, ref_weight):

    X            = opt_vars['X']
    X_nom        = opt_vars['X_nom']
    U_nom        = opt_vars['U_nom']
    Eps          = opt_vars['Eps']
    Eps_nom      = opt_vars['Eps_nom']
    B_volumes    = opt_vars['B_volumes']
    B_prices     = opt_vars['B_prices']
    Est_prices   = opt_vars['CM_est_prices']

    N_bids = min(N_bids, N_TH)
    max_N_TH = U_nom.shape[1]

    CM_prices_up, CM_prices_down = market.CM.get_estimated_clearing_prices(start_date = MTU, n_data = N_TH)
    CM_prices = np.vstack((CM_prices_up, CM_prices_down))

    model.opti_dynamic_process_constraints(opti, N_TH, X,     Eps,     ref_weight, past_X = past_X)
    model.opti_dynamic_process_constraints(opti, N_TH, X_nom, Eps_nom, ref_weight, past_X = past_X)
    
    U = model.get_u_CM(N_TH, N_bids, U_nom, B_volumes, B_prices, spot_prices, CM = market.CM, AM = market.AM, CM_clearing_prices=Est_prices)
    
    # U = self.model.get_u(N_TH, U_nom, B_volumes, B_prices, spot_prices, self.market.CM)
    obj = model.elcost_obj_function(N_TH, spot_prices, U) \
        + model.CM_bidding_obj_function(N_bids, spot_prices, B_volumes, B_prices, market.CM, MTU_start = MTU)\
        + model.terminal_cost(controller, X, U, Eps)\
        + model.terminal_cost(controller, X_nom, U_nom, Eps_nom)

    opti.minimize(obj)

    # Specify initial guesses

    U_nom_initguess = model.LIGHT_INTY * np.tile(np.hstack((np.ones((1,int(96*model.PHOTOPERIOD/24))), np.zeros((1, int(96*(24 - model.PHOTOPERIOD)/24))))), int(controller.T))[:,:max_N_TH]
    lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices = model.get_bidding_bounds(N_TH, U_nom_initguess, market.CM)
    B_volumes_initguess     = lb_B_volumes
    B_prices_initguess      = lb_B_prices

    
    U_initguess         = U_nom_initguess
    X_initguess         = model.simulate_growth(x0, U_initguess)
    X_nom_initguess     = model.simulate_growth(x0, U_nom_initguess)
    Eps_initguess       = model.get_eps(N_TH, ref_weight, X_initguess,      past_X)
    Eps_nom_initguess   = model.get_eps(N_TH, ref_weight, X_nom_initguess,  past_X)


    opti.set_initial(X,         X_initguess)
    opti.set_initial(X_nom,     X_nom_initguess)
    opti.set_initial(U_nom,     U_nom_initguess)
    opti.set_initial(B_volumes, B_volumes_initguess[:,:N_bids])
    opti.set_initial(B_prices,  B_prices_initguess[:,:N_bids])
    opti.set_initial(Eps,       Eps_initguess)
    opti.set_initial(Eps_nom,   Eps_nom_initguess)


    # Specify parameter values
    opti.set_value(opt_vars['x0'],                                      x0)
    opti.set_value(opt_vars['ref_weight'],                              ref_weight)
    opti.set_value(opt_vars['spot_prices'][:,:N_TH].reshape((-1,1)),    spot_prices)
    opti.set_value(opt_vars['CM_est_prices'][:,:N_TH],                  CM_prices)

    if not check_initguess_manually(opti, print_only_if_false = True):
        check_if_feasible(opti)

    return opti

def update_optimizer_AM_bids(controller, market: Market, model: PlantModel, opti: ca.Opti, MTU, N_TH, N_bids, opt_vars, spot_prices, x0, past_X, U_nom, ref_weight, B_volumes_min, B_prices_max):

    X           = opt_vars['X']
    Eps         = opt_vars['Eps']
    B_volumes   = opt_vars['B_volumes']
    B_prices    = opt_vars['B_prices']
    Est_prices  = opt_vars['AM_est_prices']

    N_bids = min(N_bids, N_TH)
    full_N_TH = X.shape[1] - 1

    model.opti_dynamic_process_constraints(opti, N_TH, X, Eps, ref_weight, past_X = past_X)
    
    U = model.get_u(N_TH, N_bids, U_nom, B_volumes, B_prices, spot_prices, market.AM, clearing_prices=Est_prices)
    obj_fun = model.elcost_obj_function(N_TH, spot_prices, U)\
            + model.AM_bidding_obj_function(N_bids, spot_prices, B_volumes, B_prices, market.AM, MTU_start=MTU)\
            + model.terminal_cost(controller, X, U, Eps)
            
    opti.minimize(obj_fun)

    # update parameters
    opti.set_value(opt_vars['x0'],                     x0)
    opti.set_value(opt_vars['U_nom'][:,:N_TH],         U_nom)
    opti.set_value(opt_vars['Req_volumes'][:,:B_volumes_min.shape[1]], B_volumes_min)
    opti.set_value(opt_vars['Max_prices'][:,:B_prices_max.shape[1]],   B_prices_max)
    opti.set_value(opt_vars['ref_weight'],             ref_weight)
    opti.set_value(opt_vars['spot_prices'][:,:N_TH],   spot_prices)
    
    AM_prices_up, AM_prices_down = market.AM.get_estimated_clearing_prices(start_date = MTU, n_data = N_TH)
    AM_prices = np.vstack((AM_prices_up, AM_prices_down))
    opti.set_value(opt_vars['AM_est_prices'][:,:N_TH], AM_prices)


    # Set initial guesses
    lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices = model.get_bidding_bounds(N_bids, U_nom, market.AM)
    B_volumes_initguess = np.maximum(lb_B_volumes, B_volumes_min)
    B_prices_initguess = np.minimum(lb_B_prices, B_prices_max)


    U_initguess             = np.zeros((1, full_N_TH))
    U_initguess[:,:N_TH]    = model.get_u(N_TH, N_bids, U_nom, B_volumes_initguess, B_prices_initguess, spot_prices, balancing_market = market.AM, clearing_prices = AM_prices)
    X_initguess             = model.simulate_growth(x0, U_initguess)
    Eps_initguess           = model.get_eps(N_TH, ref_weight, X_initguess, past_X)


    opti.set_initial(opt_vars['X'], X_initguess)
    opti.set_initial(B_volumes[:,:N_bids], B_volumes_initguess[:,:N_bids])
    opti.set_initial(B_prices[:,:N_bids],  B_prices_initguess[:,:N_bids])
    opti.set_initial(Eps,       Eps_initguess)

    if not check_initguess_manually(opti, print_only_if_false = True):
        check_if_feasible(opti)

    return opti




def check_violated_constraints(opti, opt_vars, tol=1e-6, print_opt_vars = True):
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


    print(f"Number of violations = {len(violations)}")

    return len(violations)


def check_if_feasible(opti: ca.Opti) -> bool:
    """
    Check if the current initial values in an Opti stack constitute a feasible solution.
    Returns:
        True if feasible, False otherwise.
    """
    # Make a copy of the original problem
    opti_copy = opti.copy()

    # Minimize a dummy cost to trigger constraint evaluation only
    opti_copy.minimize(0)

    # Set solver options
    opts = {
        'ipopt': {
            'print_level': 0,          # Suppress IPOPT output
            'sb': 'yes'                # Suppress IPOPT banner
        },
        'print_time': False
    }

    # Attach solver
    opti_copy.solver('ipopt', opts)

    try:
        # Solve the problem
        sol = opti_copy.solve()

        # Get solver stats
        stats = opti_copy.stats()
        num_iter = stats.get('iter_count', None)

        print(f"✅ Problem is feasible. Solver used {num_iter} iteration(s).")
        return True

    except RuntimeError as e:
        print("❌ Problem is infeasible.")
        return False


def check_initguess_feasibility(opti: ca.Opti, show_infeasibilities = False) -> bool:
    """
    Check if the current initial values in an Opti stack constitute a feasible solution.

    Returns:
        True if feasible, False otherwise.
    """
    # Make a copy of the original problem
    opti_copy = opti.copy()

    # Minimize a dummy cost to trigger constraint evaluation only
    opti_copy.minimize(0)

    # Set solver options
    opts = {
        'ipopt': {
            'print_level': 0,          # Keep output minimal
            'sb': 'yes',               # Suppress IPOPT banner
            'max_iter': 1,             # Only check feasibility, not optimize
            'tol': 1e-4,
            'acceptable_tol': 1e-4,
            'constr_viol_tol': 1e-4
        },
        'print_time': False
    }

    # Attach solver
    opti_copy.solver('ipopt', opts)

    try:
        # Attempt to solve the dummy optimization problem
        opti_copy.solve()
        print("✅ Initial guess is feasible.")
        return True
    except RuntimeError as e:
        # status = opti_copy.debug.value(opti_copy.stats()['return_status']) \
            # if hasattr(opti_copy, 'debug') else "Unknown"
        if show_infeasibilities: opti_copy.debug.show_infeasibilities()
        print(f"❌ Initial guess is infeasible. {e}")
        return False


def check_initguess_manually(opti: ca.Opti, tol: float = 1e-6, print_only_if_false = False):
    """
    Manually checks whether the initial values of the Opti problem satisfy the constraints.
    
    Args:
        opti (ca.Opti): The Opti problem to check.
        tol (float): Tolerance for constraint satisfaction.

    Returns:
        (bool, list): Tuple of:
            - True if all constraints are satisfied within tolerance, False otherwise.
            - List of violated constraints with (index, value, lower_bound, upper_bound).
    """

    opti_copy = opti.copy()
    opts = {
        'ipopt': {
            'print_level': 0,          # Keep output minimal
            'sb': 'yes',               # Suppress IPOPT banner
            'max_iter': 0,             # Only check feasibility, not optimize
            'tol': 1e-4,
            'acceptable_tol': 1e-4,
            'constr_viol_tol': 1e-4
        },
        'print_time': False
    }
    opti_copy.solver('ipopt', opts)

    try:
        sol = opti_copy.solve()
    except:
        sol = opti_copy.debug


    # Extract constraint expression and bounds
    g = opti_copy.g
    lbg = np.array(opti_copy.value(opti_copy.lbg))
    ubg = np.array(opti_copy.value(opti_copy.ubg))

    # Evaluate g at initial values
    g_fun = ca.Function('g_fun', [opti.x, opti.p], [opti.g])

    # Evaluate g at initial values
    x0 = sol.value(opti.x)
    p0 = sol.value(opti.p)
    g_val = np.array(g_fun(x0, p0)).flatten()

    violated_constraints = []

    assert lbg.shape[0] == g.shape[0], f"lengths of g and lbg are inconsistent. Lengths: g: {g.shape[0]}, lbg: {lbg.shape[0]}"
    assert ubg.shape[0] == g.shape[0], f"lengths of g and ubg are inconsistent. Lengths: g: {g.shape[0]}, ubg: {ubg.shape[0]}"


    for i, val in enumerate(g_val):
        lb = lbg[i] if i < len(lbg) else -np.inf
        ub = ubg[i] if i < len(ubg) else np.inf

        if val < lb - tol or val > ub + tol:
            violated_constraints.append((i, val, lb, ub, sol.g_describe(i)))

    # Print results
    if not violated_constraints:
        if not print_only_if_false: print("✅ All constraints are satisfied within tolerance.")
        return True
    else:
        print(f"WARNING: {len(violated_constraints)} constraint(s) violated:")
        for idx, val, lb, ub, g_description in violated_constraints:
            print(f"  - Constraint {idx}: value = {val:.6g}, bounds = [{lb:.6g}, {ub:.6g}], description = [{g_description}]")
        return False




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