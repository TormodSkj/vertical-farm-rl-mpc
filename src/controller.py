import casadi as ca
import numpy as np
from market import Market, Bid
from model import *
from config import Config
from utils import *
import time
import os
import json
from globals import *
from tabulate import tabulate
from datetime import datetime

class Controller():
    """The controller is tasked with finding an optimal 
    bidding strategy while making sure the plant reaches 
    the required fresh weight mass"""

    surpress_output: bool
    import_file: str

    model: PlantModel
    market: Market
    config: Config

    N:  float
    T:  float
    dt: float
    t:  np.array

    p_spot:     np.array
    x_init:     np.array

    bids:       list[Bid]
    A_up:       list[bool]
    A_down:     list[bool]

    bidding_z_init: ca.DM
    baseline_z_init: ca.DM

    runs:           dict
    search_cache:   bool
    warm_start:     bool
    calculate_fw:   bool
    u_base:         np.array
    x_base:         np.array

    def __init__(self, timehorizon, plantmodel, market, config, 
                 baseline = 'opt', surpress_output = False, search_cache = True, 
                 import_file = '', warm_start = True, calculate_fw = True):
        self.surpress_output = surpress_output
        self.warm_start = warm_start
        self.calculate_fw = True
        self.T = timehorizon   
        self.N = timehorizon * QUARTER_HOURS_PER_DAY
        self.dt = SECONDS_PER_QUARTER_HOUR   
        self.model = plantmodel  
        self.market = market
        self.config = config
        self.search_cache = search_cache
        self.import_file = import_file

        specs_data = {
            'controller': 
                {
                    'time horizon'          : self.T,
                    'N'                     : self.N,
                },
            'model'             : self.model.specs,
            'market'            : self.market.specs
        }
        self.runs = {
            'name'              : self.config.sim_name,
            'timestamp'         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'specs'             : specs_data,
            'bidding result'    : None,
            'runs'              : {}
            }
        self.hash = generate_hash(self.runs['specs'])

        self.t = np.linspace(0, self.T, self.N)
        self.p_spot = self.market.get_spotprice()

        self.bids = []
        for i in range(self.market.n_given_bids):
            self.bids.append(Bid())

        self.A_up, self.A_down = [],[]
        for i in range(self.market.n_given_activations):
            self.A_up.append(0)
            self.A_down.append(0)


        # Set initial state
        self.x_init = self.model.x_init

        # Start the bidding and baseline solvers with the init x state
        bidding_z0 = ca.DM.zeros((self.N + 1)*self.model.nx + self.N*4 + 1)  
        bidding_z0[0:self.model.nx] = self.x_init  # enforce init state
        self.bidding_z_init = bidding_z0

        baseline_z0 = ca.DM.zeros((self.N + 1)*self.model.nx + self.N*self.model.nu + 1)  
        baseline_z0[0:self.model.nx] = self.x_init  # enforce init state
        self.baseline_z_init = baseline_z0
        
 
        
    def set_bids(self, Bid_0, Bid_1):
        self.Bid_0 = Bid_0
        self.Bid_1 = Bid_1



    def optimize_bidding(self):
        
        # 
        start_time = time.time()

        hash = generate_hash(self.runs['specs'])
        if self.search_cache:
            if self.load_from_json(hash, 'Bidding'): 
                # Identical run located. Using its solution instead
                return 0
            # No identical run located, or the needed run wasn't already produced. Optimizing from scratch
            if not self.surpress_output: print('No matching run found. Generating bidding strategy')
        

        # Just check if there is a basline before proceeding
        assert 'Baseline' in self.runs['runs'] or 'Rigid' in self.runs['runs'], "Baseline was not generated"   

        N = self.N
        T = self.T
        dt = self.dt

        # State and control dimensions
        nx = self.model.nx                      # Dimension of state x (x1, x2)
        nu = self.model.nu                      # Dimension of control u (scalar)

        # Create decision variables for the optimization problem
        X = ca.MX.sym('X', nx, N+1)             # States over time (2x(N+1) vector)
        B = ca.MX.sym('B', 4, N)                # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        Eps = ca.MX.sym('Eps', 1, 1)            # Slack variable for feasibility

        U = self.model.get_u(self, B)           # Express U in terms of bidding outcomes

        # Initialize cost function and constraints
        J = self.model.bidding_objective_function(self, X, U, B)\
                       + self.model.terminal_cost(self, X, U, Eps)\
                       + self.model.fluctuating_light_cost(self, U) # Cost function


        g_eq, g_ineq = [], []
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, X, U, Eps)
        g_eq, g_ineq = self.model.get_bidding_constraints(self, g_eq, g_ineq, B)

        
        # format constraints
        n_eq = ca.vertcat(*g_eq).size()[0]
        n_ineq = ca.vertcat(*g_ineq).size()[0]
        g = g_eq + g_ineq #Sum together the equality and inequality constraints
        lbg = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)                         # \ Eq-constraints = 0
        ubg = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)   # / Ineq-constraints >= 0


        # Extract state and bidding bounds
        lbx, ubx = self.model.get_state_bounds(self)
        lb_B, ub_B = self.model.get_bidding_bounds(self)
        lb_eps, ub_eps = 0, np.inf

        # Flatten decision variables and bounds
        Z   = ca.vertcat(ca.reshape(X,   -1, 1), ca.reshape(B,    -1, 1), ca.reshape(Eps,    -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(lb_B, -1, 1), ca.reshape(lb_eps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ub_B, -1, 1), ca.reshape(ub_eps, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        z0 = self.bidding_z_init
        if self.warm_start and 'Baseline' in self.runs['runs']: 
            z0[:nx*(N+1)]                   = self.x_base.flatten()
            z0[nx*(N+1):nx*(N+1)+4*N]       = ub_B.reshape(4*N,1)
            z0[nx*(N+1)+2*N:nx*(N+1)+3*N]   = self.market.mean_prices_up
            z0[nx*(N+1)+3*N:nx*(N+1)+4*N]   = self.market.mean_prices_dn
        
        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution

        sol = sol
        x = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        B = np.array(sol['x'][(nx*(N+1)):(nx*(N+1) + 4*N)].reshape((4, N)))
        u = np.array(self.model.get_u(self, B)).flatten()
        
        end_time = time.time()
        sol['elapsed_time'] = end_time - start_time
        
        self.save_run('Bidding', sol, x, u, B=B)

        if not self.surpress_output: print('Bids optimized')
        return 0
        

    def optimize_baseline(self):
        '''
        Optimizes the baseline light schedule purely based on spot price
        '''
        start_time = time.time()

        hash = generate_hash(self.runs['specs'])
        if self.search_cache:
            if self.load_from_json(hash, 'Baseline'): 
                # Identical run located. Using its solution instead
                self.x_base = self.runs['runs']['Baseline']['timeseries']['x']
                self.u_base = self.runs['runs']['Baseline']['timeseries']['u']
                return 0
            # If no identical run was located, generate baseline instead
            if not self.surpress_output: print('No matching run found. Generating baseline light schedule')
        

        N = self.N
        T = self.T

        # State and control dimensions
        nx = self.model.nx                              # Dimension of state x (x1, x2)
        nu = self.model.nu                              # Dimension of control u (scalar)

        # Create decision variables for the optimization problem
        X = ca.MX.sym('X', nx, N+1)                     # States over time ((N+1)x1 vector)
        U = ca.MX.sym('U', nu, N)                       # Controls over time (Nx1 vector)
        Eps = ca.MX.sym('Eps', 1, 1)                    # Slack variable for feasibility

        J = self.model.baseline_obj_function(self, X, U)\
                     + self.model.terminal_cost(self, X, U, Eps)\
                     + self.model.fluctuating_light_cost(self, U)         # Cost function

        # Get bounds
        lbx, ubx = self.model.get_state_bounds(self)
        lbu, ubu = self.model.get_input_bounds(self)
        lb_eps, ub_eps = 0, np.inf

        # Get constraints
        g_eq, g_ineq = [],[]
        g_eq, g_ineq = self.model.get_process_constraints(self, g_eq, g_ineq, X, U, Eps)
        g = g_eq + g_ineq 


        # Flatten decision variables and bounds
        Z =   ca.vertcat(ca.reshape(X, -1, 1),   ca.reshape(U, -1, 1),   ca.reshape(Eps,    -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(lbu, -1, 1), ca.reshape(lb_eps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ubu, -1, 1), ca.reshape(ub_eps, -1, 1))

        # Format constraints
        n_eq = ca.vertcat(*g_eq).size()[0]
        n_ineq = ca.vertcat(*g_ineq).size()[0]
        lbg = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)
        ubg = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        z0 = self.baseline_z_init
        if self.warm_start and 'Rigid' in self.runs['runs']: 
            z0[:nx*(N+1)]                   = self.x_base.flatten()
            z0[nx*(N+1):(nx*(N+1)+N*nu)]    = self.u_base.flatten()

        sol = solver(x0=z0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution
        f     = float(sol['f'])
        x     = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        u     = np.array(sol['x'][(nx*(N+1)):(nx*(N+1)+N*nu)].reshape((nu, N)))[0,:]
        eps   = float(sol['x'][-1])

        end_time = time.time()
        elapsed_time = end_time - start_time

        sol['elapsed_time'] = elapsed_time
        
        self.save_run('Baseline', sol, x, u)
        self.u_base = u
        self.x_base = x
        self.runs['specs']['controller']['baseline'] = 'opt'

        if not self.surpress_output: print('Baseline optimized')
        return 0


    def rigid_baseline(self):
        '''
        Temporary function to get a generic baseline lighting schedule.
        This schedule assumes 18 hours on, 6 hours off.
        '''

        start_time = time.time()

        N = self.N

        # 18 hours on, 6 hours off in 15 minute intervals
        intervals_per_hour = 4   # 4 intervals (15 minutes) per hour
        hours_on = 16
        hours_off = 8

        RIGID_INTY = 0.8 * self.model.C_PPFD_max

        # Create a pattern for one full day (96 intervals for 24 hours)
        day_schedule = np.array([RIGID_INTY] * (hours_on * intervals_per_hour) + [0] * (hours_off * intervals_per_hour))

        # Repeat the daily schedule enough times to cover N intervals
        full_schedule = np.tile(day_schedule, int(np.ceil(N / len(day_schedule))))[:N]


        u_base = full_schedule

        x0 = self.x_init
        X = np.zeros((self.model.nx, N+1))
        X[:,0] = x0.reshape(1,-1)
        for k in range(N):
            #Forward euler
            dt = self.dt
            X[:,k+1] = X[:,k] + dt*np.array(self.model.derivative(X[:,k], np.array([u_base[k]]))).reshape(1, -1)

        sol ={}
        x = X
        u = u_base

        end_time = time.time()
        elapsed_time = end_time - start_time

        sol['elapsed_time'] = elapsed_time
        sol['f'] = self.model.baseline_obj_function(self, x, u)
        sol['x'] = np.hstack((x.flatten(), u, 0))
        
        self.save_run('Rigid', sol, x, u)
        self.u_base = u
        self.x_base = x
        self.model.Final_fw_sht = float(self.model.freshweight(x[:,-1]))
        # self.runs['specs']['controller']['baseline'] = 'rigid'

        if not self.surpress_output: print('Generated rigid baseline')
        return 0

    def save_run(self, run_id, sol, x, u, B = None):

        timeseries_data = {
            't'     : self.t,
            'x'     : x,
            'u'     : u
        }
        
        f       = float(sol['f'])
        eps     = float(sol['x'][-1])


        metrics_data = {
            'elapsed_time'  : sol['elapsed_time'],
            'f'             : f,
            'eps'           : eps
        }

        metrics_data = self.model.get_metrics(self, run_id, metrics_data, x, u, B)

        if B is None:
            costs = self.model.baseline_obj_function(self, x, u)
            metrics_data['Costs'] = costs
            metrics_data['Earnings'] = 0
            metrics_data['Total'] = costs - 0
        else:
            b_p_up = B[0,:]
            b_p_dn = B[1,:]
            b_c_up = B[2,:]
            b_c_dn = B[3,:]
            b_a_up = self.market.Pr_a_up(self.p_spot, b_c_up)
            b_a_dn = self.market.Pr_a_dn(self.p_spot, b_c_dn)


            timeseries_data['P_up'] = b_p_up
            timeseries_data['P_dn'] = b_p_dn
            timeseries_data['C_up'] = b_c_up
            timeseries_data['C_dn'] = b_c_dn


            bidding_earnings_up = self.market.C_eur2nok * 1/4 * np.multiply(np.multiply(b_a_up, b_p_up), b_c_up)
            bidding_earnings_dn = self.market.C_eur2nok * 1/4 * np.multiply(np.multiply(b_a_dn, b_p_dn), b_c_dn)
            bidding_earnings    = np.sum(bidding_earnings_up) + np.sum(bidding_earnings_dn)

            bidding_costs = self.model.baseline_obj_function(self, x, u)
            bidding_total = bidding_costs - bidding_earnings

            metrics_data['Costs']       = bidding_costs
            metrics_data['Earnings']    = bidding_earnings
            metrics_data['Total']       = bidding_total

            b_a_up  = np.array(self.market.Pr_a_up(self.p_spot, b_c_up))
            b_a_dn  = np.array(self.market.Pr_a_dn(self.p_spot,b_c_dn))
            up_bids = np.where(b_a_up.flatten() > 1e-6)
            dn_bids = np.where(b_a_dn.flatten() > 1e-6)

            filtered_b_p_up = b_p_up[up_bids]
            filtered_b_p_dn = b_p_dn[dn_bids]
            filtered_b_c_up = b_c_up[up_bids]
            filtered_b_c_dn = b_c_dn[dn_bids]
            filtered_b_a_up = 100*b_a_up[up_bids]
            filtered_b_a_dn = 100*b_a_dn[dn_bids]


            bidding_data = {
                'Up-regulation'     : {
                    'Bids submitted'            : len(filtered_b_a_up),
                    'Avg bid size'              : np.average(filtered_b_p_up),
                    'Avg bid price'             : np.average(filtered_b_c_up),
                    'Avg activation rate'       : np.average(filtered_b_a_up)
                },
                'Down-regulation'   : {
                    'Bids submitted'            : len(filtered_b_a_dn),
                    'Avg bid size'              : np.average(filtered_b_p_dn),
                    'Avg bid price'             : np.average(filtered_b_c_dn),
                    'Avg activation rate'       : np.average(filtered_b_a_dn)
                }
            }

            self.runs['bidding result'] = bidding_data
            

        # Storing runs in dictionaries
        self.runs['runs'][run_id] = {
            "metrics"       : metrics_data,
            "timeseries"    : timeseries_data
        }


    def save_to_json(self):
        """
        Save all runs and their data to a JSON file.
        """

        # Add spot price to data
        self.runs['spotprice'] = self.p_spot

        # Filepath
        sim_name = self.config.sim_name
        sim_save_path = os.path.join(self.config.sim_path, f"{sim_name}.json")

        # Ensure the target json file exists
        os.makedirs(self.config.sim_path, exist_ok=True)

        # Convert the entire runs dictionary
        runs_dict = convert_np_arrays_to_lists(self.runs)
        runs_dict['hash'] = self.hash

        # Save the data for all runs
        with open(sim_save_path, "w") as json_file:
            json.dump(runs_dict, json_file, indent=4)


    def load_from_json(self, hash, run_name):
        """
        Load completed runs from saved JSON files and populate `completed_runs`.
        """
        
        for file_name in os.listdir(self.config.sim_path):
            if not file_name.endswith(".json"): continue

            file_path = os.path.join(self.config.sim_path, file_name)
            with open(file_path, "r") as json_file:
                loaded_data = json.load(json_file)
            
            # Extract the hash from the JSON content
            specs_hash = loaded_data.get('hash')
            
            if specs_hash is None:
                continue

            if specs_hash == hash:
                
                # Store the run data keyed by the extracted hash
                conv_loaded_data = convert_lists_to_np_arrays(loaded_data)

                if run_name not in conv_loaded_data['runs']:
                    return False
                
                self.runs['runs'][run_name] = conv_loaded_data['runs'][run_name]
                if run_name == 'Bidding': self.runs['bidding result'] = conv_loaded_data['bidding result']

                if not self.surpress_output: print(f"Loaded run {run_name} from simulation \'{loaded_data['name']}\' dated {loaded_data['timestamp']}")
                return True
                
        return False

    def import_baseline(self):
        '''
        Imports an already made light schedule as the baseline
        '''
        start_time = time.time()
        N = self.N

        # Open and load the JSON file
        import_path = os.path.join(self.config.data_path, self.import_file)
        with open(import_path, "r") as json_file:
            light_schedule = json.load(json_file)
        

        # Transform from hourly to quarter hourly basis
        # Scale from percentage based schedule to light intensity
        u_base = self.model.C_PPFD_max/100*np.repeat(light_schedule, 4)     

        assert len(u_base) == self.N, f"Imported light schedule not correct length. Len: {len(u_base)}, N: {N}"

        x0 = self.x_init
        X = np.zeros((self.model.nx, N+1))
        X[:,0] = x0.reshape(1,-1)
        for k in range(N):
            #Forward euler
            dt = self.dt
            X[:,k+1] = X[:,k] + dt*np.array(self.model.derivative(X[:,k], np.array([u_base[k]]))).reshape(1, -1)

        sol ={}
        x = X
        u = u_base

        end_time = time.time()
        elapsed_time = end_time - start_time

        sol['elapsed_time'] = elapsed_time
        sol['f'] = self.model.baseline_obj_function(self, x, u)
        sol['x'] = np.hstack((x.flatten(), u, 0))
        
        self.save_run('Imported', sol, x, u)
        self.u_base = u

        if not self.surpress_output: print('Imported light schedule')
        return 0



    def status_report(self):


        costs = [self.runs['runs'][run]['metrics']['Costs'] for run in self.runs['runs']]
        earnings = [self.runs['runs'][run]['metrics']['Earnings'] for run in self.runs['runs']]
        totals = [self.runs['runs'][run]['metrics']['Total'] for run in self.runs['runs']]
        cost_reduction_percent = [(totals[0] - totals[i])/totals[0] * 100 for i in range(len(totals))]

        cost_data = [
            ['Costs'] + costs,
            ['Earnings'] + earnings,
            ['Totals'] + totals,
            ['Total percentage cost reduction'] + cost_reduction_percent,
        ]

        cost_table = generate_table(cost_data, header=[run for run in self.runs['runs']])
        print(f'COST DATA: \n{cost_table}\n')


        metrics_table = get_metrics_table(self.runs['runs'])
        print(f'METRICS DATA: \n{metrics_table}\n')


        if 'Bidding' in self.runs['runs']:
            bidding_result_up = self.runs['bidding result']['Up-regulation']
            bidding_result_dn = self.runs['bidding result']['Down-regulation']
            
            bidding_data = [
                ['Avg bid size',                        bidding_result_up['Avg bid size'],                                  bidding_result_dn['Avg bid size'],                               "MW"], 
                ['Avg bid price',                       bidding_result_up['Avg bid price'],                                 bidding_result_dn['Avg bid price'],                              "€/MW"], 
                ['Avg activation rate',                 bidding_result_up['Avg activation rate'],                           bidding_result_dn['Avg activation rate'],                        "%"], 
                ['Chance of activation given demand',   bidding_result_up['Avg activation rate']/self.market.Pr_D_up(),     bidding_result_dn['Avg activation rate']/self.market.Pr_D_up(),  "%"],
                ['Submitted bids',                      bidding_result_up['Bids submitted'],                                bidding_result_dn['Bids submitted'],                             "-"]
            ]
            bidding_header = ['', 'Up-regulation', 'Down-regulation', 'Unit']

            print(f'BIDDING REPORT: \n{generate_table(bidding_data, header = bidding_header)}\n')

        
        # Print solve times
        for run in self.runs['runs']:
            minutes, seconds = divmod(self.runs['runs'][run]['metrics']['elapsed_time'], 60)
            print(f"{run} solved in: {int(minutes)} minutes and {seconds:.2f} seconds. ")
        

        # f_opt = bidding_total
        # f_base = baseline_total
        # print(f"\nCost of base: {f_base}")
        # print(f"Cost after bidding: {f_opt}")
        # print(f"Cost reduction from bidding: {f_base - f_opt}")
        # print(f"Cost reduction in percentage: {100*(f_base - f_opt)/(f_base)} \n")


