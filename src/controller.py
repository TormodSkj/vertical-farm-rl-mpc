import casadi as ca
import numpy as np
from market import Market, Bid
from model import *
from config import Config
from utils import print_cost_comparison_table, print_bidding_table
import time
import os
import json
from globals import *

class Controller():
    """The controller is tasked with finding an optimal 
    bidding strategy while making sure the plant reaches 
    the required fresh weight mass"""

    surpress_output: bool

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


    sol_base: dict
    f_base: float
    u_base: np.array
    x_base: np.array
    eps_base: float
    elapsedtime_base: float

    f_opt: float
    u_opt: np.array
    x_opt: np.array
    B_opt: np.array

    sol_bid: dict
    f_bid: float
    u_bid: np.array
    x_bid: np.array
    B_bid: np.array
    eps_bid: float
    elapsedtime_base: float

    def __init__(self, timehorizon, plantmodel, market, config, baseline: str, surpress_output = False):
        self.surpress_output = False
        self.T = timehorizon   
        self.N = timehorizon * QUARTER_HOURS_PER_DAY
        self.dt = SECONDS_PER_QUARTER_HOUR   
        self.model = plantmodel  
        self.market = market
        self.config = config
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
        
        start_time = time.time()


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
                          + self.model.final_cost(self, X, U, Eps)  # Cost function


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

        
        sol = solver(x0=self.bidding_z_init, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution

        self.sol_bid = sol
        self.f_bid = float(sol['f'])
        self.x_bid = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        self.B_bid = np.array(sol['x'][(nx*(N+1)):(nx*(N+1) + 4*N)].reshape((4, N)))
        self.eps_bid = float(sol['x'][-1])

        self.u_bid = self.model.get_u(self, self.B_bid)
        
        end_time = time.time()
        self.elapsedtime_bid = end_time - start_time


        if not self.surpress_output: 
            print('Bids optimized')
            self.status_report()
        return 0
        


    def optimize_baseline(self):
        '''
        Optimizes the baseline light schedule purely based on spot price
        '''
        start_time = time.time()


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
                     + self.model.final_cost(self, X, U, Eps)         # Cost function

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

        sol = solver(x0=self.baseline_z_init, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution
        self.sol_base   = sol
        self.f_base     = float(sol['f'])
        self.x_base     = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        self.u_base     = np.array(sol['x'][(nx*(N+1)):(nx*(N+1)+N*nu)].reshape((nu, N)))[0,:]
        self.eps_base   = float(sol['x'][-1])

        end_time = time.time()
        self.elapsedtime_base = end_time - start_time
        if not self.surpress_output: print('Baseline optimized')
        return 0


    def rigid_baseline(self):
        '''
        Temporary function to get a generic baseline lighting schedule.
        This schedule assumes 18 hours on, 6 hours off.
        '''

        N = self.N

        # 18 hours on, 6 hours off in 15 minute intervals
        intervals_per_hour = 4   # 4 intervals (15 minutes) per hour
        hours_on = 18
        hours_off = 6

        # Create a pattern for one full day (96 intervals for 24 hours)
        day_schedule = np.array([self.model.C_PPFD_max/2] * (hours_on * intervals_per_hour) + [0] * (hours_off * intervals_per_hour))

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


        self.x_base = X
        self.u_base = u_base
        return 0
    

    def save_to_json(self):
        # Convert arrays to lists for JSON serialization
        data_to_save = {
            "u_base": np.array(self.u_base).tolist(),  
            "x_base": np.array(self.x_base).tolist(),
            "u_bid": np.array(self.u_bid).tolist(),
            "x_bid": np.array(self.x_bid).tolist(),
        }

        # Filepath
        sim_name = self.config.sim_name
        sim_save_path = os.path.join(self.config.sim_path, f"{sim_name}.json")

        # Ensure the target json file exists
        os.makedirs(self.config.sim_path, exist_ok=True)

        # Save the file
        with open(sim_save_path, "w") as json_file:
            json.dump(data_to_save, json_file, indent=4)


    def status_report(self):

        b_p_up = self.B_bid[0,:]
        b_p_dn = self.B_bid[1,:]
        b_c_up = self.B_bid[2,:]
        b_c_dn = self.B_bid[3,:]
        b_a_up = self.market.Pr_a_up(b_c_up)
        b_a_dn = self.market.Pr_a_dn(b_c_dn)

        bidding_earnings_up = self.market.C_eur2nok * 1/4 * np.multiply(np.multiply(b_a_up, b_p_up), b_c_up)
        bidding_earnings_dn = self.market.C_eur2nok * 1/4 * np.multiply(np.multiply(b_a_dn, b_p_dn), b_c_dn)
        bidding_earnings = np.sum(bidding_earnings_up) + np.sum(bidding_earnings_dn)

        bidding_costs = self.model.bidding_objective_function(self, self.x_bid, self.u_bid, self.B_bid)

        f_opt = bidding_costs - bidding_earnings

        baseline_costs = self.model.baseline_obj_function(self, self.x_base, self.u_base)


        print("")
        print(f"Baseline f-val: {float(self.sol_base['f'])}")
        print(f"Bidding f-val: {float(self.sol_bid['f'])}")

        print(f"Obj function f-val: {self.model.bidding_objective_function(self, self.x_bid, self.u_bid, self.B_bid)}")

        print(f"Calculated earnings: {np.sum(bidding_earnings)}")
        print(f"Calculated costs: {np.sum(bidding_costs)}")
        print(f"Calculated total cost from bidding: {f_opt}")
        print("")


                
        cost_data = {
            'Cost of power': [baseline_costs, bidding_costs],
            'Cost of bidding': [0, -bidding_earnings],
        }

        print_cost_comparison_table("Baseline", "Bidding", cost_data)
        print("")
        
        b_a_up = np.array(self.market.Pr_a_up(b_c_up))
        b_a_dn = np.array(self.market.Pr_a_dn(b_c_dn))
        up_bids = np.where(b_a_up.flatten() > 1e-6)
        dn_bids = np.where(b_a_dn.flatten() > 1e-6)

        filtered_b_p_up = b_p_up[up_bids]
        filtered_b_p_dn = b_p_dn[dn_bids]
        filtered_b_c_up = b_c_up[up_bids]
        filtered_b_c_dn = b_c_dn[dn_bids]
        filtered_b_a_up = 100*b_a_up[up_bids]
        filtered_b_a_dn = 100*b_a_dn[dn_bids]

        bidding_data = {
            'Avg bid size': [np.average(filtered_b_p_up), np.average(filtered_b_p_dn), "MW"],
            'Avg bid price': [np.average(filtered_b_c_up), np.average(filtered_b_c_dn), "€/MW"],
            'Avg activation rate': [np.average(filtered_b_a_up), np.average(filtered_b_a_dn), "%"],
            'Chance of activation given demand': [np.average(filtered_b_a_up)/self.market.Pr_D_up(), np.average(filtered_b_a_dn)/self.market.Pr_D_up(), "%"],
            'Submitted bids': [len(filtered_b_a_up), len(filtered_b_a_dn), "-"]
        }

        print_bidding_table(bidding_data)
        print("")


        
        # Print solve times
        minutes, seconds = divmod(self.elapsedtime_base, 60)
        print(f"Baseline opt solved in: {int(minutes)} minutes and {seconds:.2f} seconds. ")
        minutes, seconds = divmod(self.elapsedtime_bid, 60)
        print(f"Bidding opt solved in: {int(minutes)} minutes and {seconds:.2f} seconds. \n")


        self.f_opt = f_opt


        f_base = self.f_base
        print(f"\nCost of base: {f_base}")
        print(f"Cost after bidding: {f_opt}")
        print(f"Cost reduction from bidding: {f_base - f_opt}")
        print(f"Reduction in percentage: {100*(f_base - f_opt)/(f_base)} \n")




