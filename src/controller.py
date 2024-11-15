import casadi as ca
import numpy as np
from market import Market
from plant import PlantModel
from config import Config
from utils import print_cost_comparison_table, print_bidding_table
import time
import os
import json

class Controller():
    """The controller is tasked with finding an optimal 
    bidding strategy while making sure the plant reaches 
    the required fresh weight mass"""

    model: PlantModel
    market: Market
    config: Config

    N: float
    T: float
    dt: float

    p_spot: np.array

    t: np.array

    sol_base: dict
    f_base: float
    eps_base: float
    u_base: np.array
    x_base: np.array
    elapsedtime_base: float

    f_opt: float
    u_opt: np.array
    x_opt: np.array
    B_opt: np.array
    Eps_opt: float

    sol_bid: dict
    f_bid: float
    u_bid: np.array
    x_bid: np.array
    B_bid: np.array
    eps_bid: float
    elapsedtime_base: float

    def __init__(self, N, T, dt, plantmodel, market, config, baseline: str):
        self.N = N      
        self.T = T   
        self.dt = dt   
        self.model = plantmodel  
        self.market = market
        self.config = config
        self.t = np.linspace(0, T, N)
        
        #
        self.p_spot = self.market.get_spotprice()

        if(baseline == 'opt'):
            self.optimize_baseline()
        elif(baseline == 'rigid'):
            self.rigid_baseline()
        
    '''
    def __post_init__(self):
        self.dt = self.T/self.N
    '''

    def optimize(self):
        
        start_time = time.time()


        B_p_up_0 = 0
        B_p_dn_0 = 0
        B_c_up_0 = 0
        B_c_dn_0 = 0

        B_c_up_1 = 0
        B_c_dn_1 = 0
        B_p_up_1 = 0
        B_p_dn_1 = 0

        B_a_up_0 = 0
        B_a_dn_0 = 0
        # B_a_up_1 = self.market.Pr_a_up(B_c_up_1)
        # B_a_dn_1 = self.market.Pr_a_dn(B_c_dn_1)

        N = self.N
        T = self.T
        dt = self.dt

        # State and control dimensions
        nx = self.model.nx            # Dimension of state x (x1, x2)
        nu = self.model.nu            # Dimension of control u (scalar)
        C_conv_PPFD = self.model.C_conv_PPFD    # Conversion from Light level to power [PPFD -> MW]

        # Create decision variables for the optimization problem
        X = ca.MX.sym('X', nx, N+1)             # States over time (2x(N+1) vector)
        # U = ca.MX.sym('U', nu, N)             # Controls over time (1xN vector)
        B = ca.MX.sym('B', 4, N)                # Bids over time (Vol_up, Vol_down, Price_up, Price_down) (4xN vector)
        Eps = ca.MX.sym('Eps')                  # Slack variable (scalar)


        # Get spot price and baseline
        p_spot = self.p_spot
        u_base = self.u_base


        # Initialize cost function and constraints
        J = self.bidding_objective_function(B, Eps)                         # Cost function
        g = []                        # Equality constraint list
        h = []                        # Inequality constraint list

        # Initial state constraint
        x0 = np.array([5, 1])         # Define the initial state: x1=0, x2=0
        g.append(X[:, 0] - x0)        # Enforce the initial condition
        g.append(B[:, 0] - np.array([B_p_up_0, B_p_dn_0, B_c_up_0, B_c_dn_0]))      # Enforce bids for Q0
        g.append(B[:, 1] - np.array([B_p_up_1, B_p_dn_1, B_c_up_1, B_c_dn_1]))      # Enforce bids for Q1

        # Define the dynamic and control constraints
        for k in range(0,N):
            # Model equalities
            # Using basic forward euler #TODO Evaluate other methods

            if(k==0):
                u_tilde = 1000*(B[1,k]*B_a_dn_0 - B[0,k]*B_a_up_0)/C_conv_PPFD
                x_next = X[:, k] + dt*self.model.derivative(X[:, k], u_base[k] + u_tilde)
                g.append(X[:, k+1] - x_next)
            else:
                u_tilde = 1000*(B[1,k]*self.market.Pr_a_dn(B[3,k]) - B[0,k]*self.market.Pr_a_up(B[2,k]))/C_conv_PPFD
                x_next = X[:, k] + dt*self.model.derivative(X[:, k], u_base[k] + u_tilde)
                g.append(X[:, k+1] - x_next)
            
        
        # Final freshweight constraint
        h.append(self.model.freshweight(X[:,-1]) + Eps - self.model.Final_fw_sht) 
        
        # Upper and lower bounds on u
        for k in range(0,N):

            u_tilde = 1000*(B[1,k]*self.market.Pr_a_dn(B[3,k]) - B[0,k]*self.market.Pr_a_up(B[2,k]))/C_conv_PPFD

            h.append(u_base[k] + u_tilde)
            h.append(self.model.C_PPFD_max - (u_base[k] + u_tilde))

        n_eq = ca.vertcat(*g).size()[0]
        n_ineq = ca.vertcat(*h).size()[0]
        g = g + h #Sum together the equality and inequality constraints

        lbg = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)                         # \ Eq-constraints = 0
        ubg = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)   # / Ineq-constraints >= 0

        # Define bounds on x and u
        lbx = 0* np.ones((nx, N+1))         # Lower bound for x (x >= 0)
        ubx = np.inf * np.ones((nx, N+1))   # Upper bound for x (no upper bound)

        lb_B = 0 * np.ones((4, N))
        ub_B = np.vstack((C_conv_PPFD * u_base/1000,                             # Bid vol up
                          C_conv_PPFD * (self.model.C_PPFD_max - u_base)/1000,   # Bid vol down
                          1000 * np.ones((1, N)),                                # Bid price up. Arbitrary limit of 1000€ / MW 
                        #   100*p_spot*1000/self.market.C_eur2nok,               # Bid price up
                        #   p_spot*1000/self.market.C_eur2nok))                  # Bid price down limited to spot price in eur/MW.  might be arbitrary
                          1000 * np.ones((1, N))))                               # Bid price down. Arbitrary limit of 1000€ / MW 
        # ub_B = 0 * np.ones((4, N)) # TODO Uncomment to set all bids to 0

        lbeps = 0
        ubeps = np.inf

        # Flatten decision variables and bounds
        Z   = ca.vertcat(ca.reshape(X,   -1, 1), ca.reshape(B,    -1, 1), Eps)
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(lb_B, -1, 1), ca.reshape(lbeps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ub_B, -1, 1), ca.reshape(ubeps, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        x0 = ca.DM.zeros(Z.size1())  
        x0[0:2] = self.model.x_init  # enforce init state

        
        sol = solver(x0=x0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution

        self.sol_bid = sol
        self.f_bid = float(sol['f'])
        self.x_bid = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        self.B_bid = np.array(sol['x'][(nx*(N+1)):(nx*(N+1) + 4*N)].reshape((4, N)))
        self.eps_bid = float(sol['x'][-1])

        self.u_bid = u_base + 1000*(np.multiply(self.B_bid[1,:], self.market.Pr_a_dn(self.B_bid[3,:]))\
                                     - np.multiply(self.B_bid[0,:], self.market.Pr_a_up(self.B_bid[2,:])))/C_conv_PPFD
        
        
        end_time = time.time()
        self.elapsedtime_bid = end_time - start_time

        print('Bids optimized')

        self.status_report()
        return 0
        


    def bidding_objective_function(self, B, eps):
        
        N = self.N
        p_spot = self.p_spot

        Bp_up = B[0,:]
        Bp_dn = B[1,:]
        Bc_up = B[2,:]
        Bc_dn = B[3,:]

        L = 0

        # for k in range(0, N): #from k = 2, to N-1. 
        #     L += (1000*p_spot[k] - self.market.C_eur2nok * Bc_dn[k]) * Bp_dn[k] * self.market.Pr_a_dn(Bc_dn[k])\
        #           - (1000*p_spot[k] + self.market.C_eur2nok * Bc_up[k]) * Bp_up[k] * self.market.Pr_a_up(Bc_up[k])
        
        for k in range(0, N): #from k = 2, to N-1. 
            L += p_spot[k] * self.model.C_conv_PPFD * self.u_base[k] \
                  + (1000*p_spot[k] - self.market.C_eur2nok * Bc_dn[k]) * Bp_dn[k] * self.market.Pr_a_dn(Bc_dn[k])\
                  - (1000*p_spot[k] + self.market.C_eur2nok * Bc_up[k]) * Bp_up[k] * self.market.Pr_a_up(Bc_up[k])

        L = L/4

        L += eps * 10**10

        return L


    import numpy as np

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

        x0 = self.model.x_init
        X = np.zeros((self.model.nx, N+1))
        X[:,0] = self.model.x_init.reshape(1,-1)
        for k in range(N):
            #Forward euler
            dt = self.dt
            X[:,k+1] = X[:,k] + dt*np.array(self.model.derivative(X[:,k], np.array([u_base[k]]))).reshape(1, -1)


        self.x_base = X
        self.u_base = u_base
        return 0
    

    def optimize_baseline(self):
        '''
        Temporary function to get a generic baseline lighting schedule.
        This schedule assumes 18 hours on, 6 hours off.
        '''

        start_time = time.time()

        u_base_ub = 1*self.model.C_PPFD_max
        u_base_lb = 0*self.model.C_PPFD_max

        N = self.N
        T = self.T
        dt = self.dt

        # State and control dimensions
        nx = self.model.nx                              # Dimension of state x (x1, x2)
        nu = self.model.nu                              # Dimension of control u (scalar)

        # Create decision variables for the optimization problem
        X = ca.MX.sym('X', nx, N+1)                     # States over time ((N+1)x1 vector)
        U = ca.MX.sym('U', nu, N)                       # Controls over time (Nx1 vector)
        Eps = ca.MX.sym('Eps')                          # Slack variable (scalar)

        J = self.baseline_obj_function(U, Eps)         # Cost function
        g = []                                          # Equality constraint list
        h = []                                          # Inequality constraint list

        # Initial state constraint
        g.append(X[:, 0] - self.model.x_init)                                       # Enforce the initial condition
        h.append(self.model.freshweight(X[:,-1]) + Eps - self.model.Final_fw_sht)   # Enforce final weight condition

        # Define the dynamic and control constraints
        for k in range(N):
            #Using basic forward euler #TODO Evaluate other integration methods

            x_next = X[:, k] + dt*self.model.derivative(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)
            

        n_eq = ca.vertcat(*g).size()[0]
        n_ineq = ca.vertcat(*h).size()[0]
        lbg = np.concatenate((np.zeros((1, n_eq + n_ineq))), axis=None)
        ubg = np.concatenate((np.zeros((1, n_eq)), np.inf * np.ones((1, n_ineq))), axis=None)


        # Define bounds on x and u
        lbx = 0* np.ones((nx, N+1))             # Lower bound for x (x >= 0)
        ubx = np.inf * np.ones((nx, N+1))       # Upper bound for x (no upper bound)

        lbu = u_base_lb * np.ones((nu, N))      # Lower bound for u (u >= 0)
        ubu = u_base_ub * np.ones((nu, N))      # Upper bound for u (u <= Max PPFD 250)

        lbeps = 0
        ubeps = np.inf

        # Flatten decision variables and bounds
        Z =   ca.vertcat(ca.reshape(X, -1, 1),   ca.reshape(U, -1, 1),   Eps)
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(lbu, -1, 1), ca.reshape(lbeps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ubu, -1, 1), ca.reshape(ubeps, -1, 1))
        g = g + h #Sum together the equality and inequality constraints

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        x0 = ca.DM.zeros(Z.size1())
        x0[0:2] = self.model.x_init  # Enforce init state

        sol = solver(x0=x0, lbg=lbg, ubg=ubg, lbx=lbz, ubx=ubz)

        # Extract solution
        self.sol_base = sol
        self.x_base = np.array(sol['x'][:(nx*(N+1))].reshape((nx, N+1)))
        self.u_base = np.array(sol['x'][(nx*(N+1)):(nx*(N+1)+N*nu)].reshape((nu, N)))[0,:]
        self.f_base = np.sum(np.multiply(self.p_spot, self.u_base))*self.model.C_conv_PPFD/4 # Scale down from p_spot*u to p_spot*u*C_conv_ppfd/4
        self.eps_base = float(sol['x'][-1])

        end_time = time.time()
        self.elapsedtime_base = end_time - start_time
        print('Baseline optimized')
        return 0


    def baseline_obj_function(self, U, eps):
        N = self.N
        p_spot = self.p_spot

        L = 0
        for k in range(N):
            L += p_spot[k] * U[k] * self.model.C_conv_PPFD/4
                  
        L += eps * 10**10

        return L



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

        u_tilde = 1000*(np.multiply(b_a_dn, b_p_dn) - np.multiply(b_a_up, b_p_up))/self.model.C_conv_PPFD

        bidding_costs = np.sum(np.multiply(self.p_spot, (self.u_base + u_tilde)))*self.model.C_conv_PPFD/4

        bidding_eps_penalty = self.eps_bid * 10**10

        f_opt = bidding_costs - bidding_earnings


        baseline_costs = np.sum(np.multiply(self.u_base, self.p_spot)) * self.model.C_conv_PPFD / 4
        baseline_eps_penalty = self.eps_base * 10**10



        print("")
        print(f"Baseline f-val minus eps: {float(self.sol_base['f']) - self.eps_base*10**10}")
        print(f"Bidding f-val minus eps: {float(self.sol_bid['f']) - self.eps_bid*10**10}")

        print(f"Obj function f-val: {self.bidding_objective_function(self.B_bid, self.eps_bid)}")

        print(f"Calculated earnings: {np.sum(bidding_earnings)}")
        print(f"Calculated costs: {np.sum(bidding_costs)}")
        print(f"Calculated total cost from bidding: {f_opt}")
        print("")


                
        cost_data = {
            'Cost of power': [baseline_costs, bidding_costs],
            'Cost of bidding': [0, -bidding_earnings],
            'Epsilon penalty': [baseline_eps_penalty, bidding_eps_penalty]
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
            'Avg feasible bid price': [np.average(filtered_b_c_up), np.average(filtered_b_c_dn), "€/MW"],
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

        print(f"Missing fresh weight: {self.eps_bid}g per plant")


        f_base = self.f_base
        print(f"\nCost of base: {f_base}")
        print(f"Cost after bidding: {f_opt}")
        print(f"Cost reduction from bidding: {f_base - f_opt}")
        print(f"Reduction in percentage: {100*(f_base - f_opt)/(f_base)} \n")




