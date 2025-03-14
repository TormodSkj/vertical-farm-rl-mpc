import numpy as np
import casadi as ca
from market import Market
from bid import Bid
from globals import *
from settings import Settings

class BatteryModel:

    name = 'Battery'

    x_init:         np.array    # Initial state of charge
    
    def __init__(self, x_init):
        self.x_init = x_init

        self.specs = {
            'type'      : self.name,
            'x0'        : x_init,
            'SOC max'   : self.SOC_max,
            'SOC min'   : self.SOC_min,
            'u_max'     : self.u_max,
            'u_min'     : self.u_min
        } 
    
    nx = 1
    nu = 1

    # state labels and units (for plotting)

    title  = "Battery"
    labels = ["State of charge"]
    x_unit = "Stored energy (kWh)"
    u_unit = "Charge/Discharge"


    #constants:
    SOC_max = 900
    SOC_min = 100
   
    u_max   = 10
    u_min   = -10


    def derivative(self, x: ca.MX.sym, u: ca.MX.sym)->ca.MX.sym:
        
        #Extract state
        SOC     = x[0]
        
        U       = u[0]

        
        SOC_dot = U/1000

        return ca.vertcat(SOC_dot)
    

    def bidding_objective_function(self, controller, X, U, B, U_nom):
        
        N = controller.N
        spot_prices = controller.spot_prices

        Bp_up = B[0,:]
        Bp_dn = B[1,:]
        Bc_up = B[2,:]
        Bc_dn = B[3,:]

        L = 0
        
        for k in range(0, N): #from k = 2, to N-1. 
            L += spot_prices[k] * U_nom[k] \
                  + (spot_prices[k] - Bc_dn[k]) * Bp_dn[k] * controller.market.activation_prob_dn(spot_prices, Bc_dn[k])\
                  - (spot_prices[k] + Bc_up[k]) * Bp_up[k] * controller.market.activation_prob_up(spot_prices, Bc_up[k])

        L = L/4

        return L

    def spotopt_obj_function(self, controller, X, U):
        N = controller.N
        spot_prices = controller.spot_prices

        L = 0
        for k in range(N):
            L += spot_prices[k] * U[k] / 4
                  
        return L

    def terminal_cost(self, controller, X, U, Eps):

        return 0


    def get_bidding_constraints(self, controller, g_eq, g_ineq, B):

        # Enforce initial bids

        for k, bid in enumerate(controller.bids):
            g_eq.append(B[:, k] - bid.as_array())
        
        return g_eq, g_ineq
    
    
    def get_process_constraints(self, controller, g_eq, g_ineq, X, U, Eps):

        N = controller.N
        dt = controller.dt


        # Initial state constraint
        g_eq.append(X[:, 0] - self.x_init)
        
        # Define the dynamic and control constraints
        for k in range(0,N):
            # Model equalities
            x_next = X[:, k] + dt*self.derivative(X[:, k], U[k])
            g_eq.append(X[:, k+1] - x_next)


        # Upper and lower bounds on u
        for k in range(N):

            # Inequality constraints g_ineq >= 0
            g_ineq.append(U[k] - self.u_min)
            g_ineq.append(self.u_max - U[k])


        return g_eq, g_ineq
    


    def get_u(self, controller, B, U_nom):

        N = controller.N
        spot_prices = controller.spot_prices
        u_bar = controller.u_base
        U = np.array([])

        for k in range(N):

            if(k<controller.market.n_given_activations):
                u_tilde = 1000*(B[1,k]*controller.A_down[k] - B[0,k]*controller.A_up[k])
            else:
                u_tilde = 1000*(B[1,k]*controller.market.activation_prob_dn(spot_prices[k], B[3,k]) - B[0,k]*controller.market.activation_prob_up(spot_prices[k], B[2,k]))

            U = np.append(U, u_bar[k] + u_tilde)

        return ca.vertcat(*U)



    def get_bidding_bounds(self, controller):

        N = controller.N

        lb_B = 0 * np.ones((4, N))
        ub_B = np.vstack(((controller.u_base - self.u_min)/1000,   # Bid vol up       abs(Dist from ubase to umin)
                          (self.u_max - controller.u_base)/1000,   # Bid vol down     abs(Dist from ubase to umax)
                          1000 * np.ones((1, N)),      # Bid price up. Arbitrary limit of 1000€ / MW 
                          1000 * np.ones((1, N))))     # Bid price down. Arbitrary limit of 1000€ / MW 
        
        return lb_B, ub_B
    
    def get_state_bounds(self, controller):

        N = controller.N

        # Define bounds on x and u
        lbx = self.SOC_min * np.ones((self.nx, N+1))         # Lower bound for x (x >= 0)
        ubx = self.SOC_max * np.ones((self.nx, N+1))   # Upper bound for x (no upper bound)


        return lbx, ubx
    
    def get_input_bounds(self, controller):
        
        N = controller.N

        lbu = self.u_min * np.ones((self.nu, N))    # Lower bound for u = -10
        ubu = self.u_max * np.ones((self.nu, N))    # Upper bound for u = 10

        return lbu, ubu
    

    def get_metrics(self, controller, metrics_data, x, u, B):

        # TODO Add custom metrics you want to track and display here

        return metrics_data
    