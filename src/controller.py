import casadi as ca
import numpy as np
from market import Market
from plant import PlantModel


class Controller:
    """The controller is tasked with finding an optimal 
    bidding strategy while making sure the plant reaches 
    the required fresh weight mass"""

    model = PlantModel
    market = Market

    N: float
    T: float
    dt: float

    def __post_init__(self):
        self.dt = self.T/self.N


    def optimize(self):
                
        N = self.N
        T = self.T
        dt = self.dt

        # State and control dimensions
        nx = 2            # Dimension of state x (x1, x2)
        nu = 1            # Dimension of control u (scalar)

        # Create decision variables for the optimization problem
        X = ca.MX.sym('X', nx, N+1)             # States over time (Nx1 vector)
        U = ca.MX.sym('U', nu, N)               # Controls over time (scalar)
        B_p_up = ca.MX.sym('Bp_up', nu, N)      # Bid volume, up
        B_p_dn = ca.MX.sym('Bp_dn', nu, N)      # Bid volume, down
        B_c_up = ca.MX.sym('Bc_up', nu, N)      # Bid volume, up
        B_c_dn = ca.MX.sym('Bc_dn', nu, N)      # Bid volume, down
        Eps = ca.MX.sym('eps')                  # Slack variable


        # Get spot price
        p_spot = self.market.generate_spotprice(N) #Generates a spot price according to a simple model


        # Initialize cost function and constraints
        J = self.cost_function(N, p_spot, B_p_up, B_p_dn, B_c_up, B_c_dn, Eps)                         # Cost function
        g = []                        # Constraint list

        # Initial state constraint
        x0 = np.array([5, 1])         # Define the initial state: x1=0, x2=0
        g.append(X[:, 0] - x0)        # Enforce the initial condition

        # Define the dynamic and control constraints
        for k in range(N):
            # Dynamics constraint: X_k+1 = X_k + dt * [x2_k, u_k]
            x_next = self.model.derivative(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)

        #inits
        g.append(U[:,0] - (p_spot[0] + B_p_dn[0] * self.market.Pr_a_dn(B_c_dn[0]) - B_p_up[0] * self.market.Pr_a_up(B_c_up[0])))


        # Define bounds on x and u
        lbx = -1* np.ones((nx, N+1))   # Upper bound for x (no upper bound)
        ubx = np.inf * np.ones((nx, N+1))   # Upper bound for x (no upper bound)

        lbu = 0 * np.ones((nu, N))         # Lower bound for u (-2 <= u)
        ubu =  250 * np.ones((nu, N))         # Upper bound for u (u <= 2)

        # Flatten decision variables and bounds
        XU = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        lbxu = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(lbu, -1, 1))
        ubxu = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ubu, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': XU, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve the problem
        sol = solver(x0=ca.DM.zeros(XU.size1()), lbg=0, ubg=0, lbx=lbxu, ubx=ubxu)

        # Extract solution
        X_opt = sol['x'][:(nx * (N+1))].reshape((nx, N+1))
        U_opt = sol['x'][(nx * (N+1)):].reshape((nu, N))


    def cost_function(self, N, p_spot, Bp_up, Bp_dn, Bc_up, Bc_dn, eps):
        
        L = 0

        for k in range(1, N): #from k = 2, to N-1. But 0 indexing makes it k = 1 to N
            L += (p_spot[k] - Bc_dn[k]) * Bp_dn[k] * self.market.Pr_a_dn(Bc_dn[k]) - (p_spot[k] + Bc_up[k]) * Bp_up[k] * self.market.Pr_a_up(Bc_up[k])

        L += eps * 10**6


