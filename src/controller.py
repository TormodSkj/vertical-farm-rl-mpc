import casadi as ca
import numpy as np
from market import Market
from plant import PlantModel


class Controller():
    """The controller is tasked with finding an optimal 
    bidding strategy while making sure the plant reaches 
    the required fresh weight mass"""

    model: PlantModel
    market: Market

    N: float
    T: float
    dt: float

    def __init__(self, N, T, dt):
        self.N = N      
        self.T = T   
        self.dt = dt   
        self.model = PlantModel()  # Instantiate the PlantModel
        self.market = Market()      # Instantiate the Market
        

    '''
    def __post_init__(self):
        self.dt = self.T/self.N
    '''

    def optimize(self):
                
        N = self.N
        T = self.T
        dt = self.dt

        # State and control dimensions
        nx = self.model.nx            # Dimension of state x (x1, x2)
        nu = self.model.nu            # Dimension of control u (scalar)

        # Create decision variables for the optimization problem
        X = ca.MX.sym('X', nx, N+1)             # States over time (Nx1 vector)
        U = ca.MX.sym('U', nu, N)               # Controls over time (scalar)
        B = ca.MX.sym('B', 4, N)                # Bids over time (Vol_up, Vol_down, Price_up, Price_down)
        Eps = ca.MX.sym('eps')                  # Slack variable


        # Get spot price and baseline
        p_spot = self.market.generate_spotprice(N) #Generates a spot price according to a simple model
        u_base = self.baseline_opt(N) #TODO optimize


        # Initialize cost function and constraints
        J = self.cost_function(p_spot, B, Eps)                         # Cost function
        g = []                        # Constraint list

        # Initial state constraint
        x0 = np.array([5, 1])         # Define the initial state: x1=0, x2=0
        g.append(X[:, 0] - x0)        # Enforce the initial condition
        g.append(U[:,0] - u_base[0])  # Assume no bid activation at time 0 TODO implement actual init bid system


        # Define the dynamic and control constraints
        for k in range(N):
            # Dynamics constraint: X_k+1 = X_k + dt * [x2_k, u_k]
            #Using basic forward euler #TODO Evaluate

            x_next = X[:, k] + dt*self.model.derivative(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)
            
            if(k == 0): continue
            g.append(U[:,k] - (u_base[k] + B[1,k]*self.market.Pr_a_dn(B[3,k]) - B[0,k]*self.market.Pr_a_up(B[2,k])))

        # Define bounds on x and u
        lbx = 0* np.ones((nx, N+1))   # Upper bound for x (no upper bound)
        ubx = np.inf * np.ones((nx, N+1))   # Upper bound for x (no upper bound)

        lbu = 0 * np.ones((nu, N))         # Lower bound for u (-2 <= u)
        ubu =  250 * np.ones((nu, N))         # Upper bound for u (u <= 2)

        lb_B = 0 * np.ones((4, N))
        ub_B = np.vstack((self.model.C_conv_PPFD * u_base, self.model.P_cap * np.ones((1, N)), np.inf * np.ones((1, N)), p_spot))

        lbeps = 0
        ubeps = np.inf


        # Flatten decision variables and bounds
        Z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), ca.reshape(B, -1, 1), Eps)
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1), ca.reshape(lbu, -1, 1), ca.reshape(lb_B, -1, 1), ca.reshape(lbeps, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1), ca.reshape(ubu, -1, 1), ca.reshape(ub_B, -1, 1), ca.reshape(ubeps, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J, 'g': ca.vertcat(*g)}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve the problem
        sol = solver(x0=ca.DM.zeros(Z.size1()), lbg=0, ubg=0, lbx=lbz, ubx=ubz)

        # Extract solution
        X_opt = sol['x'][:(nx * (N+1))].reshape((nx, N+1))
        U_opt = sol['x'][(nx*(N+1)):(nx*(N+1)+N*nu)].reshape((nu, N))
        B_opt = sol['x'][(nx*(N+1)+N*nu):(nx*(N+1) + N*nu + 4*N)].reshape((4, N))
        t = np.linspace(0, T, N)

        return t, X_opt, U_opt, B_opt


    def cost_function(self, p_spot, B, eps):
        
        N = self.N

        Bp_up = B[0,:]
        Bp_dn = B[1,:]
        Bc_up = B[2,:]
        Bc_dn = B[3,:]

        L = 0

        for k in range(2, N-1): #from k = 2, to N-1. 
            L += (p_spot[k] - Bc_dn[k]) * Bp_dn[k] * self.market.Pr_a_dn(Bc_dn[k]) - (p_spot[k] + Bc_up[k]) * Bp_up[k] * self.market.Pr_a_up(Bc_up[k])

        L += eps * 10**6

        return L


    import numpy as np

    def baseline_opt(self, N):
        '''
        Temporary function to get a generic baseline lighting schedule.
        This schedule assumes 18 hours on, 6 hours off.
        '''
        # 18 hours on, 6 hours off in 15 minute intervals
        intervals_per_hour = 4   # 4 intervals (15 minutes) per hour
        hours_on = 18
        hours_off = 6

        # Create a pattern for one full day (96 intervals for 24 hours)
        day_schedule = np.array([self.model.C_PPFD_max/2] * (hours_on * intervals_per_hour) + [0] * (hours_off * intervals_per_hour))

        # Repeat the daily schedule enough times to cover N intervals
        full_schedule = np.tile(day_schedule, int(np.ceil(N / len(day_schedule))))[:N]

        return full_schedule



