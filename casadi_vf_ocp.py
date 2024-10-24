import casadi as ca
import numpy as np
from ocp_utils import *
from utils import *

# Time horizon and discretization
T = 30.0          # Total time in days
N = int(4*24*T)   # Number of control intervals
dt = T / N        # Time step

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
p_spot = generate_spotprice(N) #Generates a spot price according to a simple model


# Initialize cost function and constraints
J = cost_function(N, p_spot, B_p_up, B_p_dn, B_c_up, B_c_dn, Eps)                         # Cost function
g = []                        # Constraint list

# Initial state constraint
x0 = np.array([5, 1])         # Define the initial state: x1=0, x2=0
g.append(X[:, 0] - x0)        # Enforce the initial condition

# Define the dynamic and control constraints
for k in range(N):
    # Dynamics constraint: X_k+1 = X_k + dt * [x2_k, u_k]
    x_next = plant_model_derivative(X[:, k], U[:, k])
    g.append(X[:, k+1] - x_next)

#inits
g.append(U[:,0] - (p_spot[0] + B_p_dn[0] * Pr_a_dn(B_c_dn[0]) - B_p_up[0] * Pr_a_up(B_c_up[0])))


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

# Display the results
print("Optimal state trajectory (X):", X_opt)
print("Optimal control trajectory (U):", U_opt)

x_ts = np.array(X_opt)
u_ts = np.array(U_opt)

x1_ts = x_ts[0,:]
x2_ts = x_ts[1,:]

plot_trajectory(x1_ts, x2_ts, "casadi_ocp_test")
plotting(np.linspace(1,N,N), u_ts, "casadi_ocp_test_u")
