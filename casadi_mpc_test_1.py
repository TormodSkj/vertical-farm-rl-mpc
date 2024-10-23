import casadi as ca
import numpy as np
from utils import *

# Time horizon and discretization
T = 10.0          # Total time
N = 50            # Number of control intervals
dt = T / N        # Time step

# State and control dimensions
nx = 2            # Dimension of state x (x1, x2)
nu = 1            # Dimension of control u (scalar)

# Create optimization variables
x = ca.MX.sym('x', nx)  # State variable (2x1: [x1, x2])
u = ca.MX.sym('u', nu)  # Control variable (1x1)

# Define the dynamic model: xdot = [x2, u]
xdot = ca.vertcat(x[1], u)

# Define cost function: x1^2 + x2^2 + u^2
L = x[0]**2 + x[1]**2 + u**2

# Discretize the dynamics using Euler integration
def system_dynamics(x, u):
    return x + dt * ca.vertcat(x[1], u)

# Create decision variables for the optimization problem
X = ca.MX.sym('X', nx, N+1)   # States over time (Nx1 vector)
U = ca.MX.sym('U', nu, N)     # Controls over time (scalar)

# Initialize cost function and constraints
J = 0                         # Cost function
g = []                        # Constraint list

# Initial state constraint
x0 = np.array([5, 10])         # Define the initial state: x1=0, x2=0
g.append(X[:, 0] - x0)        # Enforce the initial condition

# Define the dynamic and control constraints
for k in range(N):
    # Dynamics constraint: X_k+1 = X_k + dt * [x2_k, u_k]
    x_next = system_dynamics(X[:, k], U[:, k])
    g.append(X[:, k+1] - x_next)
    
    # Update the cost function: x1_k^2 + x2_k^2 + u_k^2
    J += X[0, k]**2 + X[1, k]**2 + U[:, k]**2

# Add terminal state cost (only considering state at the final time step)
J += X[0, N]**2 + X[1, N]**2

# Define bounds on x and u
lbx = -1* np.ones((nx, N+1))   # Upper bound for x (no upper bound)
ubx = np.inf * np.ones((nx, N+1))   # Upper bound for x (no upper bound)

lbu = -15 * np.ones((nu, N))         # Lower bound for u (-2 <= u)
ubu =  15 * np.ones((nu, N))         # Upper bound for u (u <= 2)

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
