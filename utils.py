import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize


def generate_spotprice():
    """
    Generates a random array of 96 elements where every four consecutive elements
    are identical. The values are drawn from a normal distribution with a specified
    mean (mu) and standard deviation (sigma).
    
    Parameters:
    mu (float): The mean of the normal distribution. Default is 50.
    sigma (float): The standard deviation of the normal distribution. Default is 10.
    
    Returns:
    np.ndarray: A 1D array of 96 elements.
    """

    sigma=0.10
    timesteps = np.linspace(0, 23, 24)
    timesteps = np.repeat(timesteps, 4)
    prices_mean = 0.7 - 0.2*np.cos(np.pi * timesteps/6) - 0.15*np.cos(np.pi * timesteps/12) 


    # Generate 24 random values from a normal distribution
    spot_prices = prices_mean + np.repeat(np.random.normal(loc=0, scale=sigma, size=24), 4)

    # Ensure that no values are below 0
    spot_prices = np.clip(spot_prices, a_min=0, a_max=None)
    
    # Repeat each value 4 times to get a total of 96 elements
    
    return spot_prices





def plotting(t, timeseries, filename, folder = 'plots'):
    #Function to plot timeseries to a given file.

    if(len(timeseries)<1):
        print("Specify one or more timeseries for plotting")
        return

    plt.figure(1)
    for ts in timeseries:
        plt.plot(t[0:len(ts)], ts, "r")

    filename = filename + ".png"
    # plot_path = os.path.join(folder, filename)
    plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
    plt.savefig(plot_path)

    import numpy as np

def plot_trajectory(x1, x2, filename, folder = 'plots'):
    #Function to plot timeseries to a given file.

    
    plt.figure(2)
    plt.plot(x1, x2, "r")

    filename = filename + ".png"
    # plot_path = os.path.join(folder, filename)
    plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
    plt.savefig(plot_path)

    import numpy as np


'''
# Objective function to minimize (based on spot prices and control input u)
def objective(u, x0, spot_prices, dt):
    x = x0
    total_cost = 0
    for t in range(96):
        # Update state using the plant derivative and Euler integration
        dx_dt = plant_model_derivative(x, u[t])
        x = x + dx_dt * dt  # Euler integration for state update
        
        # Cost function: u_t * spot_price_t
        total_cost += u[t] * spot_prices[t]
    
    return total_cost

# Main optimization loop over the time horizon
def optimize_control(x0, spot_prices):
    dt = 1  # Time step size
    u_init = np.zeros(96)  # Initial guess for control input (zeros)
    
    # Bounds for control input u (optional)
    bounds = [(0, 5) for _ in range(96)]  # Example: bounds on u between -5 and 5

    # Perform the optimization using minimize
    result = minimize(objective, u_init, args=(x0, spot_prices, dt),
                      method='SLSQP', bounds=bounds)
    
    # Optimized control inputs
    u_optimal = result.x
    return u_optimal

'''