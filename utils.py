import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize


def generate_spotprice(sigma=10):
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
        plt.plot(t[0:len(ts)], ts)

    filename = filename + ".png"
    # plot_path = os.path.join(folder, filename)
    plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
    plt.savefig(plot_path)

    import numpy as np




def plant_model_derivative(x, u):
    
    #Extract state
    x_sdw = x[0]
    x_nsdw = x[1]
    PPFD = u[0]

    #Indoor climate assumptions
    T_crop = 20     #Indoor ambient temperature [C]
    co2_in = 1200   #CO2 consentration of indoor air [PPM]

    #Constants
    c_a = 0.68              #Conversion factor CO2 -> sugar
    c_b = 0.72              #Yield factor
    c_gr_max = 5e-6         #Saturation growth rate
    c_Q_10_gr = 1.6         #Q10 growth factor
    c_lar = 0.075           #Structural leaf area ratio
    c_k = 0.9               #Extinction coefficient
    c_T = 0.15              #Ratio of root dry weight to total crop dry weight
    c_Gamma = 71.5          #CO2 Compensation point at 20C 
    c_Q_10_Gamma = 2        #Q10 value affecting Gamma
    c_resp_sht = 3.47e-7    #Maintenance respiration coeff for the shoot
    c_resp_rt = 1.16e-7     #Maintenance respiration coeff for the  root
    c_e = 17e-6             #Light use effiiency at high CO2 concentrations
    rho_c = 1.893           #Density of co2
    c_car_1 = -1.32e-5      #-
    c_car_2 = 5.94e-4       # } Carboxylation resistance 2nd order approximation coefficients
    c_car_3 = -2.64e-3      #-
    l = 0.11                #Mean leaf diameter
    u_inf = 0.15            #Uninhibited air speed
    c_p = 0.217             #Conversion factor from PPFD to PAR
    eta_light = 0.8         #LED efficiency
    c_d = 0.05              #Dry matter content
    PCD = 25                #Plant crop density

    #Abstractions
    r_gr = x_nsdw / (x_nsdw + x_sdw) * c_gr_max * c_Q_10_gr**((T_crop-20)/10)           #Growth rate

    LAI = c_lar * (1-c_T)*x_sdw                                                         #Leaf area index
    CAC = 1-np.exp(-c_k * LAI)                                                          #Cultivation area cover fraction
    Gamma = c_Gamma * c_Q_10_Gamma**(T_crop - 20)/10                                    #Co2 compensation point 
    alpha = c_e * (co2_in - Gamma)/(co2_in + 2*Gamma)                                   #Quantum yield
    U_par = c_p * PPFD                                                                  #Photosynthetically active radiation
    r_car = 1/(c_car_1 * T_crop**2 + c_car_2 * T_crop + c_car_3)                        #Carboxylation resistance
    r_bnd = 350*np.sqrt(l/u_inf) / LAI                                                  #Boundary layer resistance 
    r_stm = 60*(1500 + PPFD)/(200 + PPFD)                                               #Stomatal resistance
    r_co2 = r_bnd + r_stm + r_car                                                       #Canopy resistance 
    f_sat = rho_c * (co2_in - Gamma)/r_co2                                              #Light saturated vlaue of max photosynthesis
    f_phot_max = alpha * U_par * f_sat / (alpha * U_par + f_sat)                        #Maximum photosynthetic rate
    f_phot = f_phot_max * CAC                                                           #Gross canopy photosynthesis
    f_resp = (c_resp_sht*(1-c_T) + c_resp_rt*c_T)*x_sdw * c_Q_10_gr**((T_crop-25)/10)   #Maintenance respiration rate
    x_dw_plant = (x_sdw + x_nsdw) / PCD                                                 #X dont worry plant <3
    x_fw_sht = x_dw_plant * (1-c_T)/c_d                                                 #Fresh weight per plant

    #Derivatives
    x_sdw_dot = r_gr * x_sdw
    # x_nsdw_dot = c_a * f_phot - x_sdw_dot - f_resp - (1-c_b)/c_b * r_gr * x_sdw       Slightly inefficient implementation
    x_nsdw_dot = c_a * f_phot - f_resp - 1/c_b * x_sdw_dot                              #More efficient implementation

    return np.array([x_sdw_dot, x_nsdw_dot])



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

