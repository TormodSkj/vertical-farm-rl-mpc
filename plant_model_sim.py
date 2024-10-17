import os
import numpy as np
import matplotlib.pyplot as plt

def x_dot(x, u):
    
    #Extract state
    x_sdw = x[0]
    x_nsdw = x[1]
    PPFD = u[0]

    #Indoor climate assumptions
    T_in = 24       #Indoor ambient temperature [C]
    #T_crop = 20     #Crop temperature [C]
    T_crop = T_in     #As defined in the gandj code
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
    Gamma = c_Gamma * c_Q_10_Gamma**((T_crop - 20)/10)                                  #Co2 compensation point 
    alpha = c_e * (co2_in - Gamma)/(co2_in + 2*Gamma)                                   #Quantum yield
    U_par = c_p * PPFD                                                                  #Photosynthetically active radiation
    r_car = 1/(c_car_1 * T_crop**2 + c_car_2 * T_crop + c_car_3)                       #Carboxylation resistance from thesis
    # r_car = 1/( 1/(c_car_1 * T_crop**2) + 1/(c_car_2 * T_crop) + 1/c_car_3)             #Carboxylation resistance from code
    r_bnd = 350*np.sqrt(l/u_inf) / LAI                                                  #Boundary layer resistance 
    r_stm = 60*(1500 + PPFD)/(200 + PPFD)                                               #Stomatal resistance
    r_co2 = r_bnd + r_stm + r_car                                                       #Canopy resistance 
    f_sat = rho_c * (co2_in - Gamma)/r_co2                                              #Light saturated vlaue of max photosynthesis
    f_phot_max = alpha * U_par * f_sat / (alpha * U_par + f_sat)                        #Maximum photosynthetic rate from thesis
    # f_phot_max = (alpha * U_par * r_co2 * (co2_in - Gamma)) / (alpha * U_par + r_co2 * (co2_in - Gamma))      #Maximum photosynthetic rate from code
    f_phot = f_phot_max * CAC                                                           #Gross canopy photosynthesis
    f_resp = (c_resp_sht*(1-c_T) + c_resp_rt*c_T)*x_sdw * c_Q_10_gr**((T_crop-25)/10)   #Maintenance respiration rate
    x_dw_plant = (x_sdw + x_nsdw) / PCD                                                 #X dont worry plant <3
    x_fw_sht = x_dw_plant * (1-c_T)/c_d                                                 #Fresh weight per plant

    # f_resp = 0

    #Derivatives
    x_sdw_dot = r_gr * x_sdw
    # x_nsdw_dot = c_a * f_phot - x_sdw_dot - f_resp - (1-c_b)/c_b * r_gr * x_sdw       Slightly inefficient implementation
    x_nsdw_dot = c_a * f_phot - f_resp - 1/c_b * x_sdw_dot                              #More efficient implementation
    # x_nsdw_dot = c_a * f_phot - f_resp - c_b * x_sdw_dot                              #More efficient implementation

    return np.array([x_sdw_dot, x_nsdw_dot]), np.array([f_phot, x_fw_sht])


def simulate(x_0: np.ndarray, TH: int, dt: float):

    nx = len(x_0)
    nu = 1

    #init
    N = int(np.ceil(TH/dt))
    z = np.zeros((2, N))
    x = np.zeros((nx, N))
    u = 250 * np.ones((nu, N))      #TODO change
    x[:,0] = x_0

    start_time = 18 * 60 * 60  # 18:00 in seconds (64,800)
    end_time = 24 * 60 * 60    # 00:00 in seconds (86,400)
    day_seconds = 24 * 60 * 60  # 86,400 seconds

    for i in range(N):
        current_time = (i * dt) % day_seconds
        if start_time <= current_time < end_time:
            u[:, i] = 0  # Set u to 0 between 18:00 and 00:00

    #Forward euler. keep it simple
    for k in range(N-1):
        x_dot_k, other_data = x_dot(x[:,k],u[:,k])
        x[:,k+1] = x[:,k] + dt*x_dot_k
        z[:,k+1] = other_data
        
    
    t = np.linspace(0,TH,N)

    return x, u, t, z


def plotting(x, u, t, z):

    plot_folder = 'plots'
    os.makedirs(plot_folder, exist_ok=True)

    plt.figure(1)
    plt.plot(t, x[0,:])
    plt.plot(t, x[1,:])
    plt.plot(t, z[1,:])
    plt.legend(("Structural dry weight / m^2", "Non-structural dry weight / m^2", "Fresh weight per plant"))
    # plot_path = os.path.join(plot_folder, "Plantmodel_plot_plantweight.png")
    plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/Plantmodel_plot_plantweight.png"
    plt.savefig(plot_path)

    plt.figure(2)
    plt.plot(t, u[0,:])
    plt.legend(("Light level"))
    # plot_path = os.path.join(plot_folder, "Plantmodel_plot_lightlevel.png")
    plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/Plantmodel_plot_lightlevel.png"
    plt.savefig(plot_path)

    plt.figure(3)
    plt.plot(t, z[0,:])
    plt.legend(("F_phot"))
    plot_path = os.path.join(plot_folder, "Plantmodel_plot_f_phot.png")
    plt.savefig(plot_path)


x, u, t, z = simulate(np.array([5, 1]), 20*24*60*60, 60)
plotting(x, u, t, z)


