import numpy as np
import casadi as ca
from market import Market, Bid
from globals import *

class PlantModel:

    name = 'Lettuce shoot'

    Final_fw_sht:   float       # Final plant shoot fresh weight requirement    [g]
    x_init:         np.array    # Initial dry weights per m^2                   [g/m^2]
    x_sdw_init:     float       # Initial structural dry weight per m^2         [g/m^2]
    x_nsdw_init:    float       # Initial non-structural dry weight per m^2     [g/m^2]

    specs: dict


    nx = 3
    nu = 1
    
    def __init__(self, x_init, Final_fw_sht):
        self.Final_fw_sht = Final_fw_sht
        self.x_init = np.zeros(self.nx)
        self.x_init[:len(x_init)] = x_init
        self.x_sdw_init = x_init[0]
        self.x_nsdw_init = x_init[1]

        self.specs = {
            'type'                  : self.name,
            'x0'                    : x_init,
            'Fresh weight goal'     : Final_fw_sht,
            'Ideal DLI'             : self.IDEAL_DLI,
            'Max DLI'               : self.DLI_max,
            'Min DLI'               : self.DLI_min,
            'DLI resolution'        : self.DLI_res,
            'Total growht area'     : self.A_crop,
            'Ambient temp'          : self.T_crop,
            'CO2 concentration'     : self.co2_in,
            'Max PPFD'              : self.C_PPFD_max
        } 


    #Vertical farm specs
    T_crop = 24     #Indoor ambient temperature [C]
    co2_in = 1200   #CO2 consentration of indoor air [PPM]


    A_crop = 15000                                  # Total growth area [m^2]
    C_PPFD_max = 250                                # Max lighting capacity (or max tolerated light level for the plants) [mol / m^2/s]
    C_conv = 0.217                                  # W / PPFD
    eta_light = 0.8                                 # LED efficiency coefficient
    C_conv_PPFD = C_conv*A_crop/(eta_light*1000)    # Conversion factor between PPFD and power. Expressed in kW
    P_cap_max = C_PPFD_max*C_conv_PPFD              # Vertical farm power capacity [MW]


    PHOTOPERIOD = 16    # Hours of light in a day
    LIGHT_INTY  = 200   # Light intensity for the photoactive hours
    IDEAL_DLI   = PHOTOPERIOD * LIGHT_INTY * SECONDS_PER_HOUR * 1e-6       # Equates to 11.52

    DLI_max = 1.1 * IDEAL_DLI       # Calculated from ideal PPFD and ideal photoperiod
    DLI_min = 0.9 * IDEAL_DLI       # Only used for variable DLI schemes
    DLI_res = 2                     # DLI enforcement rate. 4 = enforce over last 24h every 6h (24/4)

    
    # state labels and units (for plotting)
    title  = "Vertical Farm"
    labels = ["Structural dry weight (g/m^2)", 
              "Non-structural dry weight (g/m^2)"]
    x_unit = "Weight (g/m^2)"
    u_unit = "PPFD (umol/m^2/s)"


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
    c_car_1 = -1.32e-5      #\
    c_car_2 = 5.94e-4       # } Carboxylation resistance 2nd order approximation coefficients
    c_car_3 = -2.64e-3      #/
    l = 0.11                #Mean leaf diameter
    u_inf = 0.15            #Uninhibited air speed
    c_p = 0.217             #Conversion factor from PPFD to PAR
    c_d = 0.05              #Dry matter content
    PCD = 25                #Plant crop density
    
   


    def derivative(self, x: ca.MX.sym, u: ca.MX.sym)->ca.MX.sym:
        
        #Extract state
        x_sdw   = x[0]      # structural dry weight
        x_nsdw  = x[1]      # non-structural dry weight
        # x_LI   = x[2]
        PPFD    = u[0]      # umol/m^2/s

        
        #Common constants
        T_crop = self.T_crop
        c_T = self.c_T


        epsilon = 1e-6      #Small constant to avoid division by zero

        #Abstractions
        r_gr = x_nsdw / (x_nsdw + x_sdw + epsilon) * self.c_gr_max * self.c_Q_10_gr**((T_crop-20)/10)      #Growth rate

        LAI = self.c_lar * (1-c_T)*x_sdw                                                         #Leaf area index
        CAC = 1-np.exp(-self.c_k * LAI)                                                          #Cultivation area cover fraction
        Gamma = self.c_Gamma * self.c_Q_10_Gamma**(T_crop - 20)/10                                    #Co2 compensation point 
        alpha = self.c_e * (self.co2_in - Gamma)/(self.co2_in + 2*Gamma)                                   #Quantum yield
        U_par = self.c_p * PPFD                                                                  #Photosynthetically active radiation
        r_car = 1/(self.c_car_1 * T_crop**2 + self.c_car_2 * T_crop + self.c_car_3)                        #Carboxylation resistance
        r_bnd = 350*np.sqrt(self.l/self.u_inf) / (LAI + epsilon)                                                  #Boundary layer resistance 
        r_stm = 60*(1500 + PPFD)/(200 + PPFD)                                               #Stomatal resistance
        r_co2 = r_bnd + r_stm + r_car                                                       #Canopy resistance 
        f_sat = self.rho_c * (self.co2_in - Gamma)/r_co2                                              #Light saturated vlaue of max photosynthesis
        f_phot_max = alpha * U_par * f_sat / (alpha * U_par + f_sat)                        #Maximum photosynthetic rate
        f_phot = f_phot_max * CAC                                                           #Gross canopy photosynthesis
        f_resp = (self.c_resp_sht*(1-c_T) + self.c_resp_rt*c_T)*x_sdw * self.c_Q_10_gr**((T_crop-25)/10)   #Maintenance respiration rate
        # x_dw_plant = (x_sdw + x_nsdw) / self.PCD                                                 #X dont worry plant <3
        # x_fw_sht = x_dw_plant * (1-c_T)/self.c_d                                                 #Fresh weight per plant

        #Derivatives
        x_sdw_dot = r_gr * x_sdw
        # x_nsdw_dot = c_a * f_phot - x_sdw_dot - f_resp - (1-c_b)/c_b * r_gr * x_sdw   # Slightly inefficient implementation
        x_nsdw_dot = self.c_a * f_phot - f_resp - 1/self.c_b * x_sdw_dot                # More efficient implementation
        x_LI_dot = PPFD*1e-6

        return ca.vertcat(x_sdw_dot, x_nsdw_dot, x_LI_dot)
    

    def freshweight(self, x):
        x_sdw = x[0]
        x_nsdw = x[1]

        x_dw = x_sdw + x_nsdw
        x_dw_plant = x_dw/self.PCD
        x_fw_sht = x_dw_plant*(1-self.c_T)/self.c_d

        return ca.vertcat(x_fw_sht)
    
    

    def bidding_objective_function(self, controller, X, U, B):
        
        N = controller.N
        p_spot = controller.p_spot

        Bp_up = B[0,:]
        Bp_dn = B[1,:]
        Bc_up = B[2,:]
        Bc_dn = B[3,:]

        L = 0

        # for k in range(0, N): #from k = 2, to N-1. 
        #     L += (1000*p_spot[k] - self.market.C_eur2nok * Bc_dn[k]) * Bp_dn[k] * self.market.Pr_a_dn(Bc_dn[k])\
        #           - (1000*p_spot[k] + self.market.C_eur2nok * Bc_up[k]) * Bp_up[k] * self.market.Pr_a_up(Bc_up[k])
        
        for k in range(0, N): #from k = 2, to N-1. 
            L += p_spot[k] * self.C_conv_PPFD * controller.u_base[k] \
                  + (1000*p_spot[k] - controller.market.C_eur2nok * Bc_dn[k]) * Bp_dn[k] * controller.market.Pr_a_dn(Bc_dn[k])\
                  - (1000*p_spot[k] + controller.market.C_eur2nok * Bc_up[k]) * Bp_up[k] * controller.market.Pr_a_up(Bc_up[k])

        L = L/4

        return L

    def baseline_obj_function(self, controller, X, U):
        N = controller.N
        p_spot = controller.p_spot

        L = 0
        for k in range(N):
            L += p_spot[k] * U[k] * self.C_conv_PPFD
                  
        L = L/4
                  
        return L

    def terminal_cost(self, controller, X, U, Eps):

        return Eps * 10**6

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
        # Final weight constraint
        g_ineq.append(self.freshweight(X[:,-1]) + Eps - self.Final_fw_sht)   

        # Define the dynamic and control constraints
        for k in range(0,N):
            # Model equalities
            x_next = X[:, k] + dt*self.derivative(X[:, k], U[k])
            g_eq.append(X[:, k+1] - x_next)


        ''' Inequality constraints: g_ineq[k] > 0 for all k '''

        # Upper and lower bounds on u
        for k in range(N):
            g_ineq.append(U[k])
            g_ineq.append(self.C_PPFD_max - U[k])


        # DLI constraint
        for k in range(N+1):
            if (k % (QUARTER_HOURS_PER_DAY/self.DLI_res) == 0 and k>=QUARTER_HOURS_PER_DAY): 
                # k = 96 +24, +48, +72 ...

                LI = (X[2,k] - X[2,k-QUARTER_HOURS_PER_DAY])
                # LI = ca.sum2(U[k-QUARTER_HOURS_PER_DAY:k])*1e-6*SECONDS_PER_QUARTER_HOUR # Convert from umol/m^2/s to mol/m^2/s
                
                g_ineq.append(self.DLI_max - LI)
                g_ineq.append(LI - self.DLI_min)



        return g_eq, g_ineq
    

    def get_u(self, controller, B):

        N = controller.N
        u_bar = controller.u_base
        U = np.array([])

        for k in range(N):

            if(k<controller.market.n_given_activations):
                u_tilde = 1000*(B[1,k]*controller.A_down[k] - B[0,k]*controller.A_up[k])/self.C_conv_PPFD
            else:
                u_tilde = 1000*(B[1,k]*controller.market.Pr_a_dn(B[3,k]) - B[0,k]*controller.market.Pr_a_up(B[2,k]))/self.C_conv_PPFD

            U = np.append(U, u_bar[k] + u_tilde)

        return ca.vertcat(*U)

    def get_bidding_bounds(self, controller):

        N = controller.N

        lb_B = 0 * np.ones((4, N))
        ub_B = np.vstack((self.C_conv_PPFD * controller.u_base/1000,                             # Bid vol up
                          self.C_conv_PPFD * (self.C_PPFD_max - controller.u_base)/1000,   # Bid vol down
                          1000 * np.ones((1, N)),                                # Bid price up. Arbitrary limit of 1000€ / MW 
                          1000 * np.ones((1, N))))                               # Bid price down. Arbitrary limit of 1000€ / MW 
        
        # ub_B = 0 * np.ones((4, N)) # TODO Uncomment to set all bids to 0

        return lb_B, ub_B
    
    def get_state_bounds(self, controller):

        N = controller.N

        # Define bounds on x and u
        lbx = 0* np.ones((self.nx, N+1))         # Lower bound for x (x >= 0)
        ubx = np.inf * np.ones((self.nx, N+1))   # Upper bound for x (no upper bound)


        return lbx, ubx
    
    def get_input_bounds(self, controller):
        
        N = controller.N

        lbu = np.zeros((self.nu, N))    # Lower bound for u (u >= 0)
        ubu = self.C_PPFD_max * np.ones((self.nu, N))                     # Upper bound for u (u <= Max PPFD 250)

        return lbu, ubu
    

    def get_metrics(self, controller, run_id, metrics_data, x, u, B):

        # DLI = [np.sum(u[int(k):int(k)+QUARTER_HOURS_PER_DAY])*1e-6*SECONDS_PER_QUARTER_HOUR for k in np.linspace(0, controller.N - QUARTER_HOURS_PER_DAY, controller.T*self.DLI_res)]

        DLI = []

        for k in range(controller.N+1):
            if (k % (QUARTER_HOURS_PER_DAY/self.DLI_res) == 0 and k>=QUARTER_HOURS_PER_DAY): 
                # k = 96 +24, +48, +72 ...
                DLI.append(x[2,k] - x[2,k-QUARTER_HOURS_PER_DAY])


        metrics_data['DLI_avg'] = np.average(DLI)
        metrics_data['DLI_max'] = np.max(DLI)
        metrics_data['DLI_min'] = np.min(DLI)
        metrics_data['Final fresh weight'] = float(self.freshweight(x[:,-1]))

        return metrics_data
    


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
    x_unit = "Charge level"
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
    

    def bidding_objective_function(self, controller, X, U, B):
        
        N = controller.N
        p_spot = controller.p_spot

        Bp_up = B[0,:]
        Bp_dn = B[1,:]
        Bc_up = B[2,:]
        Bc_dn = B[3,:]

        L = 0
        
        for k in range(0, N): #from k = 2, to N-1. 
            L += p_spot[k] * controller.u_base[k] \
                  + (1000*p_spot[k] - controller.market.C_eur2nok * Bc_dn[k]) * Bp_dn[k] * controller.market.Pr_a_dn(Bc_dn[k])\
                  - (1000*p_spot[k] + controller.market.C_eur2nok * Bc_up[k]) * Bp_up[k] * controller.market.Pr_a_up(Bc_up[k])

        L = L/4

        return L

    def baseline_obj_function(self, controller, X, U):
        N = controller.N
        p_spot = controller.p_spot

        L = 0
        for k in range(N):
            L += p_spot[k] * U[k] / 4
                  
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
    


    def get_u(self, controller, B):

        N = controller.N
        u_bar = controller.u_base
        U = np.array([])

        for k in range(N):

            if(k<controller.market.n_given_activations):
                u_tilde = 1000*(B[1,k]*controller.A_down[k] - B[0,k]*controller.A_up[k])
            else:
                u_tilde = 1000*(B[1,k]*controller.market.Pr_a_dn(B[3,k]) - B[0,k]*controller.market.Pr_a_up(B[2,k]))

            U = np.append(U, u_bar[k] + u_tilde)

        return ca.vertcat(*U)



    def get_bidding_bounds(self, controller):

        N = controller.N

        lb_B = 0 * np.ones((4, N))
        ub_B = np.vstack(((controller.u_base - self.u_min)/1000,   # Bid vol up       abs(Dist from ubase to umin)
                          (self.u_max - controller.u_base)/1000,   # Bid vol down     abs(Dist from ubase to umax)
                          1000 * np.ones((1, N)),      # Bid price up. Arbitrary limit of 1000€ / MW 
                          1000 * np.ones((1, N))))     # Bid price down. Arbitrary limit of 1000€ / MW 
        
        # ub_B = 0 * np.ones((4, N)) # TODO Uncomment to set all bids to 0

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
    





