import numpy as np
import casadi as ca
from market import Market
from globals import *
from settings import Settings
from balancingmarket import BalancingMarket
from utils import *

class PlantModel:

    name = 'Lettuce shoot'

    settings: Settings

    Final_fw_sht:   float       # Final plant shoot fresh weight requirement    [g]
    x_init:         np.array    # Initial dry weights per m^2                   [g/m^2]

    specs: dict


    nx = 3          # Number of state variables
    nu = 1          # Number of input variables
    neps = 3        # Number of slack variables
    
    def __init__(self, settings: Settings):

        self.settings = settings
        self.model_settings = settings.get_settings_group('plantmodel')

        # Sim specs
        self.Final_fw_sht   = self.model_settings['TARGET_FRESHWEIGHT'] # Target weight per plant
        self.x_init         = self.model_settings['INIT_STATE']         # Initial dry weights per m^2
        
        self.discretization     = self.model_settings['DISCRETIZATION']

        
        # Growth constraints
        self.PHOTOPERIOD    = self.model_settings['PHOTOPERIOD']        # Hours of light in a day
        self.LIGHT_INTY     = self.model_settings['LIGHT_INTENSITY']    # Light intensity for the photoactive hours
        self.IDEAL_DLI      = self.model_settings['TARGET_DLI']         # Target daily light integral. PHOTOPERIOD * LIGHT_INTY * SECONDS_PER_HOUR * 1e-6      
        self.DLI_ERROR      = self.model_settings['DLI_DEVIATION']      # Max deviation from daily light integral
        self.DLI_res        = self.model_settings['DLI_RESOLUTION']                                                # DLI enforcement rate. 4 = enforce over last 24h every 6h
        
        self.DLI_max = (1 + self.DLI_ERROR) * self.IDEAL_DLI            # Calculated from ideal PPFD and ideal photoperiod
        self.DLI_min = (1 - self.DLI_ERROR) * self.IDEAL_DLI            # Only used for variable DLI schemes
        
        # Model specs
        self.T_crop         = self.model_settings['AMBIENT_TEMP']       # Indoor ambient temperature [C]
        self.co2_in         = self.model_settings['AMBIENT_CO2']        # CO2 consentration of indoor air [PPM]
        self.A_crop         = self.model_settings['GROWTH_AREA']        # Total growth area [m^2]
        self.PPFD_max       = self.model_settings['PPFD_MAX']           # Max lighting capacity (or max tolerated light level for the plants) [mol / m^2/s]
        self.eta_light      = self.model_settings['LED_EFFICIENCY']     # LED efficiency coefficient

        self.C_conv      = 0.217                                            # (W/m^2) / PPFD
        self.k_P         = self.C_conv*self.A_crop/(self.eta_light * 1e6)   # Conversion factor between light intensity [PPFD] and power [MW]
        self.P_cap_max   = self.PPFD_max*self.k_P                   # Vertical farm power capacity [MW]



        self.specs = {
            'type'                  : self.name,
            'x0'                    : self.x_init,
            'Fresh weight goal'     : self.Final_fw_sht,
            'Ideal DLI'             : self.IDEAL_DLI,
            'Max DLI'               : self.DLI_max,
            'Min DLI'               : self.DLI_min,
            'DLI resolution'        : self.DLI_res,
            'Total growht area'     : self.A_crop,
            'Ambient temp'          : self.T_crop,
            'CO2 concentration'     : self.co2_in,
            'Max PPFD'              : self.PPFD_max
        } 


    
    # state labels and units (for plotting)
    title  = "Vertical Farm"
    labels = ["Structural dry weight (g/m^2)", 
              "Non-structural dry weight (g/m^2)"]
    x_unit = "Weight (g/m^2)"
    u_unit = "PPFD"


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
    rho_c = 1.893e-3        #Density of co2
    c_car_1 = -1.32e-5      #\
    c_car_2 = 5.94e-4       # } Carboxylation resistance 2nd order approximation coefficients
    c_car_3 = -2.64e-3      #/
    l = 0.11                #Mean leaf diameter
    u_inf = 0.15            #Uninhibited air speed
    c_p = 0.217             #Conversion factor from PPFD to PAR
    c_d = 0.05              #Dry matter content
    PCD = 25                #Plant crop density
    c_fl = 7.43e-4          # Growth loss due to fluctuations in light
    curve_nr = 0.9


    def derivative(self, x: ca.MX.sym, u: ca.MX.sym, u_last: ca.MX.sym = [None])->ca.MX.sym:
        
        #Extract state
        x_sdw   = x[0]      # structural dry weight
        x_nsdw  = x[1]      # non-structural dry weight
        # x_LI   = x[2]
        PPFD    = u[0]      # umol/m^2/s
        PPFD_last = u_last[0]
        
        #Common constants
        T_crop = self.T_crop
        c_T = self.c_T


        epsilon = 1e-6      #Small constant to avoid division by zero

        #Abstractions
        r_gr = x_nsdw / (x_nsdw + x_sdw + epsilon) * self.c_gr_max * self.c_Q_10_gr**((T_crop-20)/10)      #Growth rate

        LAI = self.c_lar * (1-c_T)*x_sdw                                                    # Leaf area index
        CAC = 1-np.exp(-self.c_k * LAI)                                                     # Cultivation area cover fraction
        Gamma = self.c_Gamma * self.c_Q_10_Gamma**(T_crop - 20)/10                          # Co2 compensation point 
        alpha = self.c_e * (self.co2_in - Gamma)/(self.co2_in + 2*Gamma)                    # Quantum yield
        U_par = self.c_p * PPFD                                                             # Photosynthetically active radiation
        r_car = 1/(self.c_car_1 * T_crop**2 + self.c_car_2 * T_crop + self.c_car_3)         # Carboxylation resistance
        r_bnd = 350*np.sqrt(self.l/self.u_inf) / (LAI + epsilon)                            # Boundary layer resistance 
        r_stm = 60*(1500 + PPFD)/(200 + PPFD)                                               # Stomatal resistance
        r_co2 = r_bnd + r_stm + r_car                                                       # Canopy resistance 
        f_sat = self.rho_c * (self.co2_in - Gamma)/r_co2                                    # Light saturated vlaue of max photosynthesis
        f_phot_max = alpha * U_par * f_sat / (alpha * U_par + f_sat)                        # Maximum photosynthetic rate
        
        # Alternative calculations
        # f_phot_max = f_phot_max * ca.exp(-ca.power(self.c_fl*(PPFD - PPFD_last), 2))
        # f_phot_max_nr = (alpha*PPFD + f_sat - ca.sqrt(epsilon + ca.power(alpha*PPFD + f_sat, 2) - 4*self.curve_nr*alpha*PPFD*f_sat))/(2*self.curve_nr)
        
        f_phot = f_phot_max * CAC                                                           # Gross canopy photosynthesis
        f_resp = (self.c_resp_sht*(1-c_T) + self.c_resp_rt*c_T)*x_sdw * self.c_Q_10_gr**((T_crop-25)/10)   # Maintenance respiration rate
        

        # State derivatives
        x_sdw_dot = r_gr * x_sdw
        # x_nsdw_dot = c_a * f_phot - x_sdw_dot - f_resp - (1-c_b)/c_b * r_gr * x_sdw   # Slightly inefficient implementation
        x_nsdw_dot = self.c_a * f_phot - f_resp - 1/self.c_b * x_sdw_dot                # More efficient implementation
        x_LI_dot = PPFD*1e-6

        return ca.vertcat(x_sdw_dot, x_nsdw_dot, x_LI_dot)
    

    def casadi_function(self, dt=SECONDS_PER_QUARTER_HOUR, discretization: str = None):
        '''Repackages the system equations as a casadi function'''

        if discretization is None: discretization = self.discretization

        states = ca.MX.sym('X', self.nx)
        controls = ca.MX.sym('U', self.nu)
        state_time_derivatives = self.derivative(states, controls)
        f = ca.Function('f', [states, controls], [state_time_derivatives], ['x', 'u'], ['ode'])
        

        if  str(discretization).lower() == 'fe':
            x_next = states + dt*f(states, controls)
        elif str(discretization).lower() == 'rk':
            intg_options = {}
            ode = {
                'x': states,
                'p': controls,
                'ode': f(states,controls)
            }
            intg = ca.integrator('intg', 'rk', ode, 0, dt, intg_options)
            res = intg(x0=states, p=controls)
            x_next = res['xf']
        else:
            assert False, "Invalid discretization method"

        F = ca.Function('F', [states, controls], [x_next], ['x', 'u_control'], ['x_next'])
        return F


    def freshweight(self, x):

        '''Calculate freshweight based on plant dry weight'''

        x = x.reshape((self.nx, -1))

        x_sdw  = x[0,:]
        x_nsdw = x[1,:]

        x_dw = x_sdw + x_nsdw
        x_dw_plant = x_dw/self.PCD
        x_fw_sht = x_dw_plant*(1-self.c_T)/self.c_d

        return ca.vertcat(x_fw_sht)
    
    

    def Bidding_obj_function(self, N_TH, spot_prices, B_volumes, B_prices, balancing_market: BalancingMarket, MTU_start = None):
        
        bid_volumes_up      = B_volumes[0,:]
        bid_volumes_down    = B_volumes[1,:]
        bid_prices_up       = B_prices[0,:]
        bid_prices_down     = B_prices[1,:]

        # expected_prices_up, expected_prices_down = balancing_market.expected_clearing_prices_up.flatten(), balancing_market.expected_clearing_prices_down.flatten()
        expected_prices_up, expected_prices_down = balancing_market.get_estimated_clearing_prices(start_date = MTU_start, n_data = N_TH)
        expected_price_variance_up, expected_price_variance_down = balancing_market.get_estimated_clearing_price_variances()
        # expected_prices_up, expected_prices_down = balancing_market.get_expected_clearing_prices(spot_prices)

        L = 0

        for k in range(0, N_TH):
            L += \
                  - expected_prices_down[:,k]   * bid_volumes_down[k]   * balancing_market.activation_prob_down(bid_prices_down[k], spot_prices[k], expected_prices_down[:,k], expected_price_variance_down)\
                  - expected_prices_up[:,k]     * bid_volumes_up[k]     * balancing_market.activation_prob_up(bid_prices_up[k], spot_prices[k], expected_prices_up[:,k], expected_price_variance_up)

        L = L/4

        return L
    

    def Bidding_obj_function_seb(self, N_TH, spot_prices, B_volumes, B_prices, balancing_market: BalancingMarket, MTU_start = None):
        
        bid_volumes_up      = B_volumes[0,:]
        bid_volumes_down    = B_volumes[1,:]
        bid_prices_up       = B_prices[0,:]
        bid_prices_down     = B_prices[1,:]

        # expected_prices_up, expected_prices_down = balancing_market.expected_clearing_prices_up.flatten(), balancing_market.expected_clearing_prices_down.flatten()
        expected_prices_up, expected_prices_down = balancing_market.get_estimated_clearing_prices(start_date = MTU_start, n_data = N_TH)
        expected_price_variance_up, expected_price_variance_down = balancing_market.get_estimated_clearing_price_variances()
        # expected_prices_up, expected_prices_down = balancing_market.get_expected_clearing_prices(spot_prices)

        expected_price_sd_up, expected_price_sd_down = ca.sqrt(expected_price_variance_up), ca.sqrt(expected_price_variance_down)

        L = 0

        for k in range(0, N_TH):
            L += \
                  - bid_volumes_down[k] * balancing_market.demand_prob_down(spot_prices[k]) * (expected_prices_down[:,k]*(1-gaussian_CDF(bid_prices_down[k], expected_prices_down[:,k], expected_price_sd_down)) + expected_price_sd_down*gaussian_PDF(bid_prices_down[k], expected_prices_down[:,k], expected_price_sd_down)) \
                  - bid_volumes_up[k] * balancing_market.demand_prob_up(spot_prices[k]) * (expected_prices_up[:,k]*(1-gaussian_CDF(bid_prices_up[k], expected_prices_up[:,k], expected_price_sd_up)) + expected_price_sd_up*gaussian_PDF(bid_prices_up[k], expected_prices_up[:,k], expected_price_sd_up))


                #   - expected_prices_down[:,k]   * bid_volumes_down[k]   * balancing_market.activation_prob_down(bid_prices_down[k], spot_prices[k], expected_prices_down[:,k], expected_price_variance_down)\
                #   - expected_prices_up[:,k]     * bid_volumes_up[k]     * balancing_market.activation_prob_up(bid_prices_up[k], spot_prices[k], expected_prices_up[:,k], expected_price_variance_up)

        L = L/4

        return L

    # def AM_bidding_obj_function(self, N_TH, spot_prices, B_volumes, B_prices, balancing_market: BalancingMarket, MTU_start = None):

    #     # Altered cost function by advice from Jay during the seminar.

    #     bid_volumes_up      = B_volumes[0,:]
    #     bid_volumes_down    = B_volumes[1,:]
    #     bid_prices_up       = B_prices[0,:]
    #     bid_prices_down     = B_prices[1,:]

    #     expected_prices_up, expected_prices_down = balancing_market.get_estimated_clearing_prices(start_date = MTU_start, n_data = N_TH)
    #     expected_price_variance_up, expected_price_variance_down = balancing_market.get_estimated_clearing_price_variances()

    #     L = 0

    #     for k in range(0, N_TH):
    #         L += \
    #               - (spot_prices[k] - expected_prices_down[:,k])   * bid_volumes_down[k]   * balancing_market.activation_prob_down(bid_prices_down[k], spot_prices[k], expected_prices_down[:,k], expected_price_variance_down)\
    #               - (expected_prices_up[:,k] - spot_prices[k])     * bid_volumes_up[k]     * balancing_market.activation_prob_up(bid_prices_up[k], spot_prices[k], expected_prices_up[:,k], expected_price_variance_up)

    #     L = L/4

    #     return L
    
    
    # def CM_bidding_obj_function(self, N_TH, spot_prices, B_volumes, B_prices, balancing_market: BalancingMarket, MTU_start = None):
        
    #     bid_volumes_up      = B_volumes[0,:]
    #     bid_volumes_down    = B_volumes[1,:]
    #     bid_prices_up       = B_prices[0,:]
    #     bid_prices_down     = B_prices[1,:]

    #     # expected_prices_up, expected_prices_down = balancing_market.expected_clearing_prices_up.flatten(), balancing_market.expected_clearing_prices_down.flatten()
        
    #     # expected_prices_up, expected_prices_down = balancing_market.get_estimated_clearing_prices(spot_prices)
    #     expected_prices_up, expected_prices_down = balancing_market.get_estimated_clearing_prices(start_date = MTU_start, n_data = N_TH)
    #     expected_price_variance_up, expected_price_variance_down = balancing_market.get_estimated_clearing_price_variances()
        
    #     L = 0

    #     for k in range(0, N_TH):
    #         L += \
    #               - expected_prices_down[:,k]   * bid_volumes_down[k]   * balancing_market.activation_prob_down(bid_prices_down[k], spot_prices[k], expected_prices_down[:,k], expected_price_variance_down)\
    #               - expected_prices_up[:,k]     * bid_volumes_up[k]     * balancing_market.activation_prob_up(bid_prices_up[k], spot_prices[k], expected_prices_up[:,k], expected_price_variance_up)

    #     L = L/4

    #     return L

    def elcost_obj_function(self, N, spot_prices, U):

        L = 0
        for k in range(N):
            L += spot_prices[k] * self.k_P * U[:,k] 
                  
        L = L/4
                  
        return L
    
    def fluctuating_light_cost(self, N, U):

        L = 0
        for k in range(1, N):
            L += self.c_fl * ca.power(U[k] - U[k-1], 2)

        return L


    def terminal_cost(self, controller, X, U, Eps):

        slack_freshweight   = Eps[0,0]
        slack_max_DLI       = Eps[1,0]
        slack_min_DLI       = Eps[2,0]

        return slack_freshweight * 10**6 + (slack_max_DLI + slack_min_DLI) * 10**5



    # YET TO BE PROPERLY IMPLEMENTED
    # def running_cost(self, market: Market, k: int, N: int, Eps):
    #     '''
    #     First draft of running cost for MPC optimizer. 
    #     Attempts to make mpc optimizer aware of market states outside its opt-window
    #     '''


    #     spot_prices = market.get_spotprice()
    #     spot_prices_integral = np.array([sum(spot_prices[:k]) for k in range(len(spot_prices))])
    #     spot_prices_avg_curve = np.linspace(0, spot_prices_integral[-1], N)
        
    #     clearing_prices_up, clearing_prices_down = market.AM.get_clearing_prices()
    #     activations_up, activations_down = market.AM.get_activations()
    #     market_potencies_up         = np.multiply(clearing_prices_up, activations_up)
    #     market_potencies_down       = np.multiply(clearing_prices_down, activations_down)
    #     market_potencies_net      = market_potencies_down - market_potencies_up
    #     market_potencies_integral   = np.array([sum(market_potencies_net[:k]) for k in range(len(market_potencies_net))])
    #     market_potencies_net_avg_curve = np.linspace(0, market_potencies_integral[-1], N)

    #     spot_status = spot_prices_integral[k] - spot_prices_avg_curve[k]
    #     market_potency_status = market_potencies_integral[k] - market_potencies_net_avg_curve[k]

    #     Q_weight = 1
    #     Q_spot = 1
    #     Q_potency = 1
    #     Q_factor = 1e4

    #     # Punish lower freshweight than the reference trajectory
    #     # Alleviate weight punishment if spot prices have been above average
    #     #   - Also applies an additional penalty to weight discrepancy if spot prices have been lower than average
    #     # Alleviate weight punishment if mfrr market potency indicates high frequency of down-activations
    #     #   - Also applies an additional penalty to weight discrepancy the market potency indicates that only up-activations are due

    #     running_cost = Q_factor * (Q_weight     * Eps\
    #                                 - Q_spot    * spot_status \
    #                                 + Q_potency * market_potency_status)

    #     return running_cost

    def get_initial_bid_constraints(self, controller, g_eq, g_ineq, B):
        # Enforce initial bids

        for k, bid in enumerate(controller.bids):
            g_eq.append(B[:, k] - bid.as_array())
        
        return g_eq, g_ineq
    
    
    def get_bidding_constraints(self, g_eq, g_ineq, N, U_nom, balancing_market: BalancingMarket, B_volumes = None, B_prices=None, B_volumes_lower_bound = None, B_prices_upper_bound = None):

        lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices = self.get_bidding_bounds(N, U_nom, balancing_market)
        if B_volumes_lower_bound is not None :   lb_B_volumes    = B_volumes_lower_bound
        if B_prices_upper_bound  is not None:    ub_B_prices     = B_prices_upper_bound

        if B_volumes is not None:
            for k in range(N):
                for bid_param in range(2):
                    
                    g_ineq.append(B_volumes[bid_param,k] - lb_B_volumes[bid_param,k])
                    g_ineq.append(- B_volumes[bid_param,k] + ub_B_volumes[bid_param,k])

        if B_prices is not None:
            for k in range(N):
                for bid_param in range(2):

                    g_ineq.append(B_prices[bid_param,k] - lb_B_prices[bid_param,k])
                    g_ineq.append(- B_prices[bid_param,k] + ub_B_prices[bid_param,k])

        return g_eq, g_ineq


    def apply_bidding_constraints(self, opti: ca.Opti, N, U_nom, balancing_market: BalancingMarket, B_volumes = None, B_prices=None, B_volumes_lower_bound = None, B_prices_upper_bound = None):

        lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices    = self.get_bidding_bounds(N, U_nom, balancing_market)
        if B_volumes_lower_bound is not None :   lb_B_volumes   = B_volumes_lower_bound
        if B_prices_upper_bound  is not None:    ub_B_prices    = B_prices_upper_bound

        if B_volumes is not None:
            for k in range(N):
                for bid_param in range(2):
                    
                    opti.subject_to(opti.bounded(0, B_volumes[bid_param,k] - lb_B_volumes[bid_param,k],   ca.inf))
                    opti.subject_to(opti.bounded(0, - B_volumes[bid_param,k] + ub_B_volumes[bid_param,k], ca.inf))

        if B_prices is not None:
            for k in range(N):
                for bid_param in range(2):

                    opti.subject_to(opti.bounded(0, B_prices[bid_param,k] - lb_B_prices[bid_param,k],   ca.inf))
                    opti.subject_to(opti.bounded(0, - B_prices[bid_param,k] + ub_B_prices[bid_param,k], ca.inf))

        return
    

    def get_process_constraints(self, controller, g_eq, g_ineq, X, U, Eps):
        '''Get plant model constraints'''

        N = controller.N
        dt = controller.dt
        F = self.casadi_function(dt=dt)
        slack_freshweight = Eps[0]
        slack_max_DLI = Eps[1]
        slack_min_DLI = Eps[2]
        U = U.reshape((1,-1))

        # Initial state constraint
        g_eq.append(X[:, 0] - self.x_init)
        # Final weight constraint
        g_ineq.append(self.freshweight(X[:,-1]) + slack_freshweight - self.Final_fw_sht)   

        # Define the dynamic and control constraints
        for k in range(0,N):
            # Model equalities
            x_next = F(X[:, k], U[:,k])
            g_eq.append(X[:, k+1] - x_next)

        ''' Inequality constraints: g_ineq[k] > 0 for all k '''

        # Upper and lower bounds on u
        for k in range(N):
            g_ineq.append(U[:,k])
            g_ineq.append(self.PPFD_max - U[:,k])

        # DLI constraint
        for k in range(N+1):
            if (k % (QUARTER_HOURS_PER_DAY/self.DLI_res) == 0 and k>=QUARTER_HOURS_PER_DAY): 
                # k = 96 +24, +48, +72 ...
                LI = (X[2,k] - X[2,k-QUARTER_HOURS_PER_DAY])
                
                g_ineq.append(slack_max_DLI + self.DLI_max - LI)
                g_ineq.append(slack_min_DLI + LI - self.DLI_min)

        return g_eq, g_ineq
    
    def apply_static_process_constraints(self, opti: ca.Opti, N, dt, X, x0, U, Eps):
        '''Creates list of constraints for the mpc optimization problem'''

        U = U.reshape((1,-1))

        F = self.casadi_function(dt=dt)

        # Define the dynamic and control constraints
        for k in range(0,N):
            x_next = F(X[:, k], U[:,k])    
            opti.subject_to(X[:, k+1] - x_next == 0)

        # Initial state constraint
        opti.subject_to(X[:,0] - x0 == 0)

        ''' Inequality constraints: g_ineq[k] > 0 for all k '''

        return
    
    def apply_static_variable_bounds(self, opti: ca.Opti, N, dt, X, x0, U, Eps):

        opti.subject_to(opti.bounded(0, X,   ca.inf))
        opti.subject_to(opti.bounded(0, Eps, ca.inf))
        opti.subject_to(opti.bounded(0, U,   self.PPFD_max))

        return



    def opti_dynamic_process_constraints(self, opti: ca.Opti, N, X, Eps, ref_weight, past_X = None):
        '''Creates list of constraints for the mpc optimization problem'''

        slack_freshweight = Eps[0]
        slack_max_DLI = Eps[1]
        slack_min_DLI = Eps[2]

        # g_ineq.append([self.freshweight(X[:,N]) + slack_freshweight - ref_weight, f"Final freshweight constraint"])
        opti.subject_to(0 <= self.freshweight(X[:,N]) + slack_freshweight - ref_weight)

        # DLI constraint
        if past_X is None:
            for k in range(N+1):
                if (k % (QUARTER_HOURS_PER_DAY/self.DLI_res) == 0 and k>=QUARTER_HOURS_PER_DAY): 
                    # k = 96 +24, +48, +72 ...
                    LI = (X[2,k] - X[2,k-QUARTER_HOURS_PER_DAY])
                    
                    opti.subject_to(LI - slack_max_DLI <= self.DLI_max)
                    opti.subject_to(self.DLI_min <= LI + slack_min_DLI)
        else:

            prev_LI = past_X[2,:].reshape((1,-1))

            combined_LI = ca.horzcat(prev_LI, X[2,:].reshape((1,-1)))
            for k in range(past_X.shape[1] + N+1):
                if (k % (QUARTER_HOURS_PER_DAY/self.DLI_res) == 0 and k>=QUARTER_HOURS_PER_DAY): 
                    # k = 96 +24, +48, +72 ...
                    LI = (combined_LI[:,k] - combined_LI[:,k-QUARTER_HOURS_PER_DAY])
                    
                    opti.subject_to(LI - slack_max_DLI <= self.DLI_max)
                    opti.subject_to(self.DLI_min <= LI + slack_min_DLI)

        return
    

    def get_hourly_bids_contraint(self, opti: ca.Opti, N_TH, B_volumes, B_prices):

        for i in range(0,N_TH, QUARTER_HOURS_PER_HOUR):
            # 0, 4, 8, ... N_TH
            for j in range(1, QUARTER_HOURS_PER_HOUR):
                opti.subject_to(opti.bounded(-1e-4, B_volumes[:, i] - B_volumes[:, i+j], 1e-4))
                opti.subject_to(opti.bounded(-1e-4, B_prices[:, i]  - B_prices[:, i+j], 1e-4))

        return    

    def simulate_growth(self, x0, u):
        
        u = u.reshape((1, -1))
        N = u.shape[1]
        X = ca.DM.zeros((self.nx, N+1))
        X[:,0] = x0

        F = self.casadi_function()

        for k in range(N):
            X[:,k+1] = F(X[:,k], u[:,k])

        return X


    def get_eps(self, N_TH, ref_weight, X, past_X: None):

        '''Creates list of constraints for the mpc optimization problem'''

        # g_ineq.append([self.freshweight(X[:,N]) + slack_freshweight - ref_weight, f"Final freshweight constraint"])
        slack_freshweight = float(max(0, ref_weight - self.freshweight(X[:,N_TH])))

        min_dlis = []
        max_dlis = []

        # DLI constraint
        if past_X is None:
            for k in range(N_TH+1):
                if (k % (QUARTER_HOURS_PER_DAY/self.DLI_res) == 0 and k>=QUARTER_HOURS_PER_DAY): 
                    # k = 96 +24, +48, +72 ...
                    LI = (X[2,k] - X[2,k-QUARTER_HOURS_PER_DAY])
                    max_dlis.append(max(0, LI - self.DLI_max))
                    min_dlis.append(max(0, self.DLI_min - LI))
        else:
            prev_LI = past_X[2,:].reshape((1,-1))

            combined_LI = ca.horzcat(prev_LI, X[2,:].reshape((1,-1)))
            for k in range(past_X.shape[1] + N_TH+1):
                if (k % (QUARTER_HOURS_PER_DAY/self.DLI_res) == 0 and k>=QUARTER_HOURS_PER_DAY): 
                    # k = 96 +24, +48, +72 ...
                    LI = (combined_LI[:,k] - combined_LI[:,k-QUARTER_HOURS_PER_DAY])
                    max_dlis.append(max(0, LI - self.DLI_max))
                    min_dlis.append(max(0, self.DLI_min - LI))


        slack_max_DLI = float(max(max_dlis))
        slack_min_DLI = float(max(min_dlis))

        Eps = np.array([slack_freshweight, slack_max_DLI, slack_min_DLI]).reshape((-1,1))


        return  Eps







    

    def get_u(self, N, N_bids, U_nom, B_volumes, B_prices, spot_prices, balancing_market: BalancingMarket, clearing_prices = None):
        U = np.zeros(N, type(U_nom))
        
        clearing_prices_up   = None if clearing_prices is None else clearing_prices[0,:].reshape((1,-1))
        clearing_prices_down = None if clearing_prices is None else clearing_prices[1,:].reshape((1,-1))

        clearing_price_variance_up   = balancing_market.estimated_price_variances[balancing_market.bidding_zone]['Up']
        clearing_price_variance_down = balancing_market.estimated_price_variances[balancing_market.bidding_zone]['Down']
    
        for k in range(N):

            if k < N_bids:

                P_tilde = B_volumes[1,k]*balancing_market.activation_prob_down(B_prices[1,k], spot_prices[k], clearing_prices_down[:,k], clearing_price_variance_down)\
                        - B_volumes[0,k]*balancing_market.activation_prob_up(B_prices[0,k], spot_prices[k], clearing_prices_up[:,k], clearing_price_variance_up)
                
                u_tilde =  P_tilde / self.k_P

            else:
                u_tilde = 0

            U[k] = U_nom[:,k] + u_tilde

        return ca.vertcat(*U).reshape((1,-1))
    
    
    def get_u_CM(self, N, N_bids, U_nom, CM_B_volumes, CM_B_prices, spot_prices, CM: BalancingMarket, AM: BalancingMarket, CM_clearing_prices = None):

        assert CM_B_prices.shape[0] == CM_B_volumes.shape[0], 'Inconsitent sizes of bid arrays'
        assert CM_B_prices.shape[1] == CM_B_volumes.shape[1], 'Inconsitent sizes of bid arrays'
        U = np.zeros(N, type(U_nom))
    
        for k in range(N):

            if k < N_bids:

                CM_Volume_up     = CM_B_volumes[0,k]
                CM_Volume_down   = CM_B_volumes[1,k]
                AM_Demand_up     = AM.demand_prob_up(spot_prices[k])   # * 0.5 
                AM_Demand_down   = AM.demand_prob_down(spot_prices[k]) # * 0.5
        
                # Activatio market participation does not require CM participation. 
                # So chances of activation is not affected by CM_Activation_chance_up or CM_Activation_chance_down
                P_tilde = AM_Demand_down * CM_Volume_down \
                        - AM_Demand_up   * CM_Volume_up
                                    

                u_tilde = P_tilde / self.k_P
            
            else:
                u_tilde = 0


            U[k] = U_nom[:,k] + u_tilde

        return ca.vertcat(*U).reshape((1,-1))


    def get_u_hat(self, N, U_nom, CM_B_volumes, CM_B_prices, AM_B_volumes, AM_B_prices, CM_activations, AM_activations, spot_prices, CM_clearing_prices, AM_clearing_prices, CM: BalancingMarket, AM: BalancingMarket):

        U = np.array([])
    
        for k in range(N):

            # If AM market has been cleared for time k. We then KNOW u
            if k < AM_activations.shape[1]:

                volume_up   = AM_B_volumes[0,k]
                volume_down = AM_B_volumes[1,k]
                activation_chance_up    = AM_activations[0,k]
                activation_chance_down  = AM_activations[1,k]

            # If AM bids have been submitted for time k. We then calculate most likely u
            elif k < AM_B_volumes.shape[1]:

                volume_up   = AM_B_volumes[0,k]
                volume_down = AM_B_volumes[1,k]
                bid_price_up   = AM_B_prices[0,k]
                bid_price_down = AM_B_prices[1,k]
                
                activation_chance_up    = AM.activation_prob_up(bid_price_up,   spot_prices[k], AM_clearing_prices[0,k])
                activation_chance_down  = AM.activation_prob_up(bid_price_down, spot_prices[k], AM_clearing_prices[1,k])


            # If only CM market has been cleared.
            elif k < CM_B_volumes.shape[1]:

                volume_up   = CM_B_volumes[0,k]
                volume_down = CM_B_volumes[1,k]
                
                activation_chance_up    = AM.demand_prob_up(spot_prices[k])
                activation_chance_down  = AM.demand_prob_down(spot_prices[k])
            
            # This last one should never occur
            else:
                assert False, 'get_u_hat calculation failed. Calculation requested for time step without CM or AM bid submissions'


            P_tilde = volume_down * activation_chance_down - volume_up * activation_chance_up
            u_tilde = P_tilde/self.k_P
            U = np.append(U, U_nom[:,k] + u_tilde)

        return ca.vertcat(*U).reshape((1,-1))



    def get_u_CM_AM(self, N, U_nom, CM_B_volumes, CM_B_prices, AM_B_volumes, AM_B_prices, spot_prices, CM: BalancingMarket, AM: BalancingMarket):

        U = np.array([])
    
        for k in range(N):
            
            CM_A_up     = CM.activation_prob_up(spot_prices[k], CM_B_prices[0,k])
            CM_A_down   = CM.activation_prob_down(spot_prices[k], CM_B_prices[1,k])
            CM_Volume_up     = CM_B_volumes[0,k]
            CM_Volume_down   = CM_B_volumes[1,k]

            AM_A_up     = AM.activation_prob_up(spot_prices[k], AM_B_prices[0,k])
            AM_A_down   = AM.activation_prob_down(spot_prices[k], AM_B_prices[1,k])
            AM_Volume_up     = AM_B_volumes[0,k]
            AM_Volume_down   = AM_B_volumes[1,k]
            
            P_tilde = AM_A_down*(CM_A_down * casadi_max(CM_Volume_down, AM_Volume_down) + (1-CM_A_down)*AM_Volume_down) \
                    - AM_A_up*(CM_A_up * casadi_max(CM_Volume_up, AM_Volume_up) + (1-CM_A_up)*AM_Volume_up)
            

            u_tilde = P_tilde/self.k_P

            U = np.append(U, U_nom[:,k] + u_tilde)

        return ca.vertcat(*U).reshape((1,-1))


    def get_bidding_bounds(self, N, U_nom, balancing_market: BalancingMarket):

        U_nom = U_nom.reshape((1,-1))                                                  # Ensure correct dimension

        lb_B_prices = - balancing_market.bid_price_limit * ca.DM.ones(2, N)
        ub_B_prices = balancing_market.bid_price_limit * ca.DM.ones(2, N)
        
        lb_B_volumes = ca.DM.zeros(2, N)
        ub_B_volumes = ca.vertcat(self.k_P * U_nom,                       # Bid vol up
                                  self.k_P * (self.PPFD_max - U_nom))     # Bid vol down

        return lb_B_volumes, ub_B_volumes, lb_B_prices, ub_B_prices
    
    
    def get_state_bounds(self, controller):

        N = controller.N

        # Define bounds on x and u
        lbx = 0* np.ones((self.nx, N+1))         # Lower bound for x (x >= 0)
        ubx = np.inf * np.ones((self.nx, N+1))   # Upper bound for x (no upper bound)


        return lbx, ubx
    
    def get_input_bounds(self, controller):
        
        N = controller.N

        lbu = np.zeros((self.nu, N))    # Lower bound for u (u >= 0)
        ubu = self.PPFD_max * np.ones((self.nu, N))                     # Upper bound for u (u <= Max PPFD 250)

        return lbu, ubu
    

    def get_metrics(self, metrics_data, controller, run_id, x, u):

        DLI = []

        for k in range(controller.N+1):
            if (k % (QUARTER_HOURS_PER_DAY/self.DLI_res) == 0 and k>=QUARTER_HOURS_PER_DAY): 
                # k = 96 +24, +48, +72 ...
                DLI.append(x[2,k] - x[2,k-QUARTER_HOURS_PER_DAY])


        metrics_data['DLI_avg'] = np.average(DLI)
        metrics_data['DLI variance'] = np.var(DLI)
        metrics_data['DLI_max'] = np.max(DLI)
        metrics_data['DLI_min'] = np.min(DLI)
        metrics_data['Final fresh weight'] = float(self.freshweight(x[:,-1]))

        return metrics_data
