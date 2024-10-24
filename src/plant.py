import numpy as np


class PlantModel:


    #constants:

    #Indoor climate assumptions
    T_crop = 24     #Indoor ambient temperature [C]
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
    c_car_1 = -1.32e-5      #\
    c_car_2 = 5.94e-4       # } Carboxylation resistance 2nd order approximation coefficients
    c_car_3 = -2.64e-3      #/
    l = 0.11                #Mean leaf diameter
    u_inf = 0.15            #Uninhibited air speed
    c_p = 0.217             #Conversion factor from PPFD to PAR
    eta_light = 0.8         #LED efficiency
    c_d = 0.05              #Dry matter content
    PCD = 25                #Plant crop density

    def derivative(self, x, u):
        
        #Extract state
        x_sdw = x[0]
        x_nsdw = x[1]
        PPFD = u[0]

        
        #Common constants
        T_crop = self.T_crop
        c_T = self.c_T


        #Abstractions
        r_gr = x_nsdw / (x_nsdw + x_sdw) * self.c_gr_max * self.c_Q_10_gr**((T_crop-20)/10)      #Growth rate

        LAI = self.c_lar * (1-c_T)*x_sdw                                                         #Leaf area index
        CAC = 1-np.exp(-self.c_k * LAI)                                                          #Cultivation area cover fraction
        Gamma = self.c_Gamma * self.c_Q_10_Gamma**(T_crop - 20)/10                                    #Co2 compensation point 
        alpha = self.c_e * (self.co2_in - Gamma)/(self.co2_in + 2*Gamma)                                   #Quantum yield
        U_par = self.c_p * PPFD                                                                  #Photosynthetically active radiation
        r_car = 1/(self.c_car_1 * T_crop**2 + self.c_car_2 * T_crop + self.c_car_3)                        #Carboxylation resistance
        r_bnd = 350*np.sqrt(self.l/self.u_inf) / LAI                                                  #Boundary layer resistance 
        r_stm = 60*(1500 + PPFD)/(200 + PPFD)                                               #Stomatal resistance
        r_co2 = r_bnd + r_stm + r_car                                                       #Canopy resistance 
        f_sat = self.rho_c * (self.co2_in - Gamma)/r_co2                                              #Light saturated vlaue of max photosynthesis
        f_phot_max = alpha * U_par * f_sat / (alpha * U_par + f_sat)                        #Maximum photosynthetic rate
        f_phot = f_phot_max * CAC                                                           #Gross canopy photosynthesis
        f_resp = (self.c_resp_sht*(1-c_T) + self.c_resp_rt*c_T)*x_sdw * self.c_Q_10_gr**((T_crop-25)/10)   #Maintenance respiration rate
        x_dw_plant = (x_sdw + x_nsdw) / self.PCD                                                 #X dont worry plant <3
        x_fw_sht = x_dw_plant * (1-c_T)/self.c_d                                                 #Fresh weight per plant

        #Derivatives
        x_sdw_dot = r_gr * x_sdw
        # x_nsdw_dot = c_a * f_phot - x_sdw_dot - f_resp - (1-c_b)/c_b * r_gr * x_sdw       Slightly inefficient implementation
        x_nsdw_dot = self.c_a * f_phot - f_resp - 1/self.c_b * x_sdw_dot                              #More efficient implementation

        return np.array([x_sdw_dot, x_nsdw_dot])