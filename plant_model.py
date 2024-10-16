
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, exp
import numpy as np

def export_lettuce_ode_model() -> AcadosModel:

    model_name = 'lettuce_ode'

    # constants
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

    # set up states & controls
    x_sdw      = SX.sym('x_sdw')
    x_nsdw   = SX.sym('x_nsdw')

    x = vertcat(x_sdw, x_nsdw)

    F = SX.sym('F')
    u = vertcat(F)

    # xdot
    x_sdw_dot      = SX.sym('x_sdw_dot')
    x_nsdw_dot   = SX.sym('x_nsdw_dot')

    xdot = vertcat(x_sdw_dot, x_nsdw_dot)

    # dynamics

    #Abstractions
    r_gr = x_nsdw / (x_nsdw + x_sdw) * c_gr_max * c_Q_10_gr**((T_crop-20)/10)           #Growth rate

    LAI = c_lar * (1-c_T)*x_sdw                                                         #Leaf area index
    CAC = 1-np.exp(-c_k * LAI)                                                          #Cultivation area cover fraction
    Gamma = c_Gamma * c_Q_10_Gamma**(T_crop - 20)/10                                    #Co2 compensation point 
    alpha = c_e * (co2_in - Gamma)/(co2_in + 2*Gamma)                                   #Quantum yield
    U_par = c_p * u                                                                  #Photosynthetically active radiation
    r_car = 1/(c_car_1 * T_crop**2 + c_car_2 * T_crop + c_car_3)                        #Carboxylation resistance
    r_bnd = 350*np.sqrt(l/u_inf) / LAI                                                  #Boundary layer resistance 
    r_stm = 60*(1500 + u)/(200 + u)                                               #Stomatal resistance
    r_co2 = r_bnd + r_stm + r_car                                                       #Canopy resistance 
    f_sat = rho_c * (co2_in - Gamma)/r_co2                                              #Light saturated vlaue of max photosynthesis
    f_phot_max = alpha * U_par * f_sat / (alpha * U_par + f_sat)                        #Maximum photosynthetic rate
    f_phot = f_phot_max * CAC                                                           #Gross canopy photosynthesis
    f_resp = (c_resp_sht*(1-c_T) + c_resp_rt*c_T)*x_sdw * c_Q_10_gr**((T_crop-25)/10)   #Maintenance respiration rate
    x_dw_plant = (x_sdw + x_nsdw) / PCD                                                 #X dont worry plant <3
    x_fw_sht = x_dw_plant * (1-c_T)/c_d                                                 #Fresh weight per plant


    f_expl = vertcat(r_gr * x_sdw,
                     c_a * f_phot - f_resp - 1/c_b * x_sdw_dot
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model

