import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
import matplotlib.pyplot as plt

import numpy as np
from market import Market
from model import PlantModel
from config import Config
from globals import *

def test_c_conv_value():
    
    plant = PlantModel()
    # expected_value = 67.8125
    expected_value = 0.00027125
    assert plant.C_conv_PPFD == expected_value, f"Expected {expected_value}, but got {plant.C_conv_PPFD}"

def test_fw_calculation():

    #Sandbox to play around in mostly :D

    plant = PlantModel()
    X = np.ones((2, 30))
    print(plant.freshweight(X))

    assert True


def test_constants():

    plant = PlantModel(np.array([0,0]), 0)
    print(plant.C_conv_PPFD)
    print(plant.P_cap_max)
    print(plant.C_conv_PPFD*2.5)
    print(250*plant.C_conv_PPFD)

    assert False


def test_phot_curve():



    def f_phot(PPFD):

        T_crop = 24     #Indoor ambient temperature [C]
        co2_in = 400   #CO2 consentration of indoor air [PPM]

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
                
        epsilon = 1e-6      #Small constant to avoid division by zero

        #Abstractions
        # r_gr = x_nsdw / (x_nsdw + x_sdw + epsilon) * c_gr_max * c_Q_10_gr**((T_crop-20)/10)      #Growth rate

        # LAI = c_lar * (1-c_T)*x_sdw                                                         #Leaf area index
        LAI = 3
        CAC = 1-np.exp(-c_k * LAI)                                                          #Cultivation area cover fraction
        Gamma = c_Gamma * c_Q_10_Gamma**((T_crop - 20)/10)                                  #Co2 compensation point 
        alpha = c_e * (co2_in - Gamma)/(co2_in + 2*Gamma)                                   #Quantum yield
        U_par = c_p * PPFD                                                                  #Photosynthetically active radiation
        r_car = 1/(c_car_1 * T_crop**2 + c_car_2 * T_crop + c_car_3)                        #Carboxylation resistance
        r_bnd = 350*np.sqrt(l/u_inf) / (LAI + epsilon)                                      #Boundary layer resistance 
        r_stm = 60*(1500 + PPFD)/(200 + PPFD)                                               #Stomatal resistance
        r_co2 = r_bnd + r_stm + r_car                                                       #Canopy resistance 
        f_sat = rho_c * (co2_in - Gamma)/r_co2
        print(f_sat)                                                                        # Light saturated vlaue of max photosynthesis
        f_phot_max = alpha * U_par * f_sat / (alpha * U_par + f_sat)                        # Maximum photosynthetic rate
        f_phot = f_phot_max * CAC                                                           # Gross canopy photosynthesis
        # f_resp = (c_resp_sht*(1-c_T) + c_resp_rt*c_T)*x_sdw * c_Q_10_gr**((T_crop-25)/10)   # Maintenance respiration rate
        # x_dw_plant = (x_sdw + x_nsdw) / PCD                                                 # X dont worry plant <3
        # x_fw_sht = x_dw_plant * (1-c_T)/c_d                                                 # Fresh weight per plant

        #Derivatives
        # x_sdw_dot = r_gr * x_sdw
        # # x_nsdw_dot = c_a * f_phot - x_sdw_dot - f_resp - (1-c_b)/c_b * r_gr * x_sdw   # Slightly inefficient implementation
        # x_nsdw_dot = c_a * f_phot - f_resp - 1/c_b * x_sdw_dot                # More efficient implementation
        # x_LI_dot = PPFD*1e-6

        alpha_nr = 0.927
        slope = alpha_nr
        A_sat = f_sat
        theta = 0.9
        curve = theta
        u_light = PPFD
        inty_to_par = c_p


        term1 = A_sat + slope * u_light * inty_to_par
        term2 = np.sqrt(1e-9 + term1**2 - 4 * curve * slope * u_light * inty_to_par * A_sat)
        phot_rate = (term1 - term2) / (2 * curve)
        light_rate = u_light

        return phot_rate
    
    f_list = []
    u_list = []
    for u in range(400):
        u_list.append(u)
        f_list.append(f_phot(u))


    config = Config()
    
    plt.plot(u_list, f_list)
    plt.xlabel("Hour")

    filename = "Phot_curve"
    foldername = "testing"
    plt.savefig(config.plot_path + foldername + "/" + filename + ".png")    

    assert False