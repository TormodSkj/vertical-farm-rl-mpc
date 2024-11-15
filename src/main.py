from controller import Controller
from plant import PlantModel
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *


SIM_NAME = "2_days_test"
HORIZON_DAYS = 2



x_init = np.array([5, 1])   # Specify init vector
Final_fw_sht = 15           # Final plant weight requirement

config = Config(SIM_NAME)
plant = PlantModel(x_init, Final_fw_sht)
market = Market(HORIZON_DAYS, 1133, 'NO4', '2023-12-01')
controller = Controller(HORIZON_DAYS, plant, market, config, "opt")     #Baseline: 'opt' / 'rigid'
plotter = Plotter(config, controller)

#################################################
controller.optimize()
controller.save_to_json()
#################################################

plotter.save_plots()

