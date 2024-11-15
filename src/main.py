from controller import Controller
from plant import PlantModel
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt

QUARTER_HOURS_PER_DAY = 96
SECONDS_PER_QUARTER_HOUR = 15 * 60




SIM_NAME = "2_days_test"

HORIZON_DAYS = 2
N = HORIZON_DAYS * QUARTER_HOURS_PER_DAY  # Number of time steps. N_days * N_hours_per_day
dt = SECONDS_PER_QUARTER_HOUR             # 15 minutes in seconds

x_init = np.array([5, 1])   # Specify init vector
Final_fw_sht = 15           # Final plant weight requirement

config = Config(SIM_NAME)
plant = PlantModel(x_init, Final_fw_sht)
market = Market(N, 1133, 'NO4', '2023-12-01')
controller = Controller(N, HORIZON_DAYS, dt, plant, market, config, "opt")     #Baseline: 'opt' / 'rigid'
plotter = Plotter(config, controller)

#################################################
controller.optimize()
controller.save_to_json()
#################################################

plotter.save_plots()

