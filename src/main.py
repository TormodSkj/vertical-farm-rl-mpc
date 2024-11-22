from controller import Controller
from model import *
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *
from simulator import Simulator


SIM_NAME = "NO1_Jan1_2024"
HORIZON_DAYS = 20
FINAL_WEIGHT = 16.8
# FINAL_WEIGHT = 30
FINAL_WEIGHT = 163

SIMULATION_DATE = '2024-01-01'
BIDDING_ZONE    = 'NO1'

baseline = 'opt'

def main():

    x_init = np.array([5, 1])   # Specify init vector [structural and non structural dry weight in grams]

    config = Config(SIM_NAME, filetype="pdf")
    plant = PlantModel(x_init, FINAL_WEIGHT)
    market = Market(HORIZON_DAYS, 1133, BIDDING_ZONE, SIMULATION_DATE)
    controller = Controller(HORIZON_DAYS, plant, market, config, baseline=baseline, search_cache = True)         #Baseline: 'opt' / 'rigid'
    
    mpc_controller = Controller(HORIZON_DAYS, plant, market, config, baseline='opt', surpress_output = True)     # Instance of controller used in mpc
    simulator = Simulator(HORIZON_DAYS, plant, market, config, mpc_controller)

    battery = BatteryModel(x_init = np.array([200]))
    battery_controller = Controller(HORIZON_DAYS, battery, market, config, 'opt')


    # battery_controller.optimize_baseline()
    # battery_controller.optimize_bidding()

    controller.generate_baseline()
    controller.optimize_bidding()
    controller.save_to_json()

    plotter = Plotter(config, controller, simulator)
    plotter.save_ocp_plots()


    # simulator.Simulate_mpc()

    # plotter.save_mpc_plots()


if __name__ == "__main__":
    main()
