from controller import Controller
from model import *
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *
from simulator import Simulator


SIM_NAME = "fixed_plant_model"
HORIZON_DAYS = 20
# FINAL_WEIGHT = 16.8
# FINAL_WEIGHT = 84
# FINAL_WEIGHT = 45
FINAL_WEIGHT = 163
# FINAL_WEIGHT = 117.34

SIMULATION_DATE = '2024-01-01'
BIDDING_ZONE    = 'NO3'

baseline = 'import'
import_file = 'scaled_optimal_intensities.json'

def main():

    x_init = np.array([5, 1])   # Specify init vector [structural and non structural dry weight in grams]

    config = Config(SIM_NAME, filetype="pdf")
    plant = PlantModel(x_init, FINAL_WEIGHT)
    # plant = Photosynthesis(x_init, FINAL_WEIGHT)
    market = Market(config, HORIZON_DAYS, 1133, BIDDING_ZONE, SIMULATION_DATE)
    controller = Controller(HORIZON_DAYS, plant, market, config, baseline=baseline, search_cache = True, warm_start=True, import_file = import_file, calculate_fw=False)         #Baseline: 'opt' / 'rigid'
    
    mpc_controller = Controller(HORIZON_DAYS, plant, market, config, baseline='opt', surpress_output = True)     # Instance of controller used in mpc
    simulator = Simulator(HORIZON_DAYS, plant, market, config, mpc_controller)

    battery = BatteryModel(x_init = np.array([200]))
    battery_controller = Controller(HORIZON_DAYS, battery, market, config, 'opt')

    # battery_controller.optimize_baseline()
    # battery_controller.optimize_bidding()

    controller.import_baseline()
    controller.rigid_baseline()
    controller.optimize_baseline()
    controller.optimize_bidding()
    controller.status_report()
    controller.save_to_json()

    plotter = Plotter(config, controller, simulator)
    plotter.save_ocp_plots()
    # plotter.plot_spot_mfrr_prices()

    # market.optimal_bidding_price_prediction(controller.p_spot)

    # simulator.Simulate_mpc()

    # plotter.save_mpc_plots()


if __name__ == "__main__":
    main()
