from controller import Controller
from model import *
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *
from simulator import Simulator


SIM_NAME = "results"
HORIZON_DAYS    = 20
# FINAL_WEIGHT  = 36.66             # 7 Days
FINAL_WEIGHT    = 136.7             # 20 Days
SIMULATION_DATE = '2024-01-01'
BIDDING_ZONE    = 'NO3'
import_file     = 'scaled_optimal_intensities.json'

def main():

    x_init = np.array([5, 1])   # Specify init vector [structural and non structural dry weight in grams]


    ''' CREATING INSTANCES '''
    config = Config(SIM_NAME, filetype="pdf", seed=1133)
    plant = PlantModel(x_init, FINAL_WEIGHT)
    # plant = Photosynthesis(x_init, FINAL_WEIGHT)
    market = Market(config, HORIZON_DAYS, BIDDING_ZONE, SIMULATION_DATE)
    controller = Controller(HORIZON_DAYS, plant, market, config, search_cache = True, warm_start=True, import_file = import_file, calculate_fw=False)         #Baseline: 'opt' / 'rigid'
    
    mpc_controller = Controller(HORIZON_DAYS, plant, market, config, surpress_output = True)     # Instance of controller used in mpc
    simulator = Simulator(HORIZON_DAYS, plant, market, config, mpc_controller)

    battery = BatteryModel(x_init = np.array([200]))
    battery_controller = Controller(HORIZON_DAYS, battery, market, config, 'opt')



    ''' OPTIMIZAION AND PLOTTING '''

    # controller.import_baseline()
    controller.rigid_baseline()
    controller.optimize_baseline()
    # controller.optimize_bidding()
    controller.status_report()
    # controller.export_intensity_to_json('Baseline')
    controller.save_to_json()

    plotter = Plotter(config, controller, simulator)
    plotter.save_ocp_plots()
    plotter.plot_spot_mfrr_prices() 

    # market.optimal_bidding_price_prediction(controller.p_spot)
    # plotter.plot_price_prediction()

    # simulator.Simulate_mpc()
    # simulator.simulate_random_activation(controller, 1)
    # plotter.save_mpc_plots()

    # plotter.plot_random_activations(10)



if __name__ == "__main__":
    main()
