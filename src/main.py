from controller import Controller
from model import *
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *
from simulator import Simulator
from utils import vertigrow_calculate_energy_consumption

SIM_NAME        = "price_comparison"
HORIZON_DAYS    = 18
FINAL_WEIGHT    = 4                 # 1 day
# FINAL_WEIGHT    = 36.66             # 7 Days
# FINAL_WEIGHT    = 136.7             # 20 Days 
# FINAL_WEIGHT    = 2.05              # 20 Days [Directly from germination]
SIMULATION_DATE = '2024-01-01'
BIDDING_ZONE    = 'NO3'
import_file     = 'scaled_optimal_intensities.json'
search_cache    = 1

MPC_TH = HORIZON_DAYS
MPC_steplength = HORIZON_DAYS

def main():

    # x_init = 0.031415 * np.array([5, 1])   # From germination [calibrated from 18d experiment]
    x_init = np.array([5, 1])   # Specify init vector [structural and non structural dry weight in grams]


    ''' CREATING OBJECTS '''
    config      = Config(SIM_NAME, filetype="pdf", seed=1133)
    plant       = PlantModel(x_init, FINAL_WEIGHT)
    mpc_plant   = MpcPlantModel(x_init, FINAL_WEIGHT)
    market      = Market(config, HORIZON_DAYS, BIDDING_ZONE, SIMULATION_DATE)
    controller  = Controller(HORIZON_DAYS, plant, mpc_plant, market, config, MPC_TH, MPC_steplength, search_cache = search_cache, warm_start=True, import_file = import_file, calculate_fw=True)         #Baseline: 'opt' / 'rigid'
    
    mpc_controller = Controller(HORIZON_DAYS, mpc_plant, None, market, config, MPC_TH, MPC_steplength, surpress_output = False, calculate_fw=True)     # Instance of controller used in mpc
    simulator = Simulator(HORIZON_DAYS, mpc_plant, market, config, mpc_controller, time_horizon=MPC_TH, time_iteration=MPC_steplength)
  

    ''' OPTIMIZAION AND PLOTTING '''

    controller.import_baseline()
    controller.optimize_baseline()
    # controller.optimize_bidding()
    # controller.optimize_bidding_mpc()

    # simulator.apply_mfrr_clearing_prices(controller)
    controller.status_report()
    # controller.export_intensity_to_json('Baseline')
    controller.save_to_json()


    plotter = Plotter(config, controller)
    plotter.save_ocp_plots()

    market.analyze_price_covariances()
    plotter.plot_spot_mfrr_prices() 
    # market.optimal_bidding_price_prediction(controller.spot_prices)
    # plotter.plot_price_prediction()

    # simulator.Simulate_mpc()
    # simulator.solve_mpc()
    # simulator.simulate_random_activation(controller, 1)
    # plotter.save_mpc_plots()

    # plotter.plot_random_activations(10)



if __name__ == "__main__":
    main()
