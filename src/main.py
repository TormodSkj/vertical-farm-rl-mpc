from controller import Controller
from model import *
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *
from simulator import Simulator
from utils import vertigrow_calculate_energy_consumption, strip_entsoe_activation_data

SIM_NAME        = "mpc_test_3"
HORIZON_DAYS    = 20
FINAL_WEIGHT    = 4                 # 1 day
# FINAL_WEIGHT    = 36.66             # 7 Days
# FINAL_WEIGHT    = 136.7             # 20 Days 
# FINAL_WEIGHT    = 2.05              # 20 Days [Directly from germination]
SIMULATION_DATE = '2024-01-01'
BIDDING_ZONE    = 'NO2'
import_file     = 'scaled_optimal_intensities.json'
optimistic      = 1
search_cache    = 1

MPC_TH = 1
MPC_steplength = 1

def main():

    # x_init = 0.031415 * np.array([5, 1])   # From germination [calibrated from 18d experiment]
    x_init = np.array([5, 1])   # Specify init vector [structural and non structural dry weight in grams]

    ''' CREATING OBJECTS '''
    config          = Config(SIM_NAME, filetype="pdf", seed=1133)
    plant           = PlantModel(x_init, FINAL_WEIGHT)
    market          = Market(config, HORIZON_DAYS, BIDDING_ZONE, SIMULATION_DATE, outlier_max_dist= 2, optimistic = optimistic)
    controller      = Controller(HORIZON_DAYS, plant, market, config, MPC_TH, MPC_steplength, search_cache = search_cache, warm_start=True, import_file = import_file, calculate_fw=True, surpress_output=False)         #Baseline: 'opt' / 'rigid'
    
    mpc_controller  = Controller(HORIZON_DAYS, plant, market, config, MPC_TH, MPC_steplength, surpress_output = False, calculate_fw=True)     # Instance of controller used in mpc
    simulator       = Simulator(HORIZON_DAYS, plant, market, config, mpc_controller, time_horizon=MPC_TH, time_iteration=MPC_steplength)
  
    plotter = Plotter(config, controller, simulator)

    ''' OPTIMIZAION AND PLOTTING '''
    # market.optimal_bidding_price_prediction(controller.spot_prices)
    
    # '''
    # controller.import_baseline('imported')

    controller.optimize_spotprice('spot_opt')         #
    # controller.optimize_mfrr('mfrr_opt', 'spot_opt')  #
    # simulator.apply_mfrr_clearing_prices(controller, 'mfrr_applied', 'mfrr_opt')
    
    controller.optimize_mfrr_mpc('mfrr_mpc')
    simulator.apply_mfrr_clearing_prices(controller, 'apply_prices_mpc', 'mfrr_mpc')

    controller.optimize_mfrr_mpc('mfrr_mpc_spot', 'spot_opt')
    simulator.apply_mfrr_clearing_prices(controller, 'apply_prices_mpc_spot', 'mfrr_mpc_spot')

    # controller.generate_optimal_bidding_strategy('abs_opt', 'spot_opt')
    # simulator.apply_mfrr_clearing_prices(controller, 'abs_applied', 'abs_opt')

    # controller.optimize_mfrr('mfrr_fixed', 'fixed')
    # simulator.apply_mfrr_clearing_prices(controller, 'mfrr_fixed_applied', 'mfrr_fixed')

    controller.status_report()
    controller.save_to_json()

    #'''
    # PLOTTING
    ''''''
    plotter.save_ocp_plots()
    plotter.plot_financial_report()

    # plotter.plot_spot_mfrr_prices() 
    ''''''

    # controller.export_intensity_to_json('Baseline')

    # market.analyze_price_covariances()

    # plotter.plot_price_prediction()

    # simulator.Simulate_mpc()
    # simulator.solve_mpc()
    # simulator.simulate_random_activation(controller, 20)
    # plotter.save_mpc_plots()

    # plotter.plot_random_activations(30)



if __name__ == "__main__":
    main()
