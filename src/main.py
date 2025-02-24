from settings import Settings
from controller import Controller
from model import *
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *
from simulator import Simulator
from utils import vertigrow_calculate_energy_consumption, strip_entsoe_activation_data
from market_utils import fetch_CM_data_nucs

SIM_NAME            = "nordpool_data"
SIMULATION_LENGTH   = 20
FINAL_FRESHWEIGHT   = 4                   # 1 day
# FINAL_WEIGHT      = 36.66             # 7 Days
# FINAL_WEIGHT      = 136.7             # 20 Days 
# FINAL_WEIGHT      = 2.05              # 20 Days [Directly from germination]
SIMULATION_DATE = '2024-01-14'
BIDDING_ZONE    = 'NO2'
OPTIMISTIC      = 1
SEARCH_CACHE    = 1

MPC_TIMEHORIZON = 1
MPC_STEPLENGTH = 0.25

def main():

    # x_init = 0.031415 * np.array([5, 1, 0])   # From germination [calibrated from 18d experiment]
    X_INIT = np.array([5, 1, 0])   # Specify init vector [structural and non structural dry weight in grams]

    ''' CREATING OBJECTS '''
    settings = Settings('general', SIMULATION_LENGTH = SIMULATION_LENGTH)
    settings.update_setting(SIM_NAME = SIM_NAME)
    settings.add_setting('controller', SEARCH_SIM_CACHE = SEARCH_CACHE)
    settings.add_setting('mpc', MPC_TIMEHORIZON = MPC_TIMEHORIZON, MPC_STEPLENGTH = MPC_STEPLENGTH)
    settings.add_setting('market', OPTIMISTIC = OPTIMISTIC, BIDDING_ZONE = BIDDING_ZONE)
    settings.add_setting('plantmodel', INIT_STATE = X_INIT, TARGET_FRESHWEIGHT = FINAL_FRESHWEIGHT)

    config          = Config(settings)
    plant           = PlantModel(settings)
    market          = Market(settings, config)
    controller      = Controller(settings, plant, market, config)
    simulator       = Simulator(settings, plant, market, config, controller)
    plotter         = Plotter(settings, config, controller, simulator)

    ''' OPTIMIZAION '''
    # market.optimal_bidding_price_prediction(controller.spot_prices)
    
    # '''
    # controller.import_baseline('imported')

    # controller.optimize_spotprice('spot_opt')         #
    # controller.optimize_mfrr('mfrr_opt', 'spot_opt')  #
    # simulator.apply_mfrr_clearing_prices('mfrr_applied', 'mfrr_opt')
    
    # controller.optimize_mfrr_mpc('mfrr_mpc')
    # simulator.apply_mfrr_clearing_prices('apply_prices_mpc', 'mfrr_mpc')

    # controller.optimize_mfrr_mpc('mfrr_mpc_spot', 'spot_opt')
    # simulator.apply_mfrr_clearing_prices('apply_prices_mpc_spot', 'mfrr_mpc_spot')

    # controller.generate_optimal_bidding_strategy('abs_opt', 'spot_opt')
    # simulator.apply_mfrr_clearing_prices('abs_applied', 'abs_opt')

    # controller.optimize_mfrr('mfrr_fixed', 'fixed')
    # simulator.apply_mfrr_clearing_prices('mfrr_fixed_applied', 'mfrr_fixed')

    # controller.status_report()
    # controller.save_to_json()

    #'''
    # 
    ''' PLOTTING '''
    # plotter.save_ocp_plots()
    # plotter.plot_financial_report()

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


    # target_file = config.data_path + 'nucs_data.csv'
    # fetch_CM_data_nucs(target_file, "02-10-2024", "31-12-2024")

if __name__ == "__main__":
    main()
