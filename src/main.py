from settings import Settings
from controller import Controller
from model import *
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *
from simulator import Simulator
# from utils import vertigrow_calculate_energy_consumption, plot_balancing_market_earnings_upper_bounds

SIM_NAME            = "Full_MPC_7_day_2"
SIMULATION_LENGTH   = 6
FINAL_FRESHWEIGHT   = 136.7             # 20 Days 
# FINAL_FRESHWEIGHT      = 2.05              # 20 Days [From vertigrow experiments]
SIMULATION_DATE     = '2024-10-13'
BIDDING_ZONE        = 'SE1'
OPTIMISTIC          = 1
SEARCH_CACHE        = 0
SEARCH_PLOT_CACHE   = 0

MPC_TIMEHORIZON     = 3
MPC_STEPLENGTH      = 16/96

DISCRETIZATION      = 'fe'

def main():

    # plot_balancing_market_earnings_upper_bounds()


    # x_init = 0.031415 * np.array([5, 1, 0])   # From germination [calibrated from 18d experiment]
    X_INIT = np.array([5, 1, 0])   # Specify init vector [structural and non structural dry weight in grams]

    ''' CREATING OBJECTS '''
    settings = Settings('general', SIMULATION_LENGTH = SIMULATION_LENGTH)
    settings.update_setting(SIM_NAME = SIM_NAME)
    settings.add_setting('controller', SEARCH_SIM_CACHE = SEARCH_CACHE, IMPORT_FILE = 'mfrr_experiment_1.json')
    settings.add_setting('mpc', MPC_TIMEHORIZON = MPC_TIMEHORIZON, MPC_STEPLENGTH = MPC_STEPLENGTH,
                         CM_N_BIDS = 96, AM_N_BIDS = 48)
    settings.add_setting('market', SIMULATION_DATE = SIMULATION_DATE, OPTIMISTIC = OPTIMISTIC, BIDDING_ZONE = BIDDING_ZONE, 
                         AM_ACTIVATION_RATE  = 0.1, CM_ACTIVATION_RATE  = 0.1, EXACT_ESTIMATION = True)
    settings.add_setting('plantmodel', INIT_STATE = X_INIT, TARGET_FRESHWEIGHT = FINAL_FRESHWEIGHT, DLI_RESOLUTION = 2, DISCRETIZATION = DISCRETIZATION)
    settings.add_setting('plotter', SEARCH_PLOT_CACHE = SEARCH_PLOT_CACHE, PLOT_EXPORT_TYPE='pdf', FILTER_BIDS = True, PLOT_ASPECT_RATIO = (14, 6))

    config      = Config(settings)
    plant       = PlantModel(settings) 
    market      = Market(settings, config)
    controller  = Controller(settings, plant, market, config)
    simulator   = Simulator(settings, plant, market, config, controller)
    plotter     = Plotter(settings, config, controller, simulator)

    
    
    ''' ANALYSIS '''
    # market.calculate_balancing_market_earnings_upper_bound(AM=True, CM=True)
    # market.nordic_markets_overview(output=True)

    # market.AM.get_predicted_clearing_prices('2024-10-22', controller.mpc_N_horizon, plot=True)
    # market.CM.get_predicted_clearing_prices('2024-10-22', controller.mpc_N_horizon, plot=True)


    ''' OPTIMIZAION '''
    
    # '''
    # controller.import_light_schedule('imported', plot_run = True)

    # controller.generate_true_optimum_BL_CM_AM('MARI_opt', plot_run = True)
    controller.optimize_MPC_complete('complete_MPC', 'fixed', plot_run = True)
    # simulator.apply_mfrr_clearing_prices('complete_MPC_applied', 'complete_MPC', plot_run = True)


    # controller.optimize_spotprice('spot_opt', 'None')         #
    # controller.optimize_mfrr('mfrr_opt', 'spot_opt', plot_run=True)  #
    # simulator.apply_mfrr_clearing_prices('mfrr_applied', 'mfrr_opt', plot_run=True)
    
    # controller.optimize_AM_mpc('mfrr_mpc')
    # simulator.apply_mfrr_clearing_prices('apply_prices_mpc', 'mfrr_mpc')

    # controller.optimize_AM_mpc('mfrr_mpc_spot', 'spot_opt')
    # simulator.apply_mfrr_clearing_prices('apply_prices_mpc_spot', 'mfrr_mpc_spot')

    # controller.generate_true_optimum_AM('abs_opt', 'spot_opt')
    # simulator.apply_mfrr_clearing_prices('abs_applied', 'abs_opt')

    # controller.generate_true_optimum_CM('optimal_CM', plot_run=True)
    # controller.generate_true_optimum_AM('optimal_AM', 'optimal_CM', plot_run = True)
    # simulator.apply_mfrr_clearing_prices('abs_applied', 'abs_opt')

    # controller.generate_true_optimum_BL_CM_AM('MARI_opt', plot_run = True)
    # controller.generate_true_optimum_CM_AM('MARI_opt', refrun_id='spot_opt', plot_run = True)

    # controller.co_optimize_CM_AM('co_opt_1')
    # simulator.apply_mfrr_clearing_prices('co_opt_1_applied', 'co_opt_1_CM')
    # controller.optimize_mfrr('co_opt_2', 'co_opt_1_nom')  #
    # simulator.apply_mfrr_clearing_prices('co_opt_2_applied', 'co_opt_2')

    # controller.optimize_mfrr('mfrr_fixed', 'fixed')
    # simulator.apply_mfrr_clearing_prices('mfrr_fixed_applied', 'mfrr_fixed')

    # '''
    
    '''Status report'''
    # controller.status_report()
    controller.save_to_json()
    ''''''

    # run_ids = ['mfrr_opt', 'abs_opt']
    # for run_id in run_ids: print(f"Upper CM participation earnings limit for {run_id}: {market.calculate_CM_earnings_upper_limit(controller, run_id)}")
          
    # market.estimate_prices()
    # market.price_estimation(plot_estimates=True)

    
    # 
    ''' PLOTTING '''
    plotter.save_ocp_plots()
    plotter.plot_financial_report()

    # plotter.plot_spot_mfrr_prices() 
    # plotter.plot_CM_data() 
    ''''''

    # controller.export_intensity_to_json('abs_applied')

    # target_file = config.mfrr_CM_data_path + 'CM_data_NO_DK_SE_FI.csv'
    # fetch_CM_data_nucs(target_file, "01-10-2024", "02-10-2024")

if __name__ == "__main__":
    main()
