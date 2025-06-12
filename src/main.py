from settings import Settings
from controller import Controller
from model import *
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *
from simulator import Simulator
from datetime import datetime, timedelta

# from utils import vertigrow_calculate_energy_consumption, plot_balancing_market_earnings_upper_bounds

SIM_NAME            = "trimmed_AM_down_prices"
SIMULATION_LENGTH   = 7
FINAL_FRESHWEIGHT   = 136.7             # 20 Days 
# FINAL_FRESHWEIGHT      = 2.05              # 20 Days [From vertigrow experiments]
SIMULATION_DATE     = '2024-04-05'
BIDDING_ZONE        = 'NO2'
OPTIMISTIC          = 1
SEARCH_CACHE        = 0
SEARCH_PLOT_CACHE   = 0
SURPRESS_OUTPUT     = False

MPC_TIMEHORIZON     = 3
MPC_STEPLENGTH      = 12/96

DISCRETIZATION      = 'fe'
SAVE_TO_CSV         = False

def main():

    # plot_balancing_market_earnings_upper_bounds()


    # x_init = 0.031415 * np.array([5, 1, 0])   # From germination [calibrated from 18d experiment]
    X_INIT = np.array([5, 1, 0])   # Specify init vector [structural and non structural dry weight in grams]

    ''' CREATING OBJECTS '''
    settings = Settings('general', SIMULATION_LENGTH = SIMULATION_LENGTH)
    settings.update_setting(SIM_NAME = SIM_NAME, SURPRESS_OUTPUT = SURPRESS_OUTPUT, OPTI_PRINT_LEVEL = 0)
    settings.add_setting('controller', SEARCH_SIM_CACHE = SEARCH_CACHE, IMPORT_FILE = 'mfrr_experiment_1.json')
    settings.add_setting('mpc', MPC_TIMEHORIZON = MPC_TIMEHORIZON, MPC_STEPLENGTH = MPC_STEPLENGTH,
                         CM_N_BIDS = 96, AM_N_BIDS = 48, CHECK_FEASIBILITY = False)
    settings.add_setting('market', SIMULATION_DATE = SIMULATION_DATE, OPTIMISTIC = OPTIMISTIC, BIDDING_ZONE = BIDDING_ZONE, 
                         AM_ACTIVATION_RATE = 0.25, 
                         CM_ACTIVATION_RATE = 0.25, 
                         EXACT_ESTIMATION   = False, 
                         RELATIVE_AM_PRICES = False, 
                         TRIM_AM_PRICES     = True)
    settings.add_setting('plantmodel', INIT_STATE = X_INIT, TARGET_FRESHWEIGHT = FINAL_FRESHWEIGHT, DLI_RESOLUTION = 2, DISCRETIZATION = DISCRETIZATION)
    settings.add_setting('plotter', SEARCH_PLOT_CACHE = SEARCH_PLOT_CACHE, PLOT_EXPORT_TYPE='pdf', FILTER_BIDS = True, PLOT_ASPECT_RATIO = (14, 6))

    config      = Config(settings)
    plant       = PlantModel(settings) 
    market      = Market(settings, config)
    controller  = Controller(settings, plant, market, config)
    simulator   = Simulator(settings, plant, market, config, controller)
    plotter     = Plotter(settings, config, controller, simulator)

    
    
    ''' ANALYSIS '''
    # market.calculate_balancing_market_earnings_upper_bound(AM=True, CM=True, plot = True)
    # market.nordic_markets_overview(output=True)

    # market.AM.get_predicted_clearing_prices('2024-10-22', controller.mpc_N_horizon, plot=True)
    # market.CM.get_predicted_clearing_prices('2024-10-22', controller.mpc_N_horizon, plot=True)


    ''' OPTIMIZAION '''
    
    # '''
    controller.optimize_MPC_complete('complete_MPC', 'fixed', plot_run = True)
    # controller.optimize_mfrr('mfrr', 'fixed', plot_run = True)
    # simulator.apply_mfrr_clearing_prices('mfrr_applied', 'mfrr', plot_run = True)
    # controller.generate_true_optimum_BL_CM_AM('true_opt_BL_CM_AM', 'fixed', plot_run=True)

    # '''
    
    '''Status report'''
    # controller.status_report()
    controller.save_to_json()
    if SAVE_TO_CSV: controller.save_performance_to_csv('simbatch8_trimmed_est', run_id='complete_MPC')
    ''''''
    
    # 
    ''' PLOTTING '''
    if not SAVE_TO_CSV: plotter.save_ocp_plots()
    if not SAVE_TO_CSV: plotter.plot_financial_report()

    # plotter.plot_spot_mfrr_prices(fontsize = 12) 
    # plotter.plot_activations() 
    # plotter.plot_CM_data() 
    # plotter.plot_simbatch('simbatch7_relative_newobj_exact')

    # plotter.plot_preliminary_analysis_geodata()

    ''''''

    # controller.export_intensity_to_json('abs_applied')


def repeat_main(surpress_output):
    start_date = datetime.strptime("2024-04-01", "%Y-%m-%d")
    # start_date = datetime.strptime("2025-01-13", "%Y-%m-%d")
    end_date = datetime.strptime("2024-12-26", "%Y-%m-%d")
    # end_date = datetime.strptime("2025-02-01", "%Y-%m-%d")
    # end_date = datetime.strptime("2024-04-28", "%Y-%m-%d")
    delta = timedelta(days=7)
    bidding_zones = ["NO2", "SE1", "DK1", "FI"]
    # bidding_zones = ["SE1", "DK1", "FI"]
    # bidding_zones = ["NO2", "SE1"]

    globals()["SURPRESS_OUTPUT"]    = surpress_output
    globals()['SAVE_TO_CSV']        = True

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        sim_suffix = current_date.strftime("%Y_%m_%d")
        
        for zone in bidding_zones:
            sim_name = f"Batch8_trimmedprices_MPC_7day_{sim_suffix}_{zone}_est_actrate025"
            
            # Set globals (you could refactor main() to accept arguments instead)
            globals()["SIM_NAME"] = sim_name
            globals()["SIMULATION_DATE"] = date_str
            globals()["BIDDING_ZONE"] = zone
            
            print(f"\n=== Running Simulation: {sim_name} ===")
            main()
        
        current_date += delta


if __name__ == "__main__":
    # main()
    repeat_main(surpress_output=True)
