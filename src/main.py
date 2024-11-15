from controller import Controller
from plant import PlantModel
from market import Market
from config import Config
from plotter import Plotter
import numpy as np
from globals import *
from simulator import Simulator


SIM_NAME = "casadi_ocp"
HORIZON_DAYS = 1
FINAL_WEIGHT = 8

SIMULATION_DATE = '2023-12-01'
BIDDING_ZONE    = 'NO4'

def main():

    x_init = np.array([5, 1])   # Specify init vector [structural and non structural dry weight in grams]

    config = Config(SIM_NAME)
    plant = PlantModel(x_init, FINAL_WEIGHT)
    market = Market(HORIZON_DAYS, 1133, BIDDING_ZONE, SIMULATION_DATE)
    controller = Controller(HORIZON_DAYS, plant, market, config, "opt")     #Baseline: 'opt' / 'rigid'
    simulator = Simulator(HORIZON_DAYS, plant, market, config, controller)
    plotter = Plotter(config, controller, simulator)
    
    simulator.Simulate_mpc()

    plotter.save_mpc_plots()

    # controller.rigid_baseline()
    # controller.optimize_baseline()
    # controller.optimize()
    # controller.save_to_json()

    # plotter.save_ocp_plots()


if __name__ == "__main__":
    main()
