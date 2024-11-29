import matplotlib.pyplot as plt
from config import Config
from controller import Controller
from model import *
from market import Market
from simulator import Simulator

class Plotter():

    config: Config
    controller: Controller
    simulator: Simulator
    # model: PlantModel
    # market: Market
    foldername: str

    def __init__(self, config, controller, simulator):
        self.config = config
        self.controller = controller
        # self.model = plantModel
        # self.market = market
        self.simulator = simulator

        self.foldername = self.config.sim_name


    def save_ocp_plots(self):

        # Define these for simplicity # TODO remove bloat
        config = self.config
        controller = self.controller
        market = controller.market


        # Get plotting details
        foldername = self.foldername

        n_runs = len(list(self.controller.runs['runs'].keys()))


        t       = self.controller.t
        p_spot  = self.controller.p_spot


        if 'Bidding' in controller.runs['runs']:

            bid_ts  = self.controller.runs['runs']['Bidding']['timeseries']
            b_p_up  = bid_ts['P_up']
            b_p_dn  = bid_ts['P_dn']
            b_c_up  = bid_ts['C_up']
            b_c_dn  = bid_ts['C_dn']
            b_a_up  = np.array(self.controller.market.Pr_a_up(p_spot, b_c_up)).flatten()
            b_a_dn  = np.array(self.controller.market.Pr_a_dn(p_spot, b_c_dn)).flatten()
            
            # Filter out the unreasonably low bid activations
            activation_th = 0.01
            filtered_b_c_up = np.where(b_a_up > activation_th, b_c_up, 0).flatten()
            filtered_b_c_dn = np.where(b_a_dn > activation_th, b_c_dn, 0).flatten()

            # volume_th = 1e-3
            # filtered_b_p_up = b_p_up[np.where(b_p_up > volume_th)]
            # filtered_b_p_dn = b_p_dn[np.where(b_p_dn > volume_th)]
            # filtered_t_up   = t[np.where(b_p_up > volume_th)]
            # filtered_t_dn   = t[np.where(b_p_dn > volume_th)]
        

        # BEGIN PLOTTING (or more like saving plots, but you get it)
        ##################################################
        
        if self.controller.model.title == "Vertical Farm" or self.controller.model.title == "Gjermund plant model":
            plt.figure(1, figsize=config.plot_format)

            for run_name in controller.runs['runs']:
                x = controller.runs['runs'][run_name]['timeseries']['x']
                x_fw = controller.model.freshweight(x[:,1:])
                plt.plot(t, x_fw, label=f"Freshweight shoot {run_name} (g/plant)")

            # plt.plot(t, controller.model.freshweight(x_bid[:,1:]), "g", label="Freshweight shoot (g/plant)")
            # plt.plot(t, controller.model.freshweight(x_base[:,1:]), color='purple', linestyle=':', label="Baseline freshweight shoot (g/plant)")
            plt.axhline(y=controller.model.Final_fw_sht, color='gray', linestyle=':', label="Required Freshweight (g/plant)")
            plt.ylabel("Weight (g/plant)")
            plt.xlabel("Time (days)")
            plt.legend()

        if self.controller.model.title == "Battery":
            plt.figure(1, figsize=config.plot_format)

            for run_name in controller.runs['runs']:
                x = controller.runs['runs'][run_name]['timeseries']['x']
                x_fw = controller.model.freshweight(x[:,1:])
                plt.plot(t, x_fw, "g", label=f"SOC {run_name}")
            # plt.step(t, x_bid[:,1:].flatten(), label="Bidding state of charge")
            # plt.step(t, x_base[:,1:].flatten(), label="Baseline state of charge")
            plt.ylabel(self.controller.model.x_unit)
            plt.xlabel("Time (days)")
            plt.legend()

        filename = "fresh_weight"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)
        
        ##################################################
        plt.figure(2, figsize=config.plot_format)

        fig, axes = plt.subplots(n_runs, 1, figsize=config.plot_format, sharex=True)

        if n_runs == 1:
            axes = [axes]

        for i, run_name in enumerate(controller.runs['runs']):
            ax = axes[i]
            u = controller.runs['runs'][run_name]['timeseries']['u']

            ax.step(t, u, label=f"U {run_name}") 
            ax.set_ylabel(self.controller.model.u_unit)
            ax.set_xlabel("Time (days)")
            ax.legend()

        # ax2.step(t, u_base, label="Baseline U") 
        # ax2.set_ylabel(self.controller.model.u_unit)
        # ax2.set_xlabel("Time (days)")
        # ax2.legend()

        # ax3.step(t, u_base-u_bid, label="Difference") 
        # ax3.set_ylabel(self.controller.model.u_unit)
        # ax3.set_xlabel("Time (days)")
        # ax3.legend()

        filename = "light_schedule"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)



        if 'Bidding' in controller.runs['runs']:
            ##################################################
            plt.figure(3, figsize=config.plot_format)
            fig, ax = plt.subplots(1, 1, figsize=config.plot_format, sharex=True)

            lb_B, ub_B = self.controller.model.get_bidding_bounds(self.controller)

            # ax.step(t, -ub_B[0,:], color='grey', label='Up-regulation volume limit')
            # ax.step(t, ub_B[1,:], color='grey', label='Down-regulation volume limit')
            ax.fill_between(t, -b_p_up, 0, color='blue', alpha=0.4, label='Up-regulation', step='post')
            ax.fill_between(t, 0, b_p_dn, color='red', alpha=0.4, label='down-regulation', step='post')
            ax.set_ylabel("Power (MW)")
            ax.set_xlabel("Time")
            ax.legend(loc="upper right")

            filename = "bidding_volume"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

            ##################################################
            plt.figure(4)

            # Create a figure with two subplots sharing the same x-axis
            fig, ax = plt.subplots(1, 1, figsize=config.plot_format, sharex=True)

            # Plot for Bidding Price Up
            ax.fill_between(t, 0, filtered_b_c_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax.fill_between(t, 0,filtered_b_c_dn, label="Bidding Price Down", color="red", step='post', alpha=0.4)
            ax.step(t, p_spot*1000/self.controller.market.C_eur2nok, label="Spot price", color="grey", linestyle="--", where='mid')
            ax.set_ylabel("Bidding Price (€/MW)", color="blue")
            ax.tick_params(axis='y', labelcolor="blue")
            ax.legend(loc="upper left")

            fig.suptitle("Bidding Prices and Spot Prices in €/MW")

            # Adjust layout to avoid overlap
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leaves space for the title

            # Save the plot
            filename = "bidding_prices"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

            ##################################################
            plt.figure(6, figsize=config.plot_format)
            plt.fill_between(t, 0, b_a_up, color='blue', label="Up-activation", alpha=0.4)
            plt.fill_between(t, 0, b_a_dn, color='red', label="Down-activation", alpha=0.4)
            # plt.step(t, b_a_up, label="Up-activation")
            # plt.step(t, b_a_dn, label="Down-activation")
            plt.ylabel("Probability of activations")
            plt.xlabel("Time (days)")
            plt.legend()

            filename = "bidding_activations"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

        ##################################################
        plt.figure(7, figsize=config.plot_format)
        plt.step(t, market.get_spotprice(), label="Spot price")
        plt.ylabel("Spot price (kr/kWh)")
        plt.xlabel("Time (days)")
        plt.legend()


        filename = "spot_price"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)



    def save_mpc_plots(self):

        # Define these for simplicity # TODO remove bloat
        config = self.config
        controller = self.controller
        market = controller.market


        # Get plotting details
        foldername = self.foldername

        t = self.controller.t
        p_spot = self.controller.p_spot
        u_bid = self.controller.u_bid.flatten()
        u_base = self.controller.u_base.flatten()
        u_mpc = self.simulator.u_mpc.flatten()

        x_bid = self.simulator.x_bid
        x1_bid = x_bid[0,1:]
        x2_bid = x_bid[1,1:]
        # x_base = self.controller.x_base
        # x1_base = x_base[0,1:]
        # x2_base = x_base[1,1:]

        x_mpc = self.simulator.x_mpc
        x1_mpc = x_mpc[0,1:]
        x2_mpc = x_mpc[1,1:]


        B_bid = self.simulator.bids_mpc
        b_p_up = B_bid[0,:]
        b_p_dn = B_bid[1,:]
        b_c_up = B_bid[2,:]
        b_c_dn = B_bid[3,:]
        b_a_up = self.controller.market.Pr_a_up(p_spot, b_c_up)
        b_a_dn = self.controller.market.Pr_a_dn(p_spot, b_c_dn)        


        # BEGIN PLOTTING (or more like saving plots, but you get it)
        ##################################################
        plt.figure(1, figsize=config.plot_format)
        plt.step(t, x1_mpc, "r", label="Structural dry weight (g/m^2)") 
        plt.step(t, x2_mpc, "b", label="Non-structural dry weight (g/m^2)")
        plt.step(t, controller.model.freshweight(x_mpc[:,1:]), color='purple', label="MPC Freshweight shoot (g/plant)")
        plt.step(t, controller.model.freshweight(x_bid[:,1:]), color='g', linestyle=':', label="Bidding freshweight shoot (g/plant)")
        plt.axhline(y=controller.model.Final_fw_sht, color='orange', linestyle=':', label="Required Freshweight (g/plant)")
        plt.ylabel("Weight")
        plt.xlabel("Time (days)")
        plt.legend()

        
        filename = "Combined_ocp_x"
        plt.savefig(config.plot_path + foldername + "/MPC_" + filename + "." + config.plot_file_type, format=config.plot_file_type)
        ##################################################


        plt.figure(2, figsize=config.plot_format)
        plt.step(t, u_mpc, label="MPC U") 
        plt.step(t, u_bid, linestyle=':', label="Bidding U") 
        plt.step(t, u_base, linestyle=':', label="Baseline U") 
        plt.ylabel("Light level (PPFD)")
        plt.xlabel("Time (days)")
        plt.legend()


        filename = "Combined_ocp_u"
        plt.savefig(config.plot_path + foldername + "/MPC_" + filename + "." + config.plot_file_type, format=config.plot_file_type)
