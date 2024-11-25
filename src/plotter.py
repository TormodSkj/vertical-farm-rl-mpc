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
            
            # Filter out the unreasonably high low bid activations
            activation_th = 1e-6
            filtered_b_c_up = np.where(b_a_up > activation_th, b_c_up, 0).flatten()
            filtered_b_c_dn = np.where(b_a_dn > activation_th, b_c_dn, 0).flatten()

            # volume_th = 1e-3
            # filtered_b_p_up = b_p_up[np.where(b_p_up > volume_th)]
            # filtered_b_p_dn = b_p_dn[np.where(b_p_dn > volume_th)]
            # filtered_t_up   = t[np.where(b_p_up > volume_th)]
            # filtered_t_dn   = t[np.where(b_p_dn > volume_th)]
        

        # BEGIN PLOTTING (or more like saving plots, but you get it)
        ##################################################
        
        if self.controller.model.title == "Vertical Farm":
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

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=config.plot_format, sharex=True)

            # plt.step(t, b_p_up, label="Bidding volume up") 
            # plt.step(t, b_p_dn, label="Bidding volume down")
            # Bidding price up
            ax1.step(t, b_p_up, label="Bidding volume up", color="blue")
            ax1.set_ylabel("Power (MW)")
            ax1.legend(loc="upper right")

            # Bidding price down
            ax2.step(t, b_p_dn, label="Bidding volume down", color="red")
            ax2.set_ylabel("Power (MW)")
            ax2.set_xlabel("Time")
            ax2.legend(loc="upper right")

            # Bidding price max
            # ax3.step(t, np.maximum(b_p_dn, b_p_up), label="Max bidding volume", color="orange")
            ax3.fill_between(t, 0, np.where(b_p_up > b_p_dn, b_p_up, 0), color='blue', alpha=0.3, label='Up-regulation', step='post')
            ax3.fill_between(t, 0, np.where(b_p_dn > b_p_up, b_p_dn, 0), color='red', alpha=0.3, label='down-regulation', step='post')
            ax3.set_ylabel("Power (MW)")
            ax3.set_xlabel("Time")
            ax3.legend(loc="upper right")

            filename = "bidding_volume"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

            ##################################################
            plt.figure(4)

            # Create a figure with two subplots sharing the same x-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)

            # Plot for Bidding Price Up
            ax1.step(t, filtered_b_c_up, label="Bidding Price Up", color="blue", where='mid')
            ax1.set_ylabel("Bidding Price Up (€/MW)", color="blue")
            ax1.tick_params(axis='y', labelcolor="blue")
            ax1.legend(loc="upper left")

            # Twin y-axis for Spot Price on the first subplot
            ax1_twin = ax1.twinx()
            ax1_twin.step(t, p_spot, label="Spot Price", color="grey", linestyle="--", where='mid')
            ax1_twin.set_ylabel("Spot Price (NOK/kW)", color="grey")
            ax1_twin.tick_params(axis='y', labelcolor="grey")
            ax1_twin.legend(loc="upper right")

            # Plot for Bidding Price Down
            ax2.step(t, filtered_b_c_dn, label="Bidding Price Down", color="red", where='mid')
            ax2.set_ylabel("Bidding Price Down (€/MW)", color="red")
            ax2.tick_params(axis='y', labelcolor="red")
            ax2.legend(loc="upper left")

            # Twin y-axis for Spot Price on the second subplot
            ax2_twin = ax2.twinx()
            ax2_twin.step(t, p_spot, label="Spot Price", color="grey", linestyle="--", where='mid')
            ax2_twin.set_ylabel("Spot Price (NOK/kW)", color="grey")
            ax2_twin.tick_params(axis='y', labelcolor="grey")
            ax2_twin.legend(loc="upper right")

            # Set the common x-axis label and title
            ax2.set_xlabel("Time")
            fig.suptitle("Bidding Prices and Spot Prices in €/MW")

            # Adjust layout to avoid overlap
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leaves space for the title

            # Save the plot
            filename = "bidding_prices"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

            ##################################################
            plt.figure(6, figsize=config.plot_format)
            plt.step(t, b_a_up, label="Up-activation")
            plt.step(t, b_a_dn, label="Down-activation")
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
