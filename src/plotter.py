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

        t = self.controller.t
        u_opt = self.controller.u_bid
        u_base = self.controller.u_base

        x_bid = self.controller.x_bid
        x_base = self.controller.x_base

        B_bid = self.controller.B_bid
        b_p_up = B_bid[0,:]
        b_p_dn = B_bid[1,:]
        b_c_up = B_bid[2,:]
        b_c_dn = B_bid[3,:]
        b_a_up = self.controller.market.Pr_a_up(b_c_up)
        b_a_dn = self.controller.market.Pr_a_dn(b_c_dn)


        # BEGIN PLOTTING (or more like saving plots, but you get it)
        ##################################################
        
        if self.controller.model.title == "Vertical Farm":
            plt.figure(1)
            plt.plot(t, controller.model.freshweight(x_bid[:,1:]), "g", label="Freshweight shoot (g/plant)")
            plt.plot(t, controller.model.freshweight(x_base[:,1:]), color='purple', linestyle=':', label="Baseline freshweight shoot (g/plant)")
            plt.axhline(y=controller.model.Final_fw_sht, color='gray', linestyle=':', label="Required Freshweight (g/plant)")
            plt.ylabel("Weight (g/plant)")
            plt.xlabel("Time (days)")
            plt.legend()

        if self.controller.model.title == "Battery":
            plt.figure(1)
            plt.plot(t, x_bid[:,1:].flatten(), label="Bidding state of charge")
            plt.plot(t, x_base[:,1:].flatten(), label="Baseline state of charge")
            plt.ylabel(self.controller.model.x_unit)
            plt.xlabel("Time (days)")
            plt.legend()

        filename = "Combined_ocp_x"
        plt.savefig(config.plot_path + foldername + "/" + filename + ".png")
        
        ##################################################
        plt.figure(2)
        plt.plot(t, u_opt, label="U") 
        plt.plot(t, u_base, label="Baseline U") 
        plt.ylabel(self.controller.model.u_unit)
        plt.xlabel("Time (days)")
        plt.legend()


        filename = "Combined_ocp_u"
        plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

        ##################################################
        plt.figure(3)
        plt.plot(t, b_p_up, label="Bidding volume up") 
        plt.plot(t, b_p_dn, label="Bidding volume down")
        plt.ylabel("Bidding volume (MW)")
        plt.xlabel("Time (days)")
        plt.legend()

        filename = "Combined_ocp_b_p"
        plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

        ##################################################
        plt.figure(4)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

        # Bidding price up
        ax1.plot(t, b_c_up, label="Bidding price up", color="blue")
        ax1.set_ylabel("Bidding Price Up")
        ax1.legend(loc="upper right")

        # Bidding price down
        ax2.plot(t, b_c_dn, label="Bidding price down", color="red")
        ax2.plot(t, market.get_spotprice(), label="Spot price", color="blue")
        ax2.set_ylabel("Bidding Price Down")
        ax2.set_xlabel("Time")
        ax2.legend(loc="upper right")

        fig.tight_layout()
        filename = "Combined_ocp_b_c"
        plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

        ##################################################
        plt.figure(6)
        plt.plot(t, b_a_up, label="Up-activation")
        plt.plot(t, b_a_dn, label="Down-activation")
        plt.ylabel("Probability of activations")
        plt.xlabel("Time (days)")
        plt.legend()

        filename = "Combined_ocp_b_a"
        plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

        ##################################################
        plt.figure(7)
        plt.plot(t, market.get_spotprice(), label="Spot price")
        plt.ylabel("Spot price (kr/kWh)")
        plt.xlabel("Time (days)")
        plt.legend()


        filename = "Combined_ocp_p_spot"
        plt.savefig(config.plot_path + foldername + "/" + filename + ".png")



    def save_mpc_plots(self):

        # Define these for simplicity # TODO remove bloat
        config = self.config
        controller = self.controller
        market = controller.market


        # Get plotting details
        foldername = self.foldername

        t = self.controller.t
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
        b_a_up = self.controller.market.Pr_a_up(b_c_up)
        b_a_dn = self.controller.market.Pr_a_dn(b_c_dn)        


        # BEGIN PLOTTING (or more like saving plots, but you get it)
        ##################################################
        plt.figure(1)
        plt.plot(t, x1_mpc, "r", label="Structural dry weight (g/m^2)") 
        plt.plot(t, x2_mpc, "b", label="Non-structural dry weight (g/m^2)")
        plt.plot(t, controller.model.freshweight(x_mpc[:,1:]), color='purple', label="MPC Freshweight shoot (g/plant)")
        plt.plot(t, controller.model.freshweight(x_bid[:,1:]), color='g', linestyle=':', label="Bidding freshweight shoot (g/plant)")
        plt.axhline(y=controller.model.Final_fw_sht, color='orange', linestyle=':', label="Required Freshweight (g/plant)")
        plt.ylabel("Weight")
        plt.xlabel("Time (days)")
        plt.legend()

        
        filename = "Combined_ocp_x"
        plt.savefig(config.plot_path + foldername + "/MPC_" + filename + ".png")
        ##################################################


        plt.figure(2)
        plt.plot(t, u_mpc, label="MPC U") 
        plt.plot(t, u_bid, linestyle=':', label="Bidding U") 
        plt.plot(t, u_base, linestyle=':', label="Baseline U") 
        plt.ylabel("Light level (PPFD)")
        plt.xlabel("Time (days)")
        plt.legend()


        filename = "Combined_ocp_u"
        plt.savefig(config.plot_path + foldername + "/MPC_" + filename + ".png")