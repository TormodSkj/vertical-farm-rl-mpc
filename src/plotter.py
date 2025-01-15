import matplotlib.pyplot as plt
from config import Config
from controller import Controller
from model import *
from market import Market
from simulator import Simulator
from utils import *
import scipy.stats as stats

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
        spot_prices  = self.controller.spot_prices


        if 'Bidding' in controller.runs['runs']:

            bid_ts  = self.controller.runs['runs']['Bidding']['timeseries']
            b_p_up  = bid_ts['P_up']
            b_p_dn  = bid_ts['P_dn']
            b_c_up  = bid_ts['C_up']
            b_c_dn  = bid_ts['C_dn']
            b_a_up  = np.array(self.controller.market.Pr_a_up(spot_prices, b_c_up)).flatten()
            b_a_dn  = np.array(self.controller.market.Pr_a_dn(spot_prices, b_c_dn)).flatten()
            
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
            plt.title(f"Expected freshweight of plant growth (g/plant). ({controller.market.date}, {controller.market.bidding_zone})")

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

        fig, axes = plt.subplots(n_runs+1, 1, figsize=config.plot_format, sharex=True)

        # if n_runs == 1:
        #     axes = [axes]

        for i, run_name in enumerate(controller.runs['runs']):
            ax = axes[i]
            u = controller.runs['runs'][run_name]['timeseries']['u']

            ax.step(t, u, label=f"U {run_name}") 
            ax.set_ylabel(self.controller.model.u_unit)
            ax.set_xlabel("Time (days)")
            ax.legend()

        ax = axes[-1]
        ax.step(t, spot_prices, label=f"Spot price") 
        ax.set_ylabel("NOK/kWh")
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
            ax.step(t, spot_prices*1000/self.controller.market.C_eur2nok, label="Spot price", color="grey", linestyle="--", where='mid')
            ax.set_ylabel("Bidding Price (€/MW)", color="blue")
            ax.tick_params(axis='y', labelcolor="blue")
            ax.legend(loc="upper left")

            fig.suptitle(f"Bidding Prices and Spot Prices in €/MW ({controller.market.date}, {controller.market.bidding_zone})")

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


            # ##################################################
            # plt.figure(8, figsize=config.plot_format)

            # freshweight_outcomes = generate_freshweight_outcomes(controller, u, b_p_up, b_p_dn, b_a_up, b_a_dn)

            # plt.plot(t, x_fw, label="Expected outcome", color='green')
            # plt.plot(t, freshweight_outcomes[0,1:], label="Constant Up-activation", color='blue')
            # plt.plot(t, freshweight_outcomes[1,1:], label="Constant Down-activation", color='red')
            # plt.ylabel("Fresh weight (g/plant)")
            # plt.xlabel("Time (days)")
            # plt.title("Edge cases of constant activation")
            # plt.legend()

            # filename = "freshweight_all_outcomes"
            # plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)


            ######################################################

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_format)

            ax1.scatter(b_p_up, b_a_up, color='blue', s=5)
            ax1.set_ylabel("Projected activation chance")
            ax1.set_xlabel("Bid volume (MW)")
            ax1.title.set_text('Up-regulation bids')

            ax2.scatter(b_p_dn, b_a_dn, color='red', s=5)
            ax2.set_ylabel("Projected activation chance")
            ax2.set_xlabel("Bid volume (MW)")
            ax2.title.set_text('Down-regulation bids')
            
            fig.suptitle(f"Bidding Prices and Spot Prices in €/MW ({controller.market.date}, {controller.market.bidding_zone})")

            # Adjust layout to avoid overlap
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leaves space for the title


            filename = "bidding_scatter_plots"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)


            ######################################################

            plt.figure(figsize=config.plot_format)
            
            u_bid = controller.runs['runs']['Bidding']['timeseries']['u']
            x_bid = controller.runs['runs']['Bidding']['timeseries']['x']
            # fw_interval = calculate_freshweight_interval(controller, u_bid, b_p_up, b_p_dn, b_c_up, b_c_dn)
            fw_variance = propagate_process_covariance(controller, x_bid, u_bid, b_p_up, b_p_dn, b_c_up, b_c_dn)
            fw_sd = np.sqrt(fw_variance)
            fw = np.array(controller.model.freshweight(x_bid[:,1:])).flatten()

            fw_ub = fw + 1.96*fw_sd
            fw_lb = fw - 1.96*fw_sd

            # plt.fill_between(t, fw-3*fw_sd, fw+3*fw_sd, color='green', alpha=0.2)
            # plt.fill_between(t, fw-2*fw_sd, fw+2*fw_sd, color='green', alpha=0.2)
            plt.fill_between(t, fw_lb, fw_ub, color='green', alpha=0.6)
            plt.axhline(y=controller.model.Final_fw_sht, color='gray', linestyle=':', label="Required Freshweight (g/plant)")
            plt.ylabel("Fresh weight (g/plant)")
            plt.xlabel("Time (days)")
            plt.title('95% confidence interval of freshweight throughout one growth cycle')

            filename = "freshweight_variance"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)


            #################################################################################
            # Bidding volumes and activation chances


            # filter out improbable activations

            activation_threshold = 1e-2
            volume_threshold = 1e-3

            b_a_up_filtered = np.where(np.logical_and(b_a_up > activation_threshold, b_p_up > volume_threshold,), b_a_up, 0)
            b_a_dn_filtered = np.where(np.logical_and(b_a_dn > activation_threshold, b_p_dn > volume_threshold,), b_a_dn, 0)
            b_p_up_filtered = np.where(np.logical_and(b_a_up > activation_threshold, b_p_up > volume_threshold,), b_p_up, 0)
            b_p_dn_filtered = np.where(np.logical_and(b_a_dn > activation_threshold, b_p_dn > volume_threshold,), b_p_dn, 0)

            plt.figure(3, figsize=config.plot_format)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)

            ax1.fill_between(t, -b_p_up_filtered, 0, color='blue', alpha=0.4, label='Up-regulation', step='post')
            ax1.fill_between(t, 0, b_p_dn_filtered, color='red', alpha=0.4, label='down-regulation', step='post')
            ax1.set_ylabel("Power (MW)")
            ax1.set_xlabel("Time")
            ax1.legend(loc="upper right")

            ax2.fill_between(t, 0, b_a_up_filtered, color='blue', label="Up-activation", alpha=0.4)
            ax2.fill_between(t, 0, b_a_dn_filtered, color='red', label="Down-activation", alpha=0.4)
            ax2.set_ylabel("Probability of activations")
            ax2.set_xlabel("Time (days)")
            ax2.legend()


            fig.suptitle(f"Bidding volumes and predicted activation chances. ({controller.market.date}, {controller.market.bidding_zone})")

            filename = "results_bidding_vol_act"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)


        ##################################################
        plt.figure(figsize=config.plot_format)
        plt.step(t, market.get_spotprice(), label="Spot price")
        plt.ylabel("Spot price (kr/kWh)")
        plt.xlabel("Time (days)")
        plt.legend()


        filename = "spot_price"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)



    def plot_random_activations(self, m: int):

        simulator = self.simulator
        controller = self.controller
        market = controller.market
        config = self.config
        foldername = self.foldername


        t = controller.t
        freshwewights = simulator.simulate_random_activation(self.controller, m)


        ##################################################
        plt.figure(31, figsize=config.plot_format)

        for case in range(freshwewights.shape[0]):
            plt.plot(t, freshwewights[case,:], color='green', alpha=0.2)
        plt.axhline(y=controller.model.Final_fw_sht, color='gray', linestyle=':', label="Required Freshweight (g/plant)")
        plt.ylabel("Fresh weight (g/plant)")
        plt.xlabel("Time (days)")
        plt.title(f'Simulated {m} different cases of plausible activations')
        # plt.legend()


        filename = "random_activations"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)






    def plot_spot_mfrr_prices(self):

        config = self.config
        controller = self.controller
        market = controller.market
        foldername = self.foldername

        timestamps = market.timestamps
        spot_prices = market.spot_price_data
        mfrr_prices_up = market.mfrr_prices_up
        mfrr_prices_dn = market.mfrr_prices_dn
        spot_prices_eur = spot_prices*1000/market.C_eur2nok

        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='same')

        # Smoothed data
        window_length = 24*3
        mfrr_prices_up_smoothed = moving_average(mfrr_prices_up, window_length)
        mfrr_prices_dn_smoothed = moving_average(mfrr_prices_dn, window_length)
        spot_prices_eur_smoothed = moving_average(spot_prices_eur, window_length)

        opacity = 0.2
        linewidth=1.5

        line_x = np.array([min(spot_prices), max(spot_prices)])

        plt.figure(20, figsize=config.plot_format)

        # plt.step(timestamps, spot_prices*1000/market.C_eur2nok, label="Spot price")
        plt.step(timestamps, mfrr_prices_up, label="Clearing price up", color='blue', alpha=opacity)
        plt.step(timestamps, mfrr_prices_up_smoothed, label="Clearing price up smoothed", color='blue', alpha=1, linewidth = linewidth)

        plt.step(timestamps, mfrr_prices_dn, label="Clearing price down", color='red', alpha=opacity)
        plt.step(timestamps, mfrr_prices_dn_smoothed, label="Clearing price down smoothed", color='red', alpha=1, linewidth = linewidth)

        # plt.fill_between(timestamps, mfrr_prices_up, max(mfrr_prices_up), label="Clearing price up", color='blue', alpha=0.4)
        # plt.fill_between(timestamps, min(mfrr_prices_dn), mfrr_prices_dn, label="Clearing price down", color='red', alpha=0.4)
        plt.step(timestamps, spot_prices_eur, label="Spot price", color='grey', alpha=opacity)
        plt.step(timestamps, spot_prices_eur_smoothed, label="Spot price smoothed", color='grey', alpha=1, linewidth = linewidth)
        plt.ylabel("Price (€/MW)")
        plt.xlabel("Time (days)")
        plt.title(f"Spot price vs activation prices smoothed using {window_length}h moving average")
        plt.legend()


        filename = "smoothed_prices_NO1_2023"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)


         ###########################################################
        plt.figure(21, figsize=config.plot_format)

        # plt.step(timestamps, spot_prices_eur, label="Spot price", color='grey', linestyle=':')
        plt.step(timestamps, mfrr_prices_up - spot_prices_eur, label="Clearing price up", color='blue')
        plt.step(timestamps, mfrr_prices_dn - spot_prices_eur, label="Clearing price down", color='red')
        plt.plot(timestamps, 0 * mfrr_prices_dn, label="Zero-line", color='grey', alpha=0.5)
        plt.ylabel("Price (€/MW)")
        plt.xlabel("Time (days)")
        plt.title("Clearing prices relative to spot price (€/MW)")
        plt.legend()

        filename = "relative_prices_2023_NO1"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)


        ###########################################################
        plt.figure(22, figsize=config.plot_format)

        n_bins = 100
        N_data_points = len(mfrr_prices_up)

        x_up = np.linspace(market.mu_up-3*market.sigma_up,market.mu_up+3*market.sigma_up, 1000)
        x_dn = np.linspace(market.mu_dn-3*market.sigma_dn,market.mu_dn+3*market.sigma_dn, 1000)
        # norm_fac_up = N_data_points*market.sigma_up
        # norm_fac_dn = N_data_points*market.sigma_dn

        # plt.step(timestamps, spot_prices_eur, label="Spot price", color='grey', linestyle=':')
        plt.hist(mfrr_prices_up, label="Clearing price up", color='blue', alpha=0.4, bins=n_bins, density=True)
        plt.hist(mfrr_prices_dn, label="Clearing price down", color='red', alpha=0.4, bins=n_bins, density=True)
        plt.hist(spot_prices_eur, label="Spot prices", color='grey', alpha=0.4, bins=n_bins, density=True)
        plt.plot(x_up, stats.norm.pdf(x_up, market.mu_up, market.sigma_up), label="Estimated Up-price distribution", color='blue')
        plt.plot(x_dn, stats.norm.pdf(x_dn, market.mu_dn, market.sigma_dn), label="Estimated Down-price distribution", color='red')
        # plt.ylabel("")
        plt.xlabel("Bidding prices (€/MW)")
        plt.title("Clearing prices histogram normalized")
        plt.legend()

        filename = "histogram_mfrr_clearing_prices_2023_NO1"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)

        ###########################################################
        plt.figure(24, figsize=config.plot_format)

        x_up = np.linspace(market.mu_up-3*market.sigma_up,market.mu_up+3*market.sigma_up, 1000)
        x_dn = np.linspace(market.mu_dn-3*market.sigma_dn,market.mu_dn+3*market.sigma_dn, 1000)

        rand_prices_up = generate_samples_from_cdf(mfrr_prices_up, 8668, config.seed)
        rand_prices_dn = generate_samples_from_cdf(mfrr_prices_dn, 8668, config.seed)

        plt.hist(rand_prices_up, label="Clearing price up", color='blue', alpha=0.4, bins=n_bins, density=True)
        plt.hist(rand_prices_dn, label="Clearing price down", color='red', alpha=0.4, bins=n_bins, density=True)
        plt.plot(x_up, stats.norm.pdf(x_up, market.mu_up, market.sigma_up), label="Estimated Up-price distribution", color='blue')
        plt.plot(x_dn, stats.norm.pdf(x_dn, market.mu_dn, market.sigma_dn), label="Estimated Down-price distribution", color='red')
        # plt.ylabel("")
        plt.xlabel("Bidding prices (€/MW)")
        plt.title("Randomly generated prices normalized")
        plt.legend()

        filename = "rand_hist_clearing_prices_2023_NO1"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)


        ################################################################
        plt.figure(23, figsize=config.plot_format)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_format, sharex=True)

        n = int(len(spot_prices)/10)
        idx = np.ceil(np.linspace(1,len(spot_prices)-1,n))
        ax1.scatter(spot_prices[idx], mfrr_prices_up[idx], color='blue', label='Clearing price up', s=0.1)
        ax1.plot(line_x, line_x * 0.78*1000/market.C_eur2nok, label='Lower limit: 0.78 x spot', color='grey', alpha=0.4)
        ax1.set_ylabel("Bidding price (€/MWh)")
        ax1.set_xlabel("Spot price (NOK/kWh)")
        ax1.set_xlim([-0.5, 4.5])
        ax1.legend()

        ax2.scatter(spot_prices[idx], mfrr_prices_dn[idx], color='red', label='Clearing proce down', s=0.1)
        ax2.plot(line_x, line_x * 0.9*1000/market.C_eur2nok, label='Upper limit: 0.9 x spot', color='grey', alpha=0.4)
        ax2.set_ylabel("Bidding price (€/MWh)")
        ax2.set_xlabel("Spot price (NOK/kWh)")
        ax2.set_xlim([-0.5, 4.5])
        ax2.legend()


        fig.suptitle("Spot prices with mFRR clearing prices €/MW")

        filename = "scatter_spot_clearing_prices_2023_NO1"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)



        ###########################################################
        plt.figure(figsize=config.plot_format)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)


        ax1.hist(mfrr_prices_up-spot_prices_eur, label="Clearing price up", color='blue', alpha=0.4, bins=2*n_bins, density=True)
        ax1.set_xlabel("Bidding prices (€/MW)")
        ax1.set_xlim([-100, 100])
        ax1.legend()


        ax2.hist(mfrr_prices_dn-spot_prices_eur, label="Clearing price down", color='red', alpha=0.4, bins=n_bins, density=True)
        ax2.set_xlabel("Bidding prices (€/MW)")
        ax2.set_xlim([-100, 100])
        ax2.legend()

        plt.suptitle('Relative clearing prices, normalized')

        filename = "histogram_relative_clearing_prices"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)

       


    def save_mpc_plots(self):

        # Define these for simplicity # TODO remove bloat
        config = self.config
        controller = self.controller
        # market = controller.market
        simulator = self.simulator


        # Get plotting details
        foldername = self.foldername

        t = self.controller.t
        spot_prices = self.controller.spot_prices
        # u_bid = self.controller.u_bid.flatten()
        # u_base = self.controller.u_base.flatten()
        u_rigid = simulator.mpc_controller.runs['runs']['Rigid']['timeseries']['u'].flatten()
        x_rigid = simulator.mpc_controller.runs['runs']['Rigid']['timeseries']['x']
        u_mpc = self.simulator.u_mpc.flatten()

        # x_bid = self.simulator.x_bid
        # x1_bid = x_bid[0,1:]
        # x2_bid = x_bid[1,1:]
        # x_base = self.controller.x_base
        # x1_base = x_base[0,1:]
        # x2_base = x_base[1,1:]

        x_mpc = self.simulator.x_mpc
        x1_mpc = x_mpc[0,1:]
        x2_mpc = x_mpc[1,1:]


        # B_bid = self.simulator.bids_mpc
        # b_p_up = B_bid[0,:]
        # b_p_dn = B_bid[1,:]
        # b_c_up = B_bid[2,:]
        # b_c_dn = B_bid[3,:]
        # b_a_up = self.controller.market.Pr_a_up(spot_prices, b_c_up)
        # b_a_dn = self.controller.market.Pr_a_dn(spot_prices, b_c_dn)        


        # BEGIN PLOTTING (or more like saving plots, but you get it)
        ##################################################
        plt.figure(1, figsize=config.plot_format)
        # plt.step(t, x1_mpc, "r", label="Structural dry weight (g/m^2)") 
        # plt.step(t, x2_mpc, "b", label="Non-structural dry weight (g/m^2)")
        plt.step(t, self.simulator.model.freshweight(x_mpc), "g", label="MPC Fresh weight (g/plant)")
        plt.step(t, self.simulator.model.freshweight(x_rigid[:,1:]), "orange", label="Rigid Fresh weight (g/plant)")
        plt.axhline(y=self.simulator.model.Final_fw_sht, color='gray', linestyle=':', label="Required Freshweight (g/plant)")
        # plt.step(t, controller.model.freshweight(x_mpc[:,1:]), color='purple', label="MPC Freshweight shoot (g/plant)")
        # plt.step(t, controller.model.freshweight(x_bid[:,1:]), color='g', linestyle=':', label="Bidding freshweight shoot (g/plant)")
        # plt.axhline(y=controller.model.Final_fw_sht, color='orange', linestyle=':', label="Required Freshweight (g/plant)")
        plt.ylabel("Weight")
        plt.xlabel("Time (days)")
        plt.legend()

        
        filename = "Combined_ocp_x"
        plt.savefig(config.plot_path + foldername + "/MPC_" + filename + "." + config.plot_file_type, format=config.plot_file_type)
        ##################################################


        plt.figure(2, figsize=config.plot_format)
        plt.step(t, u_mpc, label="MPC U") 
        plt.step(t, u_rigid, label="Rigid U") 
        # plt.step(t, u_bid, linestyle=':', label="Bidding U") 
        # plt.step(t, u_base, linestyle=':', label="Baseline U") 
        plt.ylabel("Light level (PPFD)")
        plt.xlabel("Time (days)")
        plt.legend()


        filename = "Combined_ocp_u"
        plt.savefig(config.plot_path + foldername + "/MPC_" + filename + "." + config.plot_file_type, format=config.plot_file_type)




    def plot_price_prediction(self):

        # Define these for simplicity # TODO remove bloat
        config = self.config
        controller = self.controller
        market = controller.market

        # Get plotting details
        foldername = self.foldername

        n_runs = len(list(self.controller.runs['runs'].keys()))


        t       = self.controller.t
        spot_prices  = self.controller.spot_prices


        if 'Bidding' in controller.runs['runs']:

            bid_ts  = self.controller.runs['runs']['Bidding']['timeseries']
            b_p_up  = bid_ts['P_up']
            b_p_dn  = bid_ts['P_dn']
            b_c_up  = bid_ts['C_up']
            b_c_dn  = bid_ts['C_dn']
            b_a_up  = np.array(self.controller.market.Pr_a_up(spot_prices, b_c_up)).flatten()
            b_a_dn  = np.array(self.controller.market.Pr_a_dn(spot_prices, b_c_dn)).flatten()
            

            pred_prices_up = market.opt_prices_up.flatten()
            pred_prices_dn = market.opt_prices_dn.flatten()

            pred_a_up = np.array(self.controller.market.Pr_a_up(spot_prices, pred_prices_up)).flatten()
            pred_a_dn  = np.array(self.controller.market.Pr_a_dn(spot_prices, pred_prices_dn)).flatten()

            # Filter out the unreasonably low bid activations
            activation_th = 0.01
            volume_th = 1e-3
            filtered_b_c_up = np.where(np.logical_and(b_a_up > activation_th, b_p_up>volume_th), b_c_up, 0).flatten()
            filtered_b_c_dn = np.where(np.logical_and(b_a_dn > activation_th, b_p_dn>volume_th), b_c_dn, 0).flatten()
            filtered_b_a_up = np.where(np.logical_and(b_a_up > activation_th, b_p_up>volume_th), b_a_up, 0).flatten()
            filtered_b_a_dn = np.where(np.logical_and(b_a_dn > activation_th, b_p_dn>volume_th), b_a_dn, 0).flatten()

            filtered_pred_prices_up = np.where(pred_a_up > activation_th, pred_prices_up, 0).flatten()
            filtered_pred_prices_dn = np.where(pred_a_dn > activation_th, pred_prices_dn, 0).flatten()
            filtered_pred_a_up = np.where(pred_a_up > activation_th, pred_a_up, 0).flatten()
            filtered_pred_a_dn = np.where(pred_a_dn > activation_th, pred_a_dn, 0).flatten()

            ##################################################
            # plt.figure(figsize=config.plot_format)

            # Create a figure with two subplots sharing the same x-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)

            # Plot for Bidding Price Up
            ax1.fill_between(t, 0, filtered_b_c_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax1.fill_between(t, 0,filtered_b_c_dn, label="Bidding Price Down", color="red", step='post', alpha=0.4)
            ax1.step(t, spot_prices*1000/self.controller.market.C_eur2nok, label="Spot price", color="grey", linestyle="--", where='mid')
            ax1.set_ylabel("Optimal Bidding Prices (€/MW)")
            ax1.tick_params(axis='y')
            ax1.legend(loc="upper left")

            # Plot for Bidding Price Up
            ax2.fill_between(t, 0, filtered_pred_prices_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax2.fill_between(t, 0,filtered_pred_prices_dn, label="Bidding Price Down", color="red", step='post', alpha=0.4)
            ax2.step(t, spot_prices*1000/self.controller.market.C_eur2nok, label="Spot price", color="grey", linestyle="--", where='mid')
            ax2.set_ylabel("Predicted Bidding Prices (€/MW)")
            ax2.tick_params(axis='y')
            ax2.legend(loc="upper left")

            fig.suptitle(f"Actual bidding prices vs predicted optimal bidding prices. ({controller.market.date}, {controller.market.bidding_zone})")

            # Adjust layout to avoid overlap
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leaves space for the title

            # Save the plot
            filename = "predicted_vs_actual_bidding_prices"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)





            ##################################################

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)

            # Plot for Bidding Price Up
            ax1.fill_between(t, 0, filtered_b_a_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax1.fill_between(t, 0,filtered_b_a_dn, label="Bidding Price Down", color="red", step='post', alpha=0.4)
            ax1.set_ylabel("Optimal Activation Chances ")
            ax1.tick_params(axis='y')
            ax1.legend(loc="upper left")

            # Plot for Bidding Price Up
            ax2.fill_between(t, 0, filtered_pred_a_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax2.fill_between(t, 0,filtered_pred_a_dn, label="Bidding Price Down", color="red", step='post', alpha=0.4)
            ax2.set_ylabel("Predicted Optimal Activation Chances ")
            ax2.tick_params(axis='y')
            ax2.legend(loc="upper left")

            fig.suptitle(f"Actual optimal bid activation chances vs predicted optimal bid activation prices. ({controller.market.date}, {controller.market.bidding_zone})")

            # Adjust layout to avoid overlap
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leaves space for the title

            # Save the plot
            filename = "predicted_vs_actual_activation_chances"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)


            

