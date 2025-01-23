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
    foldername: str

    color_up = 'skyblue'
    color_dn = 'lightcoral'

    def __init__(self, config, controller, simulator=None):
        self.config = config
        self.controller = controller
        self.simulator = simulator

        self.foldername = self.config.sim_name


    def save_ocp_plots(self):

        config      = self.config
        controller  = self.controller
        market      = controller.market
        foldername  = self.foldername
        n_runs      = len(list(self.controller.optimization_results['runs'].keys()))
        t           = self.controller.t
        spot_prices = self.controller.spot_prices

        bidding_runs = [run for run in controller.optimization_results['runs'] if 'bidding result' in controller.optimization_results['runs'][run]]

        # BEGIN PLOTTING

        ######################################################
        #                   FRESH WEIGHTS

        plt.figure(figsize=config.plot_format)
        

        for run_name in controller.optimization_results['runs']:

            if(run_name in bidding_runs and 'Realized' not in run_name):

                bid_ts  = self.controller.optimization_results['runs'][run_name]['timeseries']
                bid_volume_up   = bid_ts['P_up']    # Volume up
                bid_volume_dn   = bid_ts['P_dn']    # Volume down
                bid_price_up    = bid_ts['C_up']    # Price up
                bid_price_dn    = bid_ts['C_dn']    # Price down

                u_nom = controller.optimization_results['runs'][run_name]['timeseries']['u_nom'] # Unsure if this is right or if 'u' is more correct
                x_bid = controller.optimization_results['runs'][run_name]['timeseries']['x']
                fw_variance = propagate_process_covariance(controller, x_bid, u_nom, bid_volume_up, bid_volume_dn, bid_price_up, bid_price_dn)
                fw_sd = np.sqrt(fw_variance)
                fw = np.array(controller.model.freshweight(x_bid[:,1:])).flatten()
                fw_ub = fw + 1.96*fw_sd
                fw_lb = fw - 1.96*fw_sd

                plt.fill_between(t, fw_lb, fw_ub, color='green', alpha=0.2)

            x = controller.optimization_results['runs'][run_name]['timeseries']['x']
            x_fw = controller.model.freshweight(x[:,1:])
            plt.plot(t, x_fw, label=f"{run_name} (g/plant)")

        plt.axhline(y=controller.model.Final_fw_sht, color='gray', linestyle=':', label="Required Freshweight (g/plant)")
        plt.ylabel("Weight (g/plant)")
        plt.xlabel("Time (days)")
        plt.legend(loc="upper left")
        plt.title(f"Expected freshweight of plant growth (g/plant). ({controller.market.date}, {controller.market.bidding_zone})")

        filename = "fresh_weight"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)
        
        ######################################################
        #                   LIGHT SCHEDULE

        fig, axes = plt.subplots(n_runs+1, 1, figsize=config.plot_format, sharex=True)

        for i, run_name in enumerate(controller.optimization_results['runs']):
            ax = axes[i]
            u = controller.optimization_results['runs'][run_name]['timeseries']['u']
            ax.step(t, u, label=f"{run_name}") 
            ax.set_ylabel(self.controller.model.u_unit, rotation=0)
            ax.legend(loc="upper right")

        ax = axes[-1]
        ax.step(t, spot_prices, label=f"Spot price", color='gray') 
        ax.set_ylabel("NOK/kWh", rotation=0)
        ax.set_xlabel("Time (days)")
        ax.legend(loc="upper right")

        filename = "light_schedule"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

        if len(bidding_runs) > 0:
            run = bidding_runs[-1]

            bid_ts          = self.controller.optimization_results['runs'][run]['timeseries']
            if 'A_up' in bid_ts and 'A_dn' in bid_ts:
                is_realized = True                      # Bool used to check if there is real activation data to plot
            else:
                is_realized = False

            run_sanitized   = run.lower().replace(" ", "_")

            bid_volume_up   = bid_ts['P_up']    # Volume up
            bid_volume_dn   = bid_ts['P_dn']    # Volume down
            bid_price_up    = bid_ts['C_up']    # Price up
            bid_price_dn    = bid_ts['C_dn']    # Price down

            if is_realized:
                bid_activation_up = bid_ts['A_up']
                bid_activation_dn = bid_ts['A_dn']

            prob_activation_up  = np.array(self.controller.market.Pr_a_up(spot_prices, bid_price_up)).flatten()
            prob_activation_dn  = np.array(self.controller.market.Pr_a_dn(spot_prices, bid_price_dn)).flatten()
            
            # Filter out the unreasonably low bid activations
            activation_threshold   = 0.01
            volume_threshold       = 0.001
            filtered_bid_price_up  = np.where(np.logical_and(prob_activation_up > activation_threshold, bid_volume_up > volume_threshold), bid_price_up, 0).flatten()
            filtered_bid_price_dn  = np.where(np.logical_and(prob_activation_dn > activation_threshold, bid_volume_dn > volume_threshold), bid_price_dn, 0).flatten()
            filtered_bid_volume_up = np.where(np.logical_and(prob_activation_up > activation_threshold, bid_volume_up > volume_threshold), bid_volume_up, 0).flatten()
            filtered_bid_volume_dn = np.where(np.logical_and(prob_activation_dn > activation_threshold, bid_volume_dn > volume_threshold), bid_volume_dn, 0).flatten()

            ######################################################
            #                   BIDDING VOLUMES

            plt.figure(figsize=config.plot_format)

            _, ub_B = self.controller.model.get_bidding_bounds(self.controller)
            linewidth = 0.8
            plt.step(t, -ub_B[0,:], color='grey', label='Up-regulation volume limit', linewidth = linewidth)
            plt.step(t, ub_B[1,:], color='grey', label='Down-regulation volume limit',  linewidth = linewidth)
            plt.fill_between(t, -filtered_bid_volume_up, 0, color='blue', alpha=0.4, label='Up-regulation', step='post')
            plt.fill_between(t, 0, filtered_bid_volume_dn, color='red', alpha=0.4, label='down-regulation', step='post')
            plt.ylabel("Power (MW)")
            plt.xlabel("Time")
            plt.legend(loc="upper right")
            plt.title(f"{run} volumes in MW ({controller.market.date}, {controller.market.bidding_zone})")

            filename = f"{run_sanitized}_volume"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

            ######################################################
            #                   BIDDING PRICES

            fig, ax = plt.subplots(1, 1, figsize=config.plot_format, sharex=True)

            ax.fill_between(t, 0, filtered_bid_price_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax.fill_between(t, 0, filtered_bid_price_dn, label="Bidding Price Down", color="red", step='post', alpha=0.4)
            ax.step(t, spot_prices*1000/self.controller.market.C_eur2nok, label="Spot price", color="grey", linestyle="--", where='mid')
            ax.set_ylabel("Bidding Price (€/MW)", color="blue")
            ax.tick_params(axis='y', labelcolor="blue")
            ax.legend(loc="upper left")

            fig.suptitle(f"{run} Bidding Prices and Spot Prices in €/MW ({controller.market.date}, {controller.market.bidding_zone})")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 

            filename = f"{run_sanitized}_prices"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

            ######################################################
            #                ACTIVATION PROBABILITIES

            plt.figure(figsize=config.plot_format)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)

            ax1.fill_between(t, 0, prob_activation_up, color=self.color_up, alpha=0.8, label="Expected", step='post', linewidth=0)
            ax2.fill_between(t, 0, prob_activation_dn, color=self.color_dn, alpha=0.8, label="Expected", step='post', linewidth=0)

            if is_realized:
                ax1.fill_between(t, 0, np.multiply(bid_activation_up, prob_activation_up), color='navy', alpha=1, label="Activated", step='post', linewidth=0)
                ax2.fill_between(t, 0, np.multiply(bid_activation_dn, prob_activation_dn), color='maroon', alpha=1, label="Activated", step='post', linewidth=0)

            ax1.set_ylabel('Expected activation probability')
            ax2.set_ylabel("Expected activation probability")
            ax2.set_xlabel("Time (days)")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            plt.title(f"{run} Activation Chances ({controller.market.date}, {controller.market.bidding_zone})")

            filename = f"{run_sanitized}_activations"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)


            ######################################################
            #             VOLUME-ACTIVATION SCATTER
   
            plt.figure(figsize=config.plot_format)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_format)

            if is_realized:
                ax1.scatter(bid_volume_up[np.where(bid_activation_up == 1)], prob_activation_up[np.where(bid_activation_up == 1)], color='green', s=10)
                ax2.scatter(bid_volume_dn[np.where(bid_activation_dn == 1)], prob_activation_dn[np.where(bid_activation_dn == 1)], color='green', s=10)
                ax1.scatter(bid_volume_up[np.where(bid_activation_up == 0)], prob_activation_up[np.where(bid_activation_up == 0)], color='blue', s=1)
                ax2.scatter(bid_volume_dn[np.where(bid_activation_dn == 0)], prob_activation_dn[np.where(bid_activation_dn == 0)], color='red', s=1)
            
            else:
                ax1.scatter(bid_volume_up, prob_activation_up, color='blue', s=1)
                ax2.scatter(bid_volume_dn, prob_activation_dn, color='red', s=1)
            
            ax1.set_ylabel("Projected activation chance")
            ax1.set_xlabel("Bid volume (MW)")
            ax1.title.set_text('Up-regulation bids')

            ax2.set_ylabel("Projected activation chance")
            ax2.set_xlabel("Bid volume (MW)")
            ax2.title.set_text('Down-regulation bids')

            title = f"{run} Volumes vs predicted activation chances €/MW ({controller.market.date}, {controller.market.bidding_zone})"
            title += "\n(Activations shown in green)"
            
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 

            filename = f"{run_sanitized}_scatter_plots"
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)


            if is_realized:
                
                # Filter out non-activated bids
                bid_prices_up_activated = np.where(np.logical_and(bid_activation_up > activation_threshold, bid_volume_up > volume_threshold,), bid_price_up, 0)
                bid_prices_dn_activated = np.where(np.logical_and(bid_activation_dn > activation_threshold, bid_volume_dn > volume_threshold,), bid_price_dn, 0)
                bid_volumes_up_activated = np.where(np.logical_and(bid_activation_up > activation_threshold, bid_volume_up > volume_threshold,), bid_volume_up, 0)
                bid_volumes_dn_activated = np.where(np.logical_and(bid_activation_dn > activation_threshold, bid_volume_dn > volume_threshold,), bid_volume_dn, 0)
                
                
                ######################################################
                #    ACTIVATED BIDDING VOLUMES AND PRICES
                #               [FILTERED]

                plt.figure(figsize=config.plot_format)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=config.plot_format, sharex=True)

                ax1.fill_between(t, -bid_volumes_up_activated, 0, color='blue', alpha=0.4, label='Up-regulation', step='post')
                ax1.fill_between(t, 0, bid_volumes_dn_activated, color='red', alpha=0.4, label='down-regulation', step='post')
                ax1.set_ylabel("Bid Volumes (MW)")
                ax1.set_xlabel("Time")
                ax1.legend(loc="upper right")

                ax2.fill_between(t, 0, filtered_bid_price_up, color='grey', label="Submitted", alpha=0.4)
                ax2.fill_between(t, 0, bid_prices_up_activated, color='blue', label="Activated", alpha=0.4)
                ax2.set_ylabel("Bid Price Up (€/MW)")
                ax2.set_xlabel("Time (days)")
                ax2.legend()

                ax3.fill_between(t, 0, filtered_bid_price_dn, color='grey', label="Submitted", alpha=0.4)
                ax3.fill_between(t, 0, bid_prices_dn_activated, color='red', label="Down-activation", alpha=0.4)
                ax3.set_ylabel("Bid Price Down (€/MW)")
                ax3.set_xlabel("Time (days)")
                ax3.legend()

                fig.suptitle(f"{run} activated prices and volumes. ({controller.market.date}, {controller.market.bidding_zone})")

                filename = f"{run_sanitized}_results_bidding_vol_act"
                plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

                ######################################################
                #    PROJECTED VS RECORDED ACTIVATION CHANCES 
                #

                plt.figure(figsize=config.plot_format)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_format)


                demand_up = market.up_activation_occurance_rate
                demand_dn = market.down_activation_occurance_rate
                n_bins_up = int(np.ceil(demand_up*100))
                n_bins_dn = int(np.ceil(demand_dn*100))

                probs_up, probs_dn = [], []
                probs_up_idx, probs_dn_idx = [], []
                for chance in range(n_bins_up):
                    # Gather average activation rate of all bids projected to be within 0.5% of an activation chance
                    slice_up = bid_activation_up[np.where(np.logical_and(prob_activation_up*100>chance-0.5, prob_activation_up*100<chance+0.5))]
                    if len(slice_up) > 0:
                        probs_up.append(100 * np.mean(slice_up))
                        probs_up_idx.append(chance)

                for chance in range(n_bins_dn):
                    # Gather average activation rate of all bids projected to be within 0.5% of an activation chance
                    slice_dn = bid_activation_dn[np.where(np.logical_and(prob_activation_dn*100>chance-0.5, prob_activation_dn*100<chance+0.5))]
                    if len(slice_dn) > 0:
                        probs_dn.append(100 * np.mean(slice_dn))
                        probs_dn_idx.append(chance)

                ax1.plot(probs_up_idx, probs_up, color=self.color_up, marker='o')
                ax2.plot(probs_dn_idx, probs_dn, color=self.color_dn, marker='o')

                ax1.set_ylabel("Recorded bid activation rate (%)")
                ax1.set_xlabel("Expected bid activation chance (%)")
                ax2.set_ylabel("Recorded bid activation rate (%)")
                ax2.set_xlabel("Expected bid activation chance (%)")
                
                fig.suptitle('')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 

                filename = f"{run_sanitized}_expected_vs_true_probabilities"
                plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)



                ######################################################
                #      RECORDED VS HISTORICAL CLEARING PRICES
                #

                plt.figure(figsize=config.plot_format)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_format)

                price_data_raw = market.merged_data
                price_data = price_data_raw.where(price_data_raw['Up Price']<500).where(price_data_raw['Down Price']>-500).dropna()

                mfrr_prices_up  = np.array(price_data['Up Price'])
                mfrr_prices_dn  = np.array(price_data['Down Price'])

                clearing_prices_up, clearing_prices_dn = market.get_clearing_prices(market.date)
                clearing_prices_up = clearing_prices_up[np.where(clearing_prices_up<500)]
                clearing_prices_dn = clearing_prices_dn[np.where(clearing_prices_dn<500)]

                n_bins = 100

                ax1.hist(mfrr_prices_up, label="Expected", color='grey', alpha=0.8, bins=n_bins, density=True)
                ax2.hist(mfrr_prices_dn, label="Expected", color='grey', alpha=0.8, bins=n_bins, density=True)
                
                ax1.hist(clearing_prices_up, label="Recorded", color=self.color_up, alpha=0.8, bins=n_bins, density=True)
                ax2.hist(clearing_prices_dn, label="Recorded", color=self.color_dn, alpha=0.8, bins=n_bins, density=True)

                ax1.set_xlabel("Bidding prices (€/MW)")
                ax2.set_xlabel("Bidding prices (€/MW)")
                ax1.set_ylabel("Occurance rate (%)")
                ax2.set_ylabel("Occurance rate (%)")
                ax1.set_xlim([-50, 250])
                ax2.set_xlim([-50, 250])
                ax1.legend()
                ax2.legend()

                fig.suptitle(f"Distributions of clearing prices, normalized ({market.bidding_zone}, {market.date})")
                
                filename = f"clearing_prices_distributions_{market.bidding_zone}"
                plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)



        ######################################################
        #                   SPOT PRICES

        plt.figure(figsize=config.plot_format)
        plt.step(t, market.get_spotprice(), label="Spot price")
        plt.ylabel("Spot price (kr/kWh)")
        plt.xlabel("Time (days)")
        plt.legend()

        filename = "spot_price"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

        plt.close()


    def plot_random_activations(self, m: int):

        simulator = self.simulator
        controller = self.controller
        market = controller.market
        config = self.config
        foldername = self.foldername


        t = controller.t
        freshwewights = simulator.simulate_random_activation(self.controller, m)


        ##################################################
        plt.figure(figsize=config.plot_format)

        bidding_runs = [run for run in controller.optimization_results['runs'] if 'bidding result' in controller.optimization_results['runs'][run]]

        for run_name in bidding_runs:
            if 'Realized' in run_name:
                continue 
            bid_ts  = self.controller.optimization_results['runs'][run_name]['timeseries']
            bid_volume_up   = bid_ts['P_up']    # Volume up
            bid_volume_dn   = bid_ts['P_dn']    # Volume down
            bid_price_up    = bid_ts['C_up']    # Price up
            bid_price_dn    = bid_ts['C_dn']    # Price down

            u_nom = controller.optimization_results['runs'][run_name]['timeseries']['u_nom']
            x_bid = controller.optimization_results['runs'][run_name]['timeseries']['x']
            fw_variance = propagate_process_covariance(controller, x_bid, u_nom, bid_volume_up, bid_volume_dn, bid_price_up, bid_price_dn)
            fw_sd = np.sqrt(fw_variance)
            fw = np.array(controller.model.freshweight(x_bid[:,1:])).flatten()

            fw_ub = fw + 1.96*fw_sd
            fw_lb = fw - 1.96*fw_sd

            plt.fill_between(t, fw_lb, fw_ub, color='green', alpha=0.4)

        for case in range(freshwewights.shape[0]):
            plt.plot(t, freshwewights[case,:], color='blue', alpha=0.2)
        plt.axhline(y=controller.model.Final_fw_sht, color='gray', linestyle=':', label="Required Freshweight (g/plant)")
        plt.ylabel("Fresh weight (g/plant)")
        plt.xlabel("Time (days)")
        plt.title(f'Simulated {m} different cases of plausible activations')

        filename = "random_activations"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + config.plot_file_type, format=config.plot_file_type)

    def plot_spot_mfrr_prices(self):

        config = self.config
        controller = self.controller
        market = controller.market

        # Remove drastic outliers
        price_data_raw = market.merged_data
        price_data = price_data_raw.where(price_data_raw['Up Price']<500).where(price_data_raw['Down Price']>-500).dropna()

        timestamps      = price_data['Start Time']
        spot_prices     = np.array(price_data['Spot Price'])
        mfrr_prices_up  = np.array(price_data['Up Price'])
        mfrr_prices_dn  = np.array(price_data['Down Price'])
        spot_prices_eur = np.array(spot_prices)*1000/market.C_eur2nok


        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='same')

        # Smoothed data
        window_length = 24*3
        mfrr_prices_up_smoothed     = moving_average(mfrr_prices_up, window_length)
        mfrr_prices_dn_smoothed     = moving_average(mfrr_prices_dn, window_length)
        spot_prices_eur_smoothed    = moving_average(spot_prices_eur, window_length)

        opacity = 0.2
        linewidth=1.5

        line_x = np.array([min(spot_prices), max(spot_prices)])


        ###########################################
        #            SPOT VS MFRR PRICES 
        #       [SMOOTHED] [OUTLIERS REMOVED]

        plt.figure(figsize=config.plot_format)

        plt.step(timestamps, mfrr_prices_up, label="Clearing price up", color='blue', alpha=opacity)
        plt.step(timestamps, mfrr_prices_up_smoothed, label="Clearing price up smoothed", color='blue', alpha=1, linewidth = linewidth)

        plt.step(timestamps, mfrr_prices_dn, label="Clearing price down", color='red', alpha=opacity)
        plt.step(timestamps, mfrr_prices_dn_smoothed, label="Clearing price down smoothed", color='red', alpha=1, linewidth = linewidth)

        plt.step(timestamps, spot_prices_eur, label="Spot price", color='grey', alpha=opacity)
        plt.step(timestamps, spot_prices_eur_smoothed, label="Spot price smoothed", color='grey', alpha=1, linewidth = linewidth)
        plt.ylabel("Price (€/MW)")
        plt.xlabel("Time (days)")
        plt.title(f"Spot price vs activation prices smoothed using {window_length}h moving average")
        plt.legend()


        filename = f"smoothed_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)



        ####################################################
        #        CLEARING PRICES RELATIVE TO SPOT 
        #               [OUTLIERS REMOVED]
        
        plt.figure(figsize=config.plot_format)

        plt.step(timestamps, mfrr_prices_up - spot_prices_eur, label="Clearing price up", color='blue')
        plt.step(timestamps, mfrr_prices_dn - spot_prices_eur, label="Clearing price down", color='red')
        plt.plot(timestamps, 0 * mfrr_prices_dn, label="Zero-line", color='grey', alpha=0.5)
        plt.ylabel("Price (€/MW)")
        plt.xlabel("Time (days)")
        plt.title("Clearing prices relative to spot price (€/MW)")
        plt.legend()

        filename = f"relative_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)


        ###############################################################
        #             CLEARING PRICES HISTOGRAM 
        #               [OUTLIERS REMOVED]
        
        plt.figure(figsize=config.plot_format)

        n_bins = 100
        N_data_points = len(mfrr_prices_up)

        _, mu_up, mu_dn = market.price_means
        sigma_up, sigma_dn = np.sqrt(market.price_cov[1, 1]), np.sqrt(market.price_cov[2, 2])

        x_up = np.linspace(mu_up-3*sigma_up,mu_up+3*sigma_up, 1000)
        x_dn = np.linspace(mu_dn-3*sigma_dn,mu_dn+3*sigma_dn, 1000)

        plt.hist(mfrr_prices_up, label="Clearing price up", color='blue', alpha=0.4, bins=n_bins, density=True)
        plt.hist(mfrr_prices_dn, label="Clearing price down", color='red', alpha=0.4, bins=n_bins, density=True)
        plt.hist(spot_prices_eur, label="Spot prices", color='grey', alpha=0.4, bins=n_bins, density=True)
        plt.plot(x_up, stats.norm.pdf(x_up, mu_up, sigma_up), label="Estimated Up-price distribution", color='blue')
        plt.plot(x_dn, stats.norm.pdf(x_dn, mu_dn, sigma_dn), label="Estimated Down-price distribution", color='red')
        plt.xlabel("Bidding prices (€/MW)")
        plt.title(f"Clearing prices histogram normalized ({market.bidding_zone}, {market.date})")
        plt.legend()

        filename = f"histogram_mfrr_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)



        ################################################################
        plt.figure(figsize=config.plot_format)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_format, sharex=True)

        n = int(np.ceil(len(spot_prices)/10))
        idx = np.int64(np.ceil(np.linspace(1,len(spot_prices)-1,n)))
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

        fig.suptitle(f"Spot prices with mFRR clearing prices €/MW ({market.bidding_zone}, {market.date})")

        filename = f"scatter_spot_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)



        ######################################################
        #        CLEARING-SPOT RELATIVE PRICES HISTOGRAM
        #                   [OUTLIERS REMOVED]

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

        plt.suptitle(f'Relative clearing prices, normalized ({market.bidding_zone}, {market.date})')

        filename = f"histogram_relative_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)


   
        ######################################################
        #           MONTHLY mFRR ACTIVATION FREQUENCIES


        up_activations_df, down_activations_df = market.up_activation_df, market.down_activation_df

        # Process data up
        up_activations_df['Start Time'] = pd.to_datetime(up_activations_df['Start Time'])
        up_activations_df.loc[:, 'Year-Month'] = up_activations_df['Start Time'].dt.to_period('M')
        n_up_activations = up_activations_df[(up_activations_df.filter(like='Activated').sum(axis=1)) > 0]
        n_up_activations = n_up_activations.copy()

        # Count activations and total timestamps by Year-Month
        monthly_counts_up = n_up_activations['Year-Month'].value_counts().sort_index()
        total_counts_up = up_activations_df['Year-Month'].value_counts().sort_index()
        activation_rate_up = (monthly_counts_up / total_counts_up * 100).sort_index()

        # Process data down
        down_activations_df['Start Time'] = pd.to_datetime(down_activations_df['Start Time'])
        down_activations_df.loc[:, 'Year-Month'] = down_activations_df['Start Time'].dt.to_period('M')
        n_down_activations = down_activations_df[(down_activations_df.filter(like='Activated').sum(axis=1)) > 0]
        n_down_activations = n_down_activations.copy()

        # Count activations and total timestamps by Year-Month
        monthly_counts_down = n_down_activations['Year-Month'].value_counts().sort_index()
        total_counts_down = down_activations_df['Year-Month'].value_counts().sort_index()
        activation_rate_down = (monthly_counts_down / total_counts_down * 100).sort_index()

        # Combine the two datasets into a single DataFrame
        combined_monthly_activation_rates = pd.DataFrame({
            'Up Activation Rate': activation_rate_up,
            'Down Activation Rate': activation_rate_down
        }).fillna(0)  # Fill missing months with 0



        plt.figure(figsize=config.plot_format)

        combined_monthly_activation_rates.plot(kind='bar', stacked=True, figsize=(12, 6), color=['skyblue', 'lightcoral'], edgecolor='gray')
        plt.title(f'Monthly activation rates in {market.bidding_zone} bidding zone')
        plt.xlabel('Year-Month')
        plt.ylabel('Percentage of MTUs where activations occur')
        plt.xticks(rotation=45)
        plt.legend(title='Activation Direction')
        plt.tight_layout()

        filename = f"{market.bidding_zone}_mFRR_monthly_activation_frequencies"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)


        ######################################################
        #           DAILY mFRR ACTIVATION COUNTS

        plt.figure(figsize=config.plot_format)

        # Process data
        up_activations_df['Start Time'] = pd.to_datetime(up_activations_df['Start Time'])
        up_activations_df.loc[:, 'Date'] = up_activations_df['Start Time'].dt.date
        down_activations_df['Start Time'] = pd.to_datetime(down_activations_df['Start Time'])
        down_activations_df.loc[:, 'Date'] = down_activations_df['Start Time'].dt.date

        # Ensure all dates from the full range are present for counting
        date_range_up = pd.date_range(start=up_activations_df['Date'].min(), end=up_activations_df['Date'].max())
        all_dates_up = pd.DataFrame({'Date': date_range_up})
        date_range_down = pd.date_range(start=down_activations_df['Date'].min(), end=down_activations_df['Date'].max())
        all_dates_down = pd.DataFrame({'Date': date_range_down})

        # Count activations and total timestamps by Date
        daily_counts_up = up_activations_df[(up_activations_df.filter(like='Activated').sum(axis=1)) > 0].groupby('Date').size().reindex(date_range_up, fill_value=0)
        total_counts_up_daily = up_activations_df.groupby('Date').size().reindex(date_range_up, fill_value=0)

        daily_counts_down = down_activations_df[(down_activations_df.filter(like='Activated').sum(axis=1)) > 0].groupby('Date').size().reindex(date_range_down, fill_value=0)
        total_counts_down_daily = down_activations_df.groupby('Date').size().reindex(date_range_down, fill_value=0)

        # Compute activation rate
        activation_rate_up_daily = (daily_counts_up / total_counts_up_daily * 100).fillna(0)
        activation_rate_down_daily = (daily_counts_down / total_counts_down_daily * 100).fillna(0)

        # Combine the two datasets into a single DataFrame
        combined_daily_rates = pd.DataFrame({
            'Date': date_range_up,
            'Up Activation Rate': activation_rate_up_daily.values,
            'Down Activation Rate': activation_rate_down_daily.values
        })

        first_of_month = combined_daily_rates['Date'][combined_daily_rates['Date'].dt.day == 1]

        ax = combined_daily_rates.set_index('Date')[['Up Activation Rate', 'Down Activation Rate']].plot(
            kind='bar', stacked=True, figsize=(15, 6), color=['skyblue', 'lightcoral']
        )

        first_of_month_indexes = combined_daily_rates[combined_daily_rates['Date'].dt.day == 1].index
        ax.set_xticks(first_of_month_indexes)
        ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in first_of_month], rotation=45)

        plt.title(f'Daily activation rates in {market.bidding_zone} bidding zone')
        plt.xlabel('Date')
        plt.ylabel('Percentage of MTUs where activations occur')
        plt.legend(title='Activation Direction')
        plt.tight_layout()

        filename = f"{market.bidding_zone}_mFRR_daily_activation_frequencies"
        plt.savefig(config.data_analysis_path + filename + "." + config.plot_file_type, format=config.plot_file_type)

        plt.close()



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

        u_rigid = simulator.mpc_controller.optimization_results['runs']['Rigid']['timeseries']['u'].flatten()
        x_rigid = simulator.mpc_controller.optimization_results['runs']['Rigid']['timeseries']['x']
        u_mpc = self.simulator.u_mpc.flatten()

        x_mpc = self.simulator.x_mpc
        x1_mpc = x_mpc[0,1:]
        x2_mpc = x_mpc[1,1:]

        # BEGIN PLOTTING
        ##################################################
        plt.figure(figsize=config.plot_format)
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


        plt.figure(figsize=config.plot_format)
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

        n_runs = len(list(self.controller.optimization_results['runs'].keys()))


        t       = self.controller.t
        spot_prices  = self.controller.spot_prices


        if 'Bidding' in controller.optimization_results['runs']:

            bid_ts  = self.controller.optimization_results['runs']['Bidding']['timeseries']
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


            

