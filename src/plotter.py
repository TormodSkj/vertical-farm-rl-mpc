import matplotlib.pyplot as plt
from config import Config
from settings import Settings
from controller import Controller
from model import *
from market import Market
from simulator import Simulator
from utils import *
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
import os
from typing import List
from tqdm import tqdm


class Plotter():

    settings: Settings
    config: Config
    controller: Controller
    simulator: Simulator
    foldername: str

    color_up = 'skyblue'
    color_dn = 'lightcoral'

    n_plots = 0
    plot_queue = []
    progressbar: tqdm

    def __init__(self, settings: Settings, config, controller=None, simulator=None):
        self.settings = settings
        self.config = config
        self.controller = controller
        self.simulator = simulator

        self.plotter_settings = settings.get_settings_group('general', 'plotter')

        self.foldername     = self.plotter_settings['SIM_NAME']
        self.plot_file_type = self.plotter_settings['PLOT_EXPORT_TYPE']
        self.activation_th  = self.plotter_settings['ACTIVATION_THRESHOLD']
        self.volume_th      = self.plotter_settings['VOLUME_THRESHOLD']





    def save_plot(self, filename, run_id = None, fig = None, pdf= None):
        config = self.config
        foldername  = self.foldername

        directory = config.plot_path + foldername + "/"
        if run_id is not None: directory += run_id + '/'

        plt.savefig(directory + filename + "." + self.plot_file_type, format=self.plot_file_type)

        if fig is not None: 
            if pdf is not None: pdf.savefig(fig)
            plt.close(fig)
            self.progressbar.update(1)
            # self.progressbar.set_description(f"Plotting {run_id} {filename}")  # Updates dynamically
            self.progressbar.set_postfix(status=f"Plotting {run_id} {filename}")  # Adds a small status message


        

    def create_folder_environment(self):
        controller  = self.controller
        config      = self.config

        
        run_groups = build_dependency_groups(controller.optimization_results)

        for run_group in run_groups:
            top_level_run_id = run_group[0]
            
            plot_folder = os.path.join(config.plot_path, config.sim_name+'/'+ top_level_run_id+'/')
            
            os.makedirs(plot_folder, exist_ok=True)



    def save_ocp_plots(self):

        self.create_folder_environment()

        # COMMON PLOTS
        self.add_plot(self.plot_freshweights,       runs = self.controller.optimization_results['runs'])
        self.add_plot(self.plot_light_schedules,    runs = self.controller.optimization_results['runs'])
        self.add_plot(self.plot_DLI,                runs = self.controller.optimization_results['runs'])


        # INDIVIDUAL PLOTS
        run_groups = build_dependency_groups(self.controller.optimization_results)
        for run_group in run_groups:
            self.add_plot(self.plot_report, run_group = run_group)
        
        self.plot()

        # Save plot cache in file system

        print(f'Optimization plots saved to {self.config.plot_folder}')


    def add_plot(self, function, *args, **kwargs):

        self.plot_queue.append((function, args, kwargs))

        
    def plot(self):

        with tqdm(desc=f"Plotting optimization results ...") as pbar:
            self.progressbar = pbar
            for func, args, kwargs in self.plot_queue:
                
                func(*args, **kwargs)
                # pbar.update(1)

        

    def plot_report(self, run_group):
        config      = self.config
        controller  = self.controller
        market      = controller.market
        t           = self.controller.t
        spot_prices = self.controller.spot_prices

        #%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
        #           BIDDING PLOTS

        # Defining run is the first of the run group
        run_id = run_group[0]


        final_report_save_path = f'{config.plot_path}{self.foldername}/report_{run_id}.pdf'

        with PdfPages(final_report_save_path) as pdf:

            bid_ts = self.controller.optimization_results['runs'][run_id]['timeseries']
            if 'A_up' in bid_ts and 'A_dn' in bid_ts:
                is_realized = True                      # Bool used to check if there is real activation data to plot
            else:
                is_realized = False

            u_nom            = bid_ts['u_nom'].flatten()
            bid_volumes_up   = bid_ts['P_up'].flatten()    # Volume up
            bid_volumes_down = bid_ts['P_dn'].flatten()    # Volume down
            bid_prices_up    = bid_ts['C_up'].flatten()    # Price up
            bid_prices_down  = bid_ts['C_dn'].flatten()    # Price down

            if is_realized:
                bid_activation_up   = bid_ts['A_up'].flatten()
                bid_activation_down = bid_ts['A_dn'].flatten()

            prob_activation_up      = np.array(self.controller.market.activation_prob_up(spot_prices, bid_prices_up)).flatten()
            prob_activation_down    = np.array(self.controller.market.activation_prob_down(spot_prices, bid_prices_down)).flatten()
            
            # Filter out the unreasonably low bid activations
            filtered_bid_prices_up      = np.where(np.logical_and(prob_activation_up   > self.activation_th, bid_volumes_up   > self.volume_th), bid_prices_up,     0).flatten()
            filtered_bid_prices_down    = np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th), bid_prices_down,   0).flatten()
            filtered_bid_volumes_up     = np.where(np.logical_and(prob_activation_up   > self.activation_th, bid_volumes_up   > self.volume_th), bid_volumes_up,    0).flatten()
            filtered_bid_volumes_down   = np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th), bid_volumes_down,  0).flatten()





            ########################################################
            #                   COMMON PLOTS


            runs = {run: controller.optimization_results['runs'][run] for run in reversed(run_group) if run in controller.optimization_results['runs']}
            self.plot_freshweights      (runs = runs, run_id = run_id, pdf = pdf)
            self.plot_light_schedules   (runs = runs, run_id = run_id, pdf = pdf)
            self.plot_DLI               (runs = runs, run_id = run_id, pdf = pdf)


            ######################################################
            #                   BIDDING VOLUMES

            fig = plt.figure(figsize=config.plot_format)

            _, ub_B_volumes, _, _ = self.controller.model.get_bidding_bounds(controller.N, u_nom.reshape((1,-1)))
            linewidth = 0.8
            plt.step(t, -np.array(ub_B_volumes[0,:]).flatten(), color='grey', label='Up-regulation volume limit', linewidth = linewidth, where = 'post')
            plt.step(t, np.array(ub_B_volumes[1,:]).flatten(), color='grey', label='Down-regulation volume limit',  linewidth = linewidth, where = 'post')
            plt.fill_between(t, -filtered_bid_volumes_up, 0, color='blue', alpha=0.4, label='Up-regulation', step='post')
            plt.fill_between(t, 0, filtered_bid_volumes_down, color='red', alpha=0.4, label='down-regulation', step='post')
            plt.ylabel("Power (MW)")
            plt.xlabel("Time")
            plt.legend(loc="upper right")
            plt.title(f"{run_id} volumes in MW ({controller.market.date}, {controller.market.bidding_zone})")

            # filename = f"{run_sanitized}_volume"
            # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
            self.save_plot(f"bid_volumes", run_id, fig, pdf)

            ######################################################
            #                   BIDDING PRICES

            fig, ax = plt.subplots(1, 1, figsize=config.plot_format, sharex=True)

            ax.fill_between(t, 0, filtered_bid_prices_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax.fill_between(t, 0, filtered_bid_prices_down, label="Bidding Price Down", color="red", step='post', alpha=0.4)
            ax.step(t, spot_prices*1000/self.controller.market.C_eur2nok, label="Spot price", color="grey", linestyle="--", where='post')
            ax.set_ylabel("Bidding Price (€/MW)")
            ax.tick_params(axis='y')
            ax.legend(loc="upper left")

            fig.suptitle(f"{run_id} Bidding Prices and Spot Prices in €/MW ({controller.market.date}, {controller.market.bidding_zone})")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 

            # filename = f"{run_sanitized}_prices"
            # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
            self.save_plot(f"bid_prices", run_id, fig, pdf)

            ######################################################
            #                ACTIVATION PROBABILITIES

            # plt.figure(figsize=config.plot_format)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)

            ax1.fill_between(t, 0, prob_activation_up, color=self.color_up, alpha=0.8, label="Expected", step='post', linewidth=0)
            ax2.fill_between(t, 0, prob_activation_down, color=self.color_dn, alpha=0.8, label="Expected", step='post', linewidth=0)

            if is_realized:
                ax1.fill_between(t, 0, np.multiply(bid_activation_up, prob_activation_up), color='navy', alpha=1, label="Activated", step='post', linewidth=0)
                ax2.fill_between(t, 0, np.multiply(bid_activation_down, prob_activation_down), color='maroon', alpha=1, label="Activated", step='post', linewidth=0)


            ax1.step(t, controller.market.demand_prob_up(controller.spot_prices).reshape((-1,1)),   color='gray', linestyle=':', label="Max activation rate", where='post')
            ax2.step(t, controller.market.demand_prob_down(controller.spot_prices).reshape((-1,1)), color='gray', linestyle=':', label="Max activation rate", where='post')
            ax1.set_ylabel('Expected activation probability')
            ax2.set_ylabel("Expected activation probability")
            ax2.set_xlabel("Time (days)")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            fig.suptitle(f"{run_id} Activation Chances ({controller.market.date}, {controller.market.bidding_zone}) \nBar heights indicate expected activation probabilities per bid. Activated bids are highlighted in dark.")

            # filename = f"{run_sanitized}_activations"
            # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
            self.save_plot(f"bid_activations", run_id, fig, pdf)

            ######################################################
            #             VOLUME-ACTIVATION SCATTER

            # plt.figure(figsize=config.plot_format)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_format)

            if is_realized:
                ax1.scatter(bid_volumes_up[np.where(bid_activation_up == 1)], prob_activation_up[np.where(bid_activation_up == 1)], color='navy', s=10)
                ax2.scatter(bid_volumes_down[np.where(bid_activation_down == 1)], prob_activation_down[np.where(bid_activation_down == 1)], color='maroon', s=10)
                ax1.scatter(bid_volumes_up[np.where(bid_activation_up == 0)], prob_activation_up[np.where(bid_activation_up == 0)], color=self.color_up, s=1)
                ax2.scatter(bid_volumes_down[np.where(bid_activation_down == 0)], prob_activation_down[np.where(bid_activation_down == 0)], color=self.color_dn, s=1)
            
            else:
                ax1.scatter(bid_volumes_up, prob_activation_up, color=self.color_up)
                ax2.scatter(bid_volumes_down, prob_activation_down, color=self.color_dn)
            
            ax1.set_ylabel("Projected activation chance")
            ax1.set_xlabel("Bid volume (MW)")
            ax1.title.set_text('Up-regulation bids')

            ax2.set_ylabel("Projected activation chance")
            ax2.set_xlabel("Bid volume (MW)")
            ax2.title.set_text('Down-regulation bids')

            title = f"{run_id} Volumes vs predicted activation chances €/MW ({controller.market.date}, {controller.market.bidding_zone})"
            title += "\n(Activations highlighted in dark)"
            
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 

            # filename = f"{run_sanitized}_scatter_plots"
            # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
            self.save_plot(f"scatter_volumes_activations", run_id, fig, pdf)

            if is_realized:
                
                # Filter out non-activated bids
                bid_prices_up_activated     = np.where(np.logical_and(bid_activation_up     > self.activation_th, bid_volumes_up    > self.volume_th,), bid_prices_up,    0)
                bid_prices_dn_activated     = np.where(np.logical_and(bid_activation_down   > self.activation_th, bid_volumes_down  > self.volume_th,), bid_prices_down,  0)
                bid_volumes_up_activated    = np.where(np.logical_and(bid_activation_up     > self.activation_th, bid_volumes_up    > self.volume_th,), bid_volumes_up,   0)
                bid_volumes_dn_activated    = np.where(np.logical_and(bid_activation_down   > self.activation_th, bid_volumes_down  > self.volume_th,), bid_volumes_down, 0)
                
                
                ######################################################
                #    ACTIVATED BIDDING VOLUMES AND PRICES
                #               [FILTERED]

                # plt.figure(figsize=config.plot_format)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=config.plot_format, sharex=True)

                ax1.fill_between(t, -filtered_bid_volumes_up, filtered_bid_volumes_down, color='grey', label="Submitted", alpha=0.4, step='post')
                ax1.fill_between(t, -bid_volumes_up_activated, 0, color='blue', alpha=0.4, label='Up-regulation', step='post')
                ax1.fill_between(t, 0, bid_volumes_dn_activated, color='red', alpha=0.4, label='down-regulation', step='post')
                ax1.set_ylabel("Bid Volumes (MW)")
                ax1.set_xlabel("Time")
                ax1.legend(loc="upper right")

                ax2.fill_between(t, 0, filtered_bid_prices_up, color='grey', label="Submitted", alpha=0.4, step='post')
                ax2.fill_between(t, 0, bid_prices_up_activated, color='blue', label="Activated", alpha=0.4, step='post')
                ax2.set_ylabel("Bid Price Up (€/MW)")
                ax2.set_xlabel("Time (days)")
                ax2.legend()

                ax3.fill_between(t, 0, filtered_bid_prices_down, color='grey', label="Submitted", alpha=0.4, step='post')
                ax3.fill_between(t, 0, bid_prices_dn_activated, color='red', label="Activated", alpha=0.4, step='post')
                ax3.set_ylabel("Bid Price Down (€/MW)")
                ax3.set_xlabel("Time (days)")
                ax3.legend()

                fig.suptitle(f"{run_id} activated prices and volumes. ({controller.market.date}, {controller.market.bidding_zone})")

                # filename = f"{run_sanitized}_results_activated_volumes_prices"
                # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
                self.save_plot("results_activated_volumes_prices", run_id, fig, pdf)


                ######################################################
                #    EXPECTED VS RECORDED CLEARING PRICES 
                #

                # plt.figure(figsize=config.plot_format)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)


                clearing_prices_up, clearing_prices_down = market.get_clearing_prices()
                activations_up, activations_down = market.get_activation_demands()

                expected_prices_up, expected_prices_down = conditional_expectation(controller.spot_prices, market.price_means, market.price_covs)

                linewidth = 0.4

                ax1.fill_between(t, 0, np.where(activations_up > 0, clearing_prices_up, 0), color='limegreen', alpha=0.4, label="Activations", step='post')
                ax1.fill_between(t, 0, bid_prices_up_activated, color='blue', label="Activated Bid Prices", alpha=0.4, step='post')
                ax1.step(t, clearing_prices_up, color='navy',  label="Recorded", linewidth=linewidth, where='post')
                ax1.step(t, expected_prices_up.flatten(), color=self.color_up,  label="Expected", linewidth=linewidth, where='post')
                ax1.step(t, filtered_bid_prices_up, color='grey',  label="Submitted", linewidth=linewidth, where='post')
                ax1.set_ylabel("Price Up (€/MW)")
                ax1.set_xlabel("Time (days)")
                ax1.legend()

                ax2.fill_between(t, 0, np.where(activations_down > 0, clearing_prices_down, 0), color='limegreen', alpha=0.4, label="Activations", step='post')
                ax2.fill_between(t, 0, bid_prices_dn_activated, color='red', label="Activated Bid Prices", alpha=0.4, step='post')
                ax2.step(t, clearing_prices_down, color='maroon', label="Recorded", linewidth=linewidth, where='post')
                ax2.step(t, expected_prices_down.flatten(), color=self.color_dn, label="Expected", linewidth=linewidth, where='post')
                ax2.step(t, filtered_bid_prices_down, color='grey',  label="Submitted", linewidth=linewidth, where='post')
                ax2.set_ylabel("Price Down (€/MW)")
                ax2.set_xlabel("Time (days)")
                ax2.legend()



                fig.suptitle(f"{run_id} Expected vs recorded clearing prices. ({controller.market.date}, {controller.market.bidding_zone})\nExpectations made based on price covariances")

                # filename = f"{run_sanitized}_expected_vs_recorded_clearing_prices"
                # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
                self.save_plot("expected_vs_recorded_clearing_prices", run_id, fig, pdf)

                ######################################################
                #    PROJECTED VS RECORDED ACTIVATION CHANCES 
                #

                # plt.figure(figsize=config.plot_format)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_format)

                step_size = 5
                n_bins_up = int(np.ceil(max(prob_activation_up)*100))+step_size
                n_bins_dn = int(np.ceil(max(prob_activation_down)*100))+step_size

                probs_up, probs_dn = [], []
                probs_up_idx, probs_dn_idx = [], []
                for chance in range(0, n_bins_up, step_size):
                    # Gather average activation rate of all bids projected to be within 0.5% of an activation chance
                    slice_up = bid_activation_up[np.where(np.logical_and(prob_activation_up*100>chance-0.5*step_size, prob_activation_up*100<chance+0.5*step_size))]
                    if len(slice_up) > 0:
                        probs_up.append(100 * np.mean(slice_up))
                        probs_up_idx.append(chance)

                for chance in range(0, n_bins_dn, step_size):
                    # Gather average activation rate of all bids projected to be within 0.5% of an activation chance
                    slice_dn = bid_activation_down[np.where(np.logical_and(prob_activation_down*100>chance-0.5*step_size, prob_activation_down*100<chance+0.5*step_size))]
                    if len(slice_dn) > 0:
                        probs_dn.append(100 * np.mean(slice_dn))
                        probs_dn_idx.append(chance)

                ax1.plot(probs_up_idx, probs_up, color=self.color_up, marker='o')
                ax2.plot(probs_dn_idx, probs_dn, color=self.color_dn, marker='o')

                ax1.plot(np.array([0, max(probs_up_idx)]), np.array([0, max(probs_up_idx)]), color='grey', linestyle=':')
                ax2.plot(np.array([0, max(probs_dn_idx)]), np.array([0, max(probs_dn_idx)]), color='grey', linestyle=':')

                # ax1.set_aspect('equal')
                # ax2.set_aspect('equal')

                ax1.set_ylabel("Recorded bid activation rate (%)")
                ax1.set_xlabel("Expected bid activation chance (%)")
                ax2.set_ylabel("Recorded bid activation rate (%)")
                ax2.set_xlabel("Expected bid activation chance (%)")
                
                fig.suptitle('')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 

                # filename = f"{run_sanitized}_expected_vs_true_probabilities"
                # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
                self.save_plot("expected_vs_true_probabilities", run_id, fig, pdf)

                plt.close('all')

                ######################################################
                #      RECORDED VS HISTORICAL CLEARING PRICES
                #

                # plt.figure(figsize=config.plot_format)
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=config.plot_format, sharex=True)

                price_data_raw = market.prices_working_set
                price_data = price_data_raw.where(price_data_raw['Clearing Price Up']<500).where(price_data_raw['Clearing Price Down']>-500).dropna()

                mfrr_prices_up  = np.array(price_data['Clearing Price Up'])
                mfrr_prices_dn  = np.array(price_data['Clearing Price Down'])

                clearing_prices_up, clearing_prices_down = market.get_clearing_prices()
                clearing_prices_up = clearing_prices_up[np.where(clearing_prices_up<500)]
                clearing_prices_down = clearing_prices_down[np.where(clearing_prices_down<500)]
                
                activated_prices_up, activated_prices_dn = market.get_clearing_prices()
                activation_demand_up, activation_demand_dn = market.mfrr_demands_up, market.mfrr_demands_down
                activated_prices_up = activated_prices_up[np.where(np.logical_and(activated_prices_up<500, activation_demand_up > 0))]
                activated_prices_dn = activated_prices_dn[np.where(np.logical_and(activated_prices_dn<500, activation_demand_dn > 0))]

                n_bins = 50

                ax1.hist(mfrr_prices_up, label="Expected", color=self.color_up, alpha=0.8, bins=n_bins, density=True)
                ax2.hist(mfrr_prices_dn, label="Expected", color=self.color_dn, alpha=0.8, bins=n_bins, density=True)
                
                ax1.hist(clearing_prices_up, label="Recorded", color='navy', histtype='step', alpha=0.8, bins=n_bins, density=True)
                ax2.hist(clearing_prices_down, label="Recorded", color='maroon', histtype='step', alpha=0.8, bins=n_bins, density=True)

                ax3.hist(activated_prices_up, label="Activated", color=self.color_up, alpha=0.8, bins=30, density=True)
                ax4.hist(activated_prices_dn, label="Activated", color=self.color_dn, alpha=0.8, bins=30, density=True)

                # Gaussian pdfs of clearing prices
                mu_up, mu_down = market.price_means[2:4]
                sigma_up, sigma_down = np.sqrt(market.price_covs[0][1, 1]), np.sqrt(market.price_covs[1][1, 1])
                x_up = np.linspace(mu_up-3*sigma_up,mu_up+3*sigma_up, 1000)
                x_dn = np.linspace(mu_down-3*sigma_down,mu_down+3*sigma_down, 1000)
                
                ax3.plot(x_up, stats.norm.pdf(x_up, mu_up, sigma_up), label="Estimate of activated prices", color='navy')
                ax4.plot(x_dn, stats.norm.pdf(x_dn, mu_down, sigma_down), label="Estimate of activated prices", color='maroon')

                ax3.set_xlabel("Bidding prices (€/MW)")
                ax4.set_xlabel("Bidding prices (€/MW)")
                ax1.set_ylabel("Occurance rate (%)")
                ax3.set_ylabel("Occurance rate (%)")
                # ax1.set_xlim([-50, 250])
                # ax2.set_xlim([-50, 250])
                [ax.legend() for ax in (ax1, ax2, ax3, ax4)]

                fig.suptitle(f"Distributions of clearing prices, normalized ({market.bidding_zone}, {market.date})")
                
                # filename = f"{run_sanitized}_clearing_prices_distributions_{market.bidding_zone}"
                # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
                self.save_plot(f"clearing_prices_distributions_{market.bidding_zone}", run_id, fig, pdf)


                #############################################################################
                #      HISTOGRAM of BIDDING PRICES RELATIVE TO CLEARING PRICES
                #

                # plt.figure(figsize=config.plot_format)
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=config.plot_format, sharex=True)

                clearing_prices_up, clearing_prices_down = market.get_clearing_prices(market.date)
                
                n_bins = 50

                demands_up, demands_dn = market.get_activation_demands()
                relative_prices_up = bid_prices_up - clearing_prices_up
                relative_prices_dn = bid_prices_down - clearing_prices_down
                demand_relative_prices_up = relative_prices_up[np.where(np.logical_and(np.logical_and(demands_up == 1, prob_activation_up > self.activation_th), bid_volumes_up > self.volume_th))]
                demand_relative_prices_dn = relative_prices_dn[np.where(np.logical_and(np.logical_and(demands_dn == 1, prob_activation_down > self.activation_th), bid_volumes_down > self.volume_th))]

                relative_prices_up = relative_prices_up[np.where(np.logical_and(prob_activation_up > self.activation_th, bid_volumes_up > self.volume_th))]
                relative_prices_dn = relative_prices_dn[np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th))]

                ax1.hist(relative_prices_up, color=self.color_up,  alpha=0.8, bins=n_bins, density=True)
                ax2.hist(relative_prices_dn, color=self.color_dn,  alpha=0.8, bins=n_bins, density=True)
                ax1.set_title('Relative Prices Up, All time slots')
                ax2.set_title('Relative Prices Down, All time slots')

                ax3.hist(demand_relative_prices_up, color=self.color_up,    alpha=0.8, bins=n_bins, density=True)
                ax4.hist(demand_relative_prices_dn, color=self.color_dn,    alpha=0.8, bins=n_bins, density=True)
                ax3.set_title('Relative Prices Up, Activation demand only')
                ax4.set_title('Relative Prices Down, Activation demand only')
                    
                # [ax.set_xlim([-50, 150]) for ax in (ax1, ax2, ax3, ax4)]
                [ax.set_ylabel("Occurance rate (%)") for ax in (ax1, ax3)]
                [ax.set_xlabel("Bidding prices (€/MW)") for ax in (ax3, ax4)]

                fig.suptitle(f"Distribution of bidding prices relative to clearing prices ({market.bidding_zone}, {market.date}) \nCalculated as bidding price - clearing price")
                
                # filename = f"{run_sanitized}_relative_clearing_prices_{market.bidding_zone}"
                # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
                self.save_plot(f"relative_clearing_prices{market.bidding_zone}", run_id, fig, pdf)




            ######################################################
            #                   SPOT PRICES

            fig = plt.figure(figsize=config.plot_format)
            plt.step(t, self.controller.market.get_spotprice(), label="Spot price", where='post')
            plt.ylabel("Spot price (kr/kWh)")
            plt.xlabel("Time (days)")
            plt.legend()

            # filename = "spot_price"
            # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)

            self.save_plot(f"spot_price_{self.controller.market.bidding_zone}", run_id, fig, pdf)
            plt.close('all')









    def plot_freshweights(self, plot_name = 'fresh_weights', run_id = None, runs = {}, pdf = None):
        config = self.config
        controller = self.controller

        fig = plt.figure(figsize=config.plot_format)
        
        for run in runs:

            x = runs[run]['timeseries']['x']
            x_fw = controller.model.freshweight(x[:,1:])
            t = runs[run]['timeseries']['t']

            '''
            if(run_name in sorted_runs[1]):          # Level 1 runs are market based, but do not have activation data
                bid_ts  = self.controller.optimization_results['runs'][run_name]['timeseries']
                bid_volumes_up   = bid_ts['P_up']    # Volume up
                bid_volumes_dn   = bid_ts['P_dn']    # Volume down
                bid_prices_up    = bid_ts['C_up']    # Price up
                bid_prices_dn    = bid_ts['C_dn']    # Price down

                u_nom = controller.optimization_results['runs'][run_name]['timeseries']['u_nom'] # Unsure if this is right or if 'u' is more correct
                x_bid = controller.optimization_results['runs'][run_name]['timeseries']['x']
                fw_variance = propagate_process_covariance(controller, x_bid, u_nom, bid_volumes_up, bid_volumes_dn, bid_prices_up, bid_prices_dn)
                fw_sd = np.sqrt(fw_variance)
                fw = np.array(controller.model.freshweight(x_bid[:,1:])).flatten()
                fw_ub = fw + 1.96*fw_sd
                fw_lb = fw - 1.96*fw_sd

                plt.fill_between(t, fw_lb, fw_ub, color='green', alpha=0.2)
            '''
                
            plt.plot(t, x_fw, label=f"{run} (g/plant)")


        plt.axhline(y=controller.model.Final_fw_sht, color='gray', linestyle=':', label="Required Freshweight (g/plant)")
        plt.ylabel("Weight (g/plant)")
        plt.xlabel("Time (days)")
        plt.legend(loc="upper left")
        plt.title(f"Expected freshweight of plant growth (g/plant). ({controller.market.date}, {controller.market.bidding_zone})")

        self.save_plot(plot_name, run_id=run_id, fig=fig, pdf=pdf)



    def plot_light_schedules(self, plot_name = 'light_schedules', run_id = None, runs = {}, pdf = None):

        ######################################################
        #                   LIGHT SCHEDULE

        controller  = self.controller
        config      = self.config
        spot_prices = controller.spot_prices
        market      = controller.market

        n_runs = len(runs)

        fig, axes = plt.subplots(n_runs+1, 1, figsize=config.plot_format, sharex=True)

        for i, run in enumerate(runs):
            ax = axes[i]
            u = runs[run]['timeseries']['u'].flatten()
            t = runs[run]['timeseries']['t'].flatten()
            ax.step(t, u, label=f"{run}", where='post') 
            ax.set_ylabel(self.controller.model.u_unit, rotation=0)
            ax.legend(loc="upper right")

        ax = axes[-1]
        ax.step(t, spot_prices, label=f"Spot price", color='gray', where='post') 
        ax.set_ylabel("NOK/kWh", rotation=0)
        ax.set_xlabel("Time (days)")
        ax.legend(loc="upper right")
        fig.suptitle(f'Light schedules ({market.date}, {market.bidding_zone})')

        self.save_plot(plot_name, run_id = run_id, fig=fig, pdf=pdf)




    def plot_DLI(self, plot_name = 'daily_light_integrals', run_id = None, runs = [], pdf = None):
        
        ######################################################
        #          DAILY LIGHT INTEGRALS OVER TIME


        controller  = self.controller
        config      = self.config
        spot_prices = controller.spot_prices
        market      = controller.market

        n_runs = len(runs)

        fig, axes = plt.subplots(n_runs+1, 1, figsize=config.plot_format, sharex=True)

        for i, run in enumerate(runs):
            ax = axes[i]
            X = runs[run]['timeseries']['x']
            t = runs[run]['timeseries']['t']
            DLI = get_DLI(X).flatten()
            ax.plot(t[-len(DLI):], DLI, label=f"{run}") 
            ax.set_ylabel('Daily light integral', rotation=0)
            ax.axhline(y=controller.model.DLI_max, color='gray', linestyle=':')
            ax.axhline(y=controller.model.DLI_min, color='gray', linestyle=':')
            ax.legend(loc="upper right")

        ax = axes[-1]
        ax.step(t, spot_prices, label=f"Spot price", color='gray') 
        ax.set_ylabel("NOK/kWh", rotation=0)
        ax.set_xlabel("Time (days)")
        ax.legend(loc="upper right")
        fig.suptitle(f'Daily light integrals ({market.date}, {market.bidding_zone})')

        self.save_plot(plot_name, fig = fig, run_id=run_id, pdf=pdf)



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
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)















    def plot_spot_mfrr_prices(self):

        config = self.config
        controller = self.controller
        market = controller.market

        # Remove drastic outliers
        price_data_raw = market.prices_working_set
        price_data = price_data_raw.where(price_data_raw['Clearing Price Up']<500).where(price_data_raw['Clearing Price Down']>-500).dropna()

        timestamps      = price_data['Start Time']
        spot_prices     = np.array(price_data['Spot Price'])
        mfrr_prices_up  = np.array(price_data['Clearing Price Up'])
        mfrr_prices_dn  = np.array(price_data['Clearing Price Down'])
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
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)



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
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)


        ###############################################################
        #             CLEARING PRICES HISTOGRAM 
        #               [OUTLIERS REMOVED]
        
        plt.figure(figsize=config.plot_format)

        n_bins = 100

        mu_up, mu_dn = market.price_means[2:4]
        sigma_up, sigma_dn = np.sqrt(market.price_covs[0][1, 1]), np.sqrt(market.price_covs[1][1, 1])

        x_up = np.linspace(mu_up-3*sigma_up,mu_up+3*sigma_up, 1000)
        x_dn = np.linspace(mu_dn-3*sigma_dn,mu_dn+3*sigma_dn, 1000)

        plt.hist(mfrr_prices_up,    label="Clearing price up",   color='blue',  alpha=0.4, bins=n_bins, density=True)
        plt.hist(mfrr_prices_dn,    label="Clearing price down", color='red',   alpha=0.4, bins=n_bins, density=True)
        plt.hist(spot_prices_eur,   label="Spot prices",         color='grey',  alpha=0.4, bins=n_bins, density=True)
        plt.plot(x_up, stats.norm.pdf(x_up, mu_up, sigma_up), label="Estimated Up-price distribution", color='blue')
        plt.plot(x_dn, stats.norm.pdf(x_dn, mu_dn, sigma_dn), label="Estimated Down-price distribution", color='red')
        plt.xlabel("Bidding prices (€/MW)")
        plt.title(f"Clearing prices histogram normalized ({market.bidding_zone}, {market.date})")
        plt.legend()

        filename = f"histogram_mfrr_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)



        ################################################################
        #         SCATTER PLOT SPOT PRICE - CLEARING PRICES 

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

        ax2.scatter(spot_prices[idx], mfrr_prices_dn[idx], color='red', label='Clearing price down', s=0.1)
        ax2.plot(line_x, line_x * 0.9*1000/market.C_eur2nok, label='Upper limit: 0.9 x spot', color='grey', alpha=0.4)
        ax2.set_ylabel("Bidding price (€/MWh)")
        ax2.set_xlabel("Spot price (NOK/kWh)")
        ax2.set_xlim([-0.5, 4.5])
        ax2.legend()

        fig.suptitle(f"Spot prices with mFRR clearing prices €/MW ({market.bidding_zone}, {market.date})")

        filename = f"scatter_spot_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)

        plt.close('all')

        ######################################################
        #        CLEARING-SPOT RELATIVE PRICES HISTOGRAM
        #                   [OUTLIERS REMOVED]

        plt.figure(figsize=config.plot_format)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)


        ax1.hist(mfrr_prices_up-spot_prices_eur, label="Clearing price up", color='blue', alpha=0.4, bins=2*n_bins, density=True)
        ax1.set_xlabel("Bidding prices (€/MW)")
        ax1.set_xlim([-75, 25])
        ax1.legend()


        ax2.hist(mfrr_prices_dn-spot_prices_eur, label="Clearing price down", color='red', alpha=0.4, bins=n_bins, density=True)
        ax2.set_xlabel("Bidding prices (€/MW)")
        ax2.set_xlim([-75, 25])
        ax2.legend()

        plt.suptitle(f'Relative clearing prices, normalized ({market.bidding_zone}, {market.date})')

        filename = f"histogram_relative_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)


   
        ######################################################
        #           MONTHLY mFRR ACTIVATION FREQUENCIES


        activations_df = market.activations_full_set.copy()

        # Process data up
        activations_df['Start Time'] = pd.to_datetime(activations_df['Start Time'])
        activations_df.loc[:, 'Year-Month'] = activations_df['Start Time'].dt.to_period('M')
        n_up_activations    = activations_df[(activations_df.filter(like='Activated Up').sum(axis=1)) > 0].copy()
        n_down_activations  = activations_df[(activations_df.filter(like='Activated Down').sum(axis=1)) > 0].copy()

        # Count activations and total timestamps by Year-Month
        total_counts            = activations_df['Year-Month'].value_counts().sort_index()
        monthly_counts_up       = n_up_activations['Year-Month'].value_counts().sort_index()
        monthly_counts_down     = n_down_activations['Year-Month'].value_counts().sort_index()
        activation_rate_up      = (monthly_counts_up / total_counts * 100).sort_index()
        activation_rate_down    = (monthly_counts_down / total_counts * 100).sort_index()

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
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)


        ######################################################
        #           DAILY mFRR ACTIVATION COUNTS

        plt.figure(figsize=config.plot_format)

        # Process data
        activations_df['Start Time'] = pd.to_datetime(activations_df['Start Time'])
        activations_df.loc[:, 'Date'] = activations_df['Start Time'].dt.date

        # Ensure all dates from the full range are present for counting
        date_range = pd.date_range(start=activations_df['Date'].min(), end=activations_df['Date'].max())

        # Count activations and total timestamps by Date
        daily_counts_up = activations_df[(activations_df.filter(like='Activated Up').sum(axis=1)) > 0].groupby('Date').size().reindex(date_range, fill_value=0)
        daily_counts_down = activations_df[(activations_df.filter(like='Activated Down').sum(axis=1)) > 0].groupby('Date').size().reindex(date_range, fill_value=0)
        total_counts = activations_df.groupby('Date').size().reindex(date_range, fill_value=0)

        # Compute activation rate
        activation_rate_up_daily = (daily_counts_up / total_counts * 100).fillna(0)
        activation_rate_down_daily = (daily_counts_down / total_counts * 100).fillna(0)

        # Combine the two datasets into a single DataFrame
        combined_daily_rates = pd.DataFrame({
            'Date': date_range,
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
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)



        ######################################################
        #           ACTIVATION COUNTS HISTOGRAM

        activations_df = market.activations_full_set.copy()
        activation_counts_up = activations_df['Activated Up'].loc[activations_df['Activated Up'] > 0]
        activation_counts_down = activations_df['Activated Down'].loc[activations_df['Activated Down'] > 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.plot_format, sharex=True)

        n_bins = 48

        ax1.hist(activation_counts_up, label="Activated Up", color=self.color_up, alpha=0.8, bins=n_bins, density=True)
        ax2.hist(activation_counts_down, label="Activated Down", color=self.color_dn, alpha=0.8, bins=n_bins, density=True)

        p           = 1/4 * np.array([market.demand_prob_up(), market.demand_prob_down()])       # Probability of activation in each 15-min slot
        lmbda       = 1/np.array([np.mean(activation_counts_up), np.mean(activation_counts_down)])     # Exponential rate parameter (mean 200 MW per activation)
        num_hours   = 100000    # Number of simulated hours
        
        # Simulate activation occurrences
        random_activations = np.random.binomial(1, p, size=(num_hours, 4, 2))  
        capacity_exp_values = np.random.exponential(scale=1/lmbda, size=(num_hours, 4, 2))  
        # Apply activations
        random_activated_capacities = random_activations * capacity_exp_values

        # Sum over the 4 quarter-hour slots, keeping the (num_hours, 2) structure
        hourly_totals = random_activated_capacities.sum(axis=1)

        # Filter values to remove values lower than 10MW as this is the lower limit in the Norway mfrr market
        hourly_totals = hourly_totals[
            (hourly_totals[:, 0] >= 10) & (hourly_totals[:, 0] < max(activation_counts_up)) & 
            (hourly_totals[:, 1] >= 10) & (hourly_totals[:, 1] < max(activation_counts_down))
        ]

        ax1.hist(hourly_totals[:,0], label=f"Random samples P={p[0]:.2f} lambda = 1/{(1/lmbda[0]):.2f}", color='navy', histtype='step', bins=n_bins, density=True)
        ax2.hist(hourly_totals[:,1], label=f"Random samples P={p[1]:.2f}, lambda = 1/{(1/lmbda[1]):.2f}", color='maroon', histtype='step', bins=n_bins, density=True)

        plt.suptitle(f'Activated capacity volumes in {market.bidding_zone} bidding zone')
        ax1.set_xlabel('Power (MW)')
        ax2.set_xlabel('Power (MW)')
        ax1.set_ylabel('Probability of occurrence')
        ax1.set_title('Activated Up')
        ax2.set_title('Activated Down')
        ax1.legend()
        ax2.legend()
        plt.tight_layout()

        filename = f"mFRR_activation_volumes_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)


        ######################################################
        #           MARKET POTENCY BAR CHART

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=False)

        market_potency_df = market.daily_market_potency_df
        rolling_market_potency_df = market.rolling_market_potency_df

        market_potency_df.set_index('Date')[['Potency Up', 'Potency Down']].plot(
            ax = ax2, kind='bar', stacked=True, figsize=(15, 6), color=['skyblue', 'lightcoral']
        )

        rolling_market_potency_df.set_index('Date')[['Rolling Potency Up', 'Rolling Potency Down', 'Rolling Total Potency']].plot(
            ax=ax1, linewidth=2.5, linestyle='-', color=[self.color_up, self.color_dn, 'grey']
        )

        first_of_month_daily    = market_potency_df['Date'][market_potency_df['Date'].dt.day == 1]

        ax1.legend(['Rolling Potency Up', 'Rolling Potency Down', 'Rolling Total Potency'], title='Activation Direction')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Rolling Window Market potency')
        ax1.set_title(f'Market potency of next {market.T} days from given date')
        
        ax2.legend(['Potency Up', 'Potency Down'], title='Activation Direction')
        ax2.set_xticks(first_of_month_daily.index)
        ax2.set_xticklabels(first_of_month_daily.dt.strftime('%Y-%m-%d'), rotation=0)
        ax2.set_ylabel('Market potency')
        ax2.set_title('Daily market potencies')

        plt.suptitle(f'Market potencies in bidding zone: {market.bidding_zone}\nCalculated from daily activation rate times mean clearing price')
        plt.tight_layout()

        filename = f"mFRR_market_potency_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)



        plt.close('all')
        print(f'Market analysis plots saved to {self.config.data_analysis_path}')




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
        plt.savefig(config.plot_path + foldername + "/MPC_" + filename + "." + self.plot_file_type, format=self.plot_file_type)
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
        plt.savefig(config.plot_path + foldername + "/MPC_" + filename + "." + self.plot_file_type, format=self.plot_file_type)




    def plot_price_prediction(self):

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
            bid_volumes_up  = bid_ts['P_up']
            bid_volumes_dn  = bid_ts['P_dn']
            bid_prices_up  = bid_ts['C_up']
            bid_prices_dn  = bid_ts['C_dn']
            bid_activations_up  = np.array(self.controller.market.activation_prob_up(spot_prices, bid_prices_up)).flatten()
            bid_activations_dn  = np.array(self.controller.market.activation_prob_down(spot_prices, bid_prices_dn)).flatten()
            

            pred_prices_up = market.opt_prices_up.flatten()
            pred_prices_dn = market.opt_prices_down.flatten()

            pred_a_up = np.array(self.controller.market.activation_prob_up(spot_prices, pred_prices_up)).flatten()
            pred_a_dn  = np.array(self.controller.market.activation_prob_down(spot_prices, pred_prices_dn)).flatten()

            # Filter out the unreasonably low bid activations
            activation_th = 0.01
            volume_th = 1e-3
            filtered_bid_prices_up = np.where(np.logical_and(bid_activations_up > activation_th, bid_volumes_up>volume_th), bid_prices_up, 0).flatten()
            filtered_bid_prices_dn = np.where(np.logical_and(bid_activations_dn > activation_th, bid_volumes_dn>volume_th), bid_prices_dn, 0).flatten()
            filtered_prob_activations_up = np.where(np.logical_and(bid_activations_up > activation_th, bid_volumes_up>volume_th), bid_activations_up, 0).flatten()
            filtered_prob_activations_dn = np.where(np.logical_and(bid_activations_dn > activation_th, bid_volumes_dn>volume_th), bid_activations_dn, 0).flatten()

            filtered_pred_prices_up = np.where(pred_a_up > activation_th, pred_prices_up, 0).flatten()
            filtered_pred_prices_dn = np.where(pred_a_dn > activation_th, pred_prices_dn, 0).flatten()
            filtered_pred_a_up = np.where(pred_a_up > activation_th, pred_a_up, 0).flatten()
            filtered_pred_a_dn = np.where(pred_a_dn > activation_th, pred_a_dn, 0).flatten()

            ##################################################
            # plt.figure(figsize=config.plot_format)

            # Create a figure with two subplots sharing the same x-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)

            # Plot for Bidding Price Up
            ax1.fill_between(t, 0, filtered_bid_prices_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax1.fill_between(t, 0,filtered_bid_prices_dn, label="Bidding Price Down", color="red", step='post', alpha=0.4)
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
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)





            ##################################################

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format, sharex=True)

            # Plot for Bidding Price Up
            ax1.fill_between(t, 0, filtered_prob_activations_up, label="Bidding Price Up",   color="blue", step='post', alpha=0.4)
            ax1.fill_between(t, 0, filtered_prob_activations_dn, label="Bidding Price Down", color="red", step='post', alpha=0.4)
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
            plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)


    def plot_financial_report(self):

        config      = self.config
        controller  = self.controller
        market      = controller.market
        foldername  = self.foldername
        n_runs      = len(list(self.controller.optimization_results['runs'].keys()))
        t           = self.controller.t
        spot_prices = self.controller.spot_prices
                

        costs       = [controller.optimization_results['runs'][run]['metrics']['Costs']       for run in controller.optimization_results['runs']]
        earnings    = [controller.optimization_results['runs'][run]['metrics']['Earnings']    for run in controller.optimization_results['runs']]
        totals      = [controller.optimization_results['runs'][run]['metrics']['Total']       for run in controller.optimization_results['runs']]
        cost_reduction_percent = [(totals[0] - totals[i])/totals[0] * 100 for i in range(len(totals))]

        header=[run for run in controller.optimization_results['runs']]

        ######################################################
        #               FINALCIAL REPORT
        #

        plt.figure(figsize=config.plot_format)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.plot_format)

        x = np.arange(len(header))  # Bar positions
        bar_width = 0.2

        # Bar chart
        ax1.bar(x - bar_width, costs, width=bar_width, color='lightcoral', label="Costs")
        ax1.bar(x, earnings, width=bar_width, color='limegreen', label="Earnings")
        ax1.bar(x + bar_width, totals, width=bar_width, color='lightskyblue', label="Totals")
        ax1.axhline(y=min(totals), color='gray', linestyle='-', linewidth=0.1)


        ax1.set_xticks(x)
        ax1.set_xticklabels(header)
        ax1.set_ylabel("Amount ($)")
        ax1.legend(loc='lower left')
        ax1.set_title("Financial Overview of Optimization Methods")

        # Table
        metrics_table = [[''] + header] + get_metrics_table_raw(controller.optimization_results['runs'])

        ax2.axis("tight")
        ax2.axis("off")
        ax2.table(cellText=metrics_table, cellLoc='center', loc='center', colWidths=[0.2] + [0.15]*len(header))


        fig.suptitle('')
        plt.tight_layout()

        filename = f"_financial_report_{market.date.replace('-','_')}_{market.bidding_zone}"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)

        plt.close('all')



        ######################################################
        #               SPECS REPORT
        #


        plt.figure(figsize=config.plot_format)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=config.plot_format)

        # Table

        table_1 = dict_to_table(controller.specs)
        ax1.axis("tight")
        ax1.axis("off")
        ax1.table(cellText=table_1, cellLoc='center', loc='center', colWidths=[0.5, 0.4])
        ax1.set_title('Controller specs')

        table_2 = dict_to_table(controller.model.specs)
        ax2.axis("tight")
        ax2.axis("off")
        ax2.table(cellText=table_2, cellLoc='center', loc='center', colWidths=[0.5, 0.4])
        ax2.set_title('Model specs')
        
        table_3 = dict_to_table(market.specs)
        ax3.axis("tight")
        ax3.axis("off")
        ax3.table(cellText=table_3, cellLoc='center', loc='center', colWidths=[0.5, 0.4])
        ax3.set_title('Market specs')


        ax4.axis("tight")
        ax4.axis("off")

        fig.suptitle(f'Specs for {config.sim_name}')
        plt.tight_layout()

        filename = f"_specs_{market.date.replace('-','_')}_{market.bidding_zone}"
        plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)

        plt.close('all')


        return 0            

