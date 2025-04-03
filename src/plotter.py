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
import shutil
import time



class Plotter():

    settings: Settings
    config: Config
    controller: Controller
    simulator: Simulator
    foldername: str

    color_up        = 'skyblue'
    color_up_dark   = 'navy'
    color_down      = 'lightcoral'
    color_down_dark = 'maroon'


    common_dependencies = ('general', 'controller', 'market', 'plantmodel', 'plotter')
    n_plots = 0
    plot_queue = []
    progressbar: tqdm

    def __init__(self, settings: Settings, config, controller=None, simulator=None):
        self.settings = settings
        self.config = config
        self.controller = controller
        self.simulator = simulator

        self.plotter_settings = settings.get_settings_group('options', 'general', 'plotter')

        self.foldername     = self.plotter_settings['SIM_NAME']
        self.plot_file_type = self.plotter_settings['PLOT_EXPORT_TYPE']
        self.activation_th  = self.plotter_settings['ACTIVATION_THRESHOLD']
        self.volume_th      = self.plotter_settings['VOLUME_THRESHOLD']
        self.aspect_ratio   = self.plotter_settings['PLOT_ASPECT_RATIO']
        self.search_plot_cache   = self.plotter_settings['SEARCH_PLOT_CACHE']


    def save_plot(self, filename, run_id = None, fig = None, pdf= None):
        config = self.config
        foldername  = self.foldername

        directory = config.plots_path + foldername + "/"
        if run_id is not None: directory += run_id + '/'

        plt.savefig(directory + filename + "." + self.plot_file_type, format=self.plot_file_type)

        if fig is not None: 
            if pdf is not None: pdf.savefig(fig)
            plt.close(fig)
            self.progressbar.update(1)
            # self.progressbar.set_description(f"Plotting {run_id} {filename}")  # Updates dynamically
            self.progressbar.set_postfix(status=f"Plotting {run_id} {filename}")  # Adds a small status message


        

    def create_folder_environment(self, plot_runs):
        config      = self.config

        for run_id in plot_runs:
      
            plot_folder = os.path.join(config.plots_path, f"{config.sim_name}/{run_id}/")
            
            os.makedirs(plot_folder, exist_ok=True)



    def save_ocp_plots(self):

        start_time = time.time()


        # COMMON PLOTS
        runs = self.controller.optimization_results['runs']
        self.add_plot(self.plot_freshweights,       runs = runs)
        self.add_plot(self.plot_light_schedules,    runs = runs)
        self.add_plot(self.plot_DLI,                runs = runs)


        plot_runs = [run for run in runs if runs[run]['plot_run'] == True]
        self.create_folder_environment(plot_runs)

        # INDIVIDUAL PLOTS
        run_groups = build_dependency_groups(self.controller.optimization_results)
        for run_group in run_groups:
            for i, run in enumerate(run_group):
                if run in plot_runs:
                    self.add_plot(self.plot_report, run_id = run, run_group = run_group[i:])
            
            
        
        self.plot()

        # Save plot cache in file system

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f'Optimization plots saved to {self.config.current_sim_plot_path}. \nPlot time: {minutes} minutes and {seconds:.2f} seconds.')


    def add_plot(self, function, *args, **kwargs):

        self.plot_queue.append((function, args, kwargs))

        
    def plot(self):

        with tqdm(desc=f"Plotting optimization results ...") as pbar:
            self.progressbar = pbar
            for func, args, kwargs in self.plot_queue:
                
                func(*args, **kwargs)
                # pbar.update(1)


    def update_plot_log(self, run_id, dependencies):
        current_dir     = self.config.current_sim_plot_path  # Folder for plots from the current run
        dependencies    = dependencies + ('plotter',)
        current_hash    = generate_hash(self.settings.get_settings_group(*dependencies))

        # Make sure there is a plot_log.json file in the current dir.
        # Add an entry to it containing the run_id and the current_hash. Alternatively update the already existing entry

        log_file_path = os.path.join(current_dir, "plot_log.json")

        # Load existing log if present
        if os.path.exists(log_file_path):
            with open(log_file_path, "r", encoding="utf-8") as f:
                try:
                    plot_log = json.load(f)
                except json.JSONDecodeError:
                    plot_log = {}  # Reset file in case it's corrupted
        else:
            plot_log = {}

        plot_log[run_id] = current_hash

        # Save
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(plot_log, f, indent=4)


        
    def find_existing_plot(self, run_id, dependencies):
        """
        Checks if a plot matching the current run's hash exists in any of the cached plot folders.

        Returns:
            1 if a matching plot is found 
            0 if no matching plot is found.
        """
        if not self.search_plot_cache: return 0 # Don't look for any old plots

        plots_folder    = self.config.plots_path                # Parent folder of all plot folders
        current_dir     = self.config.current_sim_plot_path     # Folder for plots from the current run
        dependencies    = dependencies + ('plotter',)
        current_hash    = generate_hash(self.settings.get_settings_group(*dependencies))

        for folder in os.listdir(plots_folder):
            folder_path = os.path.join(plots_folder, folder)

            if not os.path.isdir(folder_path):
                continue

            log_file_path = os.path.join(folder_path, "plot_log.json")

            # Skip if no log file
            if not os.path.exists(log_file_path):
                continue  

            with open(log_file_path, "r") as log_file:
                try:
                    plot_log = json.load(log_file)
                except json.JSONDecodeError:
                    print(f'WARNING: \tCorrupt file found when searching for {run_id} in {log_file_path}')
                    continue

            # Check log_files
            for logged_run_id, logged_hash in plot_log.items():
                if run_id == logged_run_id and current_hash == logged_hash:
                    if folder_path == current_dir:
                        return 1  # Matching plot exists in the current directory, no need to update plot log
                    
                    # Copy matching PDF report
                    pdf_name = f"report_{run_id}.pdf"
                    pdf_src = os.path.join(folder_path, pdf_name)
                    pdf_dst = os.path.join(current_dir, pdf_name)

                    if os.path.exists(pdf_src):
                        shutil.copy2(pdf_src, pdf_dst)

                    # Copy entire run folder
                    run_folder_src = os.path.join(folder_path, str(run_id))
                    run_folder_dst = os.path.join(current_dir, str(run_id))

                    if os.path.exists(run_folder_src) and os.path.isdir(run_folder_src):
                        if os.path.exists(run_folder_dst):
                            shutil.rmtree(run_folder_dst)  # Remove existing run folder to avoid conflicts
                        shutil.copytree(run_folder_src, run_folder_dst)  # Copy the entire folder

                    self.update_plot_log(run_id, dependencies)

                    return 1  # Matching plot found and copied overto current directory

        return 0  # No matching plot found

    def plot_report(self, run_id, run_group):
        config      = self.config
        controller  = self.controller
        # market      = controller.market
        # t           = self.controller.t
        # N           = self.controller.N
        # spot_prices = self.controller.spot_prices
        # zone        = market.bidding_zone

        # run_id = run_group[0]

        # if not controller.optimization_results['runs'][run_id]['Attributes']['Bids']: return
        # if controller.optimization_results['runs'][run_id]['Attributes']['Balancing_market'] == 'None': return

        dependencies = tuple(controller.optimization_results['runs'][run_id]['dependencies'])

        if self.find_existing_plot(run_id, dependencies): return

        
        final_report_save_path = f'{config.plots_path}{self.foldername}/report_{run_id}.pdf'

        with PdfPages(final_report_save_path) as pdf:

            # bid_ts = self.controller.optimization_results['runs'][run_id]['timeseries']
            # if 'A_up' in bid_ts and 'A_dn' in bid_ts:
            #     is_realized = True                      # Bool used to check if there is real activation data to plot
            # else:
            #     is_realized = False

            # u_nom            = bid_ts.get('u_nom', np.zeros(N)).flatten()
            # bid_volumes_up   = bid_ts.get('P_up',  np.zeros(N)).flatten()
            # bid_volumes_down = bid_ts.get('P_dn',  np.zeros(N)).flatten()
            # bid_prices_up    = bid_ts.get('C_up',  np.zeros(N)).flatten()
            # bid_prices_down  = bid_ts.get('C_dn',  np.zeros(N)).flatten()

            # if is_realized:
            #     bid_activation_up   = bid_ts['A_up'].flatten()
            #     bid_activation_down = bid_ts['A_dn'].flatten()

            # balancing_market = market.get_balancing_market(controller.optimization_results['runs'][run_id]['Attributes']['Balancing_market'])
        
            # prob_activation_up      = np.array(balancing_market.activation_prob_up(spot_prices, bid_prices_up)).flatten()
            # prob_activation_down    = np.array(balancing_market.activation_prob_down(spot_prices, bid_prices_down)).flatten()

            # # Filter out the unreasonably low bid activations
            # filtered_bid_prices_up      = np.where(np.logical_and(prob_activation_up   > self.activation_th, bid_volumes_up   > self.volume_th), bid_prices_up,     0).flatten()
            # filtered_bid_prices_down    = np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th), bid_prices_down,   0).flatten()
            # filtered_bid_volumes_up     = np.where(np.logical_and(prob_activation_up   > self.activation_th, bid_volumes_up   > self.volume_th), bid_volumes_up,    0).flatten()
            # filtered_bid_volumes_down   = np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th), bid_volumes_down,  0).flatten()


            ########################################################
            #                   COMMON PLOTS


            runs = {run: controller.optimization_results['runs'][run] for run in reversed(run_group) if run in controller.optimization_results['runs']}
            self.plot_freshweights      (runs = runs, run_id = run_id, pdf = pdf)
            self.plot_light_schedules   (runs = runs, run_id = run_id, pdf = pdf)
            self.plot_DLI               (runs = runs, run_id = run_id, pdf = pdf)

            self.plot_bidding_analysis  (run_id=run_id, pdf=pdf)
            self.plot_market_report     (run_id=run_id, pdf=pdf)
            self.plot_spot_prices       (run_id=run_id, pdf=pdf)

        self.update_plot_log(run_id, dependencies)



        '''
            ######################################################
            #                   BIDDING VOLUMES

            fig = plt.figure(figsize=self.aspect_ratio)

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

            fig, ax = plt.subplots(1, 1, figsize=self.aspect_ratio, sharex=True)

            ax.fill_between(t, 0, filtered_bid_prices_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax.fill_between(t, 0, filtered_bid_prices_down, label="Bidding Price Down", color="red", step='post', alpha=0.4)
            ax.step(t, spot_prices, label="Spot price", color="grey", linestyle="--", where='post')
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

            # plt.figure(figsize=self.aspect_ratio)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.aspect_ratio, sharex=True)

            ax1.fill_between(t, 0, prob_activation_up, color=self.color_up, alpha=0.8, label="Expected", step='post', linewidth=0)
            ax2.fill_between(t, 0, prob_activation_down, color=self.color_dn, alpha=0.8, label="Expected", step='post', linewidth=0)

            if is_realized:
                ax1.fill_between(t, 0, np.multiply(bid_activation_up, prob_activation_up), color='navy', alpha=1, label="Activated", step='post', linewidth=0)
                ax2.fill_between(t, 0, np.multiply(bid_activation_down, prob_activation_down), color='maroon', alpha=1, label="Activated", step='post', linewidth=0)


            ax1.step(t, balancing_market.demand_prob_up(controller.spot_prices).reshape((-1,1)),   color='gray', linestyle=':', label="Max activation rate", where='post')
            ax2.step(t, balancing_market.demand_prob_down(controller.spot_prices).reshape((-1,1)), color='gray', linestyle=':', label="Max activation rate", where='post')
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

            # plt.figure(figsize=self.aspect_ratio)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio)

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

                # plt.figure(figsize=self.aspect_ratio)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.aspect_ratio, sharex=True)

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

                # plt.figure(figsize=self.aspect_ratio)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.aspect_ratio, sharex=True)


                clearing_prices_up   = balancing_market.clearing_prices_up
                clearing_prices_down = balancing_market.clearing_prices_down
                activations_up       = balancing_market.activations_up
                activations_down     = balancing_market.activations_down
                expected_prices_up   = balancing_market.expected_prices_up
                expected_prices_down = balancing_market.expected_prices_down

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

                # plt.figure(figsize=self.aspect_ratio)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio)

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

                # plt.figure(figsize=self.aspect_ratio)
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.aspect_ratio, sharex=True)
                
                activated_clearing_prices_up   = clearing_prices_up[np.where(activations_up)]
                activated_clearing_prices_down = clearing_prices_down[np.where(activations_down)]
                
                n_bins = 50

                #TODO This is a little redundant. Cut down
                ax1.hist(clearing_prices_up,    label="Expected", color=self.color_up, alpha=0.8, bins=n_bins, density=True)
                ax2.hist(clearing_prices_down,  label="Expected", color=self.color_dn, alpha=0.8, bins=n_bins, density=True)
                
                ax1.hist(clearing_prices_up,    label="Recorded", color='navy', histtype='step', alpha=0.8, bins=n_bins, density=True)
                ax2.hist(clearing_prices_down,  label="Recorded", color='maroon', histtype='step', alpha=0.8, bins=n_bins, density=True)

                ax3.hist(activated_clearing_prices_up,   label="Activated", color=self.color_up, alpha=0.8, bins=30, density=True)
                ax4.hist(activated_clearing_prices_down, label="Activated", color=self.color_dn, alpha=0.8, bins=30, density=True)

                # Gaussian pdfs of clearing prices
                mu_up   = balancing_market.price_stats[zone]['Up']['means'][1]
                mu_down = balancing_market.price_stats[zone]['Down']['means'][1]

                sigma_up   = np.sqrt(balancing_market.price_stats[zone]['Up']['cov'][1,1])
                sigma_down = np.sqrt(balancing_market.price_stats[zone]['Down']['cov'][1,1])
                                
                x_up = np.linspace(mu_up -3*sigma_up,     mu_up +3*sigma_up,     1000)
                x_dn = np.linspace(mu_down -3*sigma_down, mu_down +3*sigma_down, 1000)
                
                ax3.plot(x_up,  stats.norm.pdf(x_up, mu_up, sigma_up),      label="Estimate of activated prices", color='navy')
                ax4.plot(x_dn,  stats.norm.pdf(x_dn, mu_down, sigma_down),  label="Estimate of activated prices", color='maroon')

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

                # plt.figure(figsize=self.aspect_ratio)
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.aspect_ratio, sharex=True)

                n_bins = 50

                relative_prices_up = bid_prices_up - clearing_prices_up
                relative_prices_dn = bid_prices_down - clearing_prices_down
                demand_relative_prices_up = relative_prices_up[np.where(np.logical_and(np.logical_and(activations_up == 1, prob_activation_up > self.activation_th), bid_volumes_up > self.volume_th))]
                demand_relative_prices_dn = relative_prices_dn[np.where(np.logical_and(np.logical_and(activations_down == 1, prob_activation_down > self.activation_th), bid_volumes_down > self.volume_th))]

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

            fig = plt.figure(figsize=self.aspect_ratio)
            plt.step(t, self.controller.market.get_spotprice(), label="Spot price", where='post')
            plt.ylabel("Spot price (€/MWh)")
            plt.xlabel("Time (days)")
            plt.legend()

            self.save_plot(f"spot_price_{self.controller.market.bidding_zone}", run_id, fig, pdf)
            plt.close('all')
            '''


        




    def plot_market_report(self, run_id, pdf):


        config      = self.config
        controller  = self.controller
        market      = controller.market
        t           = self.controller.t
        N           = self.controller.N
        spot_prices = self.controller.spot_prices
        zone        = market.bidding_zone

        # if not controller.optimization_results['runs'][run_id]['Attributes']['Bids']: 
        #     print(f'WARNING: Run {run_id} was scheduled for analysis plots of bids without having any bids to analyze.')
        #     return
        # if not controller.optimization_results['runs'][run_id]['Attributes']['Activations']: 
        #     print(f'WARNING: Run {run_id} was scheduled for market report without having any registered activation data.')
        #     return

        for market_type, market_data in controller.optimization_results['runs'][run_id]['markets'].items():
            if market_data['Activations'] is None: continue

            balancing_market = self.controller.market.get_balancing_market(market_type)
            # bid_ts = self.controller.optimization_results['runs'][run_id]['timeseries']

            # u_nom               = bid_ts.get('u_nom', np.zeros(N)).flatten()
            bid_volumes_up      = market_data['Bids']['Up']['Volume'].flatten()
            bid_volumes_down    = market_data['Bids']['Down']['Volume'].flatten()
            bid_prices_up       = market_data['Bids']['Up']['Price'].flatten()
            bid_prices_down     = market_data['Bids']['Down']['Price'].flatten()
            bid_activation_up   = market_data['Activations']['Up'].flatten()
            bid_activation_down = market_data['Activations']['Down'].flatten()
        
            prob_activation_up      = np.array(balancing_market.activation_prob_up(spot_prices, bid_prices_up)).flatten()
            prob_activation_down    = np.array(balancing_market.activation_prob_down(spot_prices, bid_prices_down)).flatten()

            # Filter out the unreasonably low bid activations
            filtered_bid_prices_up      = np.where(np.logical_and(prob_activation_up   > self.activation_th, bid_volumes_up   > self.volume_th), bid_prices_up,     0).flatten()
            filtered_bid_prices_down    = np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th), bid_prices_down,   0).flatten()
            filtered_bid_volumes_up     = np.where(np.logical_and(prob_activation_up   > self.activation_th, bid_volumes_up   > self.volume_th), bid_volumes_up,    0).flatten()
            filtered_bid_volumes_down   = np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th), bid_volumes_down,  0).flatten()


            # Filter out non-activated bids
            bid_prices_up_activated     = np.where(np.logical_and(bid_activation_up     > self.activation_th, bid_volumes_up    > self.volume_th,), bid_prices_up,    0)
            bid_prices_dn_activated     = np.where(np.logical_and(bid_activation_down   > self.activation_th, bid_volumes_down  > self.volume_th,), bid_prices_down,  0)
            bid_volumes_up_activated    = np.where(np.logical_and(bid_activation_up     > self.activation_th, bid_volumes_up    > self.volume_th,), bid_volumes_up,   0)
            bid_volumes_dn_activated    = np.where(np.logical_and(bid_activation_down   > self.activation_th, bid_volumes_down  > self.volume_th,), bid_volumes_down, 0)
            
            
            ######################################################
            #    ACTIVATED BIDDING VOLUMES AND PRICES
            #               [FILTERED]

            # plt.figure(figsize=self.aspect_ratio)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.aspect_ratio, sharex=True)

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

            self.save_plot("results_activated_volumes_prices", run_id, fig, pdf)


            ######################################################
            #    EXPECTED VS RECORDED CLEARING PRICES 
            #

            # plt.figure(figsize=self.aspect_ratio)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.aspect_ratio, sharex=True)


            clearing_prices_up   = balancing_market.clearing_prices_up
            clearing_prices_down = balancing_market.clearing_prices_down
            activations_up       = balancing_market.activations_up
            activations_down     = balancing_market.activations_down
            expected_prices_up   = balancing_market.expected_prices_up
            expected_prices_down = balancing_market.expected_prices_down

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
            ax2.step(t, expected_prices_down.flatten(), color=self.color_down, label="Expected", linewidth=linewidth, where='post')
            ax2.step(t, filtered_bid_prices_down, color='grey',  label="Submitted", linewidth=linewidth, where='post')
            ax2.set_ylabel("Price Down (€/MW)")
            ax2.set_xlabel("Time (days)")
            ax2.legend()



            fig.suptitle(f"{run_id} Expected vs recorded clearing prices. ({controller.market.date}, {controller.market.bidding_zone})\nExpectations made based on price covariances")

            self.save_plot("expected_vs_recorded_clearing_prices", run_id, fig, pdf)

            ######################################################
            #    PROJECTED VS RECORDED ACTIVATION CHANCES 
            #

            # plt.figure(figsize=self.aspect_ratio)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio)

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
            ax2.plot(probs_dn_idx, probs_dn, color=self.color_down, marker='o')

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

            self.save_plot("expected_vs_true_probabilities", run_id, fig, pdf)

            plt.close('all')

            ######################################################
            #      RECORDED VS HISTORICAL CLEARING PRICES
            #

            # plt.figure(figsize=self.aspect_ratio)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.aspect_ratio, sharex=True)
            
            activated_clearing_prices_up   = clearing_prices_up[np.where(activations_up)]
            activated_clearing_prices_down = clearing_prices_down[np.where(activations_down)]
            
            n_bins = 50

            #TODO This is a little redundant. Cut down
            ax1.hist(clearing_prices_up,    label="Expected", color=self.color_up, alpha=0.8, bins=n_bins, density=True)
            ax2.hist(clearing_prices_down,  label="Expected", color=self.color_down, alpha=0.8, bins=n_bins, density=True)
            
            ax1.hist(clearing_prices_up,    label="Recorded", color='navy', histtype='step', alpha=0.8, bins=n_bins, density=True)
            ax2.hist(clearing_prices_down,  label="Recorded", color='maroon', histtype='step', alpha=0.8, bins=n_bins, density=True)

            ax3.hist(activated_clearing_prices_up,   label="Activated", color=self.color_up, alpha=0.8, bins=30, density=True)
            ax4.hist(activated_clearing_prices_down, label="Activated", color=self.color_down, alpha=0.8, bins=30, density=True)

            # Gaussian pdfs of clearing prices
            mu_up   = balancing_market.price_stats[zone]['Up']['means'][1]
            mu_down = balancing_market.price_stats[zone]['Down']['means'][1]

            sigma_up   = np.sqrt(balancing_market.price_stats[zone]['Up']['cov'][1,1])
            sigma_down = np.sqrt(balancing_market.price_stats[zone]['Down']['cov'][1,1])
                            
            x_up = np.linspace(mu_up -3*sigma_up,     mu_up +3*sigma_up,     1000)
            x_dn = np.linspace(mu_down -3*sigma_down, mu_down +3*sigma_down, 1000)
            
            ax3.plot(x_up,  stats.norm.pdf(x_up, mu_up, sigma_up),      label="Estimate of activated prices", color='navy')
            ax4.plot(x_dn,  stats.norm.pdf(x_dn, mu_down, sigma_down),  label="Estimate of activated prices", color='maroon')

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

            # plt.figure(figsize=self.aspect_ratio)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.aspect_ratio, sharex=True)

            n_bins = 50

            relative_prices_up = bid_prices_up - clearing_prices_up
            relative_prices_dn = bid_prices_down - clearing_prices_down
            demand_relative_prices_up = relative_prices_up[np.where(np.logical_and(np.logical_and(activations_up == 1, prob_activation_up > self.activation_th), bid_volumes_up > self.volume_th))]
            demand_relative_prices_dn = relative_prices_dn[np.where(np.logical_and(np.logical_and(activations_down == 1, prob_activation_down > self.activation_th), bid_volumes_down > self.volume_th))]

            relative_prices_up = relative_prices_up[np.where(np.logical_and(prob_activation_up > self.activation_th, bid_volumes_up > self.volume_th))]
            relative_prices_dn = relative_prices_dn[np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th))]

            ax1.hist(relative_prices_up, color=self.color_up,  alpha=0.8, bins=n_bins, density=True)
            ax2.hist(relative_prices_dn, color=self.color_down,  alpha=0.8, bins=n_bins, density=True)
            ax1.set_title('Relative Prices Up, All time slots')
            ax2.set_title('Relative Prices Down, All time slots')

            ax3.hist(demand_relative_prices_up, color=self.color_up,    alpha=0.8, bins=n_bins, density=True)
            ax4.hist(demand_relative_prices_dn, color=self.color_down,    alpha=0.8, bins=n_bins, density=True)
            ax3.set_title('Relative Prices Up, Activation demand only')
            ax4.set_title('Relative Prices Down, Activation demand only')
                
            # [ax.set_xlim([-50, 150]) for ax in (ax1, ax2, ax3, ax4)]
            [ax.set_ylabel("Occurance rate (%)") for ax in (ax1, ax3)]
            [ax.set_xlabel("Bidding prices (€/MW)") for ax in (ax3, ax4)]

            fig.suptitle(f"Distribution of bidding prices relative to clearing prices ({market.bidding_zone}, {market.date}) \nCalculated as bidding price - clearing price")
            
            # filename = f"{run_sanitized}_relative_clearing_prices_{market.bidding_zone}"
            # plt.savefig(config.plot_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
            self.save_plot(f"relative_clearing_prices{market.bidding_zone}", run_id, fig, pdf)
            

        return
    

    def plot_bidding_analysis(self, run_id, pdf):

        # return

        config      = self.config
        controller  = self.controller
        market      = controller.market
        t           = self.controller.t
        N           = self.controller.N
        spot_prices = self.controller.spot_prices
        zone        = market.bidding_zone


        for market_type, market_data in controller.optimization_results['runs'][run_id]['markets'].items():
            if market_data['Activations'] is None: continue

            balancing_market = self.controller.market.get_balancing_market(market_type)
            bid_ts = self.controller.optimization_results['runs'][run_id]['timeseries']

            u_nom               = bid_ts.get('u_nom', np.zeros(N)).flatten()
            bid_volumes_up      = market_data['Bids']['Up']['Volume'].flatten()
            bid_volumes_down    = market_data['Bids']['Down']['Volume'].flatten()
            bid_prices_up       = market_data['Bids']['Up']['Price'].flatten()
            bid_prices_down     = market_data['Bids']['Down']['Price'].flatten()
            bid_activation_up   = market_data['Activations']['Up'].flatten()
            bid_activation_down = market_data['Activations']['Down'].flatten()
            
            prob_activation_up      = np.array(balancing_market.activation_prob_up(spot_prices, bid_prices_up)).flatten()
            prob_activation_down    = np.array(balancing_market.activation_prob_down(spot_prices, bid_prices_down)).flatten()

            # Filter out the unreasonably low bid activations
            filtered_bid_prices_up      = np.where(np.logical_and(prob_activation_up   > self.activation_th, bid_volumes_up   > self.volume_th), bid_prices_up,     0).flatten()
            filtered_bid_prices_down    = np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th), bid_prices_down,   0).flatten()
            filtered_bid_volumes_up     = np.where(np.logical_and(prob_activation_up   > self.activation_th, bid_volumes_up   > self.volume_th), bid_volumes_up,    0).flatten()
            filtered_bid_volumes_down   = np.where(np.logical_and(prob_activation_down > self.activation_th, bid_volumes_down > self.volume_th), bid_volumes_down,  0).flatten()



            ######################################################
            #                   BIDDING VOLUMES

            fig = plt.figure(figsize=self.aspect_ratio)

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

            self.save_plot(f"bid_volumes", run_id, fig, pdf)

            ######################################################
            #                   BIDDING PRICES

            fig, ax = plt.subplots(1, 1, figsize=self.aspect_ratio, sharex=True)

            ax.fill_between(t, 0, filtered_bid_prices_up, label="Bidding Price Up", color="blue", step='post', alpha=0.4)
            ax.fill_between(t, 0, filtered_bid_prices_down, label="Bidding Price Down", color="red", step='post', alpha=0.4)
            ax.step(t, spot_prices, label="Spot price", color="grey", linestyle="--", where='post')
            ax.set_ylabel("Bidding Price (€/MW)")
            ax.tick_params(axis='y')
            ax.legend(loc="upper left")

            fig.suptitle(f"{run_id} Bidding Prices and Spot Prices in €/MW ({controller.market.date}, {controller.market.bidding_zone})")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 

            self.save_plot(f"bid_prices", run_id, fig, pdf)

            ######################################################
            #                ACTIVATION PROBABILITIES

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.aspect_ratio, sharex=True)

            ax1.fill_between(t, 0, prob_activation_up, color=self.color_up, alpha=0.8, label="Expected", step='post', linewidth=0)
            ax2.fill_between(t, 0, prob_activation_down, color=self.color_down, alpha=0.8, label="Expected", step='post', linewidth=0)

            ax1.step(t, balancing_market.demand_prob_up(controller.spot_prices).reshape((-1,1)),   color='gray', linestyle=':', label="Max activation rate", where='post')
            ax2.step(t, balancing_market.demand_prob_down(controller.spot_prices).reshape((-1,1)), color='gray', linestyle=':', label="Max activation rate", where='post')
            ax1.set_ylabel('Expected activation probability')
            ax2.set_ylabel("Expected activation probability")
            ax2.set_xlabel("Time (days)")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            fig.suptitle(f"{run_id} Activation Chances ({controller.market.date}, {controller.market.bidding_zone}) \nBar heights indicate expected activation probabilities per bid. Activated bids are highlighted in dark.")

            self.save_plot(f"bid_activations", run_id, fig, pdf)

            ######################################################
            #             VOLUME-ACTIVATION SCATTER

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio)

            ax1.scatter(bid_volumes_up, prob_activation_up, color=self.color_up)
            ax2.scatter(bid_volumes_down, prob_activation_down, color=self.color_down)
            
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

            self.save_plot(f"scatter_volumes_activations", run_id, fig, pdf)

            ######################################################
            #             EXPECTED BID IMPACTS

            fig, ax1 = plt.subplots(1, 1, figsize=self.aspect_ratio)

            bid_impact_up = np.multiply(prob_activation_up, bid_volumes_up)
            bid_impact_down = np.multiply(prob_activation_down, bid_volumes_down)

            ax1.fill_between(t, -bid_impact_up,    color=self.color_up, label='Up Regulation')
            ax1.fill_between(t, bid_impact_down, color=self.color_down, label='Down Regulation')
            
            ax1.set_ylabel("Expected Bid impact (MW)")
            ax1.set_xlabel("Time (days)")
            ax1.legend(title='Direction')

            title = f"{run_id} Expected consumption impact of bids (MW) ({controller.market.date}, {controller.market.bidding_zone})"
            
            fig.suptitle(title)
            fig.tight_layout() 

            self.save_plot(f"bid_impacts", run_id, fig, pdf)

        return




    def plot_freshweights(self, plot_name = 'fresh_weights', run_id = None, runs = {}, pdf = None):
        
        ######################################################
        #                   FRESHWEIGHTS
        
        config = self.config
        controller = self.controller

        # if run_id is None:
        #     if self.find_existing_plot(plot_name, self.common_dependencies): return

        fig = plt.figure(figsize=self.aspect_ratio)
        
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
        self.update_plot_log(plot_name, self.common_dependencies)



    def plot_light_schedules(self, plot_name = 'light_schedules', run_id = None, runs = {}, pdf = None):

        ######################################################
        #                   LIGHT SCHEDULE
        
        # if run_id is None:
        #     if self.find_existing_plot(plot_name, self.common_dependencies): return

        controller  = self.controller
        config      = self.config
        spot_prices = controller.spot_prices
        market      = controller.market

        n_runs = len(runs)

        fig, axes = plt.subplots(n_runs+1, 1, figsize=self.aspect_ratio, sharex=True)

        for i, run in enumerate(runs):
            ax = axes[i]
            u = runs[run]['timeseries']['u'].flatten()
            t = runs[run]['timeseries']['t'].flatten()
            ax.step(t, u, label=f"{run}", where='post') 
            ax.set_ylabel(self.controller.model.u_unit)
            ax.legend(loc="upper right")

        ax = axes[-1]
        ax.step(t, spot_prices, label=f"Spot price", color='gray', where='post') 
        ax.set_ylabel("EUR/MWh")
        ax.set_xlabel("Time (days)")
        ax.legend(loc="upper right")
        fig.suptitle(f'Light schedules ({market.date}, {market.bidding_zone})')

        self.save_plot(plot_name, run_id = run_id, fig=fig, pdf=pdf)
        self.update_plot_log(plot_name, self.common_dependencies)




    def plot_DLI(self, plot_name = 'daily_light_integrals', run_id = None, runs = [], pdf = None):
        
        ######################################################
        #          DAILY LIGHT INTEGRALS OVER TIME

        # if run_id is None:
        #     if self.find_existing_plot(plot_name, self.common_dependencies): return

        controller  = self.controller
        config      = self.config
        spot_prices = controller.spot_prices
        market      = controller.market

        n_runs = len(runs)

        fig, axes = plt.subplots(n_runs+1, 1, figsize=self.aspect_ratio, sharex=True)

        for i, run in enumerate(runs):
            ax = axes[i]
            X = runs[run]['timeseries']['x']
            t = runs[run]['timeseries']['t']
            DLI = get_DLI(X).flatten()
            ax.plot(t[-len(DLI):], DLI, label=f"{run}") 
            ax.set_ylabel('Daily light integral')
            ax.axhline(y=controller.model.DLI_max, color='gray', linestyle=':')
            ax.axhline(y=controller.model.DLI_min, color='gray', linestyle=':')
            ax.legend(loc="upper right")

        ax = axes[-1]
        ax.step(t, spot_prices, label=f"Spot price", color='gray') 
        ax.set_ylabel("EUR/MWh")
        ax.set_xlabel("Time (days)")
        ax.legend(loc="upper right")
        fig.suptitle(f'Daily light integrals ({market.date}, {market.bidding_zone})')

        self.save_plot(plot_name, fig = fig, run_id=run_id, pdf=pdf)
        self.update_plot_log(plot_name, self.common_dependencies)



    def plot_spot_prices(self, run_id = None, pdf = None):
        
        ######################################################
        #           SPOT PRICE FOR EACH TIME STEP

        t = self.controller.t

        fig = plt.figure(figsize=self.aspect_ratio)
        plt.step(t, self.controller.market.get_spotprice(), label="Spot price", where='post')
        plt.ylabel("Spot price (€/MWh)")
        plt.xlabel("Time (days)")
        plt.legend()

        self.save_plot(f"spot_price_{self.controller.market.bidding_zone}", run_id, fig, pdf)
        plt.close('all')



    '''
    def plot_random_activations(self, m: int):

        simulator = self.simulator
        controller = self.controller
        market = controller.market
        config = self.config
        foldername = self.foldername


        t = controller.t
        freshwewights = simulator.simulate_random_activation(self.controller, m)


        ##################################################
        plt.figure(figsize=self.aspect_ratio)

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
        plt.savefig(config.plots_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)
    '''




    def plot_spot_mfrr_prices(self):

        config = self.config
        controller = self.controller
        market = controller.market
        zone = market.bidding_zone

        # Remove drastic outliers
        price_data_raw = market.AM_data_working_set
        price_data = price_data_raw.where(price_data_raw[f'{zone} Up Price']<500).where(price_data_raw[f'{zone} Down Price']>-500).dropna()

        timestamps      = price_data['Start Time']
        spot_prices     = np.array(price_data[f'{zone} Spot Price'])
        mfrr_prices_up  = np.array(price_data[f'{zone} Up Price'])
        mfrr_prices_dn  = np.array(price_data[f'{zone} Down Price'])
        spot_prices_eur = np.array(spot_prices)


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

        plt.figure(figsize=self.aspect_ratio)

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
        
        plt.figure(figsize=self.aspect_ratio)

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
        
        plt.figure(figsize=self.aspect_ratio)

        n_bins = 100

        mu_up   = market.AM.price_stats[zone]['Up']['means'][1]
        mu_down = market.AM.price_stats[zone]['Down']['means'][1]
                
        sigma_up   = np.sqrt(market.AM.price_stats[zone]['Up']['cov'][1,1])
        sigma_down = np.sqrt(market.AM.price_stats[zone]['Down']['cov'][1,1])

        x_up = np.linspace(mu_up-3*sigma_up,mu_up+3*sigma_up, 1000)
        x_dn = np.linspace(mu_down-3*sigma_down,mu_down+3*sigma_down, 1000)

        plt.hist(mfrr_prices_up,    label="Clearing price up",   color='blue',  alpha=0.4, bins=n_bins, density=True)
        plt.hist(mfrr_prices_dn,    label="Clearing price down", color='red',   alpha=0.4, bins=n_bins, density=True)
        plt.hist(spot_prices_eur,   label="Spot prices",         color='grey',  alpha=0.4, bins=n_bins, density=True)
        plt.plot(x_up, stats.norm.pdf(x_up, mu_up, sigma_up), label="Estimated Up-price distribution", color='blue')
        plt.plot(x_dn, stats.norm.pdf(x_dn, mu_down, sigma_down), label="Estimated Down-price distribution", color='red')
        plt.xlabel("Bidding prices (€/MW)")
        plt.title(f"Clearing prices histogram normalized ({market.bidding_zone}, {market.date})")
        plt.legend()

        filename = f"histogram_mfrr_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)



        ################################################################
        #         SCATTER PLOT SPOT PRICE - CLEARING PRICES 

        plt.figure(figsize=self.aspect_ratio)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio, sharex=True)

        n = int(np.ceil(len(spot_prices)/10))
        idx = np.int64(np.ceil(np.linspace(1,len(spot_prices)-1,n)))
        ax1.scatter(spot_prices[idx], mfrr_prices_up[idx], color='blue', label='Clearing price up', s=0.1)
        ax1.plot(line_x, line_x * 0.78, label='Lower limit: 0.78 x spot', color='grey', alpha=0.4)
        ax1.set_ylabel("Bidding price (€/MWh)")
        ax1.set_xlabel("Spot price (€/kWh)")
        # ax1.set_xlim([-0.5, 4.5])
        ax1.legend()

        ax2.scatter(spot_prices[idx], mfrr_prices_dn[idx], color='red', label='Clearing price down', s=0.1)
        ax2.plot(line_x, line_x * 0.9, label='Upper limit: 0.9 x spot', color='grey', alpha=0.4)
        ax2.set_ylabel("Bidding price (€/MWh)")
        ax2.set_xlabel("Spot price (€/kWh)")
        # ax2.set_xlim([-0.5, 4.5])
        ax2.legend()

        fig.suptitle(f"Spot prices with mFRR clearing prices €/MW ({market.bidding_zone}, {market.date})")

        filename = f"scatter_spot_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)

        plt.close('all')

        ######################################################
        #        CLEARING-SPOT RELATIVE PRICES HISTOGRAM
        #                   [OUTLIERS REMOVED]

        plt.figure(figsize=self.aspect_ratio)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.aspect_ratio, sharex=False)


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


        activations_df = market.AM_data_full_set.copy()

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

        plt.figure(figsize=self.aspect_ratio)

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

        plt.figure(figsize=self.aspect_ratio)

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
        #           ACTIVATION VOLUMES HISTOGRAM
        
        activations_df = market.AM_data_full_set.copy()
        activated_capacities_up = activations_df[f'{zone} Activated Up Volume'].loc[activations_df[f'{zone} Activated Up Volume'] > 0]
        activated_capacities_down = activations_df[f'{zone} Activated Down Volume'].loc[activations_df[f'{zone} Activated Down Volume'] > 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio, sharex=True)

        n_bins = 48

        ax1.hist(activated_capacities_up, label="Activated Up", color=self.color_up, alpha=0.8, bins=n_bins, density=True)
        ax2.hist(activated_capacities_down, label="Activated Down", color=self.color_down, alpha=0.8, bins=n_bins, density=True)

        p           = 1/4 * np.array([market.AM.demand_prob_up(), market.AM.demand_prob_down()])       # Probability of activation in each 15-min slot
        lmbda       = 1/np.array([np.mean(activated_capacities_up), np.mean(activated_capacities_down)])     # Exponential rate parameter (mean 200 MW per activation)
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
            (hourly_totals[:, 0] >= 10) & (hourly_totals[:, 0] < max(activated_capacities_up)) & 
            (hourly_totals[:, 1] >= 10) & (hourly_totals[:, 1] < max(activated_capacities_down))
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
        #        ACTIVATED/OFFERED RATIO HISTOGRAM

        activations_df = market.AM_data_full_set.copy()
        offered_capacities_up = activations_df[f'{zone} Accepted Up Volume']
        offered_capacities_down = activations_df[f'{zone} Accepted Down Volume']
        activated_capacities_up = activations_df[f'{zone} Activated Up Volume']
        activated_capacities_down = activations_df[f'{zone} Activated Down Volume']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio, sharex=True)

        n_bins = 48

        activation_ratio_up     = np.divide(np.array(activated_capacities_up),   np.array(offered_capacities_up))
        activation_ratio_down   = np.divide(np.array(activated_capacities_down), np.array(offered_capacities_down))

        ax1.hist(activation_ratio_up[np.where(activation_ratio_up>0)], label="Up", color=self.color_up, alpha=0.8, bins=n_bins, density=True)
        ax2.hist(activation_ratio_down[np.where(activation_ratio_down>0)], label="Down", color=self.color_down, alpha=0.8, bins=n_bins, density=True)

        
        plt.suptitle(f'Ratio of activated vs accepted volumes in {market.bidding_zone} bidding zone \nwhen activations are made')
        ax1.set_ylabel('Probability of occurrence')
        ax1.set_title('Activation Ratio Up')
        ax2.set_title('Activation Ratio Down')
        ax1.legend()
        ax2.legend()
        plt.tight_layout()

        filename = f"mFRR_activation_ratios_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)


        ######################################################
        #           OFFERED VOLUMES HISTOGRAM

        activations_df = market.AM_data_full_set.copy()
        activated_capacities_up = activations_df[f'{zone} Accepted Up Volume']
        activated_capacities_down = activations_df[f'{zone} Accepted Down Volume']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio, sharex=True)

        n_bins = 48

        ax1.hist(activated_capacities_up, label="Accepted Up", color=self.color_up, alpha=0.8, bins=n_bins, density=True)
        ax2.hist(activated_capacities_down, label="Accepted Down", color=self.color_down, alpha=0.8, bins=n_bins, density=True)

        
        plt.suptitle(f'Accepted capacity volumes in {market.bidding_zone} bidding zone')
        ax1.set_xlabel('Power (MW)')
        ax2.set_xlabel('Power (MW)')
        ax1.set_ylabel('Probability of occurrence')
        ax1.set_title('Accepted Up')
        ax2.set_title('Accepted Down')
        ax1.legend()
        ax2.legend()
        plt.tight_layout()

        filename = f"mFRR_accepted_volumes_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)


        ######################################################
        #           MARKET POTENCY BAR CHART

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.aspect_ratio, sharex=False)

        market_potency_df = market.daily_market_potency_df
        rolling_market_potency_df = market.rolling_market_potency_df

        market_potency_df.set_index('Date')[['Potency Up', 'Potency Down']].plot(
            ax = ax2, kind='bar', stacked=True, figsize=(15, 6), color=['skyblue', 'lightcoral']
        )

        rolling_market_potency_df.set_index('Date')[['Rolling Potency Up', 'Rolling Potency Down', 'Rolling Total Potency']].plot(
            ax=ax1, linewidth=2.5, linestyle='-', color=[self.color_up, self.color_down, 'grey']
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



    def plot_CM_data(self):

        config = self.config
        controller = self.controller
        market = controller.market
        zone = market.bidding_zone

        # Remove drastic outliers
        price_data = market.CM_data_full_set
        # price_data = price_data_raw.where(price_data_raw[f'{zone} Up Price']<500).where(price_data_raw[f'{zone} Down Price']>-500).fillna(0)

        # if price_data_raw.empty: 
        #     print(f'No Capacity Market data for date: {market.date}.\nCM data not plotted')
        #     return


        timestamps          = price_data['Start Time']
        spot_prices         = np.array(price_data[f'{zone} Spot Price'])
        mfrr_prices_up      = np.array(price_data[f'{zone} Up Price'])
        mfrr_prices_down    = np.array(price_data[f'{zone} Down Price'])
        spot_prices_eur     = np.array(spot_prices)


        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='same')

        # Smoothed data
        window_length = 24*3
        mfrr_prices_up_smoothed     = moving_average(mfrr_prices_up, window_length)
        mfrr_prices_dn_smoothed     = moving_average(mfrr_prices_down, window_length)
        spot_prices_eur_smoothed    = moving_average(spot_prices_eur, window_length)

        opacity = 0.2
        linewidth=1.5

        ###########################################
        #            SPOT VS MFRR PRICES 
        #       [SMOOTHED] [OUTLIERS REMOVED]

        plt.figure(figsize=self.aspect_ratio)

        plt.step(timestamps, mfrr_prices_up, label="Clearing price up", color='blue', alpha=opacity)
        plt.step(timestamps, mfrr_prices_up_smoothed, label="Clearing price up smoothed", color='blue', alpha=1, linewidth = linewidth)

        plt.step(timestamps, mfrr_prices_down, label="Clearing price down", color='red', alpha=opacity)
        plt.step(timestamps, mfrr_prices_dn_smoothed, label="Clearing price down smoothed", color='red', alpha=1, linewidth = linewidth)

        plt.step(timestamps, spot_prices_eur, label="Spot price", color='grey', alpha=opacity)
        plt.step(timestamps, spot_prices_eur_smoothed, label="Spot price smoothed", color='grey', alpha=1, linewidth = linewidth)
        plt.ylabel("Price (€/MW)")
        plt.yscale('log')
        plt.xlabel("Time (days)")
        plt.title(f"Spot price vs activation prices smoothed using {window_length}h moving average")
        plt.legend()


        filename = f"CM_smoothed_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)



        ####################################################
        #        CLEARING PRICES RELATIVE TO SPOT 
        #               [OUTLIERS REMOVED]
        
        plt.figure(figsize=self.aspect_ratio)

        plt.step(timestamps, mfrr_prices_up - spot_prices_eur, label="Clearing price up", color='blue')
        plt.step(timestamps, mfrr_prices_down - spot_prices_eur, label="Clearing price down", color='red')
        plt.plot(timestamps, 0 * mfrr_prices_down, label="Zero-line", color='grey', alpha=0.5)
        plt.ylabel("Price (€/MW)")
        plt.xlabel("Time (days)")
        plt.title("Clearing prices relative to spot price (€/MW)")
        plt.legend()

        filename = f"CM_relative_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)


        ###############################################################
        #             CLEARING PRICES HISTOGRAM 
        #               [OUTLIERS REMOVED]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.aspect_ratio, sharex=False)
        # plt.figure(figsize=self.aspect_ratio)

        n_bins = 100

        ax1.hist(mfrr_prices_up,    label="Clearing price up",   color='blue',  alpha=0.4, bins=n_bins, density=True)
        ax1.hist(mfrr_prices_down,  label="Clearing price down", color='red',   alpha=0.4, bins=n_bins, density=True)
        ax2.hist(spot_prices_eur,   label="Spot prices",         color='grey',  alpha=0.4, bins=n_bins, density=True)
        # plt.plot(x_up, stats.norm.pdf(x_up, mu_up, sigma_up), label="Estimated Up-price distribution", color='blue')
        # plt.plot(x_dn, stats.norm.pdf(x_dn, mu_dn, sigma_dn), label="Estimated Down-price distribution", color='red')
        fig.suptitle(f"Clearing prices histogram normalized ({market.bidding_zone}, {market.date})")
        ax1.legend()
        ax1.set_yscale('log')
        # ax1.set_xlim([min(),100])
        ax2.legend()
        ax2.set_xlabel("CM Clearing Prices (€/MWh)")

        filename = f"CM_histogram_mfrr_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)



        ################################################################
        #         SCATTER PLOT SPOT PRICE - CLEARING PRICES 

        plt.figure(figsize=self.aspect_ratio)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio, sharex=True)

        n = int(np.ceil(len(spot_prices)/10))
        idx = np.int64(np.ceil(np.linspace(1,len(spot_prices)-1,n)))
        ax1.scatter(spot_prices[idx], mfrr_prices_up[idx], color='blue', label='Clearing price up', s=0.1)
        ax1.set_ylabel("Bidding price (€/MWh)")
        ax1.set_xlabel("Spot price (€/MWh)")
        # ax1.set_xlim([-0.5, 4.5])
        ax1.legend()

        ax2.scatter(spot_prices[idx], mfrr_prices_down[idx], color='red', label='Clearing price down', s=0.1)
        ax2.set_ylabel("Bidding price (€/MWh)")
        ax2.set_xlabel("Spot price (€/MWh)")
        # ax2.set_xlim([-0.5, 4.5])
        ax2.legend()

        fig.suptitle(f"Spot prices with mFRR CM clearing prices €/MW ({market.bidding_zone}, {market.date})")

        filename = f"CM_scatter_spot_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)

        plt.close('all')

        ######################################################
        #        CLEARING-SPOT RELATIVE PRICES HISTOGRAM
        #                   [OUTLIERS REMOVED]

        plt.figure(figsize=self.aspect_ratio)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.aspect_ratio, sharex=True)


        ax1.hist(mfrr_prices_up-spot_prices_eur, label="Clearing price up", color='blue', alpha=0.4, bins=2*n_bins, density=True)
        ax1.set_xlabel("Bidding prices (€/MW)")
        # ax1.set_xlim([-75, 25])
        ax1.legend()


        ax2.hist(mfrr_prices_down-spot_prices_eur, label="Clearing price down", color='red', alpha=0.4, bins=n_bins, density=True)
        ax2.set_xlabel("Bidding prices (€/MW)")
        # ax2.set_xlim([-75, 25])
        ax2.legend()

        plt.suptitle(f'Relative clearing prices, normalized ({market.bidding_zone}, {market.date})')

        filename = f"CM_histogram_relative_clearing_prices_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)


   
        ######################################################
        #           MONTHLY RESERVED CM CAPACITY


        CM_reservations_df = market.CM_data_full_set.copy()

        # Process data up
        CM_reservations_df['Start Time'] = pd.to_datetime(CM_reservations_df['Start Time'])
        CM_reservations_df.loc[:, 'Year-Month'] = CM_reservations_df['Start Time'].dt.to_period('M')

        # Calculate total activated volume per month
        monthly_volume_up   = CM_reservations_df.groupby('Year-Month')[CM_reservations_df.filter(like=f'{zone} Up Volume procured').columns].sum().sum(axis=1)
        monthly_volume_down = CM_reservations_df.groupby('Year-Month')[CM_reservations_df.filter(like=f'{zone} Down Volume procured').columns].sum().sum(axis=1)

        # Combine into a single DataFrame
        monthly_total_volumes = pd.DataFrame({
            'Total Volume Up': monthly_volume_up,
            'Total Volume Down': monthly_volume_down
        }).fillna(0)  # Fill missing months with 0

        plt.figure(figsize=self.aspect_ratio)

        monthly_total_volumes.plot(kind='bar', stacked=True, figsize=(12, 6), color=['skyblue', 'lightcoral'], edgecolor='gray')
        plt.title(f'Monthly reservation rates in {market.bidding_zone} bidding zone')
        plt.xlabel('Year-Month')
        plt.ylabel('Total Reserved Volume (MWh)')
        plt.xticks(rotation=45)
        plt.legend(title='Activation Direction')
        plt.tight_layout()

        filename = f"CM_{market.bidding_zone}_mFRR_monthly_reserved_capacity"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)


        ######################################################
        #           DAILY mFRR ACTIVATION COUNTS

        plt.figure(figsize=self.aspect_ratio)

        # Process data
        CM_reservations_df['Start Time'] = pd.to_datetime(CM_reservations_df['Start Time'])
        CM_reservations_df.loc[:, 'Date'] = CM_reservations_df['Start Time'].dt.date

        # Ensure all dates from the full range are present for counting
        date_range = pd.date_range(start=CM_reservations_df['Date'].min(), end=CM_reservations_df['Date'].max())

        # Calculate total activated volume per day
        daily_volume_up     = CM_reservations_df.groupby('Date')[CM_reservations_df.filter(like=f'{zone} Up Volume procured').columns].sum().sum(axis=1).reindex(date_range, fill_value=0)
        daily_volume_down   = CM_reservations_df.groupby('Date')[CM_reservations_df.filter(like=f'{zone} Down Volume procured').columns].sum().sum(axis=1).reindex(date_range, fill_value=0)

        # Combine into a single DataFrame
        daily_total_volumes = pd.DataFrame({
            'Date': date_range,
            'Total Volume Up': daily_volume_up.values,
            'Total Volume Down': daily_volume_down.values
        })

        first_of_month = daily_total_volumes['Date'][daily_total_volumes['Date'].dt.day == 1]

        ax = daily_total_volumes.set_index('Date')[['Total Volume Up', 'Total Volume Down']].plot(
            kind='bar', stacked=True, figsize=(15, 6), color=['skyblue', 'lightcoral']
        )

        first_of_month_indexes = daily_total_volumes[daily_total_volumes['Date'].dt.day == 1].index
        ax.set_xticks(first_of_month_indexes)
        ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in first_of_month], rotation=45)

        plt.title(f'Daily activation rates in {market.bidding_zone} bidding zone')
        plt.xlabel('Date')
        plt.ylabel('Total Reserved Capacity (MWh)')
        plt.legend(title='Activation Direction')
        plt.tight_layout()

        filename = f"CM_{market.bidding_zone}_mFRR_daily_reserved_capacity"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)

        ######################################################
        #           ACTIVATION VOLUMES HISTOGRAM

        CM_reservations_df = market.CM_data_full_set.copy()
        CM_reserved_capacities_up = CM_reservations_df[f'{zone} Up Volume procured'].loc[CM_reservations_df[f'{zone} Up Volume procured'] > 0]
        CM_reserved_capacities_down = CM_reservations_df[f'{zone} Down Volume procured'].loc[CM_reservations_df[f'{zone} Down Volume procured'] > 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.aspect_ratio, sharex=True)

        n_bins = 48

        ax1.hist(CM_reserved_capacities_up, label="Volume Up", color=self.color_up, alpha=0.8, bins=n_bins, density=True)
        ax2.hist(CM_reserved_capacities_down, label="Volume Down", color=self.color_down, alpha=0.8, bins=n_bins, density=True)

        plt.suptitle(f'Reserved capacity volumes in {market.bidding_zone} bidding zone')
        ax1.set_xlabel('Power (MW)')
        ax2.set_xlabel('Power (MW)')
        ax1.set_ylabel('Probability of occurrence')
        ax1.set_title('Reserved Up')
        ax2.set_title('Reserved Down')
        ax1.legend()
        ax2.legend()
        plt.tight_layout()

        filename = f"CM_mFRR_reserved_volumes_{market.bidding_zone}"
        plt.savefig(config.data_analysis_path + filename + "." + self.plot_file_type, format=self.plot_file_type)






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
        plt.figure(figsize=self.aspect_ratio)
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
        plt.savefig(config.plots_path + foldername + "/MPC_" + filename + "." + self.plot_file_type, format=self.plot_file_type)
        ##################################################


        plt.figure(figsize=self.aspect_ratio)
        plt.step(t, u_mpc, label="MPC U") 
        plt.step(t, u_rigid, label="Rigid U") 
        # plt.step(t, u_bid, linestyle=':', label="Bidding U") 
        # plt.step(t, u_base, linestyle=':', label="Baseline U") 
        plt.ylabel("Light level (PPFD)")
        plt.xlabel("Time (days)")
        plt.legend()


        filename = "Combined_ocp_u"
        plt.savefig(config.plots_path + foldername + "/MPC_" + filename + "." + self.plot_file_type, format=self.plot_file_type)


    def plot_financial_report(self):

        config      = self.config
        controller  = self.controller
        market      = controller.market
        foldername  = self.foldername
        n_runs      = len(list(self.controller.optimization_results['runs'].keys()))
        t           = self.controller.t
        spot_prices = self.controller.spot_prices
                


        costs           = [controller.optimization_results['runs'][run]['metrics']['Costs']             for run in controller.optimization_results['runs']]
        totals          = [controller.optimization_results['runs'][run]['metrics']['Total']             for run in controller.optimization_results['runs']]
        CM_earnings     = [controller.optimization_results['runs'][run]['markets'].get('CM',{}).get('Earnings', 0) for run in controller.optimization_results['runs']]
        AM_earnings     = [controller.optimization_results['runs'][run]['markets'].get('AM',{}).get('Earnings', 0) for run in controller.optimization_results['runs']]
        cost_reduction_percent = [(totals[0] - totals[i])/totals[0] * 100 for i in range(len(totals))]

        header=[run for run in controller.optimization_results['runs']]

        ######################################################
        #               FINALCIAL REPORT
        #

        plt.figure(figsize=self.aspect_ratio)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.aspect_ratio)

        x = np.arange(len(header))  # Bar positions
        bar_width = 0.2

        # Bar chart
        ax1.bar(x - bar_width, costs, width=bar_width,               color='lightgrey',      label="Cost of energy")
        ax1.bar(x, CM_earnings, width=bar_width,                     color='lightskyblue',   label="Capacity Market Earnings")
        ax1.bar(x, AM_earnings, bottom=CM_earnings, width=bar_width, color='lightcoral',     label="Activation Market Earnings")
        ax1.bar(x + bar_width, totals, width=bar_width,              color='slategrey',      label="Total")
        ax1.axhline(y=min(totals), color='gray', linestyle='-', linewidth=0.1)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.2)

        ax1.set_xticks(x)
        ax1.set_xticklabels(header, rotation=0)
        ax1.set_ylabel("Amount (€)")
        ax1.legend(loc='lower left')
        ax1.set_title(f"Financial Overview of Optimization Methods (Date: {market.date}, Bidding Zone: {market.bidding_zone}) ")

        # Table
        metrics_table = [[''] + header] + get_metrics_table_raw(controller.optimization_results['runs'])

        ax2.axis("tight")
        ax2.axis("off")
        ax2.table(cellText=metrics_table, cellLoc='center', loc='center', colWidths=[0.2] + [0.15]*len(header))


        fig.suptitle('')
        plt.tight_layout()

        filename = f"_financial_report_{market.date.replace('-','_')}_{market.bidding_zone}"
        plt.savefig(config.plots_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)

        plt.close('all')



        ######################################################
        #               SPECS REPORT
        #


        plt.figure(figsize=self.aspect_ratio)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.aspect_ratio)

        # Table

        table_1 = dict_to_table(controller.settings_data['general'])
        ax1.axis("tight")
        ax1.axis("off")
        ax1.table(cellText=table_1, cellLoc='center', loc='center', colWidths=[0.5, 0.4])
        ax1.set_title('General specs')

        table_2 = dict_to_table(controller.settings_data['controller'])
        ax2.axis("tight")
        ax2.axis("off")
        ax2.table(cellText=table_2, cellLoc='center', loc='center', colWidths=[0.5, 0.4])
        ax2.set_title('Controller specs')

        table_3 = dict_to_table(controller.settings_data['model'])
        ax3.axis("tight")
        ax3.axis("off")
        ax3.table(cellText=table_3, cellLoc='center', loc='center', colWidths=[0.5, 0.4])
        ax3.set_title('Model specs')
        
        table_4 = dict_to_table(controller.settings_data['market'])
        ax4.axis("tight")
        ax4.axis("off")
        ax4.table(cellText=table_4, cellLoc='center', loc='center', colWidths=[0.5, 0.4])
        ax4.set_title('Market specs')


        fig.suptitle(f'Specs for {config.sim_name}')
        plt.tight_layout()

        filename = f"_specs_{market.date.replace('-','_')}_{market.bidding_zone}"
        plt.savefig(config.plots_path + foldername + "/" + filename + "." + self.plot_file_type, format=self.plot_file_type)

        plt.close('all')


        return 0            

