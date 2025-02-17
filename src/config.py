import os
from settings import Settings

class Config():
    path:       str 
    plots_path: str
    current_sim_plot_path:   str
    sim_name:   str
    data_path:  str
    data_analysis_path: str
    output_path: str

    spotprice_data_path: str
    mfrr_clearing_prices_path: str

    def __init__(self, settings: Settings):

        # Get project root path
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/"

    
        self.config_settings    = settings.get_settings_group('sim_name', 'general', 'config')

        self.sim_name           = self.config_settings['SIM_NAME']

        self.plots_path         = os.path.join(self.path, self.config_settings['PLOTS_SUBDIR'])
        self.simulations_path   = os.path.join(self.path, self.config_settings['SIMULATIONS_SUBDIR'])
        self.data_path          = os.path.join(self.path, self.config_settings['DATA_SUBDIR'])
        self.data_analysis_path = os.path.join(self.path, self.config_settings['DATA_ANALYSIS_SUBDIR'])
        self.output_path        = os.path.join(self.path, self.config_settings['OUTPUT_SUBDIR'])

        self.current_sim_plot_path  = os.path.join(self.plots_path, self.sim_name)

        self.mfrr_clearing_prices_path  = os.path.join(self.data_path, self.config_settings['MFRR_CLEARING_PRICES_PATH'])
        self.mfrr_activation_data_path  = os.path.join(self.data_path, self.config_settings['MFRR_ACTIVATION_DATA_PATH'])
        self.mfrr_CBMP_data_path        = os.path.join(self.data_path, self.config_settings['MFRR_CBMP_DATA_PATH'])

        # Paths for different datasets
        self.spotprice_data_path        = os.path.join(self.data_path, self.config_settings['SPOTPRICES_PATH'])

        self.ensure_dirs(self.current_sim_plot_path,
                         self.plots_path,
                         self.simulations_path,
                         self.data_path,
                         self.data_analysis_path,
                         self.output_path,
                         self.mfrr_clearing_prices_path, 
                         self.mfrr_activation_data_path, 
                         self.mfrr_CBMP_data_path                                 
                         )
     

    def ensure_dirs(self, *paths):
        for path in paths:
            os.makedirs(path, exist_ok=True)