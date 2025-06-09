import os
from settings import Settings
from utils import *

class Config():
    path:       str 
    plots_path: str
    current_sim_plot_path:   str
    sim_name:   str
    data_path:  str
    data_analysis_path: str
    output_path: str

    spotprice_data_path: str
    mfrr_AM_clearing_prices_path: str

    def __init__(self, settings: Settings):

        # Get project root path
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/"

    
        self.config_settings    = settings.get_settings_group('options', 'general', 'config')

        self.sim_name           = self.config_settings['SIM_NAME']

        self.plots_path         = os.path.join(self.path, self.config_settings['PLOTS_SUBDIR'])
        self.simulations_path   = os.path.join(self.path, self.config_settings['SIMULATIONS_SUBDIR'])
        self.data_path          = os.path.join(self.path, self.config_settings['DATA_SUBDIR'])
        self.data_analysis_path = os.path.join(self.path, self.config_settings['DATA_ANALYSIS_SUBDIR'])
        self.output_path        = os.path.join(self.path, self.config_settings['OUTPUT_SUBDIR'])
        self.geodata_path       = os.path.join(self.path, self.config_settings['GEOJSON_DATA_PATH'])

        self.current_sim_plot_path  = os.path.join(self.plots_path, self.sim_name)

        # self.mfrr_AM_clearing_prices_path   = os.path.join(self.data_path, self.config_settings['MFRR_AM_CLEARING_PRICES_PATH'])
        # self.mfrr_AM_activation_data_path   = os.path.join(self.data_path, self.config_settings['MFRR_AM_ACTIVATION_DATA_PATH'])
        self.mfrr_AM_data_path              = os.path.join(self.data_path, self.config_settings['MFRR_AM_DATA_PATH'])
        self.mfrr_CM_data_path              = os.path.join(self.data_path, self.config_settings['MFRR_CM_DATA_PATH'])
        self.mfrr_CBMP_data_path            = os.path.join(self.data_path, self.config_settings['MFRR_CBMP_DATA_PATH'])

        # Paths for different datasets
        self.spotprices_data_path        = os.path.join(self.data_path, self.config_settings['SPOTPRICES_PATH'])
        self.spotprice_data_path        = os.path.join(self.spotprices_data_path, self.config_settings['SPOTPRICES_NORWAY'])

        self.ensure_dirs(self.plots_path,
                         self.simulations_path,
                         self.data_path,
                         self.data_analysis_path,
                         self.output_path,
                         self.mfrr_CBMP_data_path,
                         self.geodata_path
                         )

        # Scan all data and store metrics/metadata in a readme
        
        for _, path in settings.get_settings_group('data').items():
            if not str(path).endswith('/'): continue
            update_dataset_readmes(os.path.join(self.data_path, path))

            

    def ensure_dirs(self, *paths):
        for path in paths:
            os.makedirs(path, exist_ok=True)