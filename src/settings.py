import numpy as np
from globals import *

class Settings():

    all_settings = {}
    grouped_keys = {}


    def __init__(self, *groups, **kwargs):

        self.add_setting('options',
                         SIM_NAME           = "placeholder_name",  # Name of simulation used for plotting and archiving
                         SURPRESS_OUTPUT    = False
                         )


        # General settings
        self.add_setting('general', 
                         SIMULATION_LENGTH  = 3,                   # Timehorizon of entire simulation  [days]
                         SIM_RESOLUTION     = QUARTER_HOURS_PER_DAY,
                         SIM_TIMEDELTA      = SECONDS_PER_QUARTER_HOUR,
                         SEED               = 1133                 # Seed used for random generation
                         )


        # Market settings
        self.add_setting('market',  
                         SIMULATION_DATE     = '2024-01-14',
                         BIDDING_ZONE        = 'NO2',
                         OPTIMISTIC          = 1,
                         n_given_bids        = 2,
                         n_given_activations = 1,
                         OUTLIER_DIST_LIMIT  = 3,
                         BID_PRICE_LIMIT     = 100,
                         AM_ACTIVATION_RATE  = 0.25,
                         CM_ACTIVATION_RATE  = 0.25,
                         EXACT_ESTIMATION    = False
                         )

        # Model settings
        self.add_setting('plantmodel', 
                         X_INIT             = 1, 
                         TARGET_FRESHWEIGHT = np.array([5, 1, 0]),
                         AMBIENT_TEMP       = 24,
                         AMBIENT_CO2        = 600,
                         PHOTOPERIOD        = 16,
                         LIGHT_INTENSITY    = 200,
                         TARGET_DLI         = 11.52,
                         DLI_DEVIATION      = 0.1,
                         DLI_RESOLUTION     = 2,
                         GROWTH_AREA        = 15000,
                         PPFD_MAX           = 230,
                         LED_EFFICIENCY     = 0.8,
                         DISCRETIZATION     = 'fe'
                         )


        # Controller settings
        self.add_setting('controller', 
                         SEARCH_SIM_CACHE   = True,
                         WARM_START         = True,
                         CALCULATE_FW       = True,
                         IMPORT_FILE        = 'scaled_optimal_intensities.json'
                         )
                        
        
        # MPC settings
        self.add_setting('mpc', 
                         MPC_TIMEHORIZON    = 1, 
                         MPC_STEPLENGTH     = 0.5
                         )

        # Plotter settings
        self.add_setting('plotter', 
                         PLOT_EXPORT_TYPE       = 'pdf',
                         PLOT_ASPECT_RATIO      = (10, 6),
                         SEARCH_PLOT_CACHE      = True,
                         ACTIVATION_THRESHOLD   = 0.01,
                         VOLUME_THRESHOLD       = 0.001,
                         FILTER_BIDS            = False
                         )


        self.add_setting('config',
                         PLOTS_SUBDIR             = 'plots/',
                         SIMULATIONS_SUBDIR       = 'simulations/',
                         DATA_SUBDIR              = 'data/',
                         DATA_ANALYSIS_SUBDIR     = 'data_analysis/',
                         OUTPUT_SUBDIR            = 'output/'
                         )
        
        self.add_setting('data', 'config',
                         SPOTPRICES_NORWAY          = 'spotprices_norway_jan_2020_dec_2024.csv',          
                         SPOTPRICES_PATH            = 'spot_market/',          
                         MFRR_CBMP_DATA_PATH        = 'mFRR_CBMP/',
                         MFRR_CM_DATA_PATH          = 'mFRR_capacity_market/',
                         MFRR_AM_DATA_PATH          = 'mFRR_activation_market/'
                         )


        # Add settings from init args
        self.add_setting(*groups, **kwargs)
    
        # Update settings based on updates to settings
        self.add_setting('general',
                         SIM_N_TIMESTEPS        = int(np.ceil(self.get_setting('SIMULATION_LENGTH') * self.get_setting('SIM_RESOLUTION')))
                         )
        
        return




    def add_setting(self, *groups, **kwargs):

        for key, value in kwargs.items():
            self.all_settings[key] = value
            for group in groups:
                if group not in self.grouped_keys: self.grouped_keys[group] = []
                self.grouped_keys[group].append(key)


    
    def update_setting(self, **kwargs):

        for key in kwargs:
            assert key in self.all_settings, f'{key} not a registered setting'
            self.all_settings[key] = kwargs[key]
            


    def get_settings_group(self, *groups):

        if not groups:
            return self.all_settings
        
        keys = set()
        for group in groups:
            assert group in self.grouped_keys, f'Group {group} is not a registered group.'
            [keys.add(item) for item in self.grouped_keys[group]]

        filtered_dict = {k: self.all_settings[k] for k in list(keys) if k in self.all_settings}
        
        return filtered_dict


    def get_setting(self, setting):

        assert setting in self.all_settings, f'Setting {setting} is not registered'

        return self.all_settings[setting]
   

# settings = Settings('general', SIM_NAME = 'testname')
# settings.add_setting()



# settings.add_setting('bokstaver', 'a', a=1)
# settings.add_setting('bokstaver', 'b', b=2)


# print(settings.get_settings('a'))
# print(settings.get_settings('b'))
# print(settings.get_settings('a', 'b'))
# print(settings.get_settings('bokstaver'))
# print(settings.get_settings('bokstaver', 'a'))
# print(settings.get_settings('bokstaver', 'b'))
# print(settings.get_settings('bokstaver', 'a', 'b'))


# print(settings.get_settings())
