

class Settings():

    all_settings = {}
    grouped_keys = {}


    def __init__(self, *groups, **kwargs):

        self.add_setting(*groups, **kwargs)

        # General settings
        self.add_setting('general', 
                         SIM_NAME            = "placeholder_name",
                         SIMULATION_LENGTH   = 3,
                         FINAL_WEIGHT        = 10,
                         SEED                = 1133
                         )


        # Market settings
        self.add_setting('market',  
                         SIMULATION_DATE     = '2024-01-14',
                         BIDDING_ZONE        = 'NO2',
                         OPTIMISTIC          = 1,
                         C_eur2nok           = 11.76,
                         n_given_bids        = 2,
                         n_given_activations = 1,
                         outlier_max_dist    = 3
                         )

        # Model settings


        # Controller settings
        self.add_setting('controller', 
                         SEARCH_CACHE     = 1,
                         SURPRESS_OUTPUT  = False,
                         WARM_START       = True,
                         CALCULATE_FW     = True,
                         IMPORT_FILE      = 'scaled_optimal_intensities.json'
                         )
                        
        
        # MPC settings
        self.add_setting('mpc', 
                         MPC_TIMEHORIZON = 1, 
                         MPC_STEPLENGTH = 0.5
                         )



        # Plotter settings


        return



    def add_setting(self, *groups, **kwargs):

        for key, value in kwargs.items():
            self.all_settings[key] = value
            for group in groups:
                if group not in self.grouped_keys: self.grouped_keys[group] = []
                self.grouped_keys[group].append(key)


    
    def change_setting(self, **kwargs):

        for key in kwargs:
            assert key in self.all_settings, f'{key} not a registered setting'
            self.all_settings[key] = kwargs[key]
            


    def get_settings(self, *groups):

        if not groups:
            return self.all_settings
        
        keys = set()
        for group in groups:
            assert group in self.grouped_keys, f'Group {group} is not a registered group.'
            [keys.add(item) for item in self.grouped_keys[group]]

        filtered_dict = {k: self.all_settings[k] for k in list(keys) if k in self.all_settings}
        
        return filtered_dict

   

# settings = Settings(testvar = 1)
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
