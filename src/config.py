import os

class Config():
    path:       str 
    plot_path:  str
    sim_path:   str
    sim_name:   str
    data_path:  str
    data_analysis_path: str
    output_path: str

    plot_file_type: str
    plot_format = (10, 6)

    spotprice_data_path: str
    mfrr_clearing_price_data_path: str

    seed: int

    def __init__(self, simulation_name="custom", filetype="pdf", seed = 1133):

        self.sim_name = simulation_name
        self.plot_file_type = filetype
        self.seed = seed

        # Get project root path
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/"

        # Paths for plots
        self.plot_path = os.path.join(self.path, "plots/")
        self.plot_folder = os.path.join(self.plot_path, simulation_name)
        os.makedirs(self.plot_folder, exist_ok=True)  # Ensure plot folders exist

        # Path for simulations
        self.sim_path = os.path.join(self.path, "simulations/")
        os.makedirs(self.sim_path, exist_ok=True)  # Ensure simulations folder exists

        # Path for data
        self.data_path = os.path.join(self.path, "data/")
        self.spotprice_data_path = os.path.join(self.data_path, 'spotprices_norway_jan_2020_dec_2024.csv')
        self.mfrr_clearing_price_data_path = os.path.join(self.data_path, 'mFRR_balancing_prices/')
        self.mfrr_activation_data_path = os.path.join(self.data_path, 'mFRR_activations/')
        self.mfrr_CBMP_data_path = os.path.join(self.data_path, 'mFRR_CBMP/')

        # Path for data
        self.data_analysis_path = os.path.join(self.path, "data_analysis/")
        os.makedirs(self.data_analysis_path, exist_ok=True)  # Ensure simulations folder exists

        # Path for data
        self.output_path = os.path.join(self.path, "output/")
        os.makedirs(self.output_path, exist_ok=True)  # Ensure simulations folder exists

