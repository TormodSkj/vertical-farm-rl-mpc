import os

class Config():
    path:       str 
    plot_path:  str
    sim_path:   str
    sim_name:   str

    plot_file_type: str
    plot_format = (10, 6)

    def __init__(self, simulation_name="custom", filetype="pdf"):

        self.sim_name = simulation_name
        self.plot_file_type = filetype

        # Get project root path
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/"

        # Paths for plots
        self.plot_path = os.path.join(self.path, "plots/")
        self.plot_folder = os.path.join(self.plot_path, simulation_name)
        os.makedirs(self.plot_folder, exist_ok=True)  # Ensure plot folders exist

        # Path for simulations
        self.sim_path = os.path.join(self.path, "simulations/")
        os.makedirs(self.sim_path, exist_ok=True)  # Ensure simulations folder exists
