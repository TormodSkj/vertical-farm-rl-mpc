import os

class Config():
    path:       str 
    plot_path:  str

    def __init__(self):
        # Get the directory one level above the current script's location
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/"
        self.plot_path = os.path.join(self.path, "plots/")
