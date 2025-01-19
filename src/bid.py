import numpy as np

class Bid():

    volume_up:      float
    volume_down:    float
    price_up:       float
    price_down:     float

    def __init__(self, volume_up = 0, volume_down = 0, 
                        price_up = 0, price_down = 0, 
                        activated_up = None, activated_down = None):
        
        self.volume_up      = volume_up
        self.volume_down    = volume_down
        self.price_up       = price_up
        self.price_down     = price_down
        
    def as_array(self):

        return np.array([self.volume_up, self.volume_down, self.price_up, self.price_up])