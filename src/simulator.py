import numpy as np
from market import Market, Bid
from model import *
from config import Config
from controller import Controller
from globals import *
import time


class Simulator():
    
    model: PlantModel
    market: Market
    config: Config
    controller: Controller

    N: float
    T: float
    dt: float

    x_mpc: np.array
    u_mpc: np.array
    bids_mpc: np.array


    def __init__(self, timehorizon, plantmodel, market, config, controller):
        self.T = timehorizon
        self.N = timehorizon * QUARTER_HOURS_PER_DAY
        self.model = plantmodel  
        self.market = market
        self.config = config
        self.controller = controller
        self.dt = self.controller.dt
        self.t = np.linspace(0, self.T, self.N)

        self.bids_mpc = np.zeros((4, self.N))
        

    def Simulate_mpc(self):

        # TODO let's get to work

        X = np.zeros((self.model.nx, self.N+1))
        X[:,0] = self.model.x_init

        U = np.zeros((self.model.nu, self.N))

        # Init bid list with two empty bids
        Bids = [Bid(), Bid()]

        bidding_z_opt = self.controller.bidding_z_init
        baseline_z_opt = self.controller.baseline_z_init
        self.controller.surpress_output = True

        for k in range(self.N-1):
        # for k in range(4):     # TODO remove 

            iter_starttime = time.time()

            # Idk why i do this, but i have a feeling it's right B-)
            TH = self.N - k
            self.controller.N = TH

            # Init controller with current state
            self.controller.set_bids(Bids[k], Bids[k+1])
            self.controller.x_init = X[:,k]
            

            # Decide next bids
            self.controller.optimize_baseline()
            self.controller.optimize_bidding()

            next_bid = self.controller.B_bid[:,0].flatten()
            Bids.append(Bid(next_bid[0], next_bid[1], next_bid[2], next_bid[3]))
            u0 = np.array(self.controller.u_bid)[0]
            
            bidding_z_opt = np.array(self.controller.sol_bid['x'])
            baseline_z_opt = np.array(self.controller.sol_base['x'])
            
            # Give optimizer a more optimal starting point next iteration
            self.controller.bidding_z_init = np.vstack((bidding_z_opt[2:TH], bidding_z_opt[TH+4:]))     # Skip first state and first bid 
            self.controller.baseline_z_init = np.vstack((baseline_z_opt[2:TH], baseline_z_opt[TH+1:]))  # Skip first state and first u

            # Grow the plant
            U[:,k] = u0
            X[:,k+1] = X[:,k] + self.dt*np.array(self.model.derivative(X[:,k], u0)).flatten()

            # Simulate market response
            self.controller.A_up = 1
            self.controller.A_down = 0

            # Move spot price one step forwards in time
            self.controller.p_spot = self.controller.p_spot[1:]

            # Status update
            iter_time = time.time()-iter_starttime
            minutes, seconds = divmod(iter_time, 60)
            print(f"Completed iteration {k} of {self.N-2} in: {int(minutes)} minutes and {seconds:.2f} seconds.")


        self.x_mpc = X
        self.u_mpc = U

        for i in range(self.N):
            self.bids_mpc[:, i] = Bids[i].as_array()



        return 0
            



            





        