import numpy as np
from scipy.stats import norm
import casadi as ca
import pandas as pd
from config import Config
from globals import *
import utils
import json
import os
import time

class Market:

    N: int
    seed: int
    bidding_zone: str
    date: str
    
    config: Config

    C_eur2nok = 11.76               # € -> NOK conversion rate as of nov 14 2024
    price_means: np.ndarray
    price_cov: np.ndarray

    mu_dn = 30          # € / MW
    mu_up = 50          # € / MW
    sigma_dn: float     # € / MW
    sigma_up: float     # € / MW

    mean_prices_up: np.array
    mean_prices_dn: np.array
    opt_prices_up: np.array
    opt_prices_dn: np.array

    spot_prices:    np.array
    mfrr_prices_up: np.array
    mfrr_prices_dn: np.array
    timestamps: np.array

    n_given_bids = 2            # Number of time intervals with previously submitted bids
    n_given_activations = 1     # Number of time intervals with received activations

    specs: dict


    def __init__(self, config, time_horizon, seed, bidding_zone, date):
        self.config = config
        self.N = time_horizon * QUARTER_HOURS_PER_DAY
        self.seed = seed
        self.bidding_zone = bidding_zone
        self.date = date

        self.analyze_price_covariances()
        conditional_variance_up, conditional_variance_dn = utils.conditional_covariance(self.price_cov)
        self.sigma_up = np.sqrt(conditional_variance_up)
        self.sigma_dn = np.sqrt(conditional_variance_dn)
        self.p_spot = self.get_spotprice()
        self.mean_prices_up = utils.conditional_expectation(self.p_spot, self.price_means, self.price_cov)[0]
        self.mean_prices_dn = utils.conditional_expectation(self.p_spot, self.price_means, self.price_cov)[1]
        self.opt_prices_up = np.zeros((1,self.N))
        self.opt_prices_dn = np.zeros((1,self.N))

        self.specs = {
            'bidding zone'                  : self.bidding_zone,
            'simdate'                       : self.date,
            'bidding zone'                  : self.bidding_zone,
            'eur to nok'                    : self.C_eur2nok,
            'Avg activation price up'       : self.price_means.iloc[1],
            'Avg activation price down'     : self.price_means.iloc[2],
            'Cond covariance spot - up'     : conditional_variance_up,
            'Cond covariance spot - Down'   : conditional_variance_dn,
            }
            


    

    def generate_spotprice(self):
        '''
        Returns an array of spot prices for each quarter hour. Number of quarter hours amounts to N. 
        N = 96 -> List of spot prices for a whole 24h period. 
        N = 94 -> 2 first QH are cut off. Assuming that the mpc ends on a whole 

        This is in kr/Kw. In order to work with mwh, you must factor that in :)
        '''

        N = self.N
        sigma=0.10

        hours = int(np.ceil(N/4))
        timesteps = np.linspace(0, hours-1, hours)
        timesteps = np.repeat(timesteps, 4)
        
        prices_mean = 0.7 - 0.2*np.cos(np.pi * timesteps/6) - 0.15*np.cos(np.pi * timesteps/12) 


        # Generate 24 random values from a normal distribution
        np.random.seed(self.seed)  # You can choose any integer value for the seed
        spot_prices = prices_mean + np.repeat(np.random.normal(loc=0, scale=sigma, size=hours), 4)

        # Ensure that no values are below 0
        spot_prices = np.clip(spot_prices, a_min=0, a_max=None)
        
        # Repeat each value 4 times to get a total of 96 elements
        return spot_prices[(hours*4)-N:]
    

    def get_spotprice(self) -> np.array:

        N = self.N
        n_hours = int(np.ceil(N/4))


        # df = pd.read_csv('../data/Spotprices_norway.csv', delimiter=';')
        df = pd.read_csv(self.config.path + 'data/Spotprices_norway.csv', delimiter=';')
        
        # Convert to datetime format
        df['Dato/klokkeslett'] = pd.to_datetime(df['Dato/klokkeslett'].str.split().str[0])
        start_idx = df[df['Dato/klokkeslett'] == pd.to_datetime(self.date)].index[0]
        
        P_spot_hours = np.array(df[self.bidding_zone].iloc[start_idx:start_idx+n_hours].values)
        P_spot = np.repeat(P_spot_hours, 4)[0:N]
        return P_spot
    

    def Pr_a_up(self, p_spot, Bc_up):
        #TODO Find real numbers here
        
        mu_up = utils.conditional_expectation(p_spot, self.price_means, self.price_cov)[0]
        sigma_up = self.sigma_up

        Bc_up_norm = (Bc_up - mu_up)/sigma_up

        # return norm.cdf(-Bc_up_norm)
        return self.Pr_D_up() * (1.0 + ca.erf(-Bc_up_norm / ca.sqrt(2.0))) / 2.0

    def Pr_a_dn(self, p_spot, Bc_dn):
        #TODO Find real numbers here

        mu_dn = utils.conditional_expectation(p_spot, self.price_means, self.price_cov)[1]
        sigma_dn = self.sigma_dn
        
        Bc_dn_norm = (Bc_dn - mu_dn)/sigma_dn

        # return norm.cdf(-Bc_dn_norm)
        return self.Pr_D_dn() * (1.0 + ca.erf(-Bc_dn_norm / ca.sqrt(2.0))) / 2.0


    def Pr_D_dn(self):  
        # Probability of the grid needing down regulation.
        # TODO implement actual model from Erlend when that's ready

        return 1/3
    
    def Pr_D_up(self):
        # Probability of the grid needing up regulation.
        # TODO implement actual model from Erlend when that's ready

        return 1/3


    def analyze_price_covariances(self):
        """
        Analyze price covariances or load precomputed results from a JSON file if it exists.
        """
        # Define the path to the JSON file
        analysis_file = os.path.join(self.config.data_path, "price_analysis.json")

        # Check if the JSON file exists
        if os.path.exists(analysis_file):
            print("Loading precomputed price analysis data from JSON file.")
            try:
                with open(analysis_file, 'r') as file:
                    analysis_data = json.load(file)
                
                # Load means and covariance matrix
                self.price_means = pd.Series(analysis_data["means"])
                self.price_cov = np.array(analysis_data["covariance_matrix"])
                
                # Load additional data arrays into a DataFrame
                merged_data = pd.DataFrame({
                    "Timestamp": pd.to_datetime(analysis_data["Timestamp"]),
                    "Spot Price": analysis_data["spot_prices"],
                    "Up Price": analysis_data["up_prices"],
                    "Down Price": analysis_data["down_prices"]
                })
                
                self.spot_prices = merged_data['Spot Price']
                self.mfrr_prices_up = merged_data['Up Price']
                self.mfrr_prices_dn = merged_data['Down Price']
                self.timestamps = merged_data['Timestamp']
                self.merged_data = merged_data

            except Exception as e:
                print(f"Error loading JSON file: {e}")
                # return -1  # Indicate an error
        else:
            print("Performing price analysis as no precomputed data found.")
            # Paths to CSV files
            spot_price_file = self.config.spotprice_data_path
            mfrr_price_file = self.config.mfrr_data_path

            # Column names to extract
            spot_timestamp_col = "DatoTid"  # Spot price timestamp column
            spot_price_col = 'NO1'  # Spot price column of interest
            mfrr_time_interval_col = "Time Interval"  # mFRR time interval column
            mfrr_up_price_col = "Up price"  # mFRR up price column
            mfrr_down_price_col = "Down Price"  # mFRR down price column

            # Load data
            spot_prices = utils.load_spot_prices(spot_price_file, spot_timestamp_col, spot_price_col)
            mfrr_prices = utils.load_mfrr_prices(mfrr_price_file, mfrr_time_interval_col, mfrr_up_price_col, mfrr_down_price_col)

            # Merge datasets
            merged_data = utils.merge_and_align(spot_prices, mfrr_prices)

            if merged_data.empty:
                print("The merged dataset is empty. Please check the alignment of timestamps.")
                return -1  # Indicate an error
            else:
                # Calculate covariance matrix
                covariance_matrix = utils.calculate_covariance_matrix(merged_data, ['Spot Price', 'Up Price', 'Down Price'])
                means = merged_data[['Spot Price', 'Up Price', 'Down Price']].mean()

                # Save results
                self.price_means = means
                self.price_cov = covariance_matrix
                self.spot_prices = np.array(merged_data['Spot Price'])
                self.mfrr_prices_up = np.array(merged_data['Up Price'])
                self.mfrr_prices_dn = np.array(merged_data['Down Price'])
                self.timestamps = merged_data['Timestamp']
                self.merged_data = merged_data

                # Ensure the directory exists
                os.makedirs(self.config.data_path, exist_ok=True)
                try:
                    with open(analysis_file, 'w') as file:
                        json.dump({
                            "means": self.price_means.to_dict(),
                            "covariance_matrix": self.price_cov.tolist(),
                            "Timestamp": merged_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                            "spot_prices": self.spot_prices.tolist(),
                            "up_prices": self.mfrr_prices_up.tolist(),
                            "down_prices": self.mfrr_prices_dn.tolist()
                        }, file)
                    print("Price analysis data saved to JSON file.")
                except Exception as e:
                    print(f"Error saving JSON file: {e}")
                    return -1  # Indicate an error

        return 0



    def optimal_bidding_price_prediction(self, p_spot):    

        start_time = time.time()
        print("Starting price prediction")

        mu_up = utils.conditional_expectation(p_spot, self.price_means, self.price_cov)[0]
        mu_dn = utils.conditional_expectation(p_spot, self.price_means, self.price_cov)[1]

        # Create decision variables for the optimization problem
        N = len(p_spot)
        X = ca.MX.sym('X', 2, N)

        J = 0
        for k in range(N):  J -= X[0,k] * self.Pr_a_up(p_spot[k], X[0,k])
        for k in range(N):  J -= X[1,k] * self.Pr_a_dn(p_spot[k], X[1,k])
        
        lbx = 0
        ubx = 1000

        # Flatten decision variables and bounds
        Z =   ca.vertcat(ca.reshape(X, -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        u0 = np.hstack((mu_up, mu_dn))
        sol = solver(x0 = u0, lbx=lbz, ubx=ubz)

        # Extract solution
        x    = np.array(sol['x'])
        x_up = x[:N]
        x_dn = x[N:]

        self.opt_prices_up = x_up
        self.opt_prices_dn = x_dn

        # print(f"Predicted optimal bidding prices: \n Up:   {x_up[range(0,N,24)]} \n Down: {x_dn[range(0,N,24)]}")
        print(f"Predicted Avg bidding prices: \n Up:   {np.average(x_up):.2f} \n Down: {np.average(x_dn):.2f}")

        # print(f"Predicted optimal bidding price is on average: \n Up:   {np.average(delta_up)} compared to predicted clearing price \n Down: {np.average(delta_dn)} compared to clearing price")

        end_time = time.time()
        print(f"Completed prediction in {end_time-start_time:.2f} seconds")
        return 0



    
    
class Bid():

    volume_up:      float
    volume_down:    float
    price_up:       float
    price_down:     float

    '''
    activated_up:   float
    activated_down: float
    '''


    def __init__(self, volume_up = 0, volume_down = 0, 
                        price_up = 0, price_down = 0, 
                        activated_up = None, activated_down = None):
        
        self.volume_up      = volume_up
        self.volume_down    = volume_down
        self.price_up       = price_up
        self.price_down     = price_down

        '''

        if activated_up: 
            self.activated_up = activated_up
        else:
            self.activated_up = Market.Pr_a_up(price_up)

        if activated_down: 
            self.activated_down = activated_down
        else:
            self.activated_down = Market.Pr_a_dn(price_down)

        '''
        
    def as_array(self):

        return np.array([self.volume_up, self.volume_down, self.price_up, self.price_up])