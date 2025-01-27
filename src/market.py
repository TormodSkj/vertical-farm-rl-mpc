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
    optimistic: bool            # Optimistic market model assumes completely accurate forecasting of spot prices, clearing prices and activation occurances.
    
    config: Config

    C_eur2nok = 11.76               # € -> NOK conversion rate as of nov 14 2024
    price_means:        np.ndarray
    price_cov:          np.ndarray

    mean_prices_up:     np.array    # Array of most likely clearing prices for up-regulation at each time step      (length: N)
    mean_prices_dn:     np.array    # Array of most likely clearing prices for down-regulation at each time step    (length: N)
    opt_prices_up:      np.array    # Array of most profitable bidding prices for up-regulation at each time step   (length: N)
    opt_prices_dn:      np.array    # Array of most profitable bidding prices for down-regulation at each time step (length: N)
    spot_prices:        np.array    # Array of spot prices used in optimization                                     (length: N)

    mfrr_prices_up:     np.array    # mFRR clearing prices up used in data analysis     (Entire dataset)
    mfrr_prices_dn:     np.array    # mFRR clearing prices down used in data analysis   (Entire dataset)
    spot_price_data:    np.array    # Array of spot prices used in data analysis        (Entire dataset)
    timestamps:         np.array

    up_activation_occurance_rate: float      # Probability of an up-activation happening evey MTU   (Avg of entire dataset) [0, 1]
    down_activation_occurance_rate: float    # Probability of a down-activation happening evey MTU  (Avg of entire dataset) [0, 1]
    both_activation_occurance_rate: float
    mfrr_demands_up: np.array   # Array of when activations are made during current growth cycle (lenght: N)
    mfrr_demands_dn: np.array   # Array of when activations are made during current growth cycle (length: N)

    n_given_bids = 2            # Number of time intervals with previously submitted bids
    n_given_activations = 1     # Number of time intervals with received activations

    specs: dict


    def __init__(self, config, time_horizon, bidding_zone, date, optimistic = False):
        self.config = config
        self.N = time_horizon * QUARTER_HOURS_PER_DAY
        self.seed = config.seed
        self.bidding_zone = bidding_zone
        self.date = date
        self.optimistic = optimistic
        self.spot_prices = self.get_spotprice()

        self.analyze_price_covariances()
        conditional_variance_up, conditional_variance_dn = utils.conditional_covariance(self.price_cov)
        self.sigma_up = np.sqrt(conditional_variance_up)
        self.sigma_dn = np.sqrt(conditional_variance_dn)
        self.mean_prices_up = utils.conditional_expectation(self.spot_prices, self.price_means, self.price_cov)[0]
        self.mean_prices_dn = utils.conditional_expectation(self.spot_prices, self.price_means, self.price_cov)[1]
        self.opt_prices_up = np.zeros((1,self.N))
        self.opt_prices_dn = np.zeros((1,self.N))

        self.specs = {
            'bidding zone'                  : self.bidding_zone,
            'simdate'                       : self.date,
            'optimistic'                    : self.optimistic,
            'eur to nok'                    : self.C_eur2nok,
            'Avg activation price up'       : self.price_means[1],
            'Avg activation price down'     : self.price_means[2],          
            'Cond covariance spot - up'     : conditional_variance_up,
            'Cond covariance spot - Down'   : conditional_variance_dn,
            }
            
        self.mfrr_activation_data_analysis()
        self.mfrr_demands_up, self.mfrr_demands_dn = self.get_activation_demands(self.date)
    

    def get_spotprice(self) -> np.array:

        N = self.N
        n_hours = int(np.ceil(N/4))


        # df = pd.read_csv('../data/Spotprices_norway.csv', delimiter=';')
        df = pd.read_csv(self.config.path + 'data/Spotprices_norway.csv', delimiter=';')
        
        # Convert to datetime format
        df['Dato/klokkeslett'] = pd.to_datetime(df['Dato/klokkeslett'].str.split().str[0])
        start_idx = df[df['Dato/klokkeslett'] == pd.to_datetime(self.date)].index[0]
        
        spot_prices_hours = np.array(df[self.bidding_zone].iloc[start_idx:start_idx+n_hours].values)
        spot_prices = np.repeat(spot_prices_hours, 4)[0:N]
        assert len(spot_prices) == N, "Insufficient spot price data"
        return spot_prices
    

    def activation_prob_up(self, spot_prices, Bc_up):
        
        mu_up = utils.conditional_expectation(spot_prices, self.price_means, self.price_cov)[0]
        sigma_up = self.sigma_up

        Bc_up_norm = (Bc_up - ca.vertcat(*mu_up))/sigma_up

        # return norm.cdf(-Bc_up_norm)
        return self.demand_prob_up() * (1.0 + self.error_function(-Bc_up_norm / ca.sqrt(2.0))) / 2.0

    def activation_prob_dn(self, spot_prices, Bc_dn):

        mu_dn = utils.conditional_expectation(spot_prices, self.price_means, self.price_cov)[1]
        sigma_dn = self.sigma_dn
        
        Bc_dn_norm = (Bc_dn - ca.vertcat(*mu_dn))/sigma_dn

        # return norm.cdf(-Bc_dn_norm)
        # return self.demand_prob_dn() * (1.0 + ca.erf(-Bc_dn_norm / ca.sqrt(2.0))) / 2.0
        return self.demand_prob_dn() * (1.0 + self.error_function(-Bc_dn_norm / ca.sqrt(2.0))) / 2.0


    def error_function(self, x):
        '''
        Function implemented to explore alternatives to ca.erf
        Some quick testing shows that ca.erf is sufficiently fast, most likely due to the simpler derivative
        '''

        # tanh error funciton approx:
        # return ca.tanh(x)
        
        # Casadi error function 
        return ca.erf(x)



    def demand_prob_up(self):
        # Probability of the grid needing up regulation.
        # TODO implement actual model from Erlend when that's ready

        # return self.up_activation_occurance_rate      # Use predicted demand rate from data
        return np.mean(self.mfrr_demands_up)            # Use actual demand rate
    
    def demand_prob_dn(self):  
        # Probability of the grid needing down regulation.
        # TODO implement actual model from Erlend when that's ready

        # return self.down_activation_occurance_rate    # Use predicted demand rate from data
        return np.mean(self.mfrr_demands_dn)            # Use actual demand rate


    def analyze_price_covariances(self):
        """
        Analyze price covariances or load precomputed results from a JSON file if it exists.
        """
        # Define the path to the JSON file
        analysis_file = os.path.join(self.config.data_path, "price_analysis.json")

        # Check if the JSON file exists
        #TODO remove the false here. Wanted to disable it for a while
        if False and os.path.exists(analysis_file):
            print("Loading precomputed price analysis data from JSON file.")

            with open(analysis_file, 'r') as file:
                analysis_data = json.load(file)
            
            # Load means and covariance matrix
            self.price_means = pd.Series(analysis_data["means"])
            self.price_cov = np.array(analysis_data["covariance_matrix"])
            
            # Load additional data arrays into a DataFrame
            merged_data = pd.DataFrame({
                "Start Time": pd.to_datetime(analysis_data["Start Time"]),
                "Spot Price": analysis_data["spot_prices"],
                "Up Price": analysis_data["up_prices"],
                "Down Price": analysis_data["down_prices"]
            })
            
            self.spot_price_data = merged_data['Spot Price']
            self.mfrr_prices_up = merged_data['Up Price']
            self.mfrr_prices_dn = merged_data['Down Price']
            self.timestamps = merged_data['Start Time']
            self.merged_data = merged_data

        else:
            print("Performing price analysis as no precomputed data found.")
            # Paths to CSV files
            spot_price_file = self.config.spotprice_data_path
            mfrr_price_datapath = self.config.mfrr_clearing_price_data_path


            # Load data
            spot_prices = utils.load_spot_prices(spot_price_file, self.bidding_zone)
            mfrr_prices = utils.load_mfrr_prices(mfrr_price_datapath, self.bidding_zone)

            # Merge datasets
            merged_data = utils.merge_and_align(spot_prices, mfrr_prices)

            if merged_data.empty:
                print("The merged dataset is empty. Please check the alignment of timestamps.")
                return -1  # Indicate an error
            else:
                # Calculate covariance matrix
                
                clearing_prices_up, clearing_prices_dn = self.get_clearing_prices(self.date)
                data = np.vstack((self.spot_prices, clearing_prices_up, clearing_prices_dn))
                covariance_matrix = np.cov(data)
                means = np.mean(data, axis=1)

                # covariance_matrix = utils.calculate_covariance_matrix(merged_data, ['Spot Price', 'Up Price', 'Down Price'])
                # means = merged_data[['Spot Price', 'Up Price', 'Down Price']].mean()

                # Save results
                self.price_means = means
                self.price_cov = covariance_matrix
                self.spot_price_data = np.array(merged_data['Spot Price'])
                self.mfrr_prices_up = np.array(merged_data['Up Price'])
                self.mfrr_prices_dn = np.array(merged_data['Down Price'])
                self.timestamps = merged_data['Start Time']
                self.merged_data = merged_data

                # Ensure the directory exists
                os.makedirs(self.config.data_path, exist_ok=True)
                try:
                    with open(analysis_file, 'w') as file:
                        json.dump({
                            "means": self.price_means.to_dict(),
                            "covariance_matrix": self.price_cov.tolist(),
                            "Start Time": merged_data['Start Time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                            "spot_prices": self.spot_price_data.tolist(),
                            "up_prices": self.mfrr_prices_up.tolist(),
                            "down_prices": self.mfrr_prices_dn.tolist()
                        }, file)
                    print("Price analysis data saved to JSON file.")
                except Exception as e:
                    print(f"Error saving JSON file: {e}")
                    return -1  # Indicate an error

        return 0



    def optimal_bidding_price_prediction(self, spot_prices):    

        start_time = time.time()
        print("Starting price prediction")

        mu_up, mu_dn = utils.conditional_expectation(spot_prices, self.price_means, self.price_cov)

        # Create decision variables for the optimization problem
        N = len(spot_prices)
        X = ca.MX.sym('X', 2, N)

        J = 0
        for k in range(N):  J -= X[0,k] * self.activation_prob_up(spot_prices[k], X[0,k])
        for k in range(N):  J -= X[1,k] * self.activation_prob_dn(spot_prices[k], X[1,k])
        
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

        u0 = np.vstack((mu_up, mu_dn)).ravel(order='F')
        sol = solver(x0 = u0, lbx=lbz, ubx=ubz)

        # Extract solution
        x    = np.array(sol['x'])
        x_up = x[::2]  
        x_dn = x[1::2] 

        self.opt_prices_up = x_up
        self.opt_prices_dn = x_dn

        # print(f"Predicted optimal bidding prices: \n Up:   {x_up[range(0,N,24)]} \n Down: {x_dn[range(0,N,24)]}")
        print(f"Predicted Avg bidding prices: \n Up:   {np.average(x_up):.2f} \n Down: {np.average(x_dn):.2f}")

        # print(f"Predicted optimal bidding price is on average: \n Up:   {np.average(delta_up)} compared to predicted clearing price \n Down: {np.average(delta_dn)} compared to clearing price")

        end_time = time.time()
        print(f"Completed prediction in {end_time-start_time:.2f} seconds")
        return 0



    def generate_activation_demands(self, N, seed):
        '''
        Generates a list of ints where 0 means no demand for activation, -1 means down activation and 1 means up
        '''

        D_up = self.demand_prob_up()
        D_dn = self.demand_prob_dn()
        no_D = 1 - D_up - D_dn

        activation_demands = utils.generate_weighted_samples([-1, 0, 1], [D_dn, no_D, D_up], N, seed)

        return activation_demands

    

    def mfrr_activation_data_analysis(self):
        
        data_path = self.config.mfrr_activation_data_path
        # utils.clean_mfrr_csv_file(filepath)

        up_activation_df, down_activation_df = utils.load_mfrr_activation_data(data_path, self.bidding_zone)
        self.up_activation_df, self.down_activation_df = up_activation_df, down_activation_df

        # Activation rate of each offered MW of capacity 
        self.up_activation_ratio    = np.sum(up_activation_df['Activated']) / np.sum(up_activation_df['Offered'])
        self.down_activation_ratio  = np.sum(down_activation_df['Activated']) / np.sum(down_activation_df['Offered'])
        
        # Arrays denoting activation occurances
        up_activation_occurances   = np.where(np.array(up_activation_df['Activated'])>0, 1, 0)
        down_activation_occurances = np.where(np.array(down_activation_df['Activated'])>0, 1, 0)
        both_activation_occurances = np.where(np.logical_and(np.array(up_activation_df['Activated'])>0,np.array(down_activation_df['Activated'])>0), 1, 0)

        # % of QH where activations occur
        self.up_activation_occurance_rate     = np.mean(up_activation_occurances)
        self.down_activation_occurance_rate   = np.mean(down_activation_occurances)
        self.both_activation_occurance_rate   = np.mean(both_activation_occurances)

        return 0
    
    def get_clearing_prices(self, date=None):

        if date==None:
            date = self.date

        clearing_price_datapath = self.config.mfrr_clearing_price_data_path

        clearing_prices_df = utils.load_mfrr_prices(clearing_price_datapath, self.bidding_zone)

        # Remove dates before simdate
        clearing_prices_df = clearing_prices_df.where(clearing_prices_df['Start Time']>pd.to_datetime(date)).dropna()

        clearing_prices_up = np.array(clearing_prices_df['Up Price']).repeat(4)[:self.N]
        clearing_prices_dn = np.array(clearing_prices_df['Down Price']).repeat(4)[:self.N]

        return clearing_prices_up, clearing_prices_dn
    
    def get_activation_demands(self, date):
        '''
        Returns numpy arrays of length N containing the MW total activated during each quarter hour from the start time.
        Start time is always assumed at 00:00 at the given start date.
        '''


        up_activation_df, down_activation_df = self.up_activation_df, self.down_activation_df

        # Remove dates before simdate
        up_activations_df = up_activation_df.where(up_activation_df['Start Time']>pd.to_datetime(date)).dropna()
        down_activations_df = down_activation_df.where(down_activation_df['Start Time']>pd.to_datetime(date)).dropna()

        demands_up = np.array(up_activations_df['Activated'])[:self.N]
        demands_dn = np.array(down_activations_df['Activated'])[:self.N]

        return np.where(demands_up > 0, 1, 0), np.where(demands_dn > 0, 1, 0)
    
