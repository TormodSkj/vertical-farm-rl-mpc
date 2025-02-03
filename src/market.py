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
    T: float
    seed: int
    bidding_zone: str
    date: str
    optimistic: bool            # Optimistic market model restricts analysis dataset to be only that of the growth cycle. Makes price distribution analysis more accurate
    
    config: Config

    C_eur2nok = 11.76                           # € -> NOK conversion rate as of nov 14 2024
    price_means:                np.ndarray
    price_cov:                  np.ndarray

    prices_full_set:            pd.DataFrame    # Full dataset of spot prices, clearing prices
    prices_working_set:         pd.DataFrame    # Slice of full dataset used in market model for control. 
    activations_full_set:       pd.DataFrame    # Full dataset of mfrr activations
    activations_working_set:    pd.DataFrame    # Slice of full dataset used in market model for control.

    expected_prices_up:         np.array        # Array of most likely clearing prices for up-regulation at each time step      (length: N)
    expected_prices_dn:         np.array        # Array of most likely clearing prices for down-regulation at each time step    (length: N)
    opt_prices_up:              np.array        # Array of most profitable bidding prices for up-regulation at each time step   (length: N)
    opt_prices_dn:              np.array        # Array of most profitable bidding prices for down-regulation at each time step (length: N)
    spot_prices:                np.array        # Array of spot prices used in optimization                                     (length: N)

    up_activation_occurance_rate:   float       # Probability of an up-activation happening evey MTU   (Avg of entire dataset) [0, 1]
    down_activation_occurance_rate: float       # Probability of a down-activation happening evey MTU  (Avg of entire dataset) [0, 1]
    both_activation_occurance_rate: float
    mfrr_demands_up: np.array   # Array of when activations are made during current growth cycle (lenght: N)
    mfrr_demands_dn: np.array   # Array of when activations are made during current growth cycle (length: N)

    n_given_bids = 2            # Number of time intervals with previously submitted bids
    n_given_activations = 1     # Number of time intervals with received activations

    specs: dict


    def __init__(self, config, time_horizon, bidding_zone, date, outlier_max_dist, optimistic = False):
        self.config = config
        self.N = time_horizon * QUARTER_HOURS_PER_DAY
        self.T = time_horizon
        self.seed = config.seed
        self.bidding_zone = bidding_zone
        self.date = date
        self.optimistic = optimistic 
        self.outlier_max_dist = outlier_max_dist

        self.import_spot_mfrr_data()

        self.spot_prices = self.get_spotprice() 

        self.analyze_price_covariances()
        conditional_variance_up, conditional_variance_dn = utils.conditional_covariance(self.price_covs)
        self.expected_prices_up, self.expected_prices_dn = utils.conditional_expectation(self.spot_prices, self.price_means, self.price_covs)
        
        epsilon = 1e-6  # For numerical stability. Avoids 0-variance
        self.sigma_up = np.sqrt(conditional_variance_up) + epsilon
        self.sigma_dn = np.sqrt(conditional_variance_dn) + epsilon
        
        # Initialize optimal prices on expected value.
        self.opt_prices_up = self.expected_prices_up
        self.opt_prices_dn = self.expected_prices_dn

        self.specs = {
            'bidding zone'                  : self.bidding_zone,
            'simdate'                       : self.date,
            'optimistic'                    : self.optimistic,
            'eur to nok'                    : self.C_eur2nok,
            'Avg activation price up'       : self.price_means[2],
            'Avg activation price down'     : self.price_means[3],          
            'Cond covariance spot - up'     : conditional_variance_up,
            'Cond covariance spot - Down'   : conditional_variance_dn,
            'Outlier max std-dev distance'  : self.outlier_max_dist
            }
            
        self.mfrr_activation_data_analysis()
        self.mfrr_demands_up, self.mfrr_demands_dn = self.get_activation_demands(self.date)
        self.analyze_market_potency(T = self.T)
    

    def get_spotprice(self) -> np.array:

        N = self.N
        n_hours = int(np.ceil(N/4))

        df = self.prices_full_set
        start_idx = df[df['Start Time'] == pd.to_datetime(self.date)].index[0]
        
        spot_prices_hours = np.array(df['Spot Price'].iloc[start_idx:start_idx+n_hours].values)
        spot_prices = np.repeat(spot_prices_hours, 4)[0:N]
        assert len(spot_prices) == N, "Insufficient spot price data"
        return spot_prices
    

    def activation_prob_up(self, spot_price, bid_price_up):
        
        mu_up = utils.conditional_expectation(spot_price, self.price_means, self.price_covs)[0]
        sigma_up = self.sigma_up

        bid_price_dn_normalized = (bid_price_up - ca.vertcat(*mu_up))/sigma_up

        # return norm.cdf(-Bc_up_norm)
        return self.demand_prob_up() * (1.0 + self.error_function(-bid_price_dn_normalized / ca.sqrt(2.0))) / 2.0

    def activation_prob_dn(self, spot_price, bid_price_dn):

        mu_dn = utils.conditional_expectation(spot_price, self.price_means, self.price_covs)[1]
        sigma_dn = self.sigma_dn
        
        bid_price_dn_normalized = (bid_price_dn - ca.vertcat(*mu_dn))/sigma_dn

        # return norm.cdf(-Bc_dn_norm)
        # return self.demand_prob_dn() * (1.0 + ca.erf(-Bc_dn_norm / ca.sqrt(2.0))) / 2.0
        return self.demand_prob_dn() * (1.0 + self.error_function(-bid_price_dn_normalized / ca.sqrt(2.0))) / 2.0


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


    def import_spot_mfrr_data(self):
        
        spot_price_file = self.config.spotprice_data_path
        mfrr_price_datapath = self.config.mfrr_clearing_price_data_path
        mfrr_activation_datapath = self.config.mfrr_activation_data_path
        # Load data
        spot_prices = utils.load_spot_prices(spot_price_file, self.bidding_zone)
        mfrr_prices = utils.load_mfrr_prices(mfrr_price_datapath, self.bidding_zone)

        # Merge datasets
        price_data = pd.merge(spot_prices, mfrr_prices, on='Start Time', how='inner')
        self.prices_full_set = price_data
        activation_data = utils.load_mfrr_activation_data(mfrr_activation_datapath, self.bidding_zone)
        self.activations_full_set = activation_data

        up_prices_merged_data = pd.merge(price_data[['Start Time', 'Spot Price', 'Clearing Price Up']], activation_data[['Start Time', 'Offered Up', 'Activated Up']], on='Start Time', how='inner')
        down_prices_merged_data = pd.merge(price_data[['Start Time', 'Spot Price', 'Clearing Price Down']], activation_data[['Start Time', 'Offered Down', 'Activated Down']], on='Start Time', how='inner')
        self.up_prices_full_set = up_prices_merged_data
        self.down_prices_full_set = down_prices_merged_data

        if self.optimistic:
            self.prices_working_set         = price_data[price_data['Start Time'] >= pd.to_datetime(self.date)].head(int(np.ceil(self.N/QUARTER_HOURS_PER_HOUR)))
            self.activations_working_set    = activation_data[activation_data['Start Time'] >= pd.to_datetime(self.date)].head(int(np.ceil(self.N/QUARTER_HOURS_PER_HOUR)))
            up_prices_working_set      = up_prices_merged_data.loc[(up_prices_merged_data['Start Time'] >= pd.to_datetime(self.date))].head(int(np.ceil(self.N / QUARTER_HOURS_PER_HOUR))).loc[(up_prices_merged_data['Activated Up'] > 0)]
            down_prices_working_set    = down_prices_merged_data.loc[(down_prices_merged_data['Start Time'] >= pd.to_datetime(self.date))].head(int(np.ceil(self.N / QUARTER_HOURS_PER_HOUR))).loc[(down_prices_merged_data['Activated Down'] > 0)]
    
        else:
            self.prices_working_set = self.prices_full_set
            self.activations_working_set = self.activations_full_set
            up_prices_working_set = self.up_prices_full_set
            down_prices_working_set = self.down_prices_full_set
        
        # Remove clear outliers
        
        mean_up = np.mean(up_prices_working_set['Clearing Price Up'])
        std_up = np.sqrt(np.var(up_prices_working_set['Clearing Price Up']))
        mean_down = np.mean(down_prices_working_set['Clearing Price Down'])
        std_down = np.sqrt(np.var(down_prices_working_set['Clearing Price Down']))
        
        # remove all entries outside 4 standard deviations. These should only be extreme cases
        epsilon = 1e-6
        self.up_prices_working_set      = up_prices_working_set[np.abs(up_prices_working_set['Clearing Price Up'] - mean_up) / (std_up + epsilon) < self.outlier_max_dist]
        self.down_prices_working_set    = down_prices_working_set[np.abs(down_prices_working_set['Clearing Price Down'] - mean_down) / (std_down + epsilon) < self.outlier_max_dist]        
        
        return 0 



    def analyze_price_covariances(self):
        """
        Analyze price covariances or load precomputed results from a JSON file if it exists.
        """
        
        print("Performing price analysis.")
        # Paths to CSV files
    
        up_price_data = self.up_prices_working_set
        down_price_data = self.down_prices_working_set

        if up_price_data.empty:
            print("The working dataset for up prices is empty. Using full dataset")
            up_price_data = self.up_prices_full_set
            assert False, 'Up price data is empty, date is likely not supported in the dataset. Or there are no activations of this type during the simulation time'
        elif down_price_data.empty:
            print("The working dataset for down prices is empty. Using full dataset")
            down_price_data = self.down_prices_full_set
            assert False, 'Down price data is empty, date is likely not supported in the dataset Or there are no activations of this type during the simulation time'

        # covariance_matrix = utils.calculate_covariance_matrix(price_data, ['Spot Price', 'Clearing Price Up', 'Clearing Price Down'])
        
        spot_up_cov     = np.cov(up_price_data[['Spot Price', 'Clearing Price Up']].T)
        spot_down_cov   = np.cov(down_price_data[['Spot Price', 'Clearing Price Down']].T)
        
        mean_price_up       = up_price_data['Clearing Price Up'].mean()
        mean_spot_price_up  = up_price_data['Spot Price'].mean()

        mean_price_down = down_price_data['Clearing Price Down'].mean()
        mean_spot_price_down = down_price_data['Spot Price'].mean()

        # Save results
        self.price_means = np.array([mean_spot_price_up, mean_spot_price_down, mean_price_up, mean_price_down])
        self.price_covs = np.array([spot_up_cov, spot_down_cov])

        return 0



    def optimal_bidding_price_prediction(self, spot_prices):    

        start_time = time.time()
        print("Starting price prediction")

        mu_up, mu_dn = utils.conditional_expectation(spot_prices, self.price_means, self.price_covs)

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
        
        activation_df = self.activations_working_set

        # Activation rate of each offered MW of capacity 
        self.up_activation_capacity_ratio    = np.sum(activation_df['Activated Up']) / np.sum(activation_df['Offered Up'])
        self.down_activation_capacity_ratio  = np.sum(activation_df['Activated Down']) / np.sum(activation_df['Offered Down'])
        
        # Arrays denoting activation occurances
        up_activation_occurances   = np.where(np.array(activation_df['Activated Up'])>0, 1, 0)
        down_activation_occurances = np.where(np.array(activation_df['Activated Down'])>0, 1, 0)
        both_activation_occurances = np.where(np.logical_and(np.array(activation_df['Activated Up'])>0,np.array(activation_df['Activated Down'])>0), 1, 0)

        # % of QH where activations occur
        self.up_activation_occurance_rate     = np.mean(up_activation_occurances)
        self.down_activation_occurance_rate   = np.mean(down_activation_occurances)
        self.both_activation_occurance_rate   = np.mean(both_activation_occurances)

        return 0
    
    def get_clearing_prices(self, date=None):

        if date==None:
            date = self.date

        clearing_prices_df = self.prices_full_set

        # Remove dates before simdate
        clearing_prices_df = clearing_prices_df.where(clearing_prices_df['Start Time']>=pd.to_datetime(date)).dropna()

        clearing_prices_up = np.array(clearing_prices_df['Clearing Price Up']).repeat(QUARTER_HOURS_PER_HOUR)[:self.N]
        clearing_prices_dn = np.array(clearing_prices_df['Clearing Price Down']).repeat(QUARTER_HOURS_PER_HOUR)[:self.N]

        return clearing_prices_up, clearing_prices_dn
    
    def get_activation_demands(self, date = None):
        '''
        Returns numpy arrays of length N with balancing demands during each quarter hour from the start time.
        For every MTU, a 1 indicates that an activation was made and a 0 indicates that no activation was made.
        Start time is always assumed at 00:00 at the given start date.
        '''

        if date is None:
            date = self.date

        activations_df = self.activations_working_set

        # Remove dates before simdate
        activations_df = activations_df.where(activations_df['Start Time']>=pd.to_datetime(date)).dropna()

        demands_up = np.array(activations_df['Activated Up']).repeat(QUARTER_HOURS_PER_HOUR)[:self.N]
        demands_dn = np.array(activations_df['Activated Down']).repeat(QUARTER_HOURS_PER_HOUR)[:self.N]

        return np.where(demands_up > 0, 1, 0), np.where(demands_dn > 0, 1, 0)
    


    def analyze_market_potency(self, T=20):
        '''
        Comb through clearing prices and activations to find the timespan of length `T`
        with the highest potential profit in the mFRR market

        `T`: Time window of market participation
        '''

        activaion_df = self.activations_full_set
        clearing_prices_df = self.prices_full_set

        # Merge dataframes to ensure data is present at all applicable time stamps
        merged_df = pd.merge(clearing_prices_df, activaion_df, on='Start Time', how='inner')

        merged_df.loc[:, 'Start Time'] = merged_df['Start Time'].dt.date
        date_range = pd.date_range(start=merged_df['Start Time'].min(), end=merged_df['Start Time'].max())

        market_potency_up = (
            merged_df.loc[merged_df['Activated Up'] > 0, ['Start Time', 'Clearing Price Up']]
            .groupby('Start Time', as_index=False)
            .sum()
        ).rename(columns={'Start Time': 'Date', 'Clearing Price Up': 'Potency Up'})

        market_potency_down = (
            merged_df.loc[merged_df['Activated Down'] > 0, ['Start Time', 'Clearing Price Down']]
            .groupby('Start Time', as_index=False)
            .sum()
        ).rename(columns={'Start Time': 'Date', 'Clearing Price Down': 'Potency Down'})

        # TODO Manage negative potency values better 
        market_potency_up['Potency Up']     = np.maximum(0, market_potency_up['Potency Up'])
        market_potency_down['Potency Down'] = np.maximum(0, market_potency_down['Potency Down'])

        market_potency_df = pd.DataFrame({'Date': date_range})
        market_potency_df = market_potency_df.merge(market_potency_up, on='Date',   how='left')
        market_potency_df = market_potency_df.merge(market_potency_down, on='Date', how='left')

        market_potency_df.fillna(0, inplace=True)

        self.daily_market_potency_df = market_potency_df

        # Rolling window potency:
        rolling_market_potency_df = market_potency_df.copy()
        rolling_market_potency_df['Total Potency'] = rolling_market_potency_df['Potency Up'] + rolling_market_potency_df['Potency Down']

        rolling_market_potency_df['Rolling Potency Up']     = rolling_market_potency_df['Potency Up'].rolling(window=T,    min_periods=1).sum().shift(-(T-1))
        rolling_market_potency_df['Rolling Potency Down']   = rolling_market_potency_df['Potency Down'].rolling(window=T,  min_periods=1).sum().shift(-(T-1))
        rolling_market_potency_df['Rolling Total Potency']  = rolling_market_potency_df['Total Potency'].rolling(window=T, min_periods=1).sum().shift(-(T-1))

        rolling_market_potency_df.dropna(inplace=True)

        self.rolling_market_potency_df = rolling_market_potency_df

        return 0

