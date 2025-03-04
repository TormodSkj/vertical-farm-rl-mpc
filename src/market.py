import numpy as np
from scipy.stats import norm
import casadi as ca
import pandas as pd
from config import Config
from globals import *
from utils import *
from market_utils import *
import json
import os
import time
from settings import Settings

class Market:

    N: int
    T: float
    seed: int
    bidding_zone: str
    date: str
    optimistic: bool            # Optimistic market model restricts analysis dataset to be only that of the growth cycle. Makes price distribution analysis more accurate
    
    config: Config
    settings: Settings

    C_eur2nok = 11.76                           # € -> NOK conversion rate as of nov 14 2024
    price_means:                np.ndarray
    price_cov:                  np.ndarray

    prices_full_set:            pd.DataFrame    # Full dataset of spot prices, clearing prices
    prices_working_set:         pd.DataFrame    # Slice of full dataset used in market model for control. 
    activations_full_set:       pd.DataFrame    # Full dataset of mfrr activations
    activations_working_set:    pd.DataFrame    # Slice of full dataset used in market model for control.

    expected_prices_up:         np.array        # Array of most likely clearing prices for up-regulation at each time step      (length: N)
    expected_prices_down:         np.array        # Array of most likely clearing prices for down-regulation at each time step    (length: N)
    opt_prices_up:              np.array        # Array of most profitable bidding prices for up-regulation at each time step   (length: N)
    opt_prices_down:              np.array        # Array of most profitable bidding prices for down-regulation at each time step (length: N)
    spot_prices:                np.array        # Array of spot prices used in optimization                                     (length: N)

    up_activation_occurance_rate:   float       # Probability of an up-activation happening evey MTU   (Avg of entire dataset) [0, 1]
    down_activation_occurance_rate: float       # Probability of a down-activation happening evey MTU  (Avg of entire dataset) [0, 1]
    both_activation_occurance_rate: float
    mfrr_demands_up: np.array   # Array of when activations are made during current growth cycle (lenght: N)
    mfrr_demands_down: np.array   # Array of when activations are made during current growth cycle (length: N)

    n_given_bids = 2            # Number of time intervals with previously submitted bids
    n_given_activations = 1     # Number of time intervals with received activations

    specs: dict


    def __init__(self, settings: Settings, config: Config):
        self.config = config
        self.settings = settings

        self.market_settings = settings.get_settings_group('general', 'market')

        self.T                  = self.market_settings['SIMULATION_LENGTH']
        self.bidding_zone       = self.market_settings['BIDDING_ZONE']
        self.date               = self.market_settings['SIMULATION_DATE']
        self.optimistic         = self.market_settings['OPTIMISTIC']
        self.outlier_max_dist   = self.market_settings['C_eur2nok']
        self.seed               = self.market_settings['SEED']
        self.N = self.T * QUARTER_HOURS_PER_DAY

        self.import_spot_mfrr_data()
        self.import_CM_data()

        self.spot_prices = self.get_spotprice() 

        self.analyze_price_covariances()
        conditional_variance_up, conditional_variance_dn    = conditional_covariance(self.price_covs)
        self.expected_prices_up, self.expected_prices_down  = conditional_expectation(self.spot_prices, self.price_means, self.price_covs)
        
        self.analyze_activation_covariances()


        epsilon = 1e-6  # For numerical stability. Avoids 0-variance
        self.sigma_up = np.sqrt(conditional_variance_up) + epsilon
        self.sigma_dn = np.sqrt(conditional_variance_dn) + epsilon
        
        # Initialize optimal prices on expected value.
        self.opt_prices_up = self.expected_prices_up
        self.opt_prices_down = self.expected_prices_down

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
        self.mfrr_demands_up, self.mfrr_demands_down = self.get_activation_demands(self.date)
        self.analyze_market_potency(T = self.T)
    

    def get_spotprice(self, date=None, n_days=None) -> np.array:

        if date==None:
            date = self.date

        if n_days==None:
            n_days = self.T

        start_date = pd.to_datetime(date)
        end_date = start_date + pd.DateOffset(n_days)
        
        
        # N = self.N
        # n_hours = int(np.ceil(N/4))

        df = self.prices_full_set.copy()
        # start_idx = df[df['Start Time'] == pd.to_datetime(self.date)].index[0]

        # Remove dates before simdate
        df = df[
            (df['Start Time']   >= start_date)    &
            (df['Start Time']   <  end_date) 
            ]
        
        df.fillna(df.mean(), inplace=True)
        
        spot_prices_hours = np.array(df['Spot Price'].values)
        spot_prices = np.repeat(spot_prices_hours, QUARTER_HOURS_PER_HOUR)
        return spot_prices
    

    def activation_prob_up(self, spot_price, bid_price_up):
        bid_price_up = bid_price_up.reshape((1,-1))

        mu_up = conditional_expectation(spot_price, self.price_means, self.price_covs)[0]
        sigma_up = self.sigma_up

        bid_price_up_normalized = ((bid_price_up - ca.vertcat(*mu_up).reshape((1,-1)))/sigma_up).reshape((1,-1))

        # return norm.cdf(-Bc_up_norm)
        return np.multiply(self.demand_prob_up(spot_price), (1.0 + self.error_function(-bid_price_up_normalized / ca.sqrt(2.0))) / 2.0)

    def activation_prob_down(self, spot_price, bid_price_down):
        bid_price_down = bid_price_down.reshape((1,-1))

        mu_down = conditional_expectation(spot_price, self.price_means, self.price_covs)[1]
        sigma_down = self.sigma_dn
        
        # bid_price_down_normalized = (bid_price_down - ca.vertcat(*mu_down))/sigma_down
        bid_price_down_normalized = ((bid_price_down - ca.vertcat(*mu_down).reshape((1,-1)))/sigma_down).reshape((1,-1))

        # return norm.cdf(-Bc_dn_norm)
        # return self.demand_prob_dn() * (1.0 + ca.erf(-Bc_dn_norm / ca.sqrt(2.0))) / 2.0
        return np.multiply(self.demand_prob_down(spot_price), (1.0 + self.error_function(-bid_price_down_normalized / ca.sqrt(2.0))) / 2.0)


    def error_function(self, x):
        '''
        Function implemented to explore alternatives to ca.erf
        Some quick testing shows that ca.erf is sufficiently fast, most likely due to the simpler derivative
        '''

        # tanh error funciton approx:
        # return ca.tanh(x)
        
        # Casadi error function 
        return ca.erf(x)



    def demand_prob_up(self, spot_price = None):
        # Probability of the grid needing up regulation.
        # TODO implement actual model from Erlend when that's ready

        if spot_price is None: return np.mean(self.mfrr_demands_up)    # Use mean spot_price if none other is specified

        # return self.up_activation_occurance_rate      # Use predicted demand rate from data
        # return np.mean(self.mfrr_demands_up)            # Use actual demand rate
        
        expected_activation_up = ca.horzcat(*conditional_expectation(spot_price, self.activation_means, self.activation_covs)[0]).reshape((1,-1))
        
        # Custom function which bounds activation chance between 0 and 1. (1 + abs(x) - abs(x-1))/2 with abs(x) = sqrt(x^2)
        expected_activation_up = 0.5 + 0.5*(ca.sqrt(ca.power(expected_activation_up, 2)) - ca.sqrt(ca.power(expected_activation_up - 1, 2)))
        
        return expected_activation_up
    
    def demand_prob_down(self, spot_price = None):  
        # Probability of the grid needing down regulation.
        # TODO implement actual model from Erlend when that's ready

        if spot_price is None: return np.mean(self.mfrr_demands_down)     # Use mean spot_price if none other is specified

        # return self.down_activation_occurance_rate    # Use predicted demand rate from data
        # return np.mean(self.mfrr_demands_down)            # Use actual demand rate
        expected_activation_down = ca.horzcat(*conditional_expectation(spot_price, self.activation_means, self.activation_covs)[1]).reshape((1,-1))

        # Custom function which bounds activation chance between 0 and 1. (1 + abs(x) - abs(x-1))/2 with abs(x) = sqrt(x^2)
        expected_activation_down = 0.5 + 0.5*(ca.sqrt(ca.power(expected_activation_down, 2)) - ca.sqrt(ca.power(expected_activation_down - 1, 2)))
        return expected_activation_down


    def import_spot_mfrr_data(self):
        '''
        Imports mfrr data and processes it into several dataframes.

        `full sets`     : All imported data available from the importing function
        `working sets`  : Select data from the full sets
        '''


        start_date  = pd.to_datetime(self.date)
        end_date    = start_date + pd.DateOffset(self.T)
        
        spot_price_file = self.config.spotprice_data_path
        mfrr_balancing_price_datapath = self.config.mfrr_clearing_prices_path
        mfrr_activation_datapath = self.config.mfrr_activation_data_path
        mfrr_CBMP_datapath = self.config.mfrr_CBMP_data_path
        # Load data
        # CBMP_prices = load_mfrr_CBMP_prices(mfrr_CBMP_datapath, self.bidding_zone)
        spot_prices = load_spot_prices(spot_price_file, self.bidding_zone)
        # mfrr_prices = load_mfrr_balancing_prices(mfrr_balancing_price_datapath, self.bidding_zone)
        mfrr_prices = load_nordpool_balancing_prices(self.config.data_path, self.bidding_zone)
        # activations = load_mfrr_activation_data(mfrr_activation_datapath, self.bidding_zone)
        activations = load_nordpool_activation_data(self.config.data_path, self.bidding_zone)


        # Merge datasets
        price_data = pd.merge(spot_prices, mfrr_prices, on='Start Time', how='inner')
        self.prices_full_set = price_data
        activation_data = pd.merge(spot_prices, activations, on='Start Time', how='inner')
        self.activations_full_set = activation_data

        up_prices_merged_data = pd.merge(price_data[['Start Time', 'Spot Price', 'Clearing Price Up']], activation_data[['Start Time', 'Offered Up', 'Activated Up']], on='Start Time', how='inner')
        down_prices_merged_data = pd.merge(price_data[['Start Time', 'Spot Price', 'Clearing Price Down']], activation_data[['Start Time', 'Offered Down', 'Activated Down']], on='Start Time', how='inner')
        self.up_prices_full_set = up_prices_merged_data
        self.down_prices_full_set = down_prices_merged_data

        if self.optimistic:
            self.prices_working_set         = price_data[(price_data['Start Time'] >= start_date) & (price_data['Start Time'] < end_date) ]
            self.activations_working_set    = activation_data[(activation_data['Start Time'] >= start_date) & (activation_data['Start Time'] < end_date) ]
            up_prices_working_set = up_prices_merged_data[
                (up_prices_merged_data['Start Time'] >= start_date)   & 
                (up_prices_merged_data['Start Time'] < end_date)      &
                (up_prices_merged_data['Activated Up'] > 0)
                ]
            down_prices_working_set = down_prices_merged_data[
                (down_prices_merged_data['Start Time'] >= start_date) & 
                (down_prices_merged_data['Start Time'] < end_date)    &
                (down_prices_merged_data['Activated Down'] > 0)
                ]
    
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


    def import_CM_data(self):
        '''
        Imports mfrr data and processes it into several dataframes.

        `full sets`     : All imported data available from the importing function
        `working sets`  : Select data from the full sets
        '''


        start_date  = pd.to_datetime(self.date)
        end_date    = start_date + pd.DateOffset(self.T)

        spot_price_file = self.config.spotprice_data_path
        spot_prices = load_spot_prices(spot_price_file, self.bidding_zone)
        CM_prices = load_CM_prices(self.config.data_path, self.bidding_zone)


        # Merge datasets
        CM_data = pd.merge(spot_prices, CM_prices, on='Start Time', how='inner')
        self.CM_data_full_set = CM_data

        if self.optimistic:
            self.CM_data_working_set = CM_data[(CM_data['Start Time'] >= start_date) & (CM_data['Start Time'] < end_date) ]
            CM_up_prices_working_set = CM_data[['Start Time', 'Clearing Price Up', 'Volume Up']][
                (CM_data['Start Time'] >= start_date)   & 
                (CM_data['Start Time'] < end_date)      &
                (CM_data['Volume Up'] > 0)
                ]
            CM_down_prices_working_set = CM_data[['Start Time', 'Clearing Price Down', 'Volume Down']][
                (CM_data['Start Time'] >= start_date) & 
                (CM_data['Start Time'] < end_date)    &
                (CM_data['Volume Down'] > 0)
                ]
    
        else:
            self.CM_data_working_set    = self.CM_data_full_set
            CM_up_prices_working_set    = CM_data[['Start Time', 'Clearing Price Up', 'Volume Up']]
            CM_down_prices_working_set  = CM_data[['Start Time', 'Clearing Price Down', 'Volume Down']]
        
        # Remove clear outliers
        
        mean_up     = np.mean(CM_up_prices_working_set['Clearing Price Up'])
        std_up      = np.sqrt(np.var(CM_up_prices_working_set['Clearing Price Up']))
        mean_down   = np.mean(CM_down_prices_working_set['Clearing Price Down'])
        std_down    = np.sqrt(np.var(CM_down_prices_working_set['Clearing Price Down']))
        
        # remove all entries outside 4 standard deviations. These should only be extreme cases
        epsilon = 1e-6
        self.CM_up_prices_working_set      = CM_up_prices_working_set[np.abs(CM_up_prices_working_set['Clearing Price Up'] - mean_up) / (std_up + epsilon) < self.outlier_max_dist]
        self.CM_down_prices_working_set    = CM_down_prices_working_set[np.abs(CM_down_prices_working_set['Clearing Price Down'] - mean_down) / (std_down + epsilon) < self.outlier_max_dist]        
        
        return 0 




    def analyze_price_covariances(self):
        """
        Analyze price covariances or load precomputed results from a JSON file if it exists.
        """
        
        print("Performing price analysis.")
        # Paths to CSV files
    
        up_price_data = self.up_prices_working_set.dropna()
        down_price_data = self.down_prices_working_set.dropna()

        if up_price_data.empty:
            print("The working dataset for up prices is empty. Using full dataset")
            up_price_data = self.up_prices_full_set
            assert False, 'Up price data is empty, date is likely not supported in the dataset. Or there are no activations of this type during the simulation time'
        elif down_price_data.empty:
            print("The working dataset for down prices is empty. Using full dataset")
            down_price_data = self.down_prices_full_set
            assert False, 'Down price data is empty, date is likely not supported in the dataset Or there are no activations of this type during the simulation time'

        # covariance_matrix = calculate_covariance_matrix(price_data, ['Spot Price', 'Clearing Price Up', 'Clearing Price Down'])
        
        spot_up_cov     = np.cov(up_price_data[['Spot Price', 'Clearing Price Up']].T)
        spot_down_cov   = np.cov(down_price_data[['Spot Price', 'Clearing Price Down']].T)
        
        mean_price_up       = up_price_data['Clearing Price Up'].mean() #-10
        mean_spot_price_up  = up_price_data['Spot Price'].mean()

        mean_price_down = down_price_data['Clearing Price Down'].mean() #+10
        mean_spot_price_down = down_price_data['Spot Price'].mean()

        # Save results
        self.price_means = np.array([mean_spot_price_up, mean_spot_price_down, mean_price_up, mean_price_down])
        self.price_covs = np.array([spot_up_cov, spot_down_cov])

        return 0
    

    def analyze_activation_covariances(self):
        """
        Analyze activation covariances
        """
        
        print("Performing activation analysis.")
    
        activation_data   = self.activations_working_set

        assert not activation_data.empty, 'Activation data is empty, date is likely not supported in the dataset. Or there are no activations of this type during the simulation time'
    
        # covariance_matrix = calculate_covariance_matrix(price_data, ['Spot Price', 'Clearing Price Up', 'Clearing Price Down'])
        
        activated_binary_up     = np.where(activation_data['Activated Up']      > 0, 1, 0)
        activated_binary_down   = np.where(activation_data['Activated Down']    > 0, 1, 0)
        spot_prices             = np.array(activation_data['Spot Price'])

        activation_cov_up     = np.cov(np.array([spot_prices, activated_binary_up]))
        activation_cov_down   = np.cov(np.array([spot_prices, activated_binary_down]))
        
        mean_activations_up   = np.mean(activated_binary_up)
        mean_activations_down = np.mean(activated_binary_down)
        mean_spot_price       = np.mean(spot_prices)

        # Save results
        self.activation_means = np.array([mean_spot_price, mean_spot_price, mean_activations_up, mean_activations_down])
        self.activation_covs = np.array([activation_cov_up, activation_cov_down])

        return 0



    def optimal_bidding_price_prediction(self, spot_prices):    

        start_time = time.time()
        print("Starting price prediction")

        mu_up, mu_dn = conditional_expectation(spot_prices, self.price_means, self.price_covs)

        # Create decision variables for the optimization problem
        N = len(spot_prices)
        X = ca.MX.sym('X', 2, N)

        J = 0
        for k in range(N):  J -= X[0,k] * self.activation_prob_up(spot_prices[k], X[0,k])
        for k in range(N):  J -= X[1,k] * self.activation_prob_down(spot_prices[k], X[1,k])
        
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
        self.opt_prices_down = x_dn

        # print(f"Predicted optimal bidding prices: \n Up:   {x_up[range(0,N,24)]} \n Down: {x_dn[range(0,N,24)]}")
        print(f"Predicted Avg bidding prices: \n Up:   {np.average(x_up):.2f} \n Down: {np.average(x_dn):.2f}")

        # print(f"Predicted optimal bidding price is on average: \n Up:   {np.average(delta_up)} compared to predicted clearing price \n Down: {np.average(delta_dn)} compared to clearing price")

        end_time = time.time()
        print(f"Completed prediction in {end_time-start_time:.2f} seconds")
        return 0



    def generate_activation_demands(self, N, seed, spot_price):
        '''
        Generates a list of ints where 0 means no demand for activation, -1 means down activation and 1 means up
        '''

        D_up = self.demand_prob_up(spot_price)
        D_dn = self.demand_prob_down(spot_price)
        no_D = 1 - D_up - D_dn

        activation_demands = generate_weighted_samples([-1, 0, 1], [D_dn, no_D, D_up], N, seed)

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
    
    def get_clearing_prices(self, date=None, n_days = None):

        if date==None:
            date = self.date

        if n_days==None:
            n_days = self.T

        start_date = pd.to_datetime(date)
        end_date = start_date + pd.DateOffset(n_days)

        clearing_prices_df = self.prices_full_set.copy()

        # Remove dates before simdate
        clearing_prices_df = clearing_prices_df[
            (clearing_prices_df['Start Time']   >= start_date)    &
            (clearing_prices_df['Start Time']   <  end_date) 
            ]
        
        clearing_prices_df.fillna(clearing_prices_df.mean(), inplace=True)

        clearing_prices_up = np.array(clearing_prices_df['Clearing Price Up']).repeat(QUARTER_HOURS_PER_HOUR)
        clearing_prices_down = np.array(clearing_prices_df['Clearing Price Down']).repeat(QUARTER_HOURS_PER_HOUR)

        return clearing_prices_up, clearing_prices_down
    
    def get_CM_clearing_prices(self, date=None, n_days = None):

        if date==None:
            date = self.date

        if n_days==None:
            n_days = self.T

        start_date = pd.to_datetime(date)
        end_date = start_date + pd.DateOffset(n_days)

        clearing_prices_df = self.CM_data_full_set.copy()

        # Remove dates before simdate
        clearing_prices_df = clearing_prices_df[
            (clearing_prices_df['Start Time']   >= start_date)    &
            (clearing_prices_df['Start Time']   <  end_date) 
            ]
        
        clearing_prices_df.fillna(clearing_prices_df.mean(), inplace=True)

        clearing_prices_up = np.array(clearing_prices_df['Clearing Price Up']).repeat(QUARTER_HOURS_PER_HOUR)
        clearing_prices_down = np.array(clearing_prices_df['Clearing Price Down']).repeat(QUARTER_HOURS_PER_HOUR)

        return clearing_prices_up, clearing_prices_down
    
    def get_activation_demands(self, date = None, n_days = None):
        '''
        Returns numpy arrays of length N with balancing demands during each quarter hour from the start time.
        For every MTU, a 1 indicates that an activation was made and a 0 indicates that no activation was made.
        Start time is always assumed at 00:00 at the given start date.
        '''

        if date is None:
            date = self.date

        if n_days==None:
            n_days = self.T

        start_date = pd.to_datetime(date)
        end_date = start_date + pd.DateOffset(n_days)

        activations_df = self.activations_full_set

        # Remove dates before simdate
        activations_df = activations_df[
            (activations_df['Start Time']   >= start_date)    &
            (activations_df['Start Time']   <  end_date) 
            ].fillna(0)

        demands_up = np.array(activations_df['Activated Up']).repeat(QUARTER_HOURS_PER_HOUR)
        demands_dn = np.array(activations_df['Activated Down']).repeat(QUARTER_HOURS_PER_HOUR)

        return np.where(demands_up > 0, 1, 0), np.where(demands_dn > 0, 1, 0)
    


    def analyze_market_potency(self, T=20):
        '''
        Comb through clearing prices and activations to find the timespan of length `T`
        with the highest potential profit in the mFRR market

        `T`: Time window of market participation
        '''

        activaion_df = self.activations_full_set.copy()
        clearing_prices_df = self.prices_full_set.copy()

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
    


    def calculate_CM_earnings_upper_limit(self, controller, run_id='fixed'):

        runs = controller.optimization_results['runs']
        assert run_id in runs, f"Run {run_id} is not registered in optimization results"
        
        run = runs[run_id]

        if 'bidding result' not in run:
            return 0
        
        CM_clearing_prices_up, CM_clearing_prices_down = self.get_CM_clearing_prices()

        bid_volumes_up = run['timeseries']["P_up"]
        bid_volumes_down = run['timeseries']["P_dn"]

        return self.C_eur2nok * np.sum(np.multiply(CM_clearing_prices_up, bid_volumes_up) + np.multiply(CM_clearing_prices_down, bid_volumes_down))
        




    def estimate_prices(self):

        AM_clearing_prices_up, AM_clearing_prices_down = self.get_clearing_prices()
        CM_clearing_prices_up, CM_clearing_prices_down = self.get_CM_clearing_prices()
        spot_prices = self.spot_prices.reshape((1,-1))[:,0::4]
        

        CM_prices = np.vstack((CM_clearing_prices_up, 
                               CM_clearing_prices_down))[:,0::4]
        
        AM_prices = np.vstack((AM_clearing_prices_up, 
                               AM_clearing_prices_down))[:,0::4]
        
        CM_price_estimate = Estimator(CM_prices, spot_prices,                           n_lags=10)
        AM_price_estimate = Estimator(AM_prices, np.vstack((CM_prices, spot_prices)),   n_lags=10)

        print(f"CM Estimator MSE: {CM_price_estimate.mse}\t lag: {CM_price_estimate.n_lags}")
        print(f"AM Estimator MSE: {AM_price_estimate.mse}\t lag: {AM_price_estimate.n_lags}")

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)


        t = np.arange(spot_prices.shape[1])
        ax1.plot(t, CM_price_estimate[0,:],     label='Est CM clearing up')
        ax2.plot(t, CM_price_estimate[1,:],     label='Est CM clearing down')
        ax3.plot(t, AM_price_estimate[0,:],     label='Est AM clearing up')
        ax4.plot(t, AM_price_estimate[1,:],     label='Est AM clearing down')

        ax1.plot(t, CM_clearing_prices_up[0::4],      linestyle = ':', color='gray', label='CM clearing up')
        ax2.plot(t, CM_clearing_prices_down[0::4],    linestyle = ':', color='gray', label='CM clearing down')
        ax3.plot(t, AM_clearing_prices_up[0::4],      linestyle = ':', color='gray', label='AM clearing up')
        ax4.plot(t, AM_clearing_prices_down[0::4],    linestyle = ':', color='gray', label='AM clearing down')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        plt.show()
        

    # def calculate_AM_upper_bound(self, start_date = '01-01-2024', end_date = '31-12-2024'):


    #     # Start at 01.01.2024
    #     # increment by one day each iteration
    #     dates = []  # placeholder
    #     daily_earnings = []

    #     # Get spot market data for current day
    #     # Get market clearing prices up and down for current day
    #     # Get market activation volumes for current day
    #     # Based on activations and clearing prices, choose the 16 most profitable hours to have light on, and the 8 most profitable hours to have it off. 
    #     # Store the

    #     for date in dates:

    #         AM_clearing_prices_up, AM_clearing_prices_down = self.get_clearing_prices(date = date, n_days = 1)
    #         AM_activation_demands_up, AM_activation_demands_down = self.get_activation_demands(date = date, n_days = 1)

    #         AM_earnings_up = np.multiply(AM_clearing_prices_up, AM_activation_demands_up)
    #         AM_earnings_down = np.multiply(AM_clearing_prices_down, AM_activation_demands_down)

    #         daily_earnings.append(np.sum(np.sort(np.hstack((AM_earnings_up, AM_earnings_down)))[:24]))
            
            
    #     yearly_earnings = np.sum(daily_earnings)

    #     print(yearly_earnings)

    def calculate_AM_upper_bound(self, start_date='01-01-2024', end_date='31-12-2024'):
        """
        Calculates the upper bound for earnings from the AM market by optimizing activation periods.
        
        Strategy:
        - For each day, retrieve activation demands and clearing prices.
        - Compute earnings for each hour.
        - Choose the 16 most profitable hours of down_activation and 8 least profitable to be "off".
        - Sum daily earnings over the year.
        """
        
        # Dates for which to accumulate earnings over.
        dates = pd.date_range(start=pd.to_datetime(start_date, format='%d-%m-%Y'),
                            end=pd.to_datetime(end_date, format='%d-%m-%Y'))

        yearly_earnings_eur = 0
        
        electricity_costs_fixed = 0
        electricity_costs_min   = 0
        electricity_costs_AM  = 0


        for date in dates:
            # Get clearing prices and activation demands for the current day
            AM_clearing_prices_up, AM_clearing_prices_down = self.get_clearing_prices(date=date, n_days=1)          # Clearing prices in eur/MWh
            AM_activation_demands_up, AM_activation_demands_down = self.get_activation_demands(date=date, n_days=1) # Activation market demands, 0 or 1
            spot_prices = self.get_spotprice(date=date, n_days=1)

            # Calculate earnings per quarter hour
            AM_cost_up      = - np.multiply(AM_clearing_prices_up, AM_activation_demands_up)
            AM_cost_down    = - np.multiply(AM_clearing_prices_down, AM_activation_demands_down)# + spot_prices*1000/self.C_eur2nok

            cost_up_df = pd.DataFrame({'MTU': list(range(len(spot_prices))), 
                                      'Cost': AM_cost_up})
            
            cost_down_df = pd.DataFrame({'MTU': list(range(len(spot_prices))), 
                                      'Cost': AM_cost_down})

            cost_up_df.sort_values('Cost', inplace=True)
            cost_down_df.sort_values('Cost', inplace=True)

            # light_schedule = np.zeros_like(spot_prices)
            # qh_on = 0
            # qh_off = 0

            # for i in range(len(spot_prices)):

            #     if qh_off < 32 or cost_up_df.iloc[i]['Cost'] < cost_down_df.iloc[i]['Cost']:
            #         mtu = cost_up_df.loc[i, 'MTU']
            #         light_schedule[mtu] = 0
            #         cost_up_df = cost_up_df.drop(index=i)
            #         cost_down_df = cost_down_df[cost_down_df['MTU'] != mtu]
            #         qh_off += 1
            #         continue

            #     elif qh_on < 64:
            #         mtu = cost_down_df.loc[i, 'MTU']
            #         light_schedule[mtu] = 1
            #         cost_down_df = cost_down_df.drop(index=i)
            #         cost_up_df = cost_up_df[cost_up_df['MTU'] != mtu]
            #         qh_on += 1
            #         continue

            # Light schedule (default 1 = light ON)
            light_schedule = np.ones(len(spot_prices))

            # Counters for 32 up (off) and 64 down (on) activations
            qh_on, qh_off = 0, 0

            # Iterators for sorted data
            i, j = 0, 0

            while qh_off < 32 or qh_on < 64:
                if i >= len(cost_up_df) or j >= len(cost_down_df):
                    break  # Safety check in case of edge cases

                mtu_up, cost_up = int(cost_up_df.iloc[i]['MTU']), cost_up_df.iloc[i]['Cost']
                mtu_down, cost_down = int(cost_down_df.iloc[j]['MTU']), cost_down_df.iloc[j]['Cost']

                if (qh_off < 32 and cost_up <= cost_down) or qh_on >= 64:  
                    # Prefer up (off) regulation if cheaper or if we already have 64 down activations
                    light_schedule[mtu_up] = 0  
                    cost_down_df = cost_down_df[cost_down_df['MTU'] != mtu_up]  # Remove conflicting MTU
                    qh_off += 1
                    i += 1  
                else:  
                    # Prefer down (on) regulation if cheaper or if we already have 32 up activations
                    light_schedule[mtu_down] = 1  
                    cost_up_df = cost_up_df[cost_up_df['MTU'] != mtu_down]  # Remove conflicting MTU
                    qh_on += 1
                    j += 1  

            cost_up     = np.sum(AM_cost_up[np.where(light_schedule==0)])
            cost_down   = np.sum(AM_cost_down[np.where(light_schedule==1)])
            energy_cost = np.sum(spot_prices[np.where(light_schedule==1)])

            max_earnings = - cost_up - cost_down
            yearly_earnings_eur += max_earnings
            electricity_costs_AM += energy_cost
            electricity_costs_fixed += np.sum(spot_prices[:16*4])

            # cheapest_32_up      = []
            # cheapest_64_down    = []

            # # Sort in ascending order and choose the n most profiable quarter hours
            # top_8_up_earnings = np.sort(AM_cost_up)[-8*4:] 
            # top_16_down_earnings = np.sort(AM_cost_down)[-16*4:]

            # top_16_down_indices = np.argsort(AM_cost_down)[-16*4:]
            # top_16_down_spot_prices = spot_prices[top_16_down_indices]

            # lowest_16_spot_prices = np.sort(spot_prices)[:16*4]

            # max_earnings = np.sum(top_8_up_earnings) + np.sum(top_16_down_earnings)
            # yearly_earnings_eur += max_earnings
            # electricity_costs_min += np.sum(lowest_16_spot_prices)
            # electricity_costs_AM += np.sum(top_16_down_spot_prices)
            # electricity_costs_fixed += np.sum(spot_prices[:16*4])

        yearly_earnings_nok = yearly_earnings_eur

        # convert from nok/kwh to eur/mwh
        # electricity_costs_min  = electricity_costs_min  / self.C_eur2nok * 1000 
        electricity_costs_AM = electricity_costs_AM / self.C_eur2nok * 1000
        electricity_costs_fixed = electricity_costs_fixed / self.C_eur2nok * 1000


        print(f'\nOptimistic earnigns calculations for mFRR AM participation in {self.bidding_zone}')
        print(f"Total electricity costs without AM participation: \t{electricity_costs_fixed:.2f}")
        print(f"Total estimated upper bound of AM earnings: \t\t{yearly_earnings_nok:.2f}")
        print(f"Total electricity costs with AM participation: \t\t{electricity_costs_AM:.2f}")
        print(f"Net cost reduction from AM participation: \t\t{electricity_costs_fixed - electricity_costs_AM + yearly_earnings_nok:.2f}")
        print(f"Net cost reduction in percentage: \t\t\t{(electricity_costs_fixed - electricity_costs_AM + yearly_earnings_nok)/electricity_costs_fixed * 100 :.2f}")
        return yearly_earnings_nok