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

    AM_price_stats: dict
    CM_price_stats: dict

    expected_AM_prices_up:         np.array        # Array of most likely clearing prices for up-regulation at each time step      (length: N)
    expected_AM_prices_down:         np.array        # Array of most likely clearing prices for down-regulation at each time step    (length: N)
    expected_CM_prices_up:         np.array        # Array of most likely clearing prices for up-regulation at each time step      (length: N)
    expected_CM_prices_down:         np.array        # Array of most likely clearing prices for down-regulation at each time step    (length: N)
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
        self.seed               = self.market_settings['SEED']
        self.outlier_max_dist   = self.market_settings['OUTLIER_DIST_LIMIT']
        self.N = self.T * QUARTER_HOURS_PER_DAY

        self.import_market_data()
        # self.import_CM_data()

        self.spot_prices = self.get_spotprice() 

        self.statistical_analysis_AM()
        conditional_variance_AM_up     = conditional_covariance(self.AM_price_stats[self.bidding_zone]['Up']['cov'])
        conditional_variance_AM_down   = conditional_covariance(self.AM_price_stats[self.bidding_zone]['Down']['cov'])
        self.expected_AM_prices_up     = conditional_expectation(self.spot_prices, self.AM_price_stats[self.bidding_zone]['Up']['means'],    self.AM_price_stats[self.bidding_zone]['Up']['cov'])
        self.expected_AM_prices_down   = conditional_expectation(self.spot_prices, self.AM_price_stats[self.bidding_zone]['Down']['means'],  self.AM_price_stats[self.bidding_zone]['Down']['cov'])
        
        self.statistical_analysis_CM()
        conditional_variance_CM_up     = conditional_covariance(self.CM_price_stats[self.bidding_zone]['Up']['cov'])
        conditional_variance_CM_down   = conditional_covariance(self.CM_price_stats[self.bidding_zone]['Down']['cov'])
        self.expected_CM_prices_up     = conditional_expectation(self.spot_prices, self.CM_price_stats[self.bidding_zone]['Up']['means'],    self.CM_price_stats[self.bidding_zone]['Up']['cov'])
        self.expected_CM_prices_down   = conditional_expectation(self.spot_prices, self.CM_price_stats[self.bidding_zone]['Down']['means'],  self.CM_price_stats[self.bidding_zone]['Down']['cov'])
        

        epsilon = 1e-6  # For numerical stability. Avoids 0-variance
        self.sigma_AM_up   = max(np.sqrt(conditional_variance_AM_up),   epsilon)
        self.sigma_AM_down = max(np.sqrt(conditional_variance_AM_down), epsilon)
        self.sigma_CM_up   = max(np.sqrt(conditional_variance_CM_up),   epsilon)
        self.sigma_CM_down = max(np.sqrt(conditional_variance_CM_down), epsilon)
        
        # Initialize optimal prices on expected value.
        self.opt_prices_up = self.expected_AM_prices_up
        self.opt_prices_down = self.expected_AM_prices_down

        self.specs = {
            'bidding zone'                  : self.bidding_zone,
            'simdate'                       : self.date,
            'optimistic'                    : self.optimistic,
            'Avg activation price up'       : self.AM_price_stats[self.bidding_zone]['Up']['means'][1],
            'Avg activation price down'     : self.AM_price_stats[self.bidding_zone]['Down']['means'][1],          
            'Cond covariance spot - up'     : conditional_variance_AM_up,
            'Cond covariance spot - Down'   : conditional_variance_AM_down,
            'Outlier max std-dev distance'  : self.outlier_max_dist
            }
            
        self.mfrr_activation_data_analysis()
        self.mfrr_demands_up, self.mfrr_demands_down = self.get_activation_demands(self.date)
        self.analyze_market_potency(T = self.T)
    
    

    def activation_prob_up(self, spot_price, bid_price_up):
        bid_price_up = bid_price_up.reshape((1,-1))
        
        mu_up = conditional_expectation(spot_price, self.AM_price_stats[self.bidding_zone]['Up']['means'], self.AM_price_stats[self.bidding_zone]['Up']['cov'])
        sigma_up = self.sigma_AM_up

        bid_price_up_normalized = ((bid_price_up - ca.vertcat(*mu_up).reshape((1,-1)))/sigma_up).reshape((1,-1))

        # return norm.cdf(-Bc_up_norm)
        return np.multiply(self.demand_prob_up(spot_price), (1.0 + ca.erf(-bid_price_up_normalized / ca.sqrt(2.0))) / 2.0)

    def activation_prob_down(self, spot_price, bid_price_down):
        bid_price_down = bid_price_down.reshape((1,-1))

        mu_down = conditional_expectation(spot_price, self.AM_price_stats[self.bidding_zone]['Down']['means'], self.AM_price_stats[self.bidding_zone]['Down']['cov'])
        sigma_down = self.sigma_AM_down
        
        # bid_price_down_normalized = (bid_price_down - ca.vertcat(*mu_down))/sigma_down
        bid_price_down_normalized = ((bid_price_down - ca.vertcat(*mu_down).reshape((1,-1)))/sigma_down).reshape((1,-1))

        # return norm.cdf(-Bc_dn_norm)
        # return self.demand_prob_dn() * (1.0 + ca.erf(-Bc_dn_norm / ca.sqrt(2.0))) / 2.0
        return np.multiply(self.demand_prob_down(spot_price), (1.0 + ca.erf(-bid_price_down_normalized / ca.sqrt(2.0))) / 2.0)


    def demand_prob_up(self, spot_price = None):
        # Probability of the grid needing up regulation.

        if spot_price is None: return np.mean(self.mfrr_demands_up)    # Use mean spot_price if none other is specified

        expected_activation_up = ca.horzcat(*conditional_expectation(spot_price, 
                                                                     self.AM_activation_stats[self.bidding_zone]['Up']['means'], 
                                                                     self.AM_activation_stats[self.bidding_zone]['Up']['cov']
                                                                     )).reshape((1,-1))
                
        return casadi_saturate(expected_activation_up, 0, 1)
    
    def demand_prob_down(self, spot_price = None):  
        # Probability of the grid needing down regulation.

        if spot_price is None: return np.mean(self.mfrr_demands_down)     # Use mean spot_price if none other is specified

        expected_activation_down = ca.horzcat(*conditional_expectation(spot_price, 
                                                                       self.AM_activation_stats[self.bidding_zone]['Down']['means'], 
                                                                       self.AM_activation_stats[self.bidding_zone]['Down']['cov']
                                                                       )).reshape((1,-1))

        return casadi_saturate(expected_activation_down, 0, 1)


    def import_market_data(self):
        '''
        Imports mfrr data and processes it into several dataframes.

        `full sets`     : All imported data available from the importing function
        `working sets`  : Select data from the full sets
        '''

        start_date  = pd.to_datetime(self.date, format='%Y-%m-%d')
        end_date    = start_date + pd.DateOffset(self.T)
        
        # spot_prices_path = self.config.spotprices_data_path
        spot_data       = load_spot_data(self.config.spotprices_data_path)
        mfrr_AM_data    = load_mfrr_AM_data(self.config.mfrr_AM_data_path)
        mfrr_CM_data    = load_mfrr_CM_data(self.config.mfrr_CM_data_path)
        AM_data_full_set    = pd.merge(spot_data, mfrr_AM_data, on='Start Time', how='inner')
        CM_data_full_set    = pd.merge(spot_data, mfrr_CM_data, on='Start Time', how='inner')

        
        mfrr_data_full_set  = pd.merge(AM_data_full_set.copy().rename(columns = {col: f'AM {col}' for col in AM_data_full_set.columns if 'Start Time' not in col}), 
                                       mfrr_CM_data.copy().rename(columns = {col: f'CM {col}' for col in mfrr_CM_data.columns if 'Start Time' not in col}), 
                                       on='Start Time', how='inner')

        if self.optimistic:

            AM_data_working_set = AM_data_full_set[(AM_data_full_set['Start Time'] >= start_date) & (AM_data_full_set['Start Time'] < end_date)]
            CM_data_working_set = CM_data_full_set[(CM_data_full_set['Start Time'] >= start_date) & (CM_data_full_set['Start Time'] < end_date)]

        else:

            AM_data_working_set = AM_data_full_set
            CM_data_working_set = CM_data_full_set

        
        self.spot_data_full_set             = spot_data
        self.AM_data_full_set               = AM_data_full_set
        self.AM_data_working_set            = AM_data_working_set
        self.CM_data_full_set               = CM_data_full_set
        self.CM_data_working_set            = CM_data_working_set

        self.mfrr_data_full_set = mfrr_data_full_set

        return 0 


    def statistical_analysis_AM(self):
        """
        Analyze mFRR Activation Market prices
        Store statistical analysis in a dict:
        - zone 
            - direction 
                - 'means'
                - 'cov'
        
        optimistic bool indicates wether to use only spot prices and market clearing prices at times of activations
        basically cherrypicking the analysis for the planned usecase, which is to estimate good clearing prices.
        """
        
        print("Performing price analysis.")
        # Paths to CSV files
    
        AM_data   = self.AM_data_working_set.copy()

        price_statistics = {}
        activation_statistics = {}
        zones = [self.bidding_zone]

        for zone in zones:
            price_statistics[zone] = {}
            activation_statistics[zone] = {}

            for direction in ['Up', 'Down']:
                price_statistics[zone][direction] = {}
                activation_statistics[zone][direction] = {}

                if self.optimistic:
                    price_data = AM_data[[f'{zone} Spot Price', 
                                    f'{zone} {direction} Price']][AM_data[f'{zone} Activated {direction} Volume'] > 0]
                else:
                    price_data      = AM_data[[f'{zone} Spot Price', f'{zone} {direction} Price']]
                
                activation_data = AM_data[[f'{zone} Spot Price', f'{zone} Activated {direction} Volume']]

                price_statistics[zone][direction]['means']      = np.mean(np.array(price_data), axis=0)
                price_statistics[zone][direction]['cov']        = np.cov(price_data.T)
                activation_statistics[zone][direction]['means'] = np.mean(np.array(activation_data), axis=0)
                activation_statistics[zone][direction]['cov']   = np.cov(activation_data.T)

        self.AM_price_stats = price_statistics
        self.AM_activation_stats = price_statistics

        return 0
    
    def statistical_analysis_CM(self):
        """
        Analyze mFRR Capacity Market prices
        Store statistical analysis in a dict:
        - zone 
            - direction 
                - 'means'
                - 'cov'
        
        optimistic bool indicates wether to use only spot prices and market clearing prices at times of activations
        basically cherrypicking the analysis for the planned usecase, which is to estimate good clearing prices.
        """
        
        print("Performing price analysis.")
        # Paths to CSV files
    
        CM_data   = self.CM_data_working_set.copy()

        price_statistics = {}
        reservation_statistics = {}
        zones = [self.bidding_zone]

        for zone in zones:
            price_statistics[zone] = {}
            reservation_statistics[zone] = {}

            for direction in ['Up', 'Down']:
                price_statistics[zone][direction] = {}
                reservation_statistics[zone][direction] = {}

                if self.optimistic:
                    price_data = CM_data[[f'{zone} Spot Price', 
                                    f'{zone} {direction} Price']][CM_data[f'{zone} {direction} Volume procured'] > 0]
                else:
                    price_data      = CM_data[[f'{zone} Spot Price', f'{zone} {direction} Price']]
                
                activation_data = CM_data[[f'{zone} Spot Price', f'{zone} {direction} Volume procured']]
                
                price_statistics[zone][direction]['means']      = np.mean(np.array(price_data), axis=0)
                price_statistics[zone][direction]['cov']        = np.cov(np.array(price_data).reshape((2,-1)))
                reservation_statistics[zone][direction]['means'] = np.mean(np.array(activation_data), axis=0)
                reservation_statistics[zone][direction]['cov']   = np.cov(np.array(activation_data).reshape((2,-1)))

        self.CM_price_stats = price_statistics
        self.CM_activation_stats = price_statistics

        return 0
    

    def optimal_bidding_price_prediction(self, spot_prices):    

        start_time = time.time()
        print("Starting price prediction")

        prices_stats_up     = self.AM_price_stats[self.bidding_zone]['Up']
        prices_stats_down   = self.AM_price_stats[self.bidding_zone]['Down']

        mu_up   = conditional_expectation(spot_prices, prices_stats_up['means'],    prices_stats_up['cov'])
        mu_down = conditional_expectation(spot_prices, prices_stats_down['means'],  prices_stats_down['cov'])

        # Create decision variables for the optimization problem
        N = len(spot_prices)
        X = ca.MX.sym('X', 2, N)

        J = 0
        for k in range(N):  J -= X[0,k] * self.activation_prob_up(spot_prices[k], X[0,k])
        for k in range(N):  J -= X[1,k] * self.activation_prob_down(spot_prices[k], X[1,k])
        
        lbx = 0
        ubx = 1000

        # Flatten decision variables and bounds
        Z   = ca.vertcat(ca.reshape(X,   -1, 1))
        lbz = ca.vertcat(ca.reshape(lbx, -1, 1))
        ubz = ca.vertcat(ca.reshape(ubx, -1, 1))

        # Nonlinear problem definition
        nlp = {'x': Z, 'f': J}

        # Create the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        u0 = np.vstack((mu_up, mu_down)).ravel(order='F')
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
        
        activation_df = self.AM_data_working_set

        # Activation rate of each offered MW of capacity 
        zone = self.bidding_zone
        self.up_activation_capacity_ratio    = np.sum(activation_df[zone + ' Activated Up Volume']) / np.sum(activation_df[zone + ' Accepted Up Volume'])
        self.down_activation_capacity_ratio  = np.sum(activation_df[zone + ' Activated Down Volume']) / np.sum(activation_df[zone + ' Accepted Down Volume'])
        
        # Arrays denoting activation occurances
        up_activation_occurances   = np.where(np.array(activation_df[zone + ' Activated Up Volume'])>0, 1, 0)
        down_activation_occurances = np.where(np.array(activation_df[zone + ' Activated Down Volume'])>0, 1, 0)
        both_activation_occurances = np.where(np.logical_and(np.array(activation_df[zone + ' Activated Up Volume'])>0,np.array(activation_df[zone + ' Activated Down Volume'])>0), 1, 0)

        # % of QH where activations occur
        self.up_activation_occurance_rate     = np.mean(up_activation_occurances)
        self.down_activation_occurance_rate   = np.mean(down_activation_occurances)
        self.both_activation_occurance_rate   = np.mean(both_activation_occurances)

        return 0

    def get_spotprice(self, date=None, n_days=None, zone=None) -> np.array:

        if date     == None: date   = self.date
        if n_days   == None: n_days = self.T
        if zone     == None: zone   = self.bidding_zone

        start_date = pd.to_datetime(date, format='%Y-%m-%d')
        end_date = start_date + pd.DateOffset(n_days)
        
        df = self.spot_data_full_set.copy()

        # Remove dates before simdate
        df = df[
            (df['Start Time']   >= start_date)    &
            (df['Start Time']   <  end_date) 
            ]
        
        df.fillna(df.mean(), inplace=True)
        
        spot_prices_hours = np.array(df[f'{zone} Spot Price'].values)
        spot_prices = np.repeat(spot_prices_hours, QUARTER_HOURS_PER_HOUR)
        return spot_prices
    
    def get_AM_clearing_prices(self, date=None, n_days = None, zone = None):

        if date     == None: date   = self.date
        if n_days   == None: n_days = self.T
        if zone     == None: zone   = self.bidding_zone

        start_date = pd.to_datetime(date, format='%Y-%m-%d')
        end_date = start_date + pd.DateOffset(n_days)

        clearing_prices_df = self.AM_data_full_set.copy()

        # Remove dates before simdate
        clearing_prices_df = clearing_prices_df[
            (clearing_prices_df['Start Time']   >= start_date)    &
            (clearing_prices_df['Start Time']   <  end_date) 
            ]
        
        clearing_prices_df.fillna(clearing_prices_df.mean(), inplace=True)

        clearing_prices_up = np.array(clearing_prices_df[f'{zone} Up Price']).repeat(QUARTER_HOURS_PER_HOUR)
        clearing_prices_down = np.array(clearing_prices_df[f'{zone} Down Price']).repeat(QUARTER_HOURS_PER_HOUR)

        return clearing_prices_up, clearing_prices_down
    
    def get_CM_clearing_prices(self, date=None, n_days = None, zone = None):

        if date     == None: date   = self.date
        if n_days   == None: n_days = self.T
        if zone     == None: zone   = self.bidding_zone

        start_date = pd.to_datetime(date, format='%Y-%m-%d')
        end_date = start_date + pd.DateOffset(n_days)

        clearing_prices_df = self.CM_data_full_set.copy()

        # Remove dates before simdate
        clearing_prices_df = clearing_prices_df[
            (clearing_prices_df['Start Time']   >= start_date)    &
            (clearing_prices_df['Start Time']   <  end_date) 
            ]
        
        clearing_prices_df.fillna(self.CM_data_full_set.mean(), inplace=True)

        clearing_prices_up = np.array(clearing_prices_df[f'{zone} Up Price']).repeat(QUARTER_HOURS_PER_HOUR)
        clearing_prices_down = np.array(clearing_prices_df[f'{zone} Down Price']).repeat(QUARTER_HOURS_PER_HOUR)

        return clearing_prices_up, clearing_prices_down
    
    def get_activation_demands(self, date = None, n_days = None, zone = None):
        '''
        Returns numpy arrays of length N with balancing demands during each quarter hour from the start time.
        For every MTU, a 1 indicates that an activation was made and a 0 indicates that no activation was made.
        Start time is always assumed at 00:00 at the given start date.
        '''

        if date     == None: date   = self.date
        if n_days   == None: n_days = self.T
        if zone     == None: zone   = self.bidding_zone

        start_date = pd.to_datetime(date, format='%Y-%m-%d')
        end_date = start_date + pd.DateOffset(n_days)

        activations_df = self.AM_data_full_set

        # Remove dates before simdate
        activations_df = activations_df[
            (activations_df['Start Time']   >= start_date)    &
            (activations_df['Start Time']   <  end_date) 
            ].fillna(0)

        demands_up = np.array(activations_df[zone + ' Activated Up Volume']).repeat(QUARTER_HOURS_PER_HOUR)
        demands_dn = np.array(activations_df[zone + ' Activated Down Volume']).repeat(QUARTER_HOURS_PER_HOUR)

        return np.where(demands_up > 0, 1, 0), np.where(demands_dn > 0, 1, 0)
    


    def get_CM_reservations(self, date = None, n_days = None, zone = None):
        '''
        Returns numpy arrays of length N with Capacity market reservations during each quarter hour from the start time.
        For every MTU, a 1 indicates that a reservation was made and a 0 indicates that no reservation was made.
        Start time is always assumed at 00:00 at the given start date.
        '''

        if date     == None: date   = self.date
        if n_days   == None: n_days = self.T
        if zone     == None: zone   = self.bidding_zone

        start_date = pd.to_datetime(date, format='%Y-%m-%d')
        end_date = start_date + pd.DateOffset(n_days)

        reservations_df = self.CM_data_full_set

        # Remove dates before simdate
        reservations_df = reservations_df[
            (reservations_df['Start Time']   >= start_date)    &
            (reservations_df['Start Time']   <  end_date) 
            ].fillna(0)

        reservations_up = np.array(reservations_df[f'{zone} Up Volume procured']).repeat(QUARTER_HOURS_PER_HOUR)
        reservations_dn = np.array(reservations_df[f'{zone} Down Volume procured']).repeat(QUARTER_HOURS_PER_HOUR)

        return np.where(reservations_up > 0, 1, 0), np.where(reservations_dn > 0, 1, 0)
    

    def analyze_market_potency(self, T=20):
        '''
        Comb through clearing prices and activations to find the timespan of length `T`
        with the highest potential profit in the mFRR market

        `T`: Time window of market participation
        '''

        zone = self.bidding_zone
        activations_df = self.AM_data_full_set.copy()
        activations_df = activations_df[['Start Time'] + [col for col in activations_df.columns if zone in col]].rename(
            columns = {zone + ' Activated Up Volume': 'Activated Up',
                       zone + ' Activated Down Volume': 'Activated Down',
                       zone + ' Accepted Up Volume': 'Offered Up',
                       zone + ' Accepted Down Volume': 'Offered Down'
                       }
        )
        clearing_prices_df = self.AM_data_full_set.copy()
        clearing_prices_df = clearing_prices_df[['Start Time'] + [col for col in clearing_prices_df.columns if zone in col]].rename(
            columns = {zone + ' Up Price': 'Clearing Price Up',
                       zone + ' Down Price': 'Clearing Price Down'
                    #    zone + ' Accepted Up Volume': 'Offered Up',
                    #    zone + ' Accepted Down Volume': 'Offered Down'
                       }
        )

        # Merge dataframes to ensure data is present at all applicable time stamps
        merged_df = pd.merge(clearing_prices_df, activations_df, on='Start Time', how='inner')

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

        bid_volumes_up      = run['timeseries']["P_up"]
        bid_volumes_down    = run['timeseries']["P_dn"]

        return np.sum(np.multiply(CM_clearing_prices_up, bid_volumes_up) + np.multiply(CM_clearing_prices_down, bid_volumes_down))
        




    def estimate_prices(self, n_xlags = 10, n_ylags = 10):


        # mfrr_AM_raw_data = (self.config.mfrr_AM_data_path, None)

        price_data = self.mfrr_data_full_set.copy().dropna()



        # spot_prices = 
        


        # AM_clearing_prices_up, AM_clearing_prices_down = self.get_AM_clearing_prices()
        # CM_clearing_prices_up, CM_clearing_prices_down = self.get_CM_clearing_prices()
        # spot_prices = self.spot_prices.reshape((1,-1))[:,0::4]
        
        AM_prices_labels_up     = [col for col in price_data if 'AM' in col and 'Price' in col and 'Up' in col]
        AM_prices_labels_down   = [col for col in price_data if 'AM' in col and 'Price' in col and 'Down' in col]
        CM_prices_labels_up     = [col for col in price_data if 'CM' in col and 'Price' in col and 'Up' in col]
        CM_prices_labels_down   = [col for col in price_data if 'CM' in col and 'Price' in col and 'Down' in col]
        spot_price_labels       = [col for col in price_data if 'Spot Price' in col]
        AM_prices_labels = AM_prices_labels_up + AM_prices_labels_down
        CM_prices_labels = CM_prices_labels_up + CM_prices_labels_down
        
        AM_clearing_prices_up   = np.array(price_data[AM_prices_labels_up]).T
        AM_clearing_prices_down = np.array(price_data[AM_prices_labels_down]).T
        CM_clearing_prices_up   = np.array(price_data[CM_prices_labels_up]).T
        CM_clearing_prices_down = np.array(price_data[CM_prices_labels_down]).T
        spot_prices             = np.array(price_data[spot_price_labels]).T
        

        CM_prices = np.vstack((CM_clearing_prices_up, 
                               CM_clearing_prices_down))
        
        AM_prices = np.vstack((AM_clearing_prices_up, 
                               AM_clearing_prices_down))
        
        '''
        for i in range(CM_prices.shape[0]):
            CM_price_estimate = Estimator(CM_prices[i,:], spot_prices,   
                                        x_labels = CM_prices_labels[i], n_xlags=n_xlags, n_ylags=n_ylags)
            CM_price_estimate.measure_performance(how='array')

        for i in range(AM_prices.shape[0]):
            AM_price_estimate = Estimator(AM_prices[i,:], np.vstack((CM_prices, spot_prices)),   
                                        x_labels = AM_prices_labels[i], n_xlags=n_xlags, n_ylags=n_ylags)
            AM_price_estimate.measure_performance(how='array')
        '''
        CM_price_estimate = Estimator(CM_prices, spot_prices,   
                                        x_labels = CM_prices_labels, n_xlags=n_xlags, n_ylags=n_ylags)
        AM_price_estimate = Estimator(AM_prices, np.vstack((CM_prices, spot_prices)),   
                                        x_labels = AM_prices_labels, n_xlags=n_xlags, n_ylags=n_ylags)
        
        print(f"CM Estimator RMSE: {CM_price_estimate.rmse}\t lag: {CM_price_estimate.n_xlags}")
        print(f"AM Estimator RMSE: {AM_price_estimate.rmse}\t lag: {AM_price_estimate.n_xlags}")

        print("\n Performance measures:")
        CM_price_estimate.measure_performance()
        AM_price_estimate.measure_performance()
        

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)


        t = np.arange(spot_prices.shape[1])
        ax1.plot(t, CM_price_estimate[0,:],     label='Est CM clearing up')
        ax2.plot(t, CM_price_estimate[12,:],     label='Est CM clearing down')
        ax3.plot(t, AM_price_estimate[0,:],     label='Est AM clearing up')
        ax4.plot(t, AM_price_estimate[12,:],     label='Est AM clearing down')

        ax1.plot(t, CM_prices[0,:],    linestyle = ':', color='gray', label='CM clearing up')
        ax2.plot(t, CM_prices[12,:],    linestyle = ':', color='gray', label='CM clearing down')
        ax3.plot(t, AM_prices[0,:],    linestyle = ':', color='gray', label='AM clearing up')
        ax4.plot(t, AM_prices[12,:],    linestyle = ':', color='gray', label='AM clearing down')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        plt.show()
        


    def calculate_AM_upper_bound(self, AM = True, CM = True, zones = None, start_date='2024-02-15', end_date='2025-02-15'):
        """
        Calculates the upper bound for earnings from the AM market by optimizing activation periods.
        
        Strategy:
        - For each day, retrieve activation demands and clearing prices.
        - Compute earnings for each hour.
        - Choose the 8 most profitable hours of down-activation and 8 most profitable hours of up-activation.
        - Sum daily earnings over the year.
        """
        if type(zones) == str: zones = [zones]
        if zones is None:
            zones = [str(col).rstrip('Spot Price') for col in self.spot_data_full_set.columns if 'Start Time' not in col]

        print(f"\nPerforming upper bound earnings analysis for mFRR participation in \n{self.bidding_zone} from {start_date} to {end_date}. \nThis might take a while...")

        
        # Dates for which to accumulate earnings over.
        dates = pd.date_range(start=pd.to_datetime(start_date, format='%Y-%m-%d'),
                            end=pd.to_datetime(end_date, format='%Y-%m-%d'))
        N = len(dates)

        results = {}

        nominal_costs_df = pd.DataFrame({'Date': dates})
        mfrr_costs_df    = pd.DataFrame({'Date': dates})
   
        nominal_costs_df[zones] = np.zeros((N, len(zones)))
        mfrr_costs_df[zones]    = np.zeros((N, len(zones)))

        with tqdm(total=len(zones), desc="Calculating ...") as pbar:
            for zone in zones:

                total_earnings_AM               = 0
                total_earnings_CM               = 0
                total_electricity_costs_fixed   = 0
                total_electricity_costs_AM      = 0

                kwargs = {'date': start_date, 'n_days': N, 'zone': zone}
                try:
                    AM_clearing_prices_up_full, AM_clearing_prices_down_full        = self.get_AM_clearing_prices(**kwargs)
                    AM_activation_demands_up_full, AM_activation_demands_down_full  = self.get_activation_demands(**kwargs)
                    spot_prices_full                                                = self.get_spotprice(**kwargs)

                    CM_clearing_prices_up_full, CM_clearing_prices_down_full        = self.get_CM_clearing_prices(**kwargs)
                    CM_reservations_up_full, CM_reservations_down_full              = self.get_CM_reservations(**kwargs)
                except:
                    pbar.update(1)
                    continue

                for k, date in enumerate(dates):
                    # Precompute slice index
                    daily_slice = slice(k * QUARTER_HOURS_PER_DAY, (k + 1) * QUARTER_HOURS_PER_DAY)
                    
                    data = {
                        "AM_up": AM_clearing_prices_up_full[daily_slice],
                        "AM_down": AM_clearing_prices_down_full[daily_slice],
                        "AM_demand_up": AM_activation_demands_up_full[daily_slice],
                        "AM_demand_down": AM_activation_demands_down_full[daily_slice],
                        "spot": spot_prices_full[daily_slice],
                        "CM_up": CM_clearing_prices_up_full[daily_slice],
                        "CM_down": CM_clearing_prices_down_full[daily_slice],
                        "CM_reserve_up": CM_reservations_up_full[daily_slice],
                        "CM_reserve_down": CM_reservations_down_full[daily_slice],
                    }
                    
                    # Ensure data length consistency. Only really relevant for days with daylight savings.
                    N_max = len(data["CM_up"])
                    for key in data.keys():
                        data[key] = data[key][:N_max]

                    AM_earnings_up      = np.multiply(data["AM_up"],   data["AM_demand_up"])
                    AM_earnings_down    = np.multiply(data["AM_down"], data["AM_demand_down"])
                    CM_earnings_up      = np.multiply(data["CM_up"],   data["CM_reserve_up"])
                    CM_earnings_down    = np.multiply(data["CM_down"], data["CM_reserve_down"])

                    # Calculate earnings from activations
                    # Counting savings or added spending from changes in spot market prices
                    mfrr_earnings_up    = (data['spot'] - np.mean(data['spot']))
                    mfrr_earnings_down  = (data['spot'] - np.mean(data['spot']))
                    mfrr_earnings_up    = mfrr_earnings_up   + AM_earnings_up   if AM else mfrr_earnings_up
                    mfrr_earnings_down  = mfrr_earnings_down + AM_earnings_down if AM else mfrr_earnings_down
                    mfrr_earnings_up    = mfrr_earnings_up   + CM_earnings_up   if CM else mfrr_earnings_up
                    mfrr_earnings_down  = mfrr_earnings_down + CM_earnings_down if CM else mfrr_earnings_down


                    # Indices of most profitable mtu's to have activated
                    up_indices      = np.argsort(mfrr_earnings_up)
                    down_indices    = np.argsort(mfrr_earnings_down)

                    up_activations      = []
                    down_activations    = []
                    light_schedule      = np.ones(len(data['spot']), dtype=int)

                    qh_on, qh_off = 0, 0
                    max_act_up, max_act_down = 32, 32

                    # Goal of while loop is to find the most profitable 8h to turn light off 
                    # and most profitable 8h to keep light on
                    # Default light value is on, and the remaining 8h are unchanged. 
                    while qh_off < max_act_up or qh_on < max_act_down:
                        if qh_off >= len(up_indices) or qh_on >= len(down_indices):
                            break  # Safety check

                        mtu_up,     earnings_up     = up_indices[-1-qh_off],  mfrr_earnings_up[up_indices[-1-qh_off]]
                        mtu_down,   earnings_down   = down_indices[-1-qh_on], mfrr_earnings_down[down_indices[-1-qh_on]]

                        if (qh_off < max_act_up and earnings_up >= earnings_down) or qh_on >= max_act_down:
                            # Up activation
                            up_activations.append(mtu_up)
                            light_schedule[mtu_up] = 0
                            qh_off += 1
                            down_indices = down_indices[np.where(down_indices != mtu_up)]
                        else:
                            # Down activation
                            down_activations.append(mtu_down)
                            light_schedule[mtu_down] = 1
                            qh_on += 1
                            up_indices = up_indices[np.where(up_indices != mtu_down)]


                    # Check validity of solution:
                    assert len(set(up_activations + down_activations)) == len(up_activations+down_activations),  "One or more MTUs have been assinged to both up and down regulations"
                    if N_max == 96: assert sum(light_schedule) == 64, "Light schedule does not meet the required 64 qhs of light"


                    # Compute total earnings and costs
                    AM_earnings_up      = np.sum(AM_earnings_up[up_activations])/4      if AM else 0
                    AM_earnings_down    = np.sum(AM_earnings_down[down_activations])/4  if AM else 0
                    CM_earnings_up      = np.sum(CM_earnings_up[up_activations])/4      if CM else 0
                    CM_earnings_down    = np.sum(CM_earnings_down[down_activations])/4  if CM else 0
                    energy_cost         = np.sum(data["spot"][light_schedule == 1])/4

                    total_earnings_AM               += AM_earnings_up + AM_earnings_down
                    total_earnings_CM               += CM_earnings_up + CM_earnings_down
                    total_electricity_costs_AM      += energy_cost
                    total_electricity_costs_fixed   += np.sum(data["spot"][:16 * 4])/4

                    nominal_costs_df.at[k, zone]   = np.sum(data["spot"][:16 * 4])/4
                    mfrr_costs_df.at[k, zone]      = energy_cost - (AM_earnings_up + AM_earnings_down + CM_earnings_up + CM_earnings_down)


                total_earnings_mFRR = total_earnings_AM + total_earnings_CM

                results[zone] = {}
                results[zone]['AM_earnings'] = total_earnings_AM
                results[zone]['CM_earnings'] = total_earnings_CM
                results[zone]['mFRR_earnings'] = total_earnings_mFRR
                results[zone]['nom_cost'] = total_electricity_costs_fixed
                results[zone]['new_cost'] = total_electricity_costs_AM
                results[zone]['mFRR_cost'] = total_electricity_costs_AM - total_earnings_mFRR

                AM_earnings_perc    = 100 * total_earnings_AM / (total_earnings_AM + total_earnings_CM)
                CM_earnings_perc    = 100 * total_earnings_CM / (total_earnings_AM + total_earnings_CM)
                mFRR_cost           = total_electricity_costs_AM - total_earnings_mFRR

                cost_reduction = (total_electricity_costs_fixed - mFRR_cost)/total_electricity_costs_fixed * 100
                results[zone]['cost_reduction'] = cost_reduction
                results[zone]['message'] = f"Net cost reduction in percentage for {zone}: \t{cost_reduction:.2f}, AM: {AM_earnings_perc:.2f}, CM: {CM_earnings_perc:.2f}, \tNominal cost: {total_electricity_costs_fixed:.2f}, mFRR cost: {mFRR_cost:.2f}"
                pbar.update(1)

        for zone in results: print(results[zone]['message'])


        cmap = plt.get_cmap("Set3")
        colors = {
            "nominal": cmap(8),   # Gray 
            "AM": cmap(3),        # Red
            "CM": cmap(0),        # Blue
            "mFRR_cost": cmap(6)  # Green
        }

        zones = list(results.keys())  
        cost_reduction = np.array([results[zone]['cost_reduction'] for zone in zones])  # Total cost reduction per zone
        AM_earnings = np.array([results[zone]['AM_earnings'] for zone in zones])  # Absolute AM earnings
        CM_earnings = np.array([results[zone]['CM_earnings'] for zone in zones])  # Absolute CM earnings

        # Convert AM and CM earnings to their proportional contributions
        total_earnings = AM_earnings + CM_earnings
        AM_earnings_perc = AM_earnings / total_earnings  # Fraction of total earnings from AM
        CM_earnings_perc = CM_earnings / total_earnings  # Fraction of total earnings from CM

        # Compute actual bar heights for stacked representation
        AM_heights = cost_reduction * AM_earnings_perc
        CM_heights = cost_reduction * CM_earnings_perc

        # Create the stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(zones, CM_heights, color=colors['CM'], label="Capacity Market")
        ax.bar(zones, AM_heights, bottom=CM_heights, color=colors['AM'], label="Activation Market")

        # Labels and Title
        ax.set_xlabel("Zones")
        ax.set_ylabel("Cost Reduction (%)")
        ax.set_title("Upper estimate of cost reduction for a VF in Nordic bidding zones")
        ax.set_xticklabels(zones, rotation=0)
        ax.legend()

        plt.show()


        zones = list(results.keys())  
        nominal_cost = np.array([results[zone]['nom_cost'] for zone in zones])  # Total electricity cost without mFRR
        new_cost = np.array([results[zone]['new_cost'] for zone in zones])  # Total electricity cost with mFRR
        mfrr_cost = np.array([results[zone]['mFRR_cost'] for zone in zones])  # Remaining cost after mFRR earnings
        mfrr_earnings = np.array([results[zone]['mFRR_earnings'] for zone in zones])  # Total mfrr earnings
        AM_earnings = np.array([results[zone]['AM_earnings'] for zone in zones])  # Absolute AM earnings
        CM_earnings = np.array([results[zone]['CM_earnings'] for zone in zones])  # Absolute CM earnings
        mfrr_cost = nominal_cost - mfrr_earnings  # Remaining cost after mFRR earnings

        # Bar positions
        x = np.arange(len(zones))  # X-axis positions
        bar_width = 0.3  # Width of each bar

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x - bar_width, nominal_cost, width=bar_width, color=colors["nominal"], label="Nominal Cost")

        ax.bar(x, CM_earnings, width=bar_width, color=colors["CM"], label="Earnings from Capacity Market")
        ax.bar(x, AM_earnings, width=bar_width, bottom=CM_earnings, color=colors["AM"], label="Earnings from Activation Market")

        ax.bar(x + bar_width, mfrr_cost, width=bar_width, color=colors["mFRR_cost"], label="Cost after mFRR participation")


        # Labels and Title
        ax.set_xlabel("Zones")
        ax.set_ylabel("Cost / Earnings (€)")
        ax.set_title("Cost Breakdown before and after mFRR participation for a VF in Nordic bidding zones")
        ax.set_xticks(x)
        ax.set_xticklabels(zones, rotation=0)
        ax.legend()

        plt.show()


        ####################################################
                #   MARKET POTENCY BAR CHART

        T = 1
        def moving_average(data, window_size):
            smoothed_data = data.copy()  # Copy to avoid modifying the original DataFrame
            for col in smoothed_data.columns:
                if col == 'Date': continue
                smoothed_data[col] = np.convolve(smoothed_data[col], np.ones(window_size) / window_size, mode='same')
            return smoothed_data  # Explicitly return the modified DataFrame

        nominal_costs_df_smoothed = moving_average(nominal_costs_df, T)
        mfrr_costs_df_smoothed = moving_average(mfrr_costs_df, T)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6), sharex=False)

        nominal_costs_df_smoothed.set_index('Date')[zones].plot(
            ax          = ax1, 
            # linewidth   = 2.5, 
            linestyle   = '-',
            figsize     = (10,2.5)
        )

        mfrr_costs_df_smoothed.set_index('Date')[zones].plot(
            ax          = ax2, 
            # linewidth   = 2.5, 
            linestyle   = '-',
            figsize     = (10,2.5)
        )

        # first_of_month_daily = mfrr_costs_df['Date'][mfrr_costs_df['Date'].dt.day == 1]

        ax1.legend(zones, title='Zone')
        # ax1.set_xticks(first_of_month_daily.index)
        # ax1.set_xticklabels(first_of_month_daily.dt.strftime('%Y-%m-%d'), rotation=0)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Daily Cost (€)')
        ax1.set_title(f'Nominal cost of operations')
        
        ax2.legend(zones, title='Zone')
        # ax2.set_xticks(first_of_month_daily.index)
        # ax2.set_xticklabels(first_of_month_daily.dt.strftime('%Y-%m-%d'), rotation=0)
        ax2.set_ylabel('Daily Cost (€)')
        ax2.set_title('Cost of operations after mfrr AM+CM participation')
        # ax2.set_ylim([-1200, 600])

        plt.suptitle(f'Daily Costs. Length {T} smoothing window')
        # plt.tight_layout()

        plt.show()

        return