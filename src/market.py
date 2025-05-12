import numpy as np
from scipy.stats import norm
import casadi as ca
import pandas as pd
from config import Config
from balancingmarket import BalancingMarket
from estimator import Estimator, EstimatorDF
from globals import *
from utils import *
from market_utils import *
import json
import os
import time
from settings import Settings
from typing import List, Dict
import pmdarima as pm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import * 
import re
import joblib
from matplotlib.colors import ListedColormap


class Market:

    N: int
    T: float
    seed: int
    bidding_zone: str
    date: str
    optimistic: bool            # Optimistic market model restricts analysis dataset to be only that of the growth cycle. Makes price distribution analysis more accurate
    
    config: Config
    settings: Settings

    balancing_markets: Dict[str, BalancingMarket] = {}
    AM: BalancingMarket # Activation Market
    CM: BalancingMarket # Capacity Market

    AM_price_stats: dict
    CM_price_stats: dict

    expected_AM_prices_up:      np.array        # Array of most likely clearing prices for up-regulation at each time step      (length: N)
    expected_AM_prices_down:    np.array        # Array of most likely clearing prices for down-regulation at each time step    (length: N)
    expected_CM_prices_up:      np.array        # Array of most likely clearing prices for up-regulation at each time step      (length: N)
    expected_CM_prices_down:    np.array        # Array of most likely clearing prices for down-regulation at each time step    (length: N)
    opt_prices_up:              np.array        # Array of most profitable bidding prices for up-regulation at each time step   (length: N)
    opt_prices_down:            np.array        # Array of most profitable bidding prices for down-regulation at each time step (length: N)
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
        self.MTU_start          = pd.to_datetime(self.date, format='%Y-%m-%d')

        self.import_market_data()

        CM_estimator_config = {'dep_lags': list(range(96, 192)), 'indep_lags': list(range(96, 192)),    'training_window': [-30*QUARTER_HOURS_PER_DAY, -1*QUARTER_HOURS_PER_DAY]}
        AM_estimator_config = {'dep_lags': list(range(4,100)),    'indep_lags': list(range(4, 100)),       'training_window': [-30*QUARTER_HOURS_PER_DAY, -1*QUARTER_HOURS_PER_DAY]}
        # AM_estimator_config = {'dep_lags': list(range(1,4)),      'indep_lags': list(range(1,4))}

        self.CM = BalancingMarket(settings, 'Capacity Market',   self.CM_data_full_set, data_resolution = 24, estimator_config = CM_estimator_config)
        self.AM = BalancingMarket(settings, 'Activation Market', self.AM_data_full_set, data_resolution = 96, estimator_config = AM_estimator_config)
        self.balancing_markets['Capacity Market']   = self.CM
        self.balancing_markets['Activation Market'] = self.AM

        self.spot_prices = self.get_spotprice() 

        self.expected_AM_prices_up  , conditional_variance_AM_up   = conditional_expectation(self.spot_prices, self.AM.price_stats[self.bidding_zone]['Up']['means'],    self.AM.price_stats[self.bidding_zone]['Up']['cov'])
        self.expected_AM_prices_down, conditional_variance_AM_down = conditional_expectation(self.spot_prices, self.AM.price_stats[self.bidding_zone]['Down']['means'],  self.AM.price_stats[self.bidding_zone]['Down']['cov'])
        
        self.expected_CM_prices_up  , conditional_variance_CM_up   = conditional_expectation(self.spot_prices, self.CM.price_stats[self.bidding_zone]['Up']['means'],    self.CM.price_stats[self.bidding_zone]['Up']['cov'])
        self.expected_CM_prices_down, conditional_variance_CM_down = conditional_expectation(self.spot_prices, self.CM.price_stats[self.bidding_zone]['Down']['means'],  self.CM.price_stats[self.bidding_zone]['Down']['cov'])
        

        epsilon = 1e-6  # For numerical stability. Avoids 0-variance
        self.sigma_AM_up   = np.sqrt(max(conditional_variance_AM_up,   epsilon))
        self.sigma_AM_down = np.sqrt(max(conditional_variance_AM_down, epsilon))
        self.sigma_CM_up   = np.sqrt(max(conditional_variance_CM_up,   epsilon))
        self.sigma_CM_down = np.sqrt(max(conditional_variance_CM_down, epsilon))
        
        # Initialize optimal prices on expected value.
        self.opt_prices_up   = self.expected_AM_prices_up
        self.opt_prices_down = self.expected_AM_prices_down

        self.specs = {
            'bidding zone'                  : self.bidding_zone,
            'simdate'                       : self.date,
            'optimistic'                    : self.optimistic,
            'Avg activation price up'       : self.AM.price_stats[self.bidding_zone]['Up']['means'][1],
            'Avg activation price down'     : self.AM.price_stats[self.bidding_zone]['Down']['means'][1],          
            'Cond covariance spot - up'     : conditional_variance_AM_up,
            'Cond covariance spot - Down'   : conditional_variance_AM_down,
            'Outlier max std-dev distance'  : self.outlier_max_dist
            }
            
        self.mfrr_activation_data_analysis()
        self.mfrr_demands_up, self.mfrr_demands_down = self.AM.get_activations(self.date)
        self.analyze_market_potency(T = self.T)



    def get_balancing_market(self, key) -> BalancingMarket:
        if key == 'CM': key = 'Capacity Market'     
        if key == 'AM': key = 'Activation Market'   
        return self.balancing_markets.get(key, None)


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

        
        mfrr_data_full_set  = pd.merge(AM_data_full_set.copy().rename(columns = {col: f'AM {col}' for col in AM_data_full_set.columns if 'Start Time' not in col and 'Spot' not in col}), 
                                           mfrr_CM_data.copy().rename(columns = {col: f'CM {col}' for col in mfrr_CM_data.columns if 'Start Time' not in col}), 
                                       on='Start Time', how='inner')

        AM_data_working_set = AM_data_full_set[(AM_data_full_set['Start Time'] >= start_date) & (AM_data_full_set['Start Time'] < end_date)]
        CM_data_working_set = CM_data_full_set[(CM_data_full_set['Start Time'] >= start_date) & (CM_data_full_set['Start Time'] < end_date)]
        
        self.spot_data_full_set             = spot_data
        self.AM_data_full_set               = AM_data_full_set
        self.AM_data_working_set            = AM_data_working_set
        self.CM_data_full_set               = CM_data_full_set
        self.CM_data_working_set            = CM_data_working_set

        self.mfrr_data_full_set = mfrr_data_full_set

        return 0 


    def optimal_bidding_price_prediction(self, spot_prices):    
        '''
        DEPRECATED since addition of CM and AM objects in the market class
        '''

        start_time = time.time()
        print("Starting price prediction")

        prices_stats_up     = self.AM.price_stats[self.bidding_zone]['Up']
        prices_stats_down   = self.AM.price_stats[self.bidding_zone]['Down']

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



    def generate_activation_demands(self, market_type, N, seed, spot_price):
        '''
        Generates a list of ints where 0 means no demand for activation, -1 means down activation and 1 means up
        '''

        balancing_market = self.get_balancing_market(market_type)

        D_up = balancing_market.demand_prob_up(spot_price)
        D_dn = balancing_market.demand_prob_down(spot_price)
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
        
        CM_clearing_prices_up, CM_clearing_prices_down = self.CM.get_clearing_prices()

        bid_volumes_up      = run['markets']['CM']['Bids']["Up"]['Volume']
        bid_volumes_down    = run['markets']['CM']['Bids']["Down"]['Volume']

        return np.sum(np.multiply(CM_clearing_prices_up, bid_volumes_up) + np.multiply(CM_clearing_prices_down, bid_volumes_down))
        




    def estimate_prices(self):


        xlags = list(range(24,72))
        ylags = list(range(2,24))



        price_data = self.mfrr_data_full_set.copy().dropna()
        
        price_data = price_data[[col for col in price_data if self.bidding_zone in col]]
        
        spot_prices_df      = price_data[[col for col in price_data if self.bidding_zone in col and 'Spot' in col]].copy()
        CM_prices_df        = price_data[[col for col in price_data if self.bidding_zone in col and 'CM' in col and 'Price' in col]].copy()
        AM_prices_df        = price_data[[col for col in price_data if self.bidding_zone in col and 'AM' in col and 'Price' in col]].copy()
        spot_CM_prices_df   = price_data[[col for col in price_data if self.bidding_zone in col and 'Spot' in col or ('CM' in col and 'Price' in col)]].copy()
        
        CM_price_estimate = EstimatorDF('CM Prices', CM_prices_df, spot_prices_df,   
                                        xlags=xlags, ylags=ylags)
        AM_price_estimate = EstimatorDF('CM Prices', AM_prices_df, spot_prices_df,   
                                        xlags=xlags, ylags=ylags)
        
        print(f"CM Estimator RMSE: {CM_price_estimate.rmse_scores}\t lags: {CM_price_estimate.xlags}")
        print(f"AM Estimator RMSE: {AM_price_estimate.rmse_scores}\t lags: {AM_price_estimate.xlags}")

        print("\n Performance measures:")
        CM_price_estimate.measure_performance()
        AM_price_estimate.measure_performance()
        

        CM_prices_up    = CM_prices_df[[col for col in CM_prices_df if 'Up Price' in col]]
        CM_prices_down  = CM_prices_df[[col for col in CM_prices_df if 'Down Price' in col]]
        AM_prices_up    = AM_prices_df[[col for col in AM_prices_df if 'Up Price' in col]]
        AM_prices_down  = AM_prices_df[[col for col in AM_prices_df if 'Down Price' in col]]

        Est_CM_prices_up    = CM_price_estimate.estimated_df[[col for col in CM_price_estimate.estimated_df if 'Up Price' in col]]
        Est_CM_prices_down  = CM_price_estimate.estimated_df[[col for col in CM_price_estimate.estimated_df if 'Down Price' in col]]
        Est_AM_prices_up    = AM_price_estimate.estimated_df[[col for col in AM_price_estimate.estimated_df if 'Up Price' in col]]
        Est_AM_prices_down  = AM_price_estimate.estimated_df[[col for col in AM_price_estimate.estimated_df if 'Down Price' in col]]


        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)


        ax1.plot(Est_CM_prices_up   ,     label='Est CM clearing up')
        ax2.plot(Est_CM_prices_down ,     label='Est CM clearing down')
        ax3.plot(Est_AM_prices_up   ,     label='Est AM clearing up')
        ax4.plot(Est_AM_prices_down ,     label='Est AM clearing down')

        ax1.plot(CM_prices_up  ,   linestyle = ':', color='gray', label='CM clearing up')
        ax2.plot(CM_prices_down,    linestyle = ':', color='gray', label='CM clearing down')
        ax3.plot(AM_prices_up  ,   linestyle = ':', color='gray', label='AM clearing up')
        ax4.plot(AM_prices_down,    linestyle = ':', color='gray', label='AM clearing down')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        plt.show()
        


    def calculate_balancing_market_earnings_upper_bound(self, AM = True, CM = True, zones = None, start_date='2024-02-15', end_date='2025-02-15'):
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
                total_electricity_costs_spot    = 0

                kwargs = {'date': start_date, 'n_days': N, 'zone': zone}
                try:
                    AM_clearing_prices_up_full, AM_clearing_prices_down_full        = self.AM.get_clearing_prices(**kwargs)
                    AM_activation_demands_up_full, AM_activation_demands_down_full  = self.AM.get_activations(**kwargs)
                    spot_prices_full                                                = self.get_spotprice(**kwargs)

                    CM_clearing_prices_up_full, CM_clearing_prices_down_full        = self.CM.get_clearing_prices(**kwargs)
                    CM_reservations_up_full, CM_reservations_down_full              = self.CM.get_activations(**kwargs)
                except:
                    pbar.update(1)
                    continue


                # Calculate reference fixed schedule cost:

                offset_qh = 0
                lowest_fixed_price = np.inf
                for QH in range(96):
                    light_schedule = np.ones(96)
                    light_schedule[64-QH:96-QH]
                    np.put(light_schedule, range(64-QH,96-QH), np.zeros(32)) 

                    light_schedule_tiled = np.tile(light_schedule, len(dates)+1)[:len(spot_prices_full)]

                    fixed_price = np.sum(np.multiply(spot_prices_full,light_schedule_tiled))/4
                    
                    if fixed_price < lowest_fixed_price:
                        lowest_fixed_price = fixed_price
                        offset_qh = QH

                total_electricity_costs_fixed_shifted = lowest_fixed_price


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
                    N_max = min(len(data["CM_up"]),len(data["AM_up"]))
                    for key in data.keys():
                        data[key] = data[key][:N_max]

                    AM_earnings_up      = np.multiply(data["AM_up"] - data['spot'],   data["AM_demand_up"])
                    AM_earnings_down    = np.multiply(data['spot'] - data["AM_down"], data["AM_demand_down"])
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
                    total_electricity_costs_spot    += np.sum(np.sort(data["spot"])[:16 * 4])/4

                    # nominal_costs_df.at[k, zone]   = np.sum(data["spot"][:16 * 4])/4
                    nominal_costs_df.at[k, zone]   = np.sum(np.sort(data["spot"])[:16 * 4])/4
                    mfrr_costs_df.at[k, zone]      = energy_cost - (AM_earnings_up + AM_earnings_down + CM_earnings_up + CM_earnings_down)


                total_earnings_mFRR = total_earnings_AM + total_earnings_CM
                total_cost_mFRR           = total_electricity_costs_AM - total_earnings_mFRR

                results[zone] = {}
                results[zone]['fixed_shifted_cost'] = total_electricity_costs_fixed_shifted
                results[zone]['spot_cost']          = total_electricity_costs_spot
                results[zone]['mFRR_cost']          = total_cost_mFRR
                results[zone]['AM_earnings']    = total_earnings_AM
                results[zone]['CM_earnings']    = total_earnings_CM
                results[zone]['mFRR_earnings']  = total_earnings_mFRR

                AM_earnings_perc    = 100 * total_earnings_AM / (total_earnings_AM + total_earnings_CM)
                CM_earnings_perc    = 100 * total_earnings_CM / (total_earnings_AM + total_earnings_CM)

                spot_cost_reduction = (total_electricity_costs_fixed_shifted - total_electricity_costs_spot)/total_electricity_costs_fixed_shifted * 100
                mFRR_cost_reduction = (total_electricity_costs_fixed_shifted - total_cost_mFRR)/total_electricity_costs_fixed_shifted * 100 - spot_cost_reduction
                results[zone]['spot_cost_reduction'] = spot_cost_reduction
                results[zone]['mFRR_cost_reduction'] = mFRR_cost_reduction
                results[zone]['AM_cost_reduction']   = AM_earnings_perc/100*mFRR_cost_reduction
                results[zone]['CM_cost_reduction']   = CM_earnings_perc/100*mFRR_cost_reduction

                results[zone]['message'] = f"Net cost reduction in percentage for {zone}: \t{spot_cost_reduction:.2f}, AM: {AM_earnings_perc:.2f}, CM: {CM_earnings_perc:.2f}, \tNominal cost: {total_electricity_costs_fixed:.2f}, mFRR cost: {total_cost_mFRR:.2f}"
                pbar.update(1)

        output_dir = self.config.output_path
        # Save summary results
        with open(os.path.join(output_dir, "summary_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        # Save daily time series
        nominal_costs_df.to_csv(os.path.join(output_dir, "nominal_costs.csv"), index=False)
        mfrr_costs_df.to_csv(os.path.join(output_dir, "mfrr_costs.csv"), index=False)

        '''
        for zone in results: print(results[zone]['message'])

        cmap = plt.get_cmap("Set3")
        colors = {
            "nominal": cmap(8),   # Gray 
            "AM": cmap(3),        # Red
            "CM": cmap(0),        # Blue
            "mFRR_cost": cmap(6), # Green
            "spot" : cmap(4)
        }

        zones = list(results.keys())
        fixed_shifted_cost      = np.array([results[zone]['fixed_shifted_cost'] for zone in zones])  
        spot_cost               = np.array([results[zone]['spot_cost'] for zone in zones])
        mFRR_cost               = np.array([results[zone]['mFRR_cost'] for zone in zones])

        spot_reduction = np.array([results[zone]['spot_cost_reduction'] for zone in zones])
        AM_earnings = np.array([results[zone]['AM_earnings'] for zone in zones])            # Absolute AM earnings
        CM_earnings = np.array([results[zone]['CM_earnings'] for zone in zones])            # Absolute CM earnings
        AM_reduction = np.array([results[zone]['AM_cost_reduction'] for zone in zones])     # Cost reduction from AM (%)
        CM_reduction = np.array([results[zone]['CM_cost_reduction'] for zone in zones])     # Cost reduction from CM (%)

        ####################################################
                #   FRACTIONAL COST REDUCTION BAR CHART. Zone-wise


        # my_colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
        # my_colors = ["#3D8D7A", "#B3D8A8", "#A3D1C6"]
        # my_colors = ["#328E6E", "#67AE6E", "#90C67C", "#E1EEBC"]
        # my_colors = ["#727D73", "#AAB99A", "#D0DDD0", "#F0F0D7"]
        # my_colors = ["#4B5945", "#66785F", "#91AC8F", "#B2C9AD"]
        # my_colors = ["#BF9264", "#6F826A", "#BBD8A3", "#F0F1C5"]
        my_colors = ["#557571", "#D49A89", "#F7D1BA", "#F4F4F4"]
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=my_colors)


        # plt.title('title',**csfont)
        # plt.xlabel('xlabel', **hfont)
        
        fontname = "DejaVu Serif"
        plt.rcParams["font.family"] = fontname

        fig, ax = plt.subplots(figsize=(6, 3.5))  # Adjusted for a narrow format

        # ax.barh(zones,  spot_reduction, color='lightgrey', label="Spot Market")
        # ax.axvline(x=100, color='black', linestyle='dotted', linewidth=1, label="Break-even Point")
        ax.barh(zones, spot_reduction, label="Spot Market")
        ax.barh(zones, CM_reduction, left=spot_reduction, label="Capacity Market")
        ax.barh(zones, AM_reduction, left=spot_reduction + CM_reduction, label="Activation Market")

        # Labels and Title
        ax.set_ylabel("Bidding Zones")
        ax.set_xlabel("Cost Reduction (%)")
        # ax.set_title("Upper Estimate of Cost Reduction for a VF in Nordic Bidding Zones")
        ax.legend(prop={'size': 10}, loc='upper right')# , bbox_to_anchor=(1, 0.7))
        ax.grid(axis='x', linestyle='-', alpha=0.6)

        plt.gca().invert_yaxis()  # Ensures zones are listed top-to-bottom
        plt.tight_layout()  # Optimizes spacing for a research paper

        plt.show()

        ####################################################
                #   COST OVERVIEW BAR CHART. Zone-wise

        # Bar positions
        x = np.arange(len(zones))  # X-axis positions
        bar_width = 0.3  # Width of each bar

        fig, ax = plt.subplots(figsize=(6, 3))

        ax.bar(x - bar_width, fixed_shifted_cost, width=bar_width, color=colors["nominal"], label="Nominal Cost")
        ax.bar(x, spot_cost, width=bar_width, color=colors["spot"], label="Cost after Spot optimizing")

        ax.bar(x + bar_width, mFRR_cost, width=bar_width, color=colors["mFRR_cost"], label="Cost after mFRR participation")


        # Labels and Title
        ax.set_xlabel("Zones")
        ax.set_ylabel("Cost / Earnings (€)")
        ax.set_title("Cost Breakdown before and after mFRR participation for a VF in Nordic bidding zones")
        ax.set_xticks(x)
        ax.set_xticklabels(zones, rotation=0)
        ax.legend()

        plt.show()


        ####################################################
                #   TIMELINES OF COST REDUCTION POTENTIAL

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

        #'''
        


    def nordic_markets_overview(self, output=False):

        AM_data_raw = self.AM.get_market_data('2024-01-01', n_days=365, all=True)
        CM_data_raw = self.CM.get_market_data('2024-01-01', n_days=365, all=True)
        

        zones = ['DK1', 'DK2', 'FI', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4']
        directions = ['Up', 'Down']

        AM_total_traded_volume = {zone: {} for zone in zones}
        AM_total_market_value = {zone: {} for zone in zones}
        AM_mean_activated_prices = {zone: {} for zone in zones}
        CM_total_traded_volume = {zone: {} for zone in zones}
        CM_total_market_value = {zone: {} for zone in zones}
        CM_mean_activated_prices = {zone: {} for zone in zones}
        
        for zone in zones:
            for direction in directions:

                AM_data = AM_data_raw.copy().dropna()
                CM_data = CM_data_raw.copy().dropna()

                AM_total_traded_volume[zone][direction] = np.sum(np.array(AM_data[[f"{zone} {direction} Volume"]]))/4
                CM_total_traded_volume[zone][direction] = np.sum(np.array(CM_data[[f"{zone} {direction} Volume"]]))/4
                
                AM_volume        = np.array(AM_data[f"{zone} {direction} Volume"])
                AM_prices        = np.array(AM_data[f"{zone} {direction} Price"])
                AM_market_value  = np.sum(np.where(AM_volume > 0, np.multiply(AM_volume, AM_prices), 0))/4
                AM_total_market_value[zone][direction] = AM_market_value
            
                CM_volume        = np.array(CM_data[f"{zone} {direction} Volume"])
                CM_prices        = np.array(CM_data[f"{zone} {direction} Price"])
                CM_market_value  = np.sum(np.where(CM_volume > 0, np.multiply(CM_volume, CM_prices), 0))/4
                CM_total_market_value[zone][direction] = CM_market_value
            
                AM_activated_prices = np.mean(AM_prices[np.where(AM_volume>0)])
                AM_mean_activated_prices[zone][direction] = AM_activated_prices

                CM_activated_prices = np.mean(CM_prices[np.where(CM_volume>0)])
                CM_mean_activated_prices[zone][direction] = CM_activated_prices


            

        AM_total_volume_up = np.array([AM_total_traded_volume[zone]['Up'] for zone in zones])
        AM_total_value_up  = np.array([AM_total_market_value[zone]['Up'] for zone in zones])  
        CM_total_volume_up = np.array([CM_total_traded_volume[zone]['Up'] for zone in zones])  
        CM_total_value_up  = np.array([CM_total_market_value[zone]['Up'] for zone in zones])  

        AM_total_volume_down = np.array([AM_total_traded_volume[zone]['Down'] for zone in zones])
        AM_total_value_down  = np.array([AM_total_market_value[zone]['Down'] for zone in zones])  
        CM_total_volume_down = np.array([CM_total_traded_volume[zone]['Down'] for zone in zones])  
        CM_total_value_down  = np.array([CM_total_market_value[zone]['Down'] for zone in zones])  

        AM_mean_price_up    = np.array([AM_mean_activated_prices[zone]['Up'] for zone in zones])
        AM_mean_price_down  = np.array([AM_mean_activated_prices[zone]['Down'] for zone in zones])  
        CM_mean_price_up    = np.array([CM_mean_activated_prices[zone]['Up'] for zone in zones])  
        CM_mean_price_down  = np.array([CM_mean_activated_prices[zone]['Down'] for zone in zones])  

        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(16,9), sharex=False)
        axes = ax1, ax2, ax3, ax4, ax5, ax6
        
        x = np.arange(len(zones))
        bar_width = 0.9

        # ax.bar(x - bar_width, nominal_cost, width=bar_width, color=colors["nominal"], label="Nominal Cost"

        # bar_colors_down = ['lightgreen']*2  + ['skyblue']*1  + ['salmon']*5      + ['gold']*4
        # bar_colors_up   = ['green']*2       + ['blue']*1     + ['tomato']*5      + ['darkgoldenrod']*4

        bar_colors_up     = 'skyblue'
        bar_colors_down   = 'lightcoral'


        ax1.set_title(f'Energy Activation Market: Total activated volume')
        ax1.bar(x, AM_total_volume_up/1e3,                                  width=bar_width, color=bar_colors_up,     label="Up")
        ax1.bar(x, AM_total_volume_down/1e3, bottom=AM_total_volume_up/1e3, width=bar_width, color=bar_colors_down,   label="Down")
        ax1.set_ylabel(f'Volume (GWh)')

        ax2.set_title(f'Energy Activation Market: Total traded value')
        ax2.bar(x, AM_total_value_up/1e6,                                   width=bar_width, color=bar_colors_up,     label="Up")
        ax2.bar(x, AM_total_value_down/1e6, bottom=AM_total_value_up/1e6,   width=bar_width, color=bar_colors_down,   label="Down")
        ax2.set_ylabel(f'Total Market Value (M€)')

        ax3.set_title(f'Capacity Market: Total volume procured')
        ax3.bar(x, CM_total_volume_up/1e3,                                  width=bar_width, color=bar_colors_up,     label="Up")
        ax3.bar(x, CM_total_volume_down/1e3, bottom=CM_total_volume_up/1e3, width=bar_width, color=bar_colors_down,   label="Down")
        ax3.set_ylabel(f'Volume (GWh)')

        ax4.set_title(f'Capacity Market: Total traded value')
        ax4.bar(x, CM_total_value_up/1e6,                                   width=bar_width, color=bar_colors_up,     label="Up")
        ax4.bar(x, CM_total_value_down/1e6, bottom=CM_total_value_up/1e6,   width=bar_width, color=bar_colors_down,   label="Down")
        ax4.set_ylabel(f'Total Market Value (M€)')

        half_bar_width = bar_width/2
        shift = half_bar_width/2

        ax5.set_title(f'Energy Activation Market: Mean clearing price')
        ax5.bar(x-shift, AM_mean_price_up,   width=half_bar_width, color=bar_colors_up,     label="Up")
        ax5.bar(x+shift, AM_mean_price_down, width=half_bar_width, color=bar_colors_down,   label="Down")
        ax5.set_ylabel(f'Clearing Price (€)')

        ax6.set_title(f'Capacity Market: Mean clearing price')
        ax6.bar(x-shift, CM_mean_price_up,   width=half_bar_width, color=bar_colors_up,     label="Up")
        ax6.bar(x+shift, CM_mean_price_down, width=half_bar_width, color=bar_colors_down,   label="Down")
        ax6.set_ylabel(f'Clearing Price (€)')


        for ax in axes:
            ax.legend(title='Direction')
            ax.set_xticks(range(len(zones)))
            ax.set_xticklabels(zones)
            ax.grid(axis='y', linestyle='-', alpha=0.7)


        plt.suptitle(f'Market overview of Nordic balancing markets 2024')
        plt.tight_layout()

        plot_file_type = 'png'

        if output:
            # Save each subplot individually
            for i, ax in enumerate(axes):  
                extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.config.output_path + f'market_overview_{i+1}.' + plot_file_type, format=plot_file_type, bbox_inches=extent, dpi=300)


        plt.show()