import numpy as np
import pandas as pd
import casadi as ca
import scipy as sp
from globals import *
from utils import *
from estimator import Estimator
from market_utils import *
from config import Config
from settings import Settings

class BalancingMarket:
    '''

    '''

    market_data_working_set: pd.DataFrame
    market_data_full_set:    pd.DataFrame

    market_type:            str
    clearing_prices_up:     np.array
    clearing_prices_down:   np.array
    volume_up:              np.array
    volume_down:            np.array
    activations_up:         np.array
    activations_down:       np.array

    price_stats:        dict
    activation_stats:   dict

    def __init__(self, settings: Settings, market_type, market_data):
        
        self.settings       = settings

        self.market_settings = settings.get_settings_group('general', 'market')

        self.T                  = self.market_settings['SIMULATION_LENGTH']
        self.bidding_zone       = self.market_settings['BIDDING_ZONE']
        self.date               = self.market_settings['SIMULATION_DATE']
        self.optimistic         = self.market_settings['OPTIMISTIC']
        self.N = self.T * QUARTER_HOURS_PER_DAY

        self.market_type    = market_type

        self.market_data_full_set       = self.standardize_market_data(market_data)
        self.market_data_working_set    = self.get_market_data()

        self.spot_prices             = np.array(self.market_data_working_set[f'{self.bidding_zone} Spot Price'])
        self.clearing_prices_up      = np.array(self.market_data_working_set[f'{self.bidding_zone} Up Price'])
        self.clearing_prices_down    = np.array(self.market_data_working_set[f'{self.bidding_zone} Down Price'])
        self.volume_up               = np.array(self.market_data_working_set[f'{self.bidding_zone} Up Volume'])
        self.volume_down             = np.array(self.market_data_working_set[f'{self.bidding_zone} Down Volume'])
        self.activations_up          = np.where(np.logical_and(self.volume_up > 0,   self.volume_up >= self.volume_down), 1, 0)
        self.activations_down        = np.where(np.logical_and(self.volume_down > 0, self.volume_up < self.volume_down),  1, 0)

        self.perform_statistical_analysis()
        self.perform_price_estimation()

        self.expected_clearing_prices_up  , conditional_price_variance_up   = conditional_expectation(self.spot_prices, self.price_stats[self.bidding_zone]['Up']['means'],    self.price_stats[self.bidding_zone]['Up']['cov'])
        self.expected_clearing_prices_down, conditional_price_variance_down = conditional_expectation(self.spot_prices, self.price_stats[self.bidding_zone]['Down']['means'],  self.price_stats[self.bidding_zone]['Down']['cov'])


    def perform_statistical_analysis(self):
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
            
        market_data   = self.market_data_working_set.copy()

        price_statistics = {}
        activation_statistics = {}
        zones = [self.bidding_zone]

        for zone in zones:
            price_statistics[zone] = {}
            activation_statistics[zone] = {}

            for direction in ['Up', 'Down']:
                price_statistics[zone][direction] = {}
                activation_statistics[zone][direction] = {}

                price_data = market_data[[f'{zone} Spot Price', 
                                    f'{zone} {direction} Price']][market_data[f'{zone} {direction} Volume'] > 0]

                activation_data = market_data[[f'{zone} Spot Price', f'{zone} {direction} Volume']].copy()
                activation_data[f'{zone} {direction} Volume'] = (activation_data[f"{zone} {direction} Volume"] > 0).astype(int)

                # if there are no activations, any dataset will do
                n = len(price_data.columns)
                if len(price_data) >= 2:
                    price_statistics[zone][direction]['cov']        = np.cov(price_data.T)
                else:
                    price_statistics[zone][direction]['cov']        = np.zeros((n,n), np.float64)
                    
                if len(price_data) >= 1:
                    price_statistics[zone][direction]['means']      = np.mean(np.array(price_data), axis=0)
                else:
                    price_statistics[zone][direction]['means']      = np.zeros(n, np.float64)

                activation_statistics[zone][direction]['means'] = np.mean(np.array(activation_data), axis=0)
                activation_statistics[zone][direction]['cov']   = np.cov(activation_data.T)

        self.price_stats         = price_statistics
        self.activation_stats    = activation_statistics

        return 0


    def activation_prob_up(self, spot_price, bid_price_up = None):
        if bid_price_up is None: return np.mean(self.activations_up)
        return self.activation_prob('Up', spot_price, bid_price_up)
    def activation_prob_down(self, spot_price, bid_price_up=None):
        if bid_price_up is None: return np.mean(self.activations_down)
        return self.activation_prob('Down', spot_price, bid_price_up)
    
    def activation_prob(self, direction, spot_price, bid_price_up):
        bid_price_up = bid_price_up.reshape((1,-1))
        
        mu, sigma = conditional_expectation(spot_price, self.price_stats[self.bidding_zone][direction]['means'], self.price_stats[self.bidding_zone][direction]['cov'])

        bid_price_up_normalized = ((bid_price_up - ca.vertcat(*mu).reshape((1,-1)))/sigma).reshape((1,-1))

        return np.multiply(self.demand_prob_up(spot_price), (1.0 + ca.erf(-bid_price_up_normalized / ca.sqrt(2.0))) / 2.0)


    def demand_prob_up(self, spot_price = None): 
        if spot_price is None: return np.mean(self.activations_up)
        return self.demand_prob('Up', spot_price)
    def demand_prob_down(self, spot_price = None): 
        if spot_price is None: return np.mean(self.activations_down)
        return self.demand_prob('Down', spot_price)


    def demand_prob(self, direction, spot_price):
            # Use mean spot_price if none other is specified

        expected_activation_up, _ = conditional_expectation(spot_price, 
                                        self.activation_stats[self.bidding_zone][direction]['means'], 
                                        self.activation_stats[self.bidding_zone][direction]['cov']
                                    )
        expected_activation_up = ca.horzcat(*expected_activation_up).reshape((1,-1))
                
        return casadi_saturate(expected_activation_up, 0, 1)
    

    def get_clearing_prices(self, date=None, n_days = None, zone = None):
        if zone     == None: zone   = self.bidding_zone

        data = self.get_market_data(date=date, n_days=n_days, zone=zone, 
                                       spot_prices=False, clearing_prices=True, volumes=False)
        
        return np.array(data[f'{zone} Up Price']), np.array(data[f'{zone} Down Price'])


    def get_expected_clearing_prices(self, date=None, n_days = None, zone = None):
        if zone     == None: zone   = self.bidding_zone

        data = self.get_market_data(date=date, n_days=n_days, zone=zone, 
                                    spot_prices=False, clearing_prices=True, volumes=False)

        spot_prices = np.array(data[f'{zone} Spot Price'])

        expected_clearing_prices_up  , _ = conditional_expectation(spot_prices, self.price_stats[self.bidding_zone]['Up']['means'],    self.price_stats[self.bidding_zone]['Up']['cov'])
        expected_clearing_prices_down, _ = conditional_expectation(spot_prices, self.price_stats[self.bidding_zone]['Down']['means'],  self.price_stats[self.bidding_zone]['Down']['cov'])
        
        return expected_clearing_prices_up, expected_clearing_prices_down


    def get_activations(self, date = None, n_days = None, zone = None):
        '''
        Returns numpy arrays of length N with Capacity market reservations during each quarter hour from the start time.
        For every MTU, a 1 indicates that a reservation was made and a 0 indicates that no reservation was made.
        Start time is always assumed at 00:00 at the given start date.
        '''

        if zone     == None: zone   = self.bidding_zone

        data = self.get_market_data(date=date, n_days=n_days, zone=zone, 
                                       spot_prices=False, clearing_prices=False, volumes=True)
        
        volumes_up, volumes_down = np.array(data[f'{zone} Up Volume']), np.array(data[f'{zone} Down Volume'])

        activations_up   = np.where(np.logical_and(volumes_up     > 0, volumes_up >= volumes_down), 1, 0)
        activations_down = np.where(np.logical_and(volumes_down   > 0, volumes_up < volumes_down), 1, 0)

        return activations_up, activations_down


    def perform_price_estimation(self):
        self.expected_prices_up, _   = conditional_expectation(self.spot_prices, self.price_stats[self.bidding_zone]['Up']['means'],    self.price_stats[self.bidding_zone]['Up']['cov'])
        self.expected_prices_down, _ = conditional_expectation(self.spot_prices, self.price_stats[self.bidding_zone]['Down']['means'],  self.price_stats[self.bidding_zone]['Down']['cov'])


    def get_market_data(self, date=None, n_days = None, zone = None, spot_prices=True, clearing_prices=True, volumes=True):
        '''
        Fetch a slice form the full data set containing specified data types.
        
        returns a dataframe containing fetched data. 
        '''

        if date     == None: date   = self.date
        if n_days   == None: n_days = self.T
        if zone     == None: zone   = self.bidding_zone

        start_date = pd.to_datetime(date, format='%Y-%m-%d')
        end_date = start_date + pd.DateOffset(n_days)

        # Remove dates before simdate
        market_data_full_set = self.market_data_full_set.copy()

        market_data_working_set = market_data_full_set[
            (market_data_full_set['Start Time']   >= start_date)    &
            (market_data_full_set['Start Time']   <  end_date) 
            ]
        
        keywords = []
        if spot_prices:     keywords.append('spot')
        if clearing_prices: keywords += ['up price', 'down price']
        if volumes:         keywords.append('volume')
        columns = ['Start Time'] + [col for col in market_data_working_set.columns if any([keyword in col.lower() for keyword in keywords])]

        market_data_working_set = market_data_working_set[columns].copy()
        market_data_working_set.fillna(market_data_working_set.mean(), inplace=True)

        return market_data_working_set

    def standardize_market_data(self, raw_data):
        data = raw_data.copy()
        
        # Ensure 'Start Time' is a datetime column
        if 'Start Time' not in data.columns:
            raise ValueError("Data must have a 'Start Time' column")
        
        data['Start Time'] = pd.to_datetime(data['Start Time'])

        # Detect frequency
        data = data.sort_values(by='Start Time')  # Ensure proper ordering
        inferred_freq = pd.infer_freq(data['Start Time'].iloc[:5])  # Check first few rows

        if inferred_freq in ["H", "60T"]:  # If data is hourly, resample to 15-minute intervals
            new_index = pd.date_range(start=data['Start Time'].min(), 
                                    end=data['Start Time'].max() + pd.Timedelta(hours=1) - pd.Timedelta(minutes=15),  
                                    freq="15T")
            data = data.set_index('Start Time').reindex(new_index, method='ffill').reset_index()
            data.rename(columns={'index': 'Start Time'}, inplace=True)

        # Standardize column names
        standardized_columns = {}

        data = data.loc[:, ~data.columns.str.contains("accepted", case=False)]

        for col in data.columns:
            col_lower = col.lower()

            if "spot" in col_lower:
                zone = col.split()[0]
                standardized_columns[col] = f"{zone} Spot Price"
            elif "volume" in col_lower:
                parts = col.split()
                zone = parts[0]
                direction = "Up" if "up" in col_lower else "Down"
                standardized_columns[col] = f"{zone} {direction} Volume"
            elif any(direction in col for direction in ["Up Price", "Down Price"]):
                standardized_columns[col] = col

        data.rename(columns=standardized_columns, inplace=True)

        return data



    def subject_bids_to_market_data(self, MTU_start, Bid_volumes, Bid_prices, zone = None):
        """
        Submit time-indexed bids to the market and determine activations and earnings.

        Args:
            MTU_start (pd.Timestamp): Start time of the first MTU in the bid sequence.
            Bid_volumes (np.ndarray): [2, N] array of bid volumes (up, down).
            Bid_prices (np.ndarray): [2, N] array of bid prices (up, down).

        Returns:
            activated_volumes (np.ndarray): [2, N] array of activated volumes.
            earnings (np.ndarray): [2, N] array of earnings.
        """

        # Default values
        if zone     == None: zone   = self.bidding_zone

        # Prepare return arrays
        n_periods = Bid_volumes.shape[1]
        activated_volumes   = np.zeros_like(Bid_volumes)
        earnings            = np.zeros_like(Bid_volumes)
        activations         = np.zeros_like(Bid_volumes)

        # Ensure market data has datetime index
        market_data = self.market_data_full_set.set_index('Start Time')

        # Loop over each time period
        for i in range(n_periods):
            current_time = MTU_start + pd.Timedelta(hours=i)

            # Skip if no matching market data
            if current_time not in market_data.index:
                continue

            market_row          = market_data.loc[current_time]
            spot_price          = market_row[f'{zone} Spot Price']
            market_volume_up    = market_row[f'{zone} Up Volume']
            market_volume_down  = market_row[f'{zone} Down Volume']
            market_price_up     = market_row[f'{zone} Up Price']
            market_price_down   = market_row[f'{zone} Down Price']
            
            # Extract bids
            bid_price_up = Bid_prices[0, i]
            bid_price_down = Bid_prices[1, i]
            bid_volume_up = Bid_volumes[0, i]
            bid_volume_down = Bid_volumes[1, i]

            # Determine activations
            if market_volume_up > 0 and market_volume_up >= market_volume_down and bid_price_up <= market_price_up:
                activations[0, i] = 1
                activated_volumes[0, i] = bid_volume_up
                earnings[0, i] = bid_volume_up * (market_price_up - spot_price) / 4

            if market_volume_down > 0 and market_volume_up < market_volume_down and bid_price_down <= market_price_down:
                activations[1, i] = 1
                activated_volumes[1, i] = bid_volume_down
                earnings[1, i] = bid_volume_down * (spot_price - market_price_down) / 4

        return activations, activated_volumes, earnings