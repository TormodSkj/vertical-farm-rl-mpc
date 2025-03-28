import numpy as np
import pandas as pd
import casadi as ca
import scipy as sp
from globals import *
from utils import *
from estimator import Estimator

class BalancingMarket:
    '''

    '''
    market_type:            str
    clearing_prices_up:     np.array
    clearing_prices_down:   np.array
    volume_up:              np.array
    volume_down:            np.array
    activations_up:         np.array
    activations_down:       np.array

    price_stats:        dict
    activation_stats:   dict

    def __init__(self, market_type, bidding_zone, market_data):
        
        self.market_type    = market_type
        self.bidding_zone   = bidding_zone
        # self.date           = date
        # self.T              = T
        self.market_data    = self.standardize_market_data(market_data)

        self.spot_prices             = np.array(self.market_data[f'{self.bidding_zone} Spot Price'])
        self.clearing_prices_up      = np.array(self.market_data[f'{self.bidding_zone} Up Price'])
        self.clearing_prices_down    = np.array(self.market_data[f'{self.bidding_zone} Down Price'])
        self.volume_up               = np.array(self.market_data[f'{self.bidding_zone} Up Volume'])
        self.volume_down             = np.array(self.market_data[f'{self.bidding_zone} Down Volume'])
        self.activations_up          = np.where(np.logical_and(self.volume_up > 0,   self.volume_up >= self.volume_down), 1, 0)
        self.activations_down        = np.where(np.logical_and(self.volume_down > 0, self.volume_up < self.volume_down),  1, 0)

        self.perform_statistical_analysis()
        self.perform_price_estimation()


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
            
        market_data   = self.market_data.copy()

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


    def activation_prob_up(self, spot_price, bid_price_up):
        return self.activation_prob('Up', spot_price, bid_price_up)
    def activation_prob_down(self, spot_price, bid_price_up):
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


    def perform_price_estimation(self):
        self.expected_prices_up, _   = conditional_expectation(self.spot_prices, self.price_stats[self.bidding_zone]['Up']['means'],    self.price_stats[self.bidding_zone]['Up']['cov'])
        self.expected_prices_down, _ = conditional_expectation(self.spot_prices, self.price_stats[self.bidding_zone]['Down']['means'],  self.price_stats[self.bidding_zone]['Down']['cov'])

    
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

