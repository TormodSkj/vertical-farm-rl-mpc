import numpy as np
import pandas as pd
import casadi as ca
import scipy as sp
from globals import *
from utils import *
from estimator import Estimator, EstimatorDF
from market_utils import *
from config import Config
from settings import Settings

class BalancingMarket:
    '''

    '''
    seed: int

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

    def __init__(self, settings: Settings, market_type, market_data, data_resolution, estimator_config):
        
        self.settings       = settings

        self.market_settings = settings.get_settings_group('general', 'market')

        self.T                  = self.market_settings['SIMULATION_LENGTH']
        self.dt                 = self.market_settings['SIM_TIMEDELTA']
        self.bidding_zone       = self.market_settings['BIDDING_ZONE']
        self.date               = self.market_settings['SIMULATION_DATE']
        self.optimistic         = self.market_settings['OPTIMISTIC']
        self.bid_price_limit    = self.market_settings['BID_PRICE_LIMIT']
        self.seed               = self.market_settings['SEED']
        self.exact_estimation   = self.market_settings['EXACT_ESTIMATION']
        self.N = self.T * QUARTER_HOURS_PER_DAY

        self.market_type        = market_type
        self.data_resolution    = data_resolution

        self.market_data_full_set    = self.standardize_market_data(market_data)
        if market_type == 'Activation Market':  self.generate_activation_times(activation_chance=self.market_settings['AM_ACTIVATION_RATE'], hourly_roll=False)
        if market_type == 'Capacity Market':    self.generate_activation_times(activation_chance=self.market_settings['CM_ACTIVATION_RATE'], hourly_roll=True)
        self.market_data_working_set = self.get_market_data(all=True)

        self.spot_prices             = np.array(self.market_data_working_set[f'{self.bidding_zone} Spot Price'])
        self.clearing_prices_up      = np.array(self.market_data_working_set[f'{self.bidding_zone} Up Price'])
        self.clearing_prices_down    = np.array(self.market_data_working_set[f'{self.bidding_zone} Down Price'])
        self.volume_up               = np.array(self.market_data_working_set[f'{self.bidding_zone} Up Volume'])
        self.volume_down             = np.array(self.market_data_working_set[f'{self.bidding_zone} Down Volume'])
        self.activations_up, self.activations_down = self.get_activations()

        self.perform_statistical_analysis()
        self.perform_price_estimation(estimator_config)

        # self.expected_clearing_prices_up  , conditional_price_variance_up   = conditional_expectation(self.spot_prices, self.price_stats[self.bidding_zone]['Up']['means'],    self.price_stats[self.bidding_zone]['Up']['cov'])
        # self.expected_clearing_prices_down, conditional_price_variance_down = conditional_expectation(self.spot_prices, self.price_stats[self.bidding_zone]['Down']['means'],  self.price_stats[self.bidding_zone]['Down']['cov'])

        self.estimated_clearing_prices_up, self.estimated_clearing_prices_down = self.get_estimated_clearing_prices()
        self.conditional_price_variance_up, self.conditional_price_variance_down = self.get_estimated_clearing_price_variances()


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
                                    f'{zone} {direction} Price']][market_data[f'{zone} Activated {direction}'] > 0]

                activation_data = market_data[[f'{zone} Spot Price', f'{zone} Activated {direction}']].copy()
                activation_data[f'{zone} Activated {direction}'] = (activation_data[f"{zone} Activated {direction}"] > 0).astype(int)

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


    def activation_prob_up(self, bid_price = None, spot_price = None, clearing_price = None, clearing_price_variance = None):
        if bid_price is None: return np.mean(self.activations_up)
        return self.activation_prob('Up', bid_price, spot_price, clearing_price, clearing_price_variance)
    def activation_prob_down(self, bid_price = None, spot_price = None, clearing_price = None, clearing_price_variance = None):
        if bid_price is None: return np.mean(self.activations_down)
        return self.activation_prob('Down', bid_price, spot_price, clearing_price, clearing_price_variance)
    
    def activation_prob(self, direction, bid_price, spot_price = None, clearing_price = None, clearing_price_variance = None):
        bid_price = bid_price.reshape((1,-1))
        
        if spot_price is not None and (clearing_price is None or clearing_price_variance is None):
            clearing_price, clearing_price_variance = conditional_expectation(spot_price, self.price_stats[self.bidding_zone][direction]['means'], self.price_stats[self.bidding_zone][direction]['cov'])
        elif spot_price is None and (clearing_price is None or clearing_price_variance is None):
            assert False, 'activation_prob called without either spot price or clearing prices'

        return np.multiply(self.demand_prob(direction, spot_price), 1 - gaussian_CDF(bid_price, clearing_price, clearing_price_variance))


    def demand_prob_up(self, spot_price = None): 
        if spot_price is None: return np.mean(self.activations_up)
        return self.demand_prob('Up', spot_price)
    def demand_prob_down(self, spot_price = None): 
        if spot_price is None: return np.mean(self.activations_down)
        return self.demand_prob('Down', spot_price)


    def demand_prob(self, direction, spot_price):

        expected_activation, _ = conditional_expectation(spot_price, 
                                        self.activation_stats[self.bidding_zone][direction]['means'], 
                                        self.activation_stats[self.bidding_zone][direction]['cov']
                                    )
        expected_activation = ca.horzcat(*expected_activation).reshape((1,-1))
                
        return casadi_saturate(expected_activation, 0, 1)
    

    def get_clearing_prices(self, date=None, n_days = None, zone = None):
        if zone     == None: zone   = self.bidding_zone

        data = self.get_market_data(start_date=date, n_days=n_days, zone=zone, 
                                     times= True, clearing_prices=True)
        
        return np.array(data[f'{zone} Up Price']), np.array(data[f'{zone} Down Price'])


    def get_estimated_clearing_prices(self, spot_prices = None, start_date=None, end_date=None, n_data = None, zone = None):

        if start_date == None: start_date = self.date
        if n_data     == None: n_data     = self.N
        if end_date   == None: end_date   = pd.to_datetime(start_date, format='%Y-%m-%d') + pd.DateOffset(seconds = n_data*self.dt)
        if zone       == None: zone       = self.bidding_zone

        start_date = pd.to_datetime(start_date, format='%Y-%m-%d')

        # Remove dates before simdate
        est_prices_full_set = self.estimated_prices_data.copy()

        est_prices_working_set = est_prices_full_set[
            (est_prices_full_set['Start Time']   >= start_date)    &
            (est_prices_full_set['Start Time']   <  end_date) 
            ]
        
        estimated_clearing_prices_up   = np.array(est_prices_working_set[f'{zone} Up Price']).reshape((1,-1))
        estimated_clearing_prices_down = np.array(est_prices_working_set[f'{zone} Down Price']).reshape((1,-1))

        return estimated_clearing_prices_up, estimated_clearing_prices_down

    def get_predicted_clearing_prices(self, current_MTU=None, n_data = None, zone = None, plot = False):

        if current_MTU == None: current_MTU = self.date
        if n_data     == None: n_data     = self.N
        # if end_date   == None: end_date   = pd.to_datetime(start_date, format='%Y-%m-%d') + pd.DateOffset(seconds = n_data*self.dt)
        if zone       == None: zone       = self.bidding_zone

        current_MTU = pd.to_datetime(current_MTU, format='%Y-%m-%d')


        max_xlag = self.clearing_price_estimator.max_xlag
        max_ylag = self.clearing_price_estimator.max_ylag
        max_lag = self.clearing_price_estimator.max_lag

        
        start_date  = current_MTU - pd.DateOffset(seconds = self.dt * max_lag)
        end_date    = current_MTU + pd.DateOffset(seconds = self.dt * n_data)
        # prev_MTU    = current_MTU - pd.DateOffset(seconds = self.dt)

        past_clearing_prices = self.get_market_data(start_date=start_date, end_date=current_MTU, zone=zone, 
                                     times= False, clearing_prices=True)
        past_and_future_spot_prices = self.get_market_data(start_date=start_date, end_date=end_date, zone=zone, 
                                     times= False, spot_prices=True)
        true_clearing_prices = self.get_market_data(start_date=current_MTU, end_date=end_date, zone=zone, 
                                     times= False, clearing_prices=True)

        predicted_prices = self.clearing_price_estimator.calculate_future_prediction(true_clearing_prices, past_clearing_prices, past_and_future_spot_prices)

        predicted_prices_up   = np.array(predicted_prices[f'{zone} Up Price']).reshape((1,-1))
        predicted_prices_down = np.array(predicted_prices[f'{zone} Down Price']).reshape((1,-1))

        if plot:
            previous_prices_up      = np.array(past_clearing_prices[f'{zone} Up Price']).reshape((1,-1))
            previous_prices_down    = np.array(past_clearing_prices[f'{zone} Down Price']).reshape((1,-1))

            actual_prices_up    =  np.array(true_clearing_prices[f'{zone} Up Price']).reshape((1,-1))
            actual_prices_down  =  np.array(true_clearing_prices[f'{zone} Down Price']).reshape((1,-1))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)


            t = np.arange(len(previous_prices_up.flatten())+len(actual_prices_up.flatten()))

            ax1.step(t, np.hstack((previous_prices_up, actual_prices_up)).flatten(),      label='True Up Price',        color='slategray', alpha=1, where='post')
            ax1.step(t, np.hstack((previous_prices_up, predicted_prices_up)).flatten(),   label='Predicted Up Price',   color='blue',      alpha=1, where='post')
            ax1.axvline(x=len(previous_prices_up.flatten()), linestyle=':', alpha=0.5)

            ax2.step(t, np.hstack((previous_prices_down, actual_prices_down)).flatten(),    label='True Down Price',      color='slategray', alpha=1, where='post')
            ax2.step(t, np.hstack((previous_prices_down, predicted_prices_down)).flatten(), label='Predicted Down Price', color='red',       alpha=1, where='post')
            ax2.axvline(x=len(previous_prices_up.flatten()), linestyle=':', alpha=0.5)

            ax1.set_title(f"Clearing prices up")
            ax2.set_title(f"Clearing prices down")
            fig.suptitle(f"{self.market_type} Price prediction performances")

            plt.show()



        return predicted_prices_up, predicted_prices_down



    def get_estimated_clearing_price_variances(self, zone = None):

        if zone       == None: zone       = self.bidding_zone

        return self.estimated_price_variances[zone]['Up'], self.estimated_price_variances[zone]['Down']


    def get_activations(self, date = None, n_days = None, zone = None):
        '''
        Returns numpy arrays of length N with Capacity market reservations during each quarter hour from the start time.
        For every MTU, a 1 indicates that a reservation was made and a 0 indicates that no reservation was made.
        Start time is always assumed at 00:00 at the given start date.
        '''

        if zone     == None: zone   = self.bidding_zone

        data = self.get_market_data(start_date=date, n_days=n_days, zone=zone, 
                                     times=True, activations=True)
        
        activations_up, activations_down = np.array(data[f'{zone} Activated Up']), np.array(data[f'{zone} Activated Down'])

        return activations_up, activations_down


    def perform_price_estimation(self, estimator_config):

        dep_lags   = estimator_config['dep_lags']
        indep_lags = estimator_config['indep_lags']

        max_lag = max(dep_lags + indep_lags)

        training_window = estimator_config['training_window']

        # if training_window[0] < max_lag
        
        date        = pd.to_datetime(self.date, format='%Y-%m-%d')
        start_date  = date - pd.DateOffset(minutes= max_lag*15)
        end_date    = date + pd.DateOffset(self.T)

        training_start_date =  date + pd.DateOffset(minutes= 15 * training_window[0])
        training_end_date =  date + pd.DateOffset(minutes= 15 * training_window[1])


        timeslots           = self.get_market_data(start_date = start_date, end_date = end_date, zone = self.bidding_zone,
                                                   times = True)
        dependent_true_data      = self.get_market_data(start_date = start_date, end_date = end_date, zone = self.bidding_zone,
                                                   clearing_prices=True)
        independent_input_data    = self.get_market_data(start_date = start_date, end_date = end_date, zone = self.bidding_zone,
                                                   spot_prices=True)

        dependent_training_data      = self.get_market_data(start_date = training_start_date, end_date = training_end_date, zone = self.bidding_zone,
                                                   clearing_prices=True)
        independent_training_data    = self.get_market_data(start_date = training_start_date, end_date = training_end_date, zone = self.bidding_zone,
                                                   spot_prices=True)


        clearing_price_estimator = EstimatorDF(f"{self.market_type} Clearing Price Estimator", dependent_true_data, dependent_training_data, independent_input_data, independent_training_data, dep_lags, indep_lags, exact=self.exact_estimation)

        estimated_data = pd.concat([timeslots, clearing_price_estimator.estimated_df], axis=1).dropna().round(1)
        self.estimated_prices_data = estimated_data
        

        estimated_price_variances = {}
        for col in clearing_price_estimator.estimated_df.columns:
            zone      = col.split()[0]
            estimated_price_variances[zone] = {}
        
        for col in clearing_price_estimator.estimated_df.columns:
            zone      = col.split()[0]
            direction = col.split()[1]
            estimated_price_variances[zone][direction] = clearing_price_estimator.conditional_variance[f'{zone} {direction} Price']
        
        self.estimated_price_variances = estimated_price_variances
    
        self.clearing_price_estimator = clearing_price_estimator
        clearing_price_estimator.show_estimator_profile()
        clearing_price_estimator.measure_performance()


    def generate_activation_times(self, activation_chance=1, hourly_roll=False):
        dataset = self.market_data_full_set.copy()

        np.random.seed(self.seed)

        time_col = 'Start Time'
        if time_col not in dataset.columns:
            raise ValueError(f"'{time_col}' column is required.")

        for col in dataset.columns:
            if "Volume" in col:
                zone, direction, _ = col.split()
                new_col_name = f"{zone} Activated {direction}"

                # Identify time slots where activation is possible (volume > 0)
                eligible = dataset[col] > 0

                # Default to zero activation
                activation = np.zeros(len(dataset), dtype=int)

                if hourly_roll:
                    # Roll once per hour, but only where volume > 0
                    hours = dataset[time_col].dt.floor('H')
                    dataset['__hour'] = hours  # Temporary helper column
                    eligible_hours = hours[eligible]
                    unique_eligible_hours = eligible_hours.unique()
                    hour_randoms = {
                        hour: np.random.rand() < activation_chance for hour in unique_eligible_hours
                    }
                    random_activation = hours.map(hour_randoms).fillna(False).astype(int)
                    activation = random_activation * eligible.astype(int)
                    dataset.drop(columns='__hour', inplace=True)  # Clean up
                else:
                    # Roll per time slot, but only apply where eligible
                    random_draws = np.random.rand(len(dataset)) < activation_chance
                    activation = (random_draws & eligible).astype(int)

                dataset[new_col_name] = activation

        self.market_data_full_set = dataset
        return


    def get_market_data(self, start_date=None, end_date=None, n_days = None, zone = None, all = False, times = False, spot_prices=False, clearing_prices=False, volumes=False, activations=False):
        '''
        Fetch a slice form the full data set containing specified data types.
        
        returns a dataframe containing fetched data. 
        '''

        if start_date == None: start_date = self.date
        if n_days     == None: n_days     = self.T
        if end_date   == None: end_date   = pd.to_datetime(start_date, format='%Y-%m-%d') + pd.DateOffset(n_days)
        if zone       == None: zone       = self.bidding_zone

        start_date = pd.to_datetime(start_date, format='%Y-%m-%d')

        # Remove dates before simdate
        market_data_full_set = self.market_data_full_set.copy()

        market_data_working_set = market_data_full_set[
            (market_data_full_set['Start Time']   >= start_date)    &
            (market_data_full_set['Start Time']   <  end_date) 
            ]
        
        keywords = []
        if times            or all: keywords += ['Start Time']
        if spot_prices      or all: keywords += ['spot']
        if clearing_prices  or all: keywords += ['up price', 'down price']
        if volumes          or all: keywords += ['volume']
        if activations      or all: keywords += ['activated']
        columns = [col for col in market_data_working_set.columns if any([keyword.lower() in col.lower() for keyword in keywords])]

        if zone.lower() != 'all':
            new_columns = [col for col in columns if zone in col or 'start time' in col.lower()]
            columns = new_columns

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
            activations (np.ndarray): [2, N] array of 1s and 0s indicating when a bid was activated
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

        # clearing_prices = self.get_clearing_prices(date = MTU_start)

        # Loop over each time period
        for i in range(n_periods):
            current_time = MTU_start + pd.Timedelta(minutes=i*15)

            # Skip if no matching market data
            if current_time not in market_data.index:
                continue

            market_row          = market_data.loc[current_time]
            spot_price          = market_row[f'{zone} Spot Price']
            market_volume_up    = market_row[f'{zone} Up Volume']
            market_volume_down  = market_row[f'{zone} Down Volume']
            market_price_up     = market_row[f'{zone} Up Price']
            market_price_down   = market_row[f'{zone} Down Price']
            market_activation_up   = market_row[f'{zone} Activated Up']
            market_activation_down = market_row[f'{zone} Activated Down']
            
            # Extract bids
            bid_price_up    = Bid_prices[0, i]
            bid_price_down  = Bid_prices[1, i]
            bid_volume_up   = Bid_volumes[0, i]
            bid_volume_down = Bid_volumes[1, i]

            # Determine activations
            if market_volume_up > 0 and bid_price_up <= market_price_up and market_activation_up:
                activations[0, i] = 1
                activated_volumes[0, i] = bid_volume_up
                # earnings[0, i] = bid_volume_up * (market_price_up - spot_price) / 4
                earnings[0, i] = bid_volume_up * (market_price_up) / 4

            if market_volume_down > 0 and bid_price_down <= market_price_down and market_activation_down:
                activations[1, i] = 1
                activated_volumes[1, i] = bid_volume_down
                # earnings[1, i] = bid_volume_down * (spot_price - market_price_down) / 4
                earnings[1, i] = bid_volume_down * (market_price_down) / 4

        return activations, activated_volumes, earnings