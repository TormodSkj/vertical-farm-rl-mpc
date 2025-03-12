import matplotlib.pyplot as plt
from tabulate import tabulate
import os
import numpy as np
import json
import hashlib
import pandas as pd
import casadi as ca
import scipy as sp
from globals import *
import requests
from bs4 import BeautifulSoup
import csv
import time
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime, timedelta
import sys


class Estimator:
    '''
    Estimator assumes a stochastic variable x which has some self similarity and also some covariance with another signal. 
    The distribution of x is assumed dependent on past values of x
    The distribution of x is also assumed dependent on values for y

    The estimator trains on datasets of x and y, analyzing covariance and autocorrelations.
    
    '''

    exact: bool
    n_xlags: int

    sample_data: np.ndarray
    estimated_data: np.ndarray
    
    nx = 0
    ny = 0
    
    n_data: int

    covariances: np.ndarray
    means: np.array

    def __init__(self, dependent_data: np.ndarray, independent_data: np.ndarray = None, 
                 x_labels = [], y_labels = [], n_xlags = 0, n_ylags=0, is_exact = False):

        dependent_data, independent_data, x_labels, y_labels = self.sanitize_inputs(dependent_data, independent_data, x_labels, y_labels)
        if x_labels: assert len(x_labels) == dependent_data.shape[0]
        if y_labels: assert len(y_labels) == independent_data.shape[0]

        self.exact = is_exact
        self.n_xlags = n_xlags
        self.n_ylags = n_ylags
        self.max_lags = max(n_xlags, n_ylags)
        self.min_lags = min(n_xlags, n_ylags)
        self.sample_data = dependent_data
        self.n_data = dependent_data.shape[1]

        self.x_labels = x_labels
        self.y_labels = y_labels
        
        self.add_sample(X_sample=dependent_data, Y_sample=independent_data)
        self.calculate_estimate(Y_sample = independent_data)


    def __getitem__(self, index):
        return self.estimated_data[index]

    def __setitem__(self, index, value):
        self.estimated_data[index] = value

    def mean(self):
        return np.mean(self.estimated_data)

    def variance(self):
        return np.var(self.estimated_data)

    def __len__(self):
        return len(self.estimated_data)

    def __repr__(self):
        return f"EstimatingArray({self.estimated_data})"



    def sanitize_inputs(self, dependent_data: np.ndarray, independent_data: np.ndarray = None,  x_labels=[], y_labels=[]):

        if len(dependent_data.shape) == 1: dependent_data = dependent_data.reshape((1,-1))
        
        if independent_data is None:         independent_data = np.zeros((0, dependent_data.shape[1]))
        if len(independent_data.shape) == 1: independent_data = independent_data.reshape((1,-1))
        
        if type(x_labels) == str: x_labels = [x_labels]
        if type(y_labels) == str: y_labels = [y_labels]

        return dependent_data, independent_data, x_labels, y_labels

    def add_sample(self, X_sample: np.ndarray, Y_sample: np.ndarray):

        assert X_sample.shape[1] == Y_sample.shape[1],  f"Inconsistent lengths of dependent and independent data"
        assert X_sample.shape[1] > self.n_xlags - 1,     f"Sample data is too short for choice of lag variables"
        assert Y_sample.shape[1] == self.n_data,        f"Size inconsistency when adding input signal. Expected length {self.n_data}, received length{Y_sample.shape[1]} "
        
        self.ny = Y_sample.shape[0]
        self.nx = X_sample.shape[0]

        self.update_covariances(Y_sample)
        

    def update_covariances(self, Y_sample):

        lagged_xdata = np.zeros((self.nx*self.n_xlags, self.n_data - self.n_xlags))

        for k in range(1, self.n_xlags+1):
            lagged_xdata[self.nx*(k-1):self.nx*k, :] = self.sample_data[:,self.n_xlags-k:self.n_data-k]


        lagged_ydata = np.zeros((self.ny*self.n_ylags, self.n_data - self.n_ylags))
        for k in range(1, self.n_ylags+1):
            lagged_ydata[self.ny*(k-1):self.ny*k, :] = Y_sample[:,self.n_ylags-k:self.n_data-k]

        all_signals     = np.vstack((self.sample_data[:,self.max_lags:],
                                    lagged_xdata[:,self.max_lags - self.n_xlags:], 
                                    Y_sample[:,self.max_lags:],
                                    lagged_ydata[:,self.max_lags - self.n_ylags:])) 
        
        self.covariances = np.cov(all_signals)
        self.means = np.mean(all_signals, axis=1)
        # self.means[self.nx:self.nx+self.n_lags*self.nx] = np.repeat(self.means[:self.nx], self.n_lags)
    

    def build_stable_covariances(self, threshold=1e8):
        """
        Incrementally builds a well-conditioned covariance matrix Pyy,
        skipping lag variables that make it numerically unstable.
        
        Args:
            y_lagged: Matrix of lagged y values (shape: [n_samples, n_lags])
            x_lagged: Corresponding lagged x values (for Pxy update)
            threshold: Condition number threshold for stability
        
        Returns:
            Pyy: Well-conditioned covariance matrix
            Pxy: Corresponding cross-covariance matrix
            selected_lags: List of indices of selected lags
        """
        n = self.nx + self.n_xlags + self.ny + self.n_ylags
        selected_lags = []

        full_covariances = self.covariances[self.nx:, self.nx:]

        Pyy = np.zeros((0,0))
        Pxy = np.zeros((0, self.n_xlags + self.ny + self.n_ylags))  # Matching empty cross-matrix

        for var in range(n - self.nx):

            new_col = full_covariances[selected_lags + [var], var].reshape(-1, 1)
            new_row = new_col.T

            # Pyy_candidate = self.covariances[self.nx:self.nx+lag, self.nx:self.nx+lag]
            Pyy_candidate = np.block([
                [Pyy, new_col[:-1,:]],
                [new_row[:,:-1], new_row[-1,-1]]
            ])
            
            # Compute condition number
            cond_number = np.linalg.cond(Pyy_candidate)

            if cond_number < threshold:
                Pyy = Pyy_candidate  # Accept new column/row
                selected_lags.append(var)

                # # Update Pxy to match
                # new_pxy_col = full_covariances[:, lag].reshape(-1, 1)
                # Pxy = np.hstack((Pxy, new_pxy_col)) if Pxy.size else new_pxy_col
                # Pxy = Pxy.reshape((self.nx, -1))
            
        
        Pxy = self.covariances[:self.nx, np.array(selected_lags) + self.nx]

        return Pyy, Pxy, selected_lags


    def calculate_estimate(self, Y_sample: np.ndarray):
        
        if self.exact:
            self.estimated_data = self.sample_data
            self.mse = 0
            self.rmse = 0


        # Pxy     = self.covariances[:self.nx,self.nx:]
        # Pyy     = self.covariances[self.nx:,self.nx:]
        
        # print(self.covariances)
        Pxx = self.covariances[:self.nx, :self.nx]
        Pyy, Pxy, selected_vars = self.build_stable_covariances()
        Pyy_inv = np.linalg.inv(Pyy)

        # selected_lags   = [var for var in selected_vars if var >= self.nx and var < self.nx + self.n_lags]
        # selected_y      = [var for var in selected_vars if var >= self.nx + self.n_lags]
        # print(Pxy)
        # print(Pyy)

        a = self.means[:self.nx].reshape((-1, 1))
        b = self.means[selected_vars].reshape((-1, 1))

        # if np.linalg.cond(Pyy) < 1/sys.float_info.epsilon:
        #     Pyy_inv = np.linalg.inv(Pyy)
        # else:
        #     print('Adjusting Pyy to make it non-singular')
        #     Pyy_inv = np.linalg.inv(1e-6*np.eye(*Pyy.shape) + Pyy)
            

        x_est = np.zeros_like(self.sample_data)
        x_est[:,:self.max_lags] = np.repeat(self.means[:self.nx].reshape((-1,1)), self.max_lags, axis=1)
        # x_est[:,:self.n_lags] = self.sample_data[:,:self.n_lags]

        for k in range(self.max_lags, self.n_data):
            
            # if Y_sample is not None:
            y = np.vstack((np.flip(self.sample_data[:,k-self.n_xlags:k], axis=1).ravel(order='F').reshape((-1,1)), 
                           Y_sample[:,k].reshape((-1,1)),
                           np.flip(Y_sample[:,k-self.n_ylags:k], axis=1).ravel(order='F').reshape((-1,1))
                           ))[selected_vars, :]
            # else:
            #     y = x_est[:,k:k+self.n_lags].ravel(order='F').reshape((-1,1))

            conditional_expectation = a + Pxy @ Pyy_inv @ (y - b)
            
            x_est[:,k] = conditional_expectation.flatten()

        self.conditional_covariance  = Pxx - Pxy @ Pyy_inv @ Pxy.T

        # self.estimated_data = np.flip(x_est, axis=0)
        self.estimated_data = x_est

        self.mse = np.mean(np.square(self.estimated_data[:,self.max_lags:] - self.sample_data[:,self.max_lags:]))
        self.rmse = np.sqrt(self.mse)
                                      
        return
    
    def measure_performance(self, how='array'):
        
        if how=='single':
            print(f"RMSE: {rmse} \tWith expected covariance {self.conditional_covariance[i,i]}")

        elif how=='array':
            for i in range(self.nx):
                signal_name = self.x_labels[i] if self.x_labels else f"X{i}"
                rmse = np.sqrt(np.mean(np.square(self.estimated_data[i,self.max_lags:] - self.sample_data[i,self.max_lags:])))
                print(f"RMSE for {signal_name}: {rmse} \tWith expected covariance {self.conditional_covariance[i,i]}")


        return



    '''
    def estimate(self, past_vals: np.ndarray = None, Y_sample: np.ndarray = None):
        
        if len(past_vals.shape) == 1: past_vals = past_vals.reshape((self.nx, -1))
        if len(Y_sample.shape) == 1: Y_sample = Y_sample.reshape((self.ny, -1))

        Pyy, Pxy, selected_vars = self.build_stable_covariances()
        Pyy_inv = np.linalg.inv(Pyy)
        a = self.means[:self.nx].reshape((-1, 1))
        b = self.means[selected_vars].reshape((-1, 1))
        
        x_est = np.zeros((self.nx, self.n_lags + Y_sample.shape[1]))
        x_est[:,:self.n_lags] = past_vals[:,:n_lags]

        for k in range(self.n_lags, self.n_data):
            
            y = np.vstack((np.flip(x_est[:,k-self.n_lags:k].ravel(order='F')).reshape((-1,1)), Y_sample[:,k].reshape((-1,1))))[selected_vars, :]
            
            conditional = a + Pxy @ Pyy_inv @ (y - b)
            
            x_est[:,k] = conditional.flatten()

        return x_est[:,self.n_lags:]
        '''

''' #

# x = np.sin(np.linspace(0, 6*np.pi,signal_length)).reshape((1,-1))
# x = np.tile(np.array([[-1,-1,-2,-2],
                    #   [1,1,2,2]]), int(signal_length/4))


# y = x.repeat(1, axis=0)
# np.random.seed(1133)
# y = y + 0.2*np.random.randn(*y.shape)


signal_length = 100
y = np.linspace(1, 10, signal_length).reshape((1,-1))
x = np.vstack((2*y + np.random.randn(*y.shape),
                -0.2*y + 0.3*np.random.randn(*y.shape)))


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(np.arange(signal_length), x[0,:].flatten(), color='black', label='x', linewidth=3, linestyle=':')
ax2.plot(np.arange(signal_length), x[1,:].flatten(), color='black', label='x', linewidth=3, linestyle=':')

mse = {}

for lag in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    n_lags = lag
    est = Estimator(x, y, n_lags=n_lags, is_exact=False)

    # print(f"MSE: {est.mse}")
    mse[lag] = (est.mse)

    past_vals = np.array([0])

    output = est[:,:]
    # output = est.estimate(past_vals, y)

    ax1.plot(np.arange(signal_length), output[0,:].flatten(), label=f'lag: {lag}')
    ax2.plot(np.arange(signal_length), output[1,:].flatten(), label=f'lag: {lag}')
    ax1.legend()
    ax2.legend()

for lag in mse:
    print(f"MSE for lag {lag}: {mse[lag]}")

plt.show()

# '''


def load_spot_prices(file_path, bidding_zone):
    """
    Load spot prices, parse timestamps, and extract the price column.

    Parameters:
    - file_path: str, path to the CSV file
    - timestamp_col: str, name of the timestamp column

    Returns:
    - pandas DataFrame with 'Timestamp' and 'Spot Price' columns
    """

    # Column names to extract
    timestamp_col = "Dato/klokkeslett"  # Spot price timestamp column
    price_col = bidding_zone  # Spot price column of interest
           
    # Load the CSV file with UTF-8 encoding to prevent issues with special characters
    data = pd.read_csv(file_path, delimiter=";", encoding="utf-8")
    
    # Clean timestamp format (remove 'Kl.' and split by '-')
    data['Start Time'] = data[timestamp_col].str.replace("Kl. ", "", regex=False)  # Remove "Kl. "
    
    # Split the timestamp into date and time components (first part of '01-02' becomes '01')
    data['Start Time'] = data['Start Time'].apply(lambda x: x.split(" ")[0] + " " + x.split(" ")[1].split("-")[0] + ":00")
    
    # Convert the string to a datetime object (with date and hour set to the first hour of the range)
    data['Start Time'] = pd.to_datetime(data['Start Time'], format='%Y-%m-%d %H:%M', errors='coerce')
    
    # Return the relevant columns with the 'Spot Price' column renamed
    return data[['Start Time', price_col]].rename(columns={price_col: 'Spot Price'})



import re

def load_spot_data(data_folder):
    """
    Load spot prices from NO, SE, DK and FI from data gathered via energy-charts.info

    Parameters:
    - file_path: str, path to the CSV file
    - timestamp_col: str, name of the timestamp column

    Returns:
    - pandas DataFrame with 'Spot Price' columns
    """

    all_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    all_data = []

    for file in all_files:
        file_path = os.path.join(data_folder, file)
        
        # Read CSV with correct delimiter
        data = pd.read_csv(file_path, delimiter=";", encoding="utf-8")
        
        # Extract 'Start Time' correctly
        date_col = next((col for col in data.columns if 'date' in col.lower()), None)
        if date_col is None:
            raise ValueError(f"Date column not found in {file}")
        
        data[date_col] = data[date_col].str.extract(r"(\d{4}-\d{2}-\d{2}T\d{2}):\d{2}")
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')

        auction_cols = [col for col in data.columns if 'day ahead auction' in col.lower()]
        rename_dict = {date_col: 'Start Time'}
        rename_dict.update({col: col if (match := re.search(r"\((.*?)\)", col)) is None else match.group(1) for col in auction_cols})
        data.rename(columns=rename_dict, inplace=True)

        drop_columns = ['Renewable', 'Non-Renewable', 'Nuclear']

        # Convert drop_columns to lowercase to ensure case-insensitive matching
        drop_columns = [word.lower() for word in drop_columns]

        # Drop exact matches (case insensitive)
        data.drop(columns=[col for col in data.columns if col.lower() in drop_columns], inplace=True) 

        all_data.append(data)


    merged_df = combine_dataframe_blocks(all_data)

    # return_df = merged_df[['Start Time', bidding_zone]].rename(columns={bidding_zone: 'Spot Price'})

    return_df = merged_df.rename(
        columns={
            col: col + ' Spot Price' for col in merged_df.columns if 'Start Time' not in col
            }
        )

    return return_df



def load_mfrr_balancing_prices(data_folder, bidding_zone):
    """
    Load activation market balancing prices from all relevant files in a folder, parse timestamps,
    and merge them into a single DataFrame with 'Start Time', 'End Time', 'Up Price',
    and 'Down Price' columns.

    Parameters:
    - data_folder: str, path to the folder containing CSV files.

    Returns:
    - pandas DataFrame: Merged dataset sorted by 'Start Time'.
    """

    # standardize_balancing_price_files(data_folder)

    all_files = [f for f in os.listdir(data_folder) if "mFRR_balancing_prices" in f and f.endswith(".csv")]
    all_data = []

    for file in all_files:
        filepath = os.path.join(data_folder, file)

        # Load the mFRR data
        data = pd.read_csv(filepath, delimiter=";", encoding="utf-8")

        # Parse 'Start Time' and 'End Time' from the 'Time Interval' column
        time_interval_col = "Date/Time CET/CEST"  # Adjust if your column name is different
        
        data = data[data['MBA'] == bidding_zone]

        data[['Start Time']] = data[time_interval_col].str.extract(
            r'(\d{2}.\d{2}.\d{4}/\d{2}:\d{2})'
        )
        data['Start Time'] = pd.to_datetime(data['Start Time'], format='%d.%m.%Y/%H:%M', errors='coerce')

        up_price_col = "Up Regulation Price [EUR/MWh]"     
        down_price_col = "Down Regulation Price [EUR/MWh]" 

        data = data[['Start Time', up_price_col, down_price_col]].rename(
            columns={up_price_col: 'Clearing Price Up', down_price_col: 'Clearing Price Down'}
        )

        data['Clearing Price Up']   = pd.to_numeric(data['Clearing Price Up'].str.replace(',', '.'), errors='coerce')
        data['Clearing Price Down'] = pd.to_numeric(data['Clearing Price Down'].str.replace(',', '.'), errors='coerce')
        
        # Append processed data to the list
        all_data.append(data)

    # Merge all data and sort by 'Start Time'
    merged_data = pd.concat(all_data, ignore_index=True)
    merged_data.sort_values(by='Start Time', inplace=True)
    # merged_data.fillna(0, inplace=True)

    return merged_data



def load_nordpool_balancing_prices(data_folder, bidding_zone):
    """

    """

    mfrr_AM_data = load_mfrr_AM_data(data_folder)

    up_price_col    = f"{bidding_zone} Up Price (EUR)"     
    down_price_col  = f"{bidding_zone} Down Price (EUR)" 

    mfrr_AM_data = mfrr_AM_data[['Start Time', up_price_col, down_price_col]].rename(
        columns={up_price_col: 'Clearing Price Up', down_price_col: 'Clearing Price Down'}
    )

    mfrr_AM_data['Clearing Price Up']   = pd.to_numeric(mfrr_AM_data['Clearing Price Up'],   errors='coerce')
    mfrr_AM_data['Clearing Price Down'] = pd.to_numeric(mfrr_AM_data['Clearing Price Down'], errors='coerce')
    

    return mfrr_AM_data


def load_mfrr_AM_data(data_folder):
    """

    """

    all_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    all_data = []

    for file in all_files:
        filepath = os.path.join(data_folder, file)

        # Load the mFRR data
        data = pd.read_csv(filepath, delimiter=";", encoding="utf-8")

        # data = data[['Delivery Start (CET)'] + [column for column in data.columns if bidding_zone in column]]

        data[['Start Time']] = data['Delivery Start (CET)'].str.extract(
            r'(\d{2}.\d{2}.\d{4} \d{2}:\d{2}:\d{2})'
        )
        data['Start Time'] = pd.to_datetime(data['Start Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

        data.drop(columns=['Delivery Start (CET)', 'Delivery End (CET)'])

        # Append to the list
        all_data.append(data)
    
    # Concatenate all data into a single DataFrame
    combined_df = combine_dataframe_blocks(all_data)
    rename_dict = {}
    rename_dict.update({col: col[:-5] for col in combined_df.columns if 'Accepted' in col})
    rename_dict.update({col: col[:-5] for col in combined_df.columns if 'Activated' in col})
    rename_dict.update({col: col[:-6] for col in combined_df.columns if 'Price' in col})
    combined_df.rename(columns = rename_dict, inplace=True)

    combined_df.drop(columns = ['Delivery Start (CET)', 'Delivery End (CET)'], inplace=True)
    combined_df.drop(columns = [col for col in combined_df.columns if 'Imbalance' in col], inplace=True)

    return combined_df


def load_nordpool_activation_data(data_folder, bidding_zone):
    """

    """
    AM_data_df = load_mfrr_AM_data(data_folder, bidding_zone)

    # Convert 'Offered' and 'Activated' to numeric
    AM_data_df[['Offered Up', 'Activated Up']] = AM_data_df[[f"{bidding_zone} Accepted Up Volume (MW)" , f"{bidding_zone} Activated Up Volume (MW)" ]].apply(pd.to_numeric, errors='coerce')
    AM_data_df[['Offered Down', 'Activated Down']] = AM_data_df[[f"{bidding_zone} Accepted Down Volume (MW)" , f"{bidding_zone} Activated Down Volume (MW)" ]].apply(pd.to_numeric, errors='coerce')
    
    AM_data_df = AM_data_df[['Start Time', 'Offered Up', 'Activated Up', 'Offered Down', 'Activated Down']]

    return AM_data_df


# def combine_dataframe_blocks(blocks):
#     if not blocks:
#         raise ValueError("No dataframes provided for merging.")

#     # Ensure all blocks have 'Start Time' column and convert it to datetime
#     for i, df in enumerate(blocks):
#         if 'Start Time' not in df.columns:
#             raise ValueError(f"Block {i} is missing the required 'Start Time' column.")
#         df['Start Time'] = pd.to_datetime(df['Start Time'], errors='coerce')

#     # Concatenate all blocks (this keeps all columns)
#     full_df = pd.concat(blocks, axis=0, ignore_index=True)

#     # Group by 'Start Time' and merge overlapping data
#     full_df = full_df.groupby('Start Time', as_index=False).first()

#     # Sort chronologically
#     full_df = full_df.sort_values(by='Start Time').reset_index(drop=True)

#     return full_df

def combine_dataframe_blocks(blocks):
    if not blocks:
        raise ValueError("No dataframes provided for merging.")

    # Ensure all blocks have 'Start Time' column and convert it to datetime
    for i, df in enumerate(blocks):
        if 'Start Time' not in df.columns:
            raise ValueError(f"Block {i} is missing the required 'Start Time' column.")
        df['Start Time'] = pd.to_datetime(df['Start Time'], errors='coerce')

    # Concatenate all blocks (this keeps all columns)
    full_df = pd.concat(blocks, axis=0, ignore_index=True)

    # Group by 'Start Time' and merge overlapping data
    full_df = full_df.groupby('Start Time', as_index=False).first()

    # Sort columns: Keep 'Start Time' first, sort the rest alphabetically
    sorted_columns = ['Start Time'] + sorted([col for col in full_df.columns if col != 'Start Time'])
    full_df = full_df[sorted_columns]

    # Sort chronologically
    full_df = full_df.sort_values(by='Start Time').reset_index(drop=True)

    return full_df


def load_mfrr_CBMP_prices(data_folder, bidding_zone):
    """
    Load Cross-border marginal prices from all relevant files in a folder, parse timestamps,
    and merge them into a single DataFrame with 'Start Time', 'End Time', 'Up Price',
    and 'Down Price' columns.

    Parameters:
    - data_folder: str, path to the folder containing CSV files.

    Returns:
    - pandas DataFrame: Merged dataset sorted by 'Start Time'.
    """

    # standardize_balancing_price_files(data_folder)

    all_files = [f for f in os.listdir(data_folder) if "mFRR_balancing_prices" in f and bidding_zone in f and f.endswith(".csv")]
    all_data = []

    for file in all_files:
        filepath = os.path.join(data_folder, file)

        # Load the mFRR data
        data = pd.read_csv(filepath, delimiter=",", encoding="utf-8")

        # Parse 'Start Time' and 'End Time' from the 'Time Interval' column
        time_interval_col = "ISP (CET/CEST)"  # Adjust if your column name is different
        data[['Start Time', 'End Time']] = data[time_interval_col].str.extract(
            r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - (\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})'
        )
        data['Start Time'] = pd.to_datetime(data['Start Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        data['End Time'] = pd.to_datetime(data['End Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

        up_price_col = "Price Up (EUR/MWh)"     
        down_price_col = "Price Down (EUR/MWh)" 

        data = data[['Start Time', up_price_col, down_price_col]].rename(
            columns={up_price_col: 'Clearing Price Up', down_price_col: 'Clearing Price Down'}
        )
        
        # Append processed data to the list
        all_data.append(data)

    # Merge all data and sort by 'Start Time'
    merged_data = pd.concat(all_data, ignore_index=True)
    merged_data.sort_values(by='Start Time', inplace=True)
    # merged_data.fillna(0, inplace=True)

    return merged_data


def standardize_balancing_price_files(folder_path):
    """
    Detect 'fricked' CSV files in a folder and clean them.

    Parameters:
    - folder_path: str, path to the folder containing CSV files.
    """
    # List all CSV files in the folder
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)

        # Try reading the header to check if the file parses correctly
        df = pd.read_csv(file_path, nrows=0)
        if len(df.columns) == 1:  # Single column implies problematic formatting
            # print(f"File '{file}' is problematic. Fixing it...")

            with open(file_path, "r") as infile:
                lines = infile.readlines()

            # Remove enclosing quotation marks and replace doubled quotes
            cleaned_lines = []
            for line in lines:
                line = line.strip()

                # Remove the outermost quotation marks if they exist
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]  # Slice to remove first and last characters

                # Replace doubled quotes with single quotes
                line = line.replace('""', '"')

                cleaned_lines.append(line)

            # Write cleaned lines back to the file
            with open(file_path, "w") as outfile:
                outfile.write("\n".join(cleaned_lines) + "\n")  # Re-add a final newline

            # print(f"Successfully cleaned: {file}")
        # else:
            # print(f"File '{file}' is properly formatted. Skipping...")

        # Read the cleaned file
        df = pd.read_csv(file_path)

        # Clean price columns with commas, convert to float
        price_columns = ['Price Up (EUR/MWh)', 'Price Down (EUR/MWh)']  # Assuming these are the price columns
        for col in price_columns:
            df[col] = df[col].replace({',': ''}, regex=True)  # Remove commas
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric values

        # Save the cleaned file back
        df.to_csv(file_path, index=False)

        # print(f"Successfully cleaned and standardized: {file}")



def clean_mfrr_csv_file(filepath):
    temp_filepath = filepath + ".tmp"  # Create a temporary file path

    # Process the file safely
    with open(filepath, "r") as infile, open(temp_filepath, "w") as outfile:
        for line in infile:
            # Remove quotation marks at the start and end of the line
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            # Replace two consecutive quotation marks with a single quotation mark
            line = line.replace('""', "'")
            
            # Write the cleaned line to the temp file
            outfile.write(line + "\n")
    
    # Replace the original file with the cleaned one
    os.replace(temp_filepath, filepath)
    print(f"File cleaned successfully: {filepath}")


def load_mfrr_activation_data(data_folder, bidding_zone):
    """
    Load mFRR activation data from all relevant files in a folder, merge all data for upward 
    and downward activations into a single DataFrame, and sort them by date.

    Parameters:
    - data_folder: str, path to the folder containing CSV files.
    - bidding_zone: str, filter files by bidding zone.

    Returns:
    - A pandas DataFrame with columns: ['Start Time', 'Offered Up', 'Activated Up', 
      'Offered Down', 'Activated Down'].
    """
    all_files = [
        f for f in os.listdir(data_folder)
        if "mFRR_activations" in f and bidding_zone in f and f.endswith(".csv")
    ]
    
    all_data = []

    for file in all_files:
        filepath = os.path.join(data_folder, file)
        
        # Load the data
        data = pd.read_csv(filepath, quotechar='"', skipinitialspace=True)
        
        # Clean column names (remove extra quotes and whitespace)
        data.columns = data.columns.str.replace("'", '').str.strip()
        data = data.apply(lambda x: x.str.replace("'", '').str.strip() if x.dtype == "object" else x)
        
        # Extract 'Start Time' from ISP column
        data['Start Time'] = data['ISP'].str.extract(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})')
        data['Start Time'] = pd.to_datetime(data['Start Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        
        # Rename AREA to Bidding Zone and simplify the zone names
        data.rename(columns={'Area': 'Bidding Zone'}, inplace=True)
        data['Bidding Zone'] = data['Bidding Zone'].str.replace(' SCA', '')
        
        # Remove unnecessary columns
        data.drop(columns=['ISP', 'Reserve Type', 'Type of Product', 'Unavailable (MW)'], inplace=True)
        
        # Convert 'Offered' and 'Activated' to numeric
        data[['Offered', 'Activated']] = data[['Offered (MW)', 'Activated (MW)']].apply(pd.to_numeric, errors='coerce')
        
        # Separate upward and downward activations
        up_data = data[data['Direction'] == "Up"].reset_index(drop=True).rename(columns={
            'Offered': 'Offered Up',
            'Activated': 'Activated Up'
        })
        down_data = data[data['Direction'] == "Down"].reset_index(drop=True).rename(columns={
            'Offered': 'Offered Down',
            'Activated': 'Activated Down'
        })
        
        # Merge upward and downward activations on 'Start Time'
        merged_data = pd.merge(
            up_data[['Start Time', 'Offered Up', 'Activated Up']],
            down_data[['Start Time', 'Offered Down', 'Activated Down']],
            on='Start Time',
            how='outer'
        )
        
        # Append to the list
        all_data.append(merged_data)
    
    # Concatenate all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by Start Time
    combined_df.sort_values(by='Start Time', inplace=True)
    
    return combined_df



def merge_and_align(spot_prices, mfrr_prices):
    """
    Merge spot prices and mFRR prices on their timestamps.

    Parameters:
    - spot_prices: pandas DataFrame with spot price data
    - mfrr_prices: pandas DataFrame with mFRR price data

    Returns:
    - pandas DataFrame with aligned data
    """
    # Merge the two DataFrames on the 'Timestamp' column, ensuring alignment
    merged_data = pd.merge(spot_prices, mfrr_prices, on='Start Time', how='inner')
    
    return merged_data



def fetch_CM_data_nucs(target_file_path, start_date, end_date):
    """
    Fetch balancing reserve data for a range of dates, parse HTML tables,
    and store the data in a CSV file
    """

    base_url = "https://www.nucs.net/balancing/r2/pricesAndVolumesOfProcuredBalancingReserve/show"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    def fetch_data(url, direction_prefix):
        """Fetch and parse HTML data."""
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            table = soup.find("table", {"id": "pricesAndVolumesOfProcuredBalancingReserve"})

            if not table:
                print(f"Table not found for {url}")
                return {}

            structured_data = defaultdict(lambda: {})

            bidding_zone_row = table.find_all("tr")[0]
            bidding_zone_cols = list(np.repeat(
                [th.text.strip() for th in bidding_zone_row.find_all("th")][2:], 2
            ))

            headers_row = table.find_all("tr")[2]
            headers_cols = [th.text.strip()[:-10].strip() for th in headers_row.find_all("th")]

            new_header_cols = ['Date', 'Hour'] + [
                f"{bidding_zone_cols[i]} {direction_prefix} {headers_cols[i]}" for i in range(len(headers_cols))
            ]

            zone_indices = {zone.lstrip('MBA|'): i for i, zone in enumerate(new_header_cols) if 'MBA|' in zone}

            # new_header_cols = [str(header_col).lstrip('MBA|') for header_col in new_header_cols]

            for tr in table.find_all("tr")[3:]:  
                cols = tr.find_all("td")
                row_data = [td.text.strip() for td in cols]

                if len(row_data) < 3:
                    continue  

                date = row_data[0]
                hour = row_data[1]

                for zone, idx in zone_indices.items():
                    value = None if row_data[idx] in ["", "N/A"] else float(row_data[idx])
                    structured_data[hour][zone] = value

            return structured_data

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return {}

    # Load existing data
    try:
        existing_df = pd.read_csv(target_file_path, delimiter=';')
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    all_new_data = []

    current_date = datetime.strptime(start_date, "%d-%m-%Y")
    end_date_dt = datetime.strptime(end_date, "%d-%m-%Y")
    total_days = int((end_date_dt - current_date).days) + 1

    with tqdm(total=total_days, desc=f"Fetching data from nucs platform") as pbar:
        while current_date <= end_date_dt:
            date_str = current_date.strftime("%d.%m.%Y")
            pbar.set_postfix(status=f"Fetching data from {date_str}")

            query_params = (
                f"?name=&defaultValue=false&viewType=TABLE&areaType=MBA&atch=false"
                f"&dateTime.dateTime={date_str}+00:00|CET|DAYTIMERANGE"
                f"&dateTime.endDateTime={date_str}+00:00|CET|DAYTIMERANGE"
                f"&areaSelectType=USER_SELECTED"
                f"&marketArea.values=CTY|10YNO-0--------C!MBA|10YNO-1--------2"
                f"&marketArea.values=CTY|10YNO-0--------C!MBA|10YNO-2--------T"
                f"&marketArea.values=CTY|10YNO-0--------C!MBA|10YNO-3--------J"
                f"&marketArea.values=CTY|10YNO-0--------C!MBA|10YNO-4--------9"
                f"&marketArea.values=CTY|10YNO-0--------C!MBA|10Y1001A1001A48H"
                f"&marketArea.values=CTY|10Y1001A1001A65H!MBA|10YDK-1--------W"
                f"&marketArea.values=CTY|10Y1001A1001A65H!MBA|10YDK-2--------M"
                f"&marketArea.values=CTY|10YFI-1--------U!MBA|10YFI-1--------U"
                f"&marketArea.values=CTY|10YSE-1--------K!MBA|10Y1001A1001A44P"
                f"&marketArea.values=CTY|10YSE-1--------K!MBA|10Y1001A1001A45N"
                f"&marketArea.values=CTY|10YSE-1--------K!MBA|10Y1001A1001A46L"
                f"&marketArea.values=CTY|10YSE-1--------K!MBA|10Y1001A1001A47J"
                f"&dataItems.values=PRICE&dataItems.values=VOLUME"
                f"&reserveType.values=A97&balancingTypes=TERTIARY&reserveSource.values=ALL&aFRRmFRRType.values=A47"
            )

            # "https://www.nucs.net/balancing/r2/pricesAndVolumesOfProcuredBalancingReserve/show?name=&defaultValue=false&viewType=TABLE&areaType=MBA&atch=false&dateTime.dateTime=12.06.2024+00:00|CET|DAYTIMERANGE&dateTime.endDateTime=12.06.2024+00:00|CET|DAYTIMERANGE&areaSelectType=USER_SELECTED"
            # "&marketArea.values=CTY|10Y1001A1001A65H!MBA|10YDK-1--------W"
            # "&marketArea.values=CTY|10Y1001A1001A65H!MBA|10YDK-2--------M"
            # "&marketArea.values=CTY|10YFI-1--------U!MBA|10YFI-1--------U"
            # "&marketArea.values=CTY|10YNO-0--------C!MBA|10YNO-1--------2"
            # "&marketArea.values=CTY|10YNO-0--------C!MBA|10YNO-2--------T"
            # "&marketArea.values=CTY|10YNO-0--------C!MBA|10YNO-3--------J"
            # "&marketArea.values=CTY|10YNO-0--------C!MBA|10YNO-4--------9"
            # "&marketArea.values=CTY|10YNO-0--------C!MBA|10Y1001A1001A48H"
            # "&marketArea.values=CTY|10YSE-1--------K!MBA|10Y1001A1001A44P"
            # "&marketArea.values=CTY|10YSE-1--------K!MBA|10Y1001A1001A45N"
            # "&marketArea.values=CTY|10YSE-1--------K!MBA|10Y1001A1001A46L"
            # "&marketArea.values=CTY|10YSE-1--------K!MBA|10Y1001A1001A47J"
            # "&balancingDirection.values=A01&dataItems.values=PRICE&dataItems.values=VOLUME&reserveType.values=A97&balancingTypes=TERTIARY&reserveSource.values=ALL&aFRRmFRRType.values=A47"

            url_up = base_url + query_params + "&balancingDirection.values=A01"
            url_down = base_url + query_params + "&balancingDirection.values=A02"

            up_data = fetch_data(url_up, "Up")
            down_data = fetch_data(url_down, "Down")

            all_hours = sorted(set(up_data.keys()).union(set(down_data.keys())))
            all_columns = set()

            for hour_data in list(up_data.values()) + list(down_data.values()):
                all_columns.update(hour_data.keys())

            sorted_columns = sorted(all_columns)

            combined_data = []
            for hour in all_hours:
                row = {"Date": date_str, "Hour": hour}
                for col in sorted_columns:
                    if up_data.get(hour, {}).get(col, 0) is not None and down_data.get(hour, {}).get(col, 0) is not None:
                        row[col] = up_data.get(hour, {}).get(col, 0) + down_data.get(hour, {}).get(col, 0)
                    else: 
                        row[col] = None
                combined_data.append(row)

            all_new_data.extend(combined_data)
            current_date += timedelta(days=1)


            # Save to the csv file after every fetch
            new_df = pd.DataFrame(combined_data)

            if not new_df.empty:
                # Load existing CSV data (if it exists)
                try:
                    existing_df = pd.read_csv(target_file_path, delimiter=';')

                    # Remove old entries for the current date to avoid duplicates
                    existing_df = existing_df[existing_df["Date"] != date_str]
                except FileNotFoundError:
                    existing_df = pd.DataFrame()

                

                # Append new data and save
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_csv(target_file_path, sep=';', index=False)

                # print(f"Data for {date_str} saved to {target_file_path}")

                # Sort file to ensure it's always chronological
                df = pd.read_csv(target_file_path, delimiter=';')
                df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
                df = df.sort_values(by=["Date", "Hour"])
                df["Date"] = df["Date"].dt.strftime("%d.%m.%Y")
                df.to_csv(target_file_path, index=False, sep=';')
            
            else:
                print(f"WARNING: Data for {date_str} was empty")


            time.sleep(np.random.uniform(0.1, 0.5))
            pbar.update(1)

    
    print(f"Successfully imported data from {start_date} to {end_date}")




def load_mfrr_CM_data(data_folder):

    """

    """

    all_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    all_data = []

    for file in all_files:
        filepath = os.path.join(data_folder, file)

        # Load the mFRR data
        data = pd.read_csv(filepath, delimiter=";", encoding="utf-8")

        # data = data[['Date', 'Hour'] + [column for column in data.columns if bidding_zone in column]]

        data['Start Time'] = data['Date'] + " " + data['Hour']
        data['Start Time'] = pd.to_datetime(data['Start Time'], format='%d.%m.%Y %H:%M', errors='coerce')

        # up_price_col    = f"{bidding_zone} Up Price"     
        # up_volume_col   = f"{bidding_zone} Up Volume procured"     
        # down_price_col  = f"{bidding_zone} Down Price"     
        # down_volume_col = f"{bidding_zone} Down Volume procured"  

        # data = data[['Start Time', up_price_col, up_volume_col, down_price_col, down_volume_col]].rename(
        #     columns={up_price_col: 'Clearing Price Up', down_price_col: 'Clearing Price Down',
        #              up_volume_col: 'Volume Up',        down_volume_col: 'Volume Down'}
        # )

        # data['Clearing Price Up']   = pd.to_numeric(data['Clearing Price Up'],   errors='coerce')
        # data['Clearing Price Down'] = pd.to_numeric(data['Clearing Price Down'], errors='coerce')
        # data['Volume Up']           = pd.to_numeric(data['Volume Up'],           errors='coerce')
        # data['Volume Down']         = pd.to_numeric(data['Volume Down'],         errors='coerce')
        
        data.drop(columns=['Date', 'Hour'],inplace=True)

        # Append processed data to the list
        all_data.append(data)

    assert len(all_data) > 0, 'Expected non-empty list of data. Verify correctly specified import path.'

    # Merge all data and sort by 'Start Time'
    merged_data = combine_dataframe_blocks(all_data)
    # merged_data.sort_values(by='Start Time', inplace=True)
    # merged_data.fillna(0, inplace=True)

    return merged_data
