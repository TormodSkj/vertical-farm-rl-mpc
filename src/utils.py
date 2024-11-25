import matplotlib.pyplot as plt
from tabulate import tabulate
import os
import numpy as np
import json
import hashlib
import pandas as pd


def plotting(t, timeseries, filename, folder = 'plots'):
    #Function to plot timeseries to a given file.

    if(len(timeseries)<1):
        print("Specify one or more timeseries for plotting")
        return

    plt.figure(1)
    for ts in timeseries:
        plt.plot(t[0:len(ts)], ts, "r")

    filename = filename + ".png"
    # plot_path = os.path.join(folder, filename)
    plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
    plt.savefig(plot_path)


def saveplot(filename, foldername, config):
    filename = "Combined_ocp_b_p"
    plt.savefig(config.plot_path + foldername + "/" + filename + ".png")



def generate_table(table_data, header = None, sumrow=False, diffcol=False):
    """
    Creates a table from a list of header items and a list of rows (table_data).
    Adds a sum row at the bottom if `sum=True` and a difference column to the right if `diff=True`.
    """

    np_table_data = np.array([row[1:] for row in table_data])

    if sumrow:
        # Compute the sum for numeric columns, ignoring the first column (labels)
        numeric_sums = ["Sum"] + list(np.sum(np_table_data, axis=0))
        table_data.append(numeric_sums)

    if diffcol:
        # Compute the difference for numeric elements (ignoring the label)
        for row in table_data[:] if sumrow else table_data:
            if len(row) > 2:  # Ensure there's enough data for a difference
                difference = row[-1] - row[1]
                row.append(difference)
            else:
                row.append(None)  # Append `None` if there's not enough data
        
        if header is not None:
            header.append("Difference")
    
    if header is not None:
        return tabulate(table_data, headers=header, tablefmt="grid", floatfmt=".2f")
    else:
        return tabulate(table_data, tablefmt="grid", floatfmt=".2f")



def generate_hash(specs):

    converted_specs = convert_np_arrays_to_lists(specs)

    # Serialize specs consistently
    specs_str = json.dumps(converted_specs, sort_keys=True)
    # Compute and return SHA-256 hash
    return hashlib.sha256(specs_str.encode()).hexdigest()

    
# Convert data for JSON serialization
def convert_np_arrays_to_lists(obj):
    """
    Recursively convert np.array to lists in a nested dictionary or list.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays_to_lists(item) for item in obj]
    else:
        return obj
    

# Convert data to ensure lists become np.array
def convert_lists_to_np_arrays(obj):
    """
    Recursively convert lists to np.array in a nested dictionary or list.
    """
    if isinstance(obj, list):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {key: convert_lists_to_np_arrays(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj  # Keep np.array as-is
    else:
        return obj


def get_metrics_table(runs):
    '''
    Takes in a dict of runs containing metrics dicts. 
    Each run has its own metrics dict, which will be unraveled and displayed here
    '''
    
    any_run = next(iter(runs.values()))
    any_metrics = any_run['metrics']
    n_metrics = len(any_metrics)

    metrics_table = []

    for metric in any_metrics:      # Keys are the same for all metrics dicts regardless of run

        metrics_row = [metric]
        for run in runs:
            
            metrics_row.append(runs[run]['metrics'][metric])

        metrics_table.append(metrics_row)
    

    return generate_table(metrics_table, header = list(runs.keys()), sumrow=False, diffcol=False)

def load_spot_prices(file_path, timestamp_col, price_col):
    """
    Load spot prices, parse timestamps, and extract the price column.

    Parameters:
    - file_path: str, path to the CSV file
    - timestamp_col: str, name of the timestamp column
    - price_col: str, name of the price column

    Returns:
    - pandas DataFrame with 'Timestamp' and 'Spot Price' columns
    """
    # Load the CSV file with UTF-8 encoding to prevent issues with special characters
    data = pd.read_csv(file_path, delimiter=",", encoding="utf-8")
    
    # Clean timestamp format (remove 'Kl.' and split by '-')
    data['Timestamp'] = data[timestamp_col].str.replace("Kl. ", "", regex=False)  # Remove "Kl. "
    
    # Split the timestamp into date and time components (first part of '01-02' becomes '01')
    data['Timestamp'] = data['Timestamp'].apply(lambda x: x.split(" ")[0] + " " + x.split(" ")[1].split("-")[0] + ":00")
    
    # Convert the string to a datetime object (with date and hour set to the first hour of the range)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y-%m-%d %H:%M', errors='coerce')
    
    # Return the relevant columns with the 'Spot Price' column renamed
    return data[['Timestamp', price_col]].rename(columns={price_col: 'Spot Price'})


def load_mfrr_prices(file_path, time_interval_col, up_price_col, down_price_col):
    """
    Load mFRR prices, parse timestamps, and extract up and down price columns.

    Parameters:
    - file_path: str, path to the CSV file
    - time_interval_col: str, name of the time interval column
    - up_price_col: str, name of the up price column
    - down_price_col: str, name of the down price column

    Returns:
    - pandas DataFrame with 'Timestamp', 'Up Price', and 'Down Price' columns
    """
    # Load the mFRR data with UTF-8 encoding
    data = pd.read_csv(file_path, delimiter=",", encoding="utf-8")
    
    # Parse the timestamp from the time interval, taking the first part (start time)
    data['Timestamp'] = pd.to_datetime(
        data[time_interval_col].str.split(" - ").str[0], format='%d.%m.%Y %H:%M', errors='coerce'
    )
    
    return data[['Timestamp', up_price_col, down_price_col]].rename(
        columns={up_price_col: 'Up Price', down_price_col: 'Down Price'}
    )


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
    merged_data = pd.merge(spot_prices, mfrr_prices, on='Timestamp', how='inner')
    
    return merged_data


def calculate_covariance_matrix(data, columns):
    """
    Calculate the covariance matrix for specified columns.

    Parameters:
    - data: pandas DataFrame
    - columns: list of str, column names to include in the covariance matrix

    Returns:
    - numpy array, covariance matrix
    """
    # Use np.cov to compute covariance of the specified columns
    return np.cov(data[columns].T)


def conditional_expectation(spot_price, means, cov_matrix):
    """
    Calculate the expected Up and Down prices given a known Spot Price.

    Args:
        spot_price (float): Known Spot Price.
        means (list): Mean values for Spot, Up, and Down prices [mean_spot, mean_up, mean_down].
        cov_matrix (np.ndarray): Covariance matrix for Spot, Up, and Down prices.

    Returns:
        tuple: Expected Up Price and Down Price.
    """
    # Extract means
    mean_spot, mean_up, mean_down = means

    # Extract covariance submatrices
    var_spot = cov_matrix[0, 0]  # Variance of Spot Price
    cov_spot_up = cov_matrix[0, 1]  # Covariance between Spot and Up Price
    cov_spot_down = cov_matrix[0, 2]  # Covariance between Spot and Down Price

    # Covariances as vector
    cov_spot_others = np.array([cov_spot_up, cov_spot_down])

    # Means of Up and Down prices
    means_others = np.array([mean_up, mean_down])

    # Conditional expectation formula
    spot_price = np.array(spot_price)
    conditional_means = np.repeat(means_others.reshape(2,1), spot_price.size ,axis=1 )\
          + np.diag((cov_spot_others / var_spot)) @ (np.repeat(spot_price.reshape(1,-1),2,axis=0) - mean_spot)

    return conditional_means

def conditional_covariance(cov_matrix):
    
    # Extract covariance submatrices
    var_spot = cov_matrix[0, 0]         # Variance of Spot Price
    var_up = cov_matrix[1,1]            # Variance of up-price
    var_down = cov_matrix[2,2]          # Variance of down-price
    cov_spot_up = cov_matrix[0, 1]      # Covariance between Spot and Up Price
    cov_spot_down = cov_matrix[0, 2]    # Covariance between Spot and Down Price
    
    cond_cov_up = var_up - cov_spot_up * var_spot * cov_spot_up
    cond_cov_down = var_down - cov_spot_down * var_spot * cov_spot_down
    
    return np.array([cond_cov_up, cond_cov_down])

