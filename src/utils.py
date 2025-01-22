import matplotlib.pyplot as plt
from tabulate import tabulate
import os
import numpy as np
import json
import hashlib
import pandas as pd
import casadi as ca
import scipy as sp



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

def load_spot_prices(file_path, bidding_zone):
    """
    Load spot prices, parse timestamps, and extract the price column.

    Parameters:
    - file_path: str, path to the CSV file
    - timestamp_col: str, name of the timestamp column
    - price_col: str, name of the price column

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


def load_mfrr_prices(data_folder, bidding_zone):
    """
    Load mFRR balancing prices from all relevant files in a folder, parse timestamps,
    and merge them into a single DataFrame with 'Start Time', 'End Time', 'Up Price',
    and 'Down Price' columns.

    Parameters:
    - data_folder: str, path to the folder containing CSV files.

    Returns:
    - pandas DataFrame: Merged dataset sorted by 'Start Time'.
    """

    standardize_balancing_price_files(data_folder)

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

        # Select and rename relevant columns
        up_price_col = "Price Up (EUR/MWh)"  # Adjust if your column name is different
        down_price_col = "Price Down (EUR/MWh)"  # Adjust if your column name is different

        data = data[['Start Time', 'End Time', up_price_col, down_price_col]].rename(
            columns={up_price_col: 'Up Price', down_price_col: 'Down Price'}
        )
        
        # Append processed data to the list
        all_data.append(data)

    # Merge all data and sort by 'Start Time'
    merged_data = pd.concat(all_data, ignore_index=True)
    merged_data.sort_values(by='Start Time', inplace=True)
    merged_data.dropna(inplace=True)

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
    Load mFRR activation data from all relevant files in a folder, merge all data for 
    upward and downward activations separately, and sort them by date.

    Parameters:
    - data_folder: str, path to the folder containing CSV files.

    Returns:
    - Tuple of two pandas DataFrames: (merged_upward_activations, merged_downward_activations)
    """
    all_files = [f for f in os.listdir(data_folder) if "mFRR_activations" in f and bidding_zone in f and f.endswith(".csv")]
    
    all_upwards = []
    all_downwards = []

    for file in all_files:
        filepath = os.path.join(data_folder, file)
        
        # Load the data
        data = pd.read_csv(filepath, quotechar='"', skipinitialspace=True)
        
        # Clean column names (remove extra quotes and whitespace)
        data.columns = data.columns.str.replace("'", '').str.strip()
        data = data.apply(lambda x: x.str.replace("'", '').str.strip() if x.dtype == "object" else x)
        
        # Split ISP into 'Start Time' and 'End Time'
        data['Start Time'] = data['ISP'].str.extract(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})')
        data['Start Time'] = pd.to_datetime(data['Start Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        # data['End Time'] = pd.to_datetime(data['End Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        
        # Rename AREA to Bidding Zone and simplify the zone names
        data.rename(columns={'Area': 'Bidding Zone'}, inplace=True)
        data['Bidding Zone'] = data['Bidding Zone'].str.replace(' SCA', '')
        
        # Remove unnecessary columns
        data.drop(columns=['ISP', 'Reserve Type', 'Type of Product', 'Unavailable (MW)'], inplace=True)
        
        # Convert 'Offered' and 'Activated' to numeric
        data[['Offered', 'Activated']] = data[['Offered (MW)', 'Activated (MW)']].apply(pd.to_numeric, errors='coerce')
        
        # Split into upwards and downwards activations
        up_activation_df = data[data['Direction'] == "Up"].reset_index(drop=True).drop(columns=['Direction'])
        down_activation_df = data[data['Direction'] == "Down"].reset_index(drop=True).drop(columns=['Direction'])
        
        # Append to lists
        all_upwards.append(up_activation_df)
        all_downwards.append(down_activation_df)
    
    # Merge all upwards and downwards data separately
    merged_upwards = pd.concat(all_upwards, ignore_index=True)
    merged_downwards = pd.concat(all_downwards, ignore_index=True)
    
    # Sort each dataset by Start Time
    merged_upwards.sort_values(by='Start Time', inplace=True)
    merged_downwards.sort_values(by='Start Time', inplace=True)
    
    return merged_upwards, merged_downwards



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




def generate_freshweight_outcomes(controller, u, b_p_up, b_p_dn, b_a_up, b_a_dn):

    model = controller.model
    x_init = model.x_init
    N = controller.N
    dt = controller.dt

    activation_th = 1e-2        # lower limit for reasonable activation chance: 1%
    # volume_th = 1e-3            # lower limit for valid bid: 0.001 MW
    # th = activation_th * volume_th

    b_a_up_filtered = np.where(b_a_up>activation_th, 1, 0)
    b_a_dn_filtered = np.where(b_a_up>activation_th, 1, 0)

    u_tilde_up = 2*1000*np.multiply(b_p_up, b_a_up)/model.C_conv_PPFD
    u_tilde_dn = 2*1000*np.multiply(b_p_dn, b_a_dn)/model.C_conv_PPFD
    
    u_up = u - u_tilde_up
    u_dn = u + u_tilde_dn

    x_up = np.zeros((model.nx, N+1))
    x_dn = np.zeros((model.nx, N+1))
    x_up[:,0] = np.array(x_init)
    x_dn[:,0] = np.array(x_init)
    
    for k in range(N):
        x_up[:,k+1] = x_up[:,k] + dt*np.array(model.derivative(x_up[:,k], [u_up[k]])).flatten()
        x_dn[:,k+1] = x_dn[:,k] + dt*np.array(model.derivative(x_dn[:,k], [u_dn[k]])).flatten()


    fw_up = np.array(model.freshweight(x_up)).flatten()
    fw_dn = np.array(model.freshweight(x_dn)).flatten()

    fw = np.vstack((fw_up, fw_dn))
    
    return fw



def empirical_cdf(data, bins=100):
    """
    Compute the empirical CDF for the given data.
    Parameters:
        data (array-like): Input data
        bins (int): Number of bins to use
    Returns:
        sorted_data (numpy array): Sorted original data (inverse CDF)
        cdf (numpy array): CDF values corresponding to sorted data
    """
    # Create bins for histogram
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    cdf = np.cumsum(hist) / np.sum(hist)  # Normalize cumulative sum to [0, 1]
    
    # Use bin midpoints to represent the data points
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, cdf



def generate_samples_from_cdf(original_data, num_samples, seed):
    """
    Generate random samples similar to the original distribution using inverse CDF sampling.
    Parameters:
        original_data (array-like): Input data for CDF
        num_samples (int): Number of samples to generate
        bins (int): Number of bins for the empirical CDF
    Returns:
        synthetic_samples (numpy array): Random samples similar to the original data
    """

    if seed is not None:
        np.random.seed(seed)

    # Compute the empirical CDF
    bins = int(max(original_data))
    bin_centers, cdf = empirical_cdf(original_data, bins)
    
    # Generate random uniform samples and sort them
    # uniform_samples = np.sort(np.random.uniform(0, 1, num_samples))
    uniform_samples = np.random.uniform(0, 1, num_samples)
    
    # Interpolate inverse CDF
    synthetic_samples = np.interp(uniform_samples, cdf, bin_centers)
    return synthetic_samples



def generate_weighted_samples(values, probabilities, n, seed=None):
    """
    Generate samples from a weighted distribution.
    
    Parameters:
        values (list or array-like): The values of the weighted distribution.
        probabilities (list or array-like): The probabilities for each value.
        n (int): Number of samples to generate.
        seed (int, optional): Seed for reproducibility.
        
    Returns:
        numpy array: Array of generated samples.
    """
    # Check that probabilities sum to 1
    if not np.isclose(sum(probabilities), 1):
        raise ValueError("Probabilities must sum to 1.")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate samples using np.random.choice
    samples = np.random.choice(values, size=n, p=probabilities)
    return samples




def calculate_light_schedule_variance(controller, bidding_volumes_up, bidding_volumes_dn, bidding_prices_up, bidding_prices_dn):

    market = controller.market
    model = controller.model
    N = controller.N
    spot_prices = controller.spot_prices


    u_var = np.zeros((1,N))

    for k in range(2,N):        # Don't count the first 2 bids


        p_up = market.Pr_a_up(spot_prices[k], bidding_prices_up[k])  # Bernoulli constant p for u_tilde_up
        p_dn = market.Pr_a_dn(spot_prices[k], bidding_prices_dn[k])  # Bernoulli constant p for u_tilde_down
        E_u_up = 1000 * bidding_volumes_up[k]/model.C_conv_PPFD * p_up
        E_u_dn = 1000 * bidding_volumes_dn[k]/model.C_conv_PPFD * p_dn
        
        var_u_up = (1000 * bidding_volumes_up[k]/model.C_conv_PPFD)**2 * p_up * (1-p_up)
        var_u_dn = (1000 * bidding_volumes_dn[k]/model.C_conv_PPFD)**2 * p_dn * (1-p_dn)

        u_var[:,k] = var_u_up + var_u_dn - 2*E_u_up*E_u_dn

        assert not (u_var[:,k]) < -1e-6, 'Variance cannot be negative'

    return u_var.flatten()



def calculate_freshweight_interval(controller, u_bid, b_p_up, b_p_dn, b_c_up, b_c_dn):
    '''
    Generates a timeseries interval for one standard deviation from u_base
    '''

    u_var = calculate_light_schedule_variance(controller, b_p_up, b_p_dn, b_c_up, b_c_dn)

    model = controller.model
    N = controller.N
    dt = controller.dt
    
    u_sd = np.sqrt(u_var)       # Get standard deviation from variance

    # upper and lower bounds on u defining the interval
    u_ub = np.maximum(np.minimum(u_bid + u_sd, model.C_PPFD_max), 0)
    u_lb = np.maximum(np.minimum(u_bid - u_sd, model.C_PPFD_max), 0)

    X_ub = np.zeros((model.nx, N+1))
    X_lb = np.zeros((model.nx, N+1))
    X_ub[:,0] = model.x_init
    X_lb[:,0] = model.x_init
    for k in range(N):
        #Forward euler
        X_ub[:,k+1] = X_ub[:,k] + dt*np.array(controller.model.derivative(X_ub[:,k], np.array([u_ub[k]]))).reshape(1, -1)
        X_lb[:,k+1] = X_lb[:,k] + dt*np.array(controller.model.derivative(X_lb[:,k], np.array([u_lb[k]]))).reshape(1, -1)

    return np.vstack((np.array(model.freshweight(X_ub[:,1:])).flatten(),
                      np.array(model.freshweight(X_lb[:,1:])).flatten()))




def propagate_process_covariance(controller, x_bid, u_bid, bidding_volumes_up, bidding_volumes_dn, bidding_prices_up, bidding_prices_dn):
    model = controller.model
    market = controller.market
    N = controller.N
    dt = controller.dt
    spot_prices = controller.spot_prices

    # State dimensions
    x0 = ca.DM(model.x_init)  # Initial state
    P = 0 * ca.DM.eye(model.nx)  # Initial covariance matrix
    C = ca.DM([1,1,0]).T

    # Arrays to store results
    state_covariances = [P]
    dw_variances = np.zeros((1, N+1))
    skewness_values = np.zeros((1, N+1))  # To store skewness
    dw_variances[:,0] = float(C @ P @ C.T)

    for k in range(N):
        # Input uncertainty terms
        a, b = 1000*bidding_volumes_up[k]/model.C_conv_PPFD, 1000*bidding_volumes_dn[k]/model.C_conv_PPFD
        p_up = market.Pr_a_up(spot_prices[k], bidding_prices_up[k])
        p_dn = market.Pr_a_dn(spot_prices[k], bidding_prices_dn[k])

        # Expected values of u_a and u_b
        E_u_up = a * p_up
        E_u_dn = b * p_dn
        E_u = u_bid[k] - E_u_up + E_u_dn

        # Variance of u
        Var_u_a = a**2 * p_up * (1 - p_up)
        Var_u_b = b**2 * p_dn * (1 - p_dn)
        Var_u = Var_u_a + Var_u_b - 2 * (E_u_up * E_u_dn)

        # # Skewness of u
        # Skew_u_a = -((1 - 2*p_a) / (np.sqrt(Var_u_a) if Var_u_a > 0 else 1e-6) if Var_u_b>1 else 0)
        # Skew_u_b = ((1 - 2*p_b) / (np.sqrt(Var_u_b) if Var_u_b > 0 else 1e-6) if Var_u_b>1 else 0)
        # Skew_u = Skew_u_a + Skew_u_b  # Approximate combined skewness

        # # Store skewness for measurement
        # skewness_values[:, k+1] = Skew_u

        if Var_u < 0: Var_u = 0
        Q = ca.DM([float(Var_u)])  # Input variance matrix

        # Linearize the process model
        x = ca.MX.sym('x', model.nx)
        u = ca.MX.sym('u')
        x_dot = model.derivative(x, u)
        A = ca.jacobian(x_dot, x)
        G = ca.jacobian(x_dot, u)

        # Evaluate A and G
        A_cont_eval = ca.Function('A_cont', [x, u], [A])(x_bid[:,k], E_u)
        G_cont_eval = ca.Function('G_cont', [x, u], [G])(x_bid[:,k], E_u)

        # Discretize A and compute G_discrete
        A_discrete = sp.linalg.expm(A_cont_eval.full() * dt)
        if np.linalg.cond(A_cont_eval.full()) < 1 / np.finfo(float).eps:
            G_discrete = np.linalg.solve(A_cont_eval.full(), (A_discrete - np.eye(model.nx))) @ G_cont_eval.full()
        else:
            G_discrete = np.zeros_like(G_cont_eval.full())
            for i in range(10):
                tau = i * dt / 10
                expm_partial = sp.linalg.expm(A_cont_eval.full() * tau)
                G_discrete += expm_partial @ G_cont_eval.full() * dt / 10

        A_discrete = ca.DM(A_discrete)
        G_discrete = ca.DM(G_discrete)

        # Propagate covariance
        P = A_discrete @ P @ A_discrete.T + G_discrete @ Q @ G_discrete.T
        state_covariances.append(P)
        dw_variances[:, k+1] = float(C @ P @ C.T)

    # # Adjust confidence intervals with skewness (Cornish-Fisher expansion)
    # z = 1.96  # For 95% confidence
    # skew_adjustment = skewness_values[:, 1:].flatten() * (z**2 - 1) / 6
    # z_upper = z + skew_adjustment
    # z_lower = z - skew_adjustment

    fw_variances = ((1-model.c_T)/(model.c_d * model.PCD))**2 * dw_variances[:, 1:].flatten()

    return fw_variances #, z_upper, z_lower





def vertigrow_calculate_energy_consumption(controller):

    inty_to_power = {
        0: 0,
        10: 18,
        20: 30,
        30: 44,
        40: 57,
        50: 70,
        60: 84,
        70: 97,
        80: 112,
        90: 126,
        100: 140
    }


    for run in controller.optimization_results['runs']:
        u = controller.optimization_results['runs'][run]['timeseries']['u']

        print("Analysing: "+run)

        total_energy = 0
        total_cost = 0
        for k in range(controller.N):
            
            current_energy = inty_to_power[np.round((u[k]/controller.model.C_PPFD_max * 100)/10)*10]/4

            total_energy += current_energy
            total_cost += controller.spot_prices[k] * current_energy


        print(f'Total energy: {total_energy} Wh')
        print(f'Cost: {total_cost} NOK')



def strip_entsoe_activation_data(data_folder, filename):
    """
    Filters the rows of a CSV file, keeping only those that have 'mFRR' in the 'Reserve Type'
    column and 'Local' in the 'Type of Product' column, while also removing rows with 'n/e'
    in the columns 'Offered (MW)', 'Activated (MW)', or 'Unavailable (MW)'. Overwrites the 
    original file with the filtered data.

    Parameters:
    - filename: str, path to the CSV file to be processed.
    """

    filepath = data_folder + filename + ".csv"

    # Open the file for reading and writing
    with open(filepath, 'r') as infile:
        lines = infile.readlines()

    # Get the header (the first line) and filter the relevant columns by their names
    header = lines[0].strip()
    columns = header.split(',')
    
    reserve_type_index = columns.index('"Reserve Type"')
    type_of_product_index = columns.index('"Type of Product"')
    offered_mw_index = columns.index('"Offered (MW)"')
    activated_mw_index = columns.index('"Activated (MW)"')
    unavailable_mw_index = columns.index('"Unavailable (MW)"')

    # Filter lines based on conditions
    filtered_lines = [lines[0]]  # Start with the header
    for line in lines[1:]:  # Skip the header and check the remaining lines
        columns = line.strip().split(',')

        # Check if the line satisfies the conditions
        if len(columns) > max(reserve_type_index, type_of_product_index, offered_mw_index, activated_mw_index, unavailable_mw_index):  # Avoid IndexError
            reserve_type = columns[reserve_type_index]
            type_of_product = columns[type_of_product_index]
            offered_mw = columns[offered_mw_index]
            activated_mw = columns[activated_mw_index]
            unavailable_mw = columns[unavailable_mw_index]

            # Check for 'mFRR' in 'Reserve Type' and 'Local' in 'Type of Product'
            if '"mFRR"' in reserve_type and '"Local"' in type_of_product:
                # Check for 'n/e' in the MW columns
                if offered_mw != '"n/e"' and activated_mw != '"n/e"' and unavailable_mw != '"n/e"':
                    filtered_lines.append(line)

    # Write the filtered lines back to the file, overwriting the original file
    with open(filepath, 'w') as outfile:
        outfile.writelines(filtered_lines)

    print(f"File '{filepath}' has been filtered successfully.")