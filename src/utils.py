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
from datetime import datetime


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



def generate_hash(specs: dict):

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

def get_metrics_table_raw(runs):
    '''
    Takes in a dict of runs containing metrics dicts. 
    Each run has its own metrics dict, which will be unraveled and displayed here.
    Ensures every unique metric is included, prioritizing the order from the run with the most metrics.
    '''
    
    # Find the run with the most metrics (assuming it has the full set)
    max_metrics_run = max(runs.values(), key=lambda r: len(r['metrics']))
    ordered_metrics = list(max_metrics_run['metrics'].keys())  # Preserve the order

    # Collect all unique metrics while preserving the order from max_metrics_run
    all_metrics = set(ordered_metrics)
    for run in runs.values():
        for metric in run['metrics']:
            if metric not in all_metrics:
                ordered_metrics.append(metric)
                all_metrics.add(metric)

    metrics_table = []

    for metric in ordered_metrics:  # Use ordered list from max-metrics run
        metrics_row = [metric]

        for run in runs:
            data = runs[run]['metrics'].get(metric, 0)  # Default to 0 if missing
            if isinstance(data, (float, np.float64)):
                data = f"{data:.2f}"  # Format floats to 2 decimal places

            metrics_row.append(data)

        metrics_table.append(metrics_row)

    return metrics_table


def dict_to_table(input_dict: dict):

    table = []

    for key in input_dict:

        datapoint = input_dict[key]
        if type(datapoint) == float or type(datapoint) == np.float64:
            datapoint = f"{datapoint:.2f}"

        table.append([key, str(datapoint)])

    return table


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


def conditional_expectation(y, means, cov_matrix):
    """

    """
    y = np.array(y)

    epsilon = 1e-6
    # cov_spot_others = np.array(price_covs[0][1,1], price_covs[1][1,1])

    Pxx         = cov_matrix[1,1]
    Pxy         = cov_matrix[0,1]
    Pyy         = max(cov_matrix[0,0], epsilon)

    mean_y   = means[0]
    mean_x   = means[1]

    conditional_mean = np.repeat(mean_x, y.size) + (Pxy / Pyy) * (y - mean_y)
    conditional_covariance = Pxx - Pxy / Pyy * Pxy

    return np.array([conditional_mean]), conditional_covariance

# def conditional_covariance(cov_matrix):
#     epsilon = 1e-6
    
#     Pxx = cov_matrix[1,1]
#     Pxy = cov_matrix[0,1]
#     Pyy = max(cov_matrix[0,0], epsilon)

#     cond_cov = Pxx - Pxy /Pyy* Pxy    
#     return cond_cov




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


        p_up = market.activation_prob_up(spot_prices[k], bidding_prices_up[k])  # Bernoulli constant p for u_tilde_up
        p_dn = market.activation_prob_dn(spot_prices[k], bidding_prices_dn[k])  # Bernoulli constant p for u_tilde_down
        E_u_up = 1000 * bidding_volumes_up[k]/model.C_conv_PPFD * p_up
        E_u_dn = 1000 * bidding_volumes_dn[k]/model.C_conv_PPFD * p_dn
        
        var_u_up = (1000 * bidding_volumes_up[k]/model.C_conv_PPFD)**2 * p_up * (1-p_up)
        var_u_dn = (1000 * bidding_volumes_dn[k]/model.C_conv_PPFD)**2 * p_dn * (1-p_dn)

        u_var[:,k] = var_u_up + var_u_dn - 2*E_u_up*E_u_dn

        assert not (u_var[:,k]) < -1e-6, 'Variance cannot be negative'

    return u_var.flatten()



def calculate_freshweight_interval_old(controller, u_bid, b_p_up, b_p_dn, b_c_up, b_c_dn):
    '''
    Old function do not use. Calculates absolute worst cases what have probability of less than 1e-125 just for one day
    '''

    u_var = calculate_light_schedule_variance(controller, b_p_up, b_p_dn, b_c_up, b_c_dn)

    model = controller.model
    N = controller.N
    dt = controller.dt
    
    u_sd = np.sqrt(u_var)       # Get standard deviation from variance

    # upper and lower bounds on u defining the interval
    u_ub = np.maximum(np.minimum(u_bid + u_sd, model.PPFD_max), 0)
    u_lb = np.maximum(np.minimum(u_bid - u_sd, model.PPFD_max), 0)

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
    u_ub = np.maximum(np.minimum(u_bid + u_sd, model.PPFD_max), 0)
    u_lb = np.maximum(np.minimum(u_bid - u_sd, model.PPFD_max), 0)

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
    dw_variances[:,0] = float(C @ P @ C.T)

    for k in range(N):
        # Input uncertainty terms
        a, b = 1000*bidding_volumes_up[k]/model.C_conv_PPFD, 1000*bidding_volumes_dn[k]/model.C_conv_PPFD
        p_up = market.activation_prob_up(spot_prices[k], bidding_prices_up[k])
        p_dn = market.activation_prob_dn(spot_prices[k], bidding_prices_dn[k])

        # Expected values of u_tile_up and u_tilde_dn
        E_u_up = a * p_up
        E_u_dn = b * p_dn
        E_u = u_bid[k] - E_u_up + E_u_dn

        # Variance of u
        Var_u_a = a**2 * p_up * (1 - p_up)
        Var_u_b = b**2 * p_dn * (1 - p_dn)
        Var_u = Var_u_a + Var_u_b - 2 * (E_u_up * E_u_dn)

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

    fw_variances = ((1-model.c_T)/(model.c_d * model.PCD))**2 * dw_variances[:, 1:].flatten()

    return fw_variances



def casadi_saturate(x, min, max):
    # Custom function which bounds activation chance between 0 and 1. (1 + abs(x) - abs(x-1))/2
    
    return (max + min +  casadi_abs(x-min) - casadi_abs(x-max)) / 2

def casadi_max(x, y):
    # Custom function which returns the maximum of two values
    
    return (casadi_abs(x-y) + x-y) / 2 + y

def casadi_abs(x):
    return ca.sqrt(ca.power(x, 2))



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
            
            current_energy = inty_to_power[np.round((u[k]/controller.model.PPFD_max * 100)/10)*10]/4

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



def sort_runs(optimization_results: dict):
    '''
    Sorts runs into one of three groups:
    0:  Run is generated on its own. Can either be fixed or spot price optimized.
    1:  Run contains mfrr bids and iterates on runs from level 0.
    2:  Run subjects bids to real data. Iterates on runs from level 1.

    Inputs: 
    optimization_results    -    dictionary containing all run data

    Outputs:
    sorted_runs     -   Dict of run names sorted into the three categories mentioned above
    group_sizes     -   List of group sizes
    '''


    sorted_runs = {0: [],
                   1: [],
                   2: []}

    for run_id in optimization_results['runs']:
        run_data = optimization_results['runs'][run_id]
        
        if 'bidding result' not in run_data:
            sorted_runs[0].append(run_id)
        elif 'bidding result' in run_data and ('A_up' not in run_data['timeseries'] or 'A_dn' not in run_data['timeseries']):
            sorted_runs[1].append(run_id)
        elif 'bidding result' in run_data and 'A_up' in run_data['timeseries'] and 'A_dn' in run_data['timeseries']:
            sorted_runs[2].append(run_id)
        else:
            assert False, 'Processed run data does not fit any of the run groups'

    group_sizes = [len(sorted_runs[i]) for i in sorted_runs]

    return sorted_runs, group_sizes



def build_dependency_groups(optimization_results: dict):
    """
    Builds a tree of runs, where each run has a reference to the previous one it depends on.
    Groups each extremity (leaf node) and its ancestors from the root ('fixed' or 'imported').

    Inputs:
    optimization_results    -    Dictionary containing all run data.

    Outputs:
    dependency_groups       -   A list of lists, where each list contains a group of runs from the leaf node to the root.
    """
    # Initialize tree structure
    tree = {}

    # Step 1: Build the tree structure
    for run_id, run_data in optimization_results['runs'].items():
        parent_run = run_data.get('reference_run', None)  # Reference to the parent run (previous iteration)
        
        if run_id not in tree: tree[run_id] = []
        if parent_run != 'None' and parent_run is not None:
            if parent_run not in tree: tree[parent_run] = []
            tree[parent_run].append(run_id)

    # Step 2: Identify extremities (leaf nodes)
    extremities = []
    for node, children in tree.items():
        if not children:  # No children means it's an extremity
            extremities.append(node)

    # Step 3: Trace paths from extremities to the root
    def trace_path_to_root(node, tree):
        """Trace the path from a leaf node to the root (fixed or imported)."""
        path = []
        current_node = node
        while current_node is not None:
            path.append(current_node)
            parent_node = None
            for parent, children in tree.items():
                if current_node in children:
                    parent_node = parent
                    break
            current_node = parent_node
        return path

    # Step 4: Build the dependency groups from the extremities
    dependency_groups = []
    for extremity in extremities:
        group = trace_path_to_root(extremity, tree)
        dependency_groups.append(group)

    return dependency_groups



def get_DLI(X):

    N = X.shape[1]-1
    DLI = np.zeros((1, N-QUARTER_HOURS_PER_DAY))

    for k in range(QUARTER_HOURS_PER_DAY, N):
        # k = 96 +24, +48, +72 ...
        LI = (X[2,k] - X[2,k-QUARTER_HOURS_PER_DAY])
        
        DLI[:,k-QUARTER_HOURS_PER_DAY] = LI
            
    return DLI



def build_market_participation(B_volumes = None, B_prices = None, Activations = None):

    market_participation = {}

    market_participation['Bids'] = {
        'Up': {
            'Volume' : B_volumes[0,:],
            'Price'  : B_prices[0,:]
        },
        'Down': {
            'Volume' : B_volumes[1,:],
            'Price'  : B_prices[1,:]
        }
    }

    market_participation['Activations'] = None if Activations is None else {
        'Up'    : Activations[0,:],
        'Down'  : Activations[1,:]
    }

    # if B_volumes is not None or B_prices is not None:
    #     market_participation['Bids'] = {
    #         'Up'    : {},
    #         'Down'  : {}
    #     }

    # if B_volumes is not None:
    #     market_participation['Bids']['Up']['Volume']    = B_volumes[0,:]
    #     market_participation['Bids']['Down']['Volume']  = B_volumes[1,:]
        
    
    # if B_prices is not None:
    #     market_participation['Bids']['Up']['Price']    = B_prices[0,:]
    #     market_participation['Bids']['Down']['Price']  = B_prices[1,:]
        
    # if Activations is not None:
    #     market_participation['Activations'] = {
    #         'Up'    : Activations[0,:],
    #         'Down'  : Activations[1,:]
    #     }
    

    return market_participation






# def get_file_metadata(file_path):
#     """Retrieve creation and last modified timestamps of a file."""
#     creation_time = os.path.getctime(file_path)
#     modified_time = os.path.getmtime(file_path)
#     return (
#         datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S'),
#         datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
#     )

# def generate_readme_content(csv_files):
#     """Generate README content based on existing CSV files."""
#     content = "# Dataset Information\n\n"
#     for file in csv_files:
#         file_path = file['path']
#         filename = os.path.basename(file_path)
#         created, modified = get_file_metadata(file_path)
        
#         try:
#             df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows for efficiency
#             columns = ', '.join(df.columns)
#         except Exception as e:
#             columns = f"Could not read file: {e}"

#         content += (f"## {filename}\n"
#                     f"- **File Path:** {file_path}\n"
#                     f"- **Columns:** {columns}\n"
#                     f"- **Created:** {created}\n"
#                     f"- **Last Modified:** {modified}\n\n")
#     return content

# def update_readme_for_folder(folder_path):
#     """Scan a folder for CSV files and update the README.md file."""
#     readme_path = os.path.join(folder_path, "README.md")
    
#     # Get all CSV files in the folder
#     csv_files = [{'path': os.path.join(folder_path, f)} for f in os.listdir(folder_path) if f.endswith('.csv')]
    
#     # Generate new README content
#     new_content = generate_readme_content(csv_files)
    
#     # Write or update README.md
#     with open(readme_path, 'w', encoding='utf-8') as f:
#         f.write(new_content)
    
#     print(f"Updated README in: {folder_path}")

# def scan_data_directory(root_dir):
#     """Scan all subdirectories of root_dir, updating or creating README.md files."""
#     for subdir, _, _ in os.walk(root_dir):
#         update_readme_for_folder(subdir)




# def scan_data_directory(data_dir):
#     """
#     Updates README.md in each subfolder of data_dir.
#     - Adds entries for new CSV files with filename, columns, creation/modification dates, and first/last row datetime.
#     - Removes entries for missing CSV files.
#     - Preserves user-entered 'Source' fields.
#     """
#     for root, _, files in os.walk(data_dir):
#         readme_path = os.path.join(root, "README.md")
#         existing_entries = {}
#         source_entries = {}
#         link_entries = {}
#         description_entries = {}

#         # Load existing README if it exists
#         if os.path.exists(readme_path):
#             with open(readme_path, "r", encoding="utf-8") as f:
#                 content = f.read().split("\n\n")
#                 for section in content:
#                     lines = section.split("\n")
#                     if len(lines) < 2:
#                         continue
#                     filename = lines[0].lstrip('# ')
#                     existing_entries[filename] = section
                    
#                     # Preserve source entry
#                     for i, line in enumerate(lines):
#                         if i == 0: continue
#                         if lines[i-1].startswith("### Source:"):
#                             source_entries[filename] = line
#                         if lines[i-1].startswith("### Link:"):
#                             link_entries[filename] = line
#                         if lines[i-1].startswith("### Description:"):
#                             description_entries[filename] = line

#         # Discover current CSV files
#         new_entries = {}
#         for file in files:
#             if file.endswith(".csv"):
#                 file_path = os.path.join(root, file)
#                 created = datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d")
#                 modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d")
                
#                 # Read first and last rows
#                 try:
#                     df = pd.read_csv(file_path, delimiter=';', nrows=1)  # Read only first row
#                     df_tail = pd.read_csv(file_path, delimiter=';').tail(1)  # Read last row efficiently
#                     date_col = next((col for col in df.columns if any(word in col.lower() for word in ['date', 'dato', 'delivery start', 'time'])), None)
#                     first_date = df[date_col].values[0] if date_col else "Unknown"
#                     last_date = df_tail[date_col].values[0] if date_col else "Unknown"
#                 except Exception as e:
#                     first_date, last_date = "Error", "Error"

#                 # Restore source entry if it exists
#                 source_entry        = source_entries.get(file, " (Add source info here)")
#                 link_entry          = link_entries.get(file, " None")
#                 description_entry   = description_entries.get(file, " (no description)")
                
#                 # Format new README entry
#                 new_entries[file] = (f"# {file}\n"
#                                      f"### Description:\n{description_entry}\n"
#                                      f"### From:\n {first_date}\n"
#                                      f"### To:\n {last_date}\n"
#                                      f"### Source:\n{source_entry}\n"
#                                      f"### Link:\n{link_entry}\n"
#                                      f"### Created:\n {created}\n"
#                                      f"### Modified:\n {modified}\n"
#                                      f"### Columns:\n {', '.join(df.columns)}\n"
#                                      f"\n")
        
#         # Remove missing files from README
#         for missing_file in set(existing_entries) - set(new_entries):
#             del existing_entries[missing_file]

#         # Write updated README
#         with open(readme_path, "w", encoding="utf-8") as f:
#             f.write("\n\n".join(new_entries.values()) + "\n")

def update_dataset_readmes(data_dir):
    """
    Scans all CSV files in subdirectories of data_dir and updates their corresponding README.md.
    - Updates README only if any CSV file has been modified after the last README update.
    - Adds entries for new CSV files with filename, columns, creation/modification dates, and first/last row datetime.
    - Removes entries for missing CSV files.
    - Preserves user-entered 'Source', 'Link', and 'Description' fields.
    """

    for root, _, files in os.walk(data_dir):
        readme_path = os.path.join(root, "README.md")

        # Get last modified timestamp of README
        readme_mtime = datetime.fromtimestamp(os.path.getmtime(readme_path)) if os.path.exists(readme_path) else None

        # Get list of CSV files and their modification times
        csv_files = [f for f in files if f.endswith(".csv")]
        latest_mod_time = max(
            (datetime.fromtimestamp(os.path.getmtime(os.path.join(root, f))) for f in csv_files),
            default=None
        )

        # Skip update if no CSV file has been modified since the last README update
        if readme_mtime and latest_mod_time and latest_mod_time <= readme_mtime:
            # print(f"Skipping update for {root}, no changes detected.")
            continue

        existing_entries = {}
        source_entries = {}
        link_entries = {}
        description_entries = {}

        # Load existing README if it exists
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read().split("\n\n")
                for section in content:
                    lines = section.split("\n")
                    if len(lines) < 2:
                        continue
                    filename = lines[0].lstrip('# ')
                    existing_entries[filename] = section
                    
                    # Preserve user-entered fields
                    for i, line in enumerate(lines):
                        if i == 0: continue
                        if lines[i-1].startswith("### Source:"):
                            source_entries[filename] = line
                        if lines[i-1].startswith("### Link:"):
                            link_entries[filename] = line
                        if lines[i-1].startswith("### Description:"):
                            description_entries[filename] = line

        # Discover current CSV files
        new_entries = {}
        for file in csv_files:
            file_path = os.path.join(root, file)
            created = datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d")
            modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d")
            
            # Read first and last rows
            try:
                df = pd.read_csv(file_path, delimiter=';', nrows=1)  # Read only first row
                df_tail = pd.read_csv(file_path, delimiter=';').tail(1)  # Read last row efficiently
                date_col = next((col for col in df.columns if any(word in col.lower() for word in ['date', 'dato', 'delivery start', 'time'])), None)
                first_date = df[date_col].values[0] if date_col else "Unknown"
                last_date = df_tail[date_col].values[0] if date_col else "Unknown"
            except Exception:
                first_date, last_date = "Error", "Error"

            # Restore user-entered fields if they exist
            source_entry = source_entries.get(file, " (Add source info here)")
            link_entry = link_entries.get(file, " None")
            description_entry = description_entries.get(file, " (no description)")
            
            # Format new README entry
            new_entries[file] = (f"# {file}\n"
                                 f"### Description:\n{description_entry}\n"
                                 f"### From:\n {first_date}\n"
                                 f"### To:\n {last_date}\n"
                                 f"### Source:\n{source_entry}\n"
                                 f"### Link:\n{link_entry}\n"
                                 f"### Created:\n {created}\n"
                                 f"### Modified:\n {modified}\n"
                                 f"### Columns:\n {', '.join(df.columns)}\n"
                                 f"\n")
        
        # Remove missing files from README
        for missing_file in set(existing_entries) - set(new_entries):
            del existing_entries[missing_file]

        # Write updated README
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(new_entries.values()) + "\n")

        print(f"Updated README in {root}")