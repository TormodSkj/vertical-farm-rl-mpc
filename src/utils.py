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

def get_metrics_table_raw(runs):
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

            data = runs[run]['metrics'][metric]
            if type(data) == float or type(data) == np.float64:
                data = f"{data:.2f}"
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


def conditional_expectation(spot_price, means, covs):
    """
    Calculate the expected Up and Down prices given a known Spot Price.

    Args:
        spot_price (float): Known Spot Price.
        means (list): Mean values for Spot, Up, and Down prices [mean_spot, mean_up, mean_down].
        cov_matrix (np.ndarray): Covariance matrix for Spot, Up, and Down prices.

    Returns:
        tuple: Expected Up Price and Down Price.
    """

    
    # cov_spot_others = np.array(price_covs[0][1,1], price_covs[1][1,1])
    var_spot_up = covs[0][0,0]
    cov_spot_up = covs[0][0,1]
    var_spot_down = covs[1][0,0]
    cov_spot_down = covs[1][0,1]

    # Means of Up and Down prices
    mean_spot_up    = means[0]
    mean_spot_down  = means[1]
    mean_up_price   = means[2]
    mean_down_price = means[3]

    # Conditional expectation formula
    spot_price = np.array(spot_price)
    conditional_mean_up     = np.repeat(mean_up_price, spot_price.size)     + (cov_spot_up / var_spot_up)       * (spot_price - mean_spot_up)
    conditional_mean_down   = np.repeat(mean_down_price, spot_price.size)   + (cov_spot_down / var_spot_down)   * (spot_price - mean_spot_down)
    
    return np.array([conditional_mean_up]), np.array([conditional_mean_down])

def conditional_covariance(price_covs):
    
    price_cov_up = price_covs[0]
    price_cov_down = price_covs[1]


    # Extract covariance submatrices
    var_spot_up     = price_cov_up[0, 0]      # Variance of Spot Price for Up prices
    var_spot_down   = price_cov_down[0, 0]    # Variance of Spot Price for Down prices
    var_up          = price_cov_up[1,1]       # Variance of up-price
    var_down        = price_cov_down[1,1]     # Variance of down-price
    cov_spot_up     = price_cov_up[0, 1]      # Covariance between Spot and Up Price
    cov_spot_down   = price_cov_down[0, 1]    # Covariance between Spot and Down Price
    
    cond_cov_up = var_up - cov_spot_up * var_spot_up * cov_spot_up
    cond_cov_down = var_down - cov_spot_down * var_spot_down * cov_spot_down
    
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
