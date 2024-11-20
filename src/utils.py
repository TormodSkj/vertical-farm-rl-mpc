import matplotlib.pyplot as plt
from tabulate import tabulate
import os
import numpy as np
import json
import hashlib

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
    

    return generate_table(metrics_table, header = list(runs.keys()), sumrow=False, diffcol=True)






