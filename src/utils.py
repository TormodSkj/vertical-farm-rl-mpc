import matplotlib.pyplot as plt
from tabulate import tabulate
import os

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



def print_cost_comparison_table(method1_name, method2_name, cost_data):
    """
    Prints a comparison table for two optimization methods, showing components of the final cost.

    """
    # Define the column headers
    headers = ['Cost Component', method1_name, method2_name, 'Difference']
    
    # Format data for the table, adding the difference column
    table_data = [
        [row, cost_data[row][0], cost_data[row][1], f"{(-cost_data[row][0] + cost_data[row][1]):.2f}"]
        for row in cost_data
    ]
    
    # Add the Total row without a difference column
    total_method1 = sum(value[0] for value in cost_data.values())
    total_method2 = sum(value[1] for value in cost_data.values())
    table_data.append(['Total', total_method1, total_method2, total_method2-total_method1])
    
    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f"))


def print_bidding_table(bidding_data):
# Define the column headers
    headers = ['Attribute', 'Up-regulation', 'Down-regulation', 'Unit']
    
    # Format data for the table, adding the difference column
    table_data = [
        [row, bidding_data[row][0], bidding_data[row][1], bidding_data[row][2]]
        for row in bidding_data
    ]
    
    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f"))

