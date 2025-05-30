import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import os



def main():
    # Load files

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    filepath = os.path.join(data_path, "summary_results.json")

    with open(filepath, "r") as f:
        results = json.load(f)

    nominal_costs_df = pd.read_csv(os.path.join(data_path, "nominal_costs.csv"), parse_dates=["Date"])
    mfrr_costs_df    = pd.read_csv(os.path.join(data_path, "mfrr_costs.csv"), parse_dates=["Date"])

    zones = list(results.keys())

    fixed_shifted_cost = np.array([results[zone]["fixed_shifted_cost"] for zone in zones])
    spot_cost = np.array([results[zone]["spot_cost"] for zone in zones])
    mFRR_cost = np.array([results[zone]["mFRR_cost"] for zone in zones])

    spot_reduction = np.array([results[zone]["spot_cost_reduction"] for zone in zones])
    AM_reduction   = np.array([results[zone]["AM_cost_reduction"] for zone in zones])
    CM_reduction   = np.array([results[zone]["CM_cost_reduction"] for zone in zones])


    # my_colors = ["#557571", "#D49A89", "#F7D1BA", "#F4F4F4"]
    # my_colors = ["gray", "blue", "red"]
    # my_colors = ["green", "blue", "red"]
    my_colors = ["slategray", "#77CDFF", "#F95454"]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=my_colors)
    
    fontname = "DejaVu Serif"
    plt.rcParams["font.family"] = fontname


    # Plot 1: Fractional cost reduction bar chart
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(zones, spot_reduction,                                      label="Spot Market")        # , color='blue')
    ax.barh(zones, CM_reduction, left=spot_reduction,                   label="Capacity Market")    # , color='skyblue')
    ax.barh(zones, AM_reduction, left=spot_reduction + CM_reduction,    label="Activation Market")  # , color='red')
    ax.set_ylabel("Bidding Zones")
    ax.set_xlabel("Cost Reduction (%)")
    ax.legend(loc='upper right')
    ax.grid(axis='x', linestyle='-', alpha=0.6)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    # # Plot 2: Cost breakdown bar chart
    # x = np.arange(len(zones))
    # bar_width = 0.3
    # fig, ax = plt.subplots(figsize=(6, 3))
    # ax.bar(x - bar_width, fixed_shifted_cost, width=bar_width,  label="Nominal Cost")        # , color='blue')
    # ax.bar(x, spot_cost, width=bar_width,                       label="Cost after Spot")     # , color='skyblue'
    # ax.bar(x + bar_width, mFRR_cost, width=bar_width,           label="Cost after mFRR")     # , color='red')
    # ax.set_xticks(x)
    # ax.set_xticklabels(zones)
    # ax.set_xlabel("Zones")
    # ax.set_ylabel("Cost (€)")
    # ax.legend()
    # plt.show()

    # # Plot 3: Timeline of cost savings
    # T = 1
    # def moving_average(df, w):
    #     df = df.copy()
    #     for col in df.columns:
    #         if col == 'Date': continue
    #         df[col] = np.convolve(df[col], np.ones(w)/w, mode='same')
    #     return df

    # nominal_costs_df = moving_average(nominal_costs_df, T)
    # mfrr_costs_df = moving_average(mfrr_costs_df, T)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    # nominal_costs_df.set_index('Date')[zones].plot(ax=ax1)
    # mfrr_costs_df.set_index('Date')[zones].plot(ax=ax2)
    # ax1.set_ylabel("Nominal Cost (€)")
    # ax1.set_title("Nominal Daily Cost")
    # ax2.set_ylabel("mFRR Adjusted Cost (€)")
    # ax2.set_title("Cost after mFRR Participation")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()


    