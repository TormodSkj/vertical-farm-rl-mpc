import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.patheffects as pe

def main1():
    # Load files

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    filepath = os.path.join(data_path, "summary_results_actrate100.json")

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
    plt.rcParams.update({'font.size': 12})
        



    alpha = 0.8

    # Plot 1: Fractional cost reduction bar chart
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(zones, spot_reduction,                                   alpha=1,   label="Spot Market")        # , color='blue')
    ax.barh(zones, CM_reduction, left=spot_reduction,                alpha=alpha,   label="Capacity Market")    # , color='skyblue')
    ax.barh(zones, AM_reduction, left=spot_reduction + CM_reduction, alpha=alpha,   label="Activation Market")  # , color='red')
    ax.set_ylabel("Bidding Zones")
    ax.set_xlabel("Cost Reduction (%)")
    # ax.legend(bbox_to_anchor=(0.4, 1.4))
    # ax.legend(loc='upper right')
    ax.legend()
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


def main2():
    # List of JSON filenames to process
    json_files = [
        "summary_results_actrate100.json",
        "summary_results_actrate025.json",
        "summary_results_actrate010.json",
        "summary_results_actrate005.json",
        "summary_results_actrate001.json"
    ]

    # Get the path to the current script directory
    data_path = os.path.abspath(os.path.dirname(__file__))

    # Load common CSVs once
    nominal_costs_df = pd.read_csv(os.path.join(data_path, "nominal_costs.csv"), parse_dates=["Date"])
    mfrr_costs_df    = pd.read_csv(os.path.join(data_path, "mfrr_costs.csv"), parse_dates=["Date"])

    # Style setup
    my_colors = ["slategray", "#77CDFF", "#F95454"]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=my_colors)
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams.update({'font.size': 12})

    for json_file in json_files:
        filepath = os.path.join(data_path, json_file)

        # Load results
        with open(filepath, "r") as f:
            results = json.load(f)

        zones = list(results.keys())

        actrate = json_file.split('.')[0][-3:]

        fixed_shifted_cost = np.array([results[zone]["fixed_shifted_cost"] for zone in zones])
        spot_cost = np.array([results[zone]["spot_cost"] for zone in zones])
        mFRR_cost = np.array([results[zone]["mFRR_cost"] for zone in zones])

        spot_reduction = np.array([results[zone]["spot_cost_reduction"] for zone in zones])
        AM_reduction   = np.array([results[zone]["AM_cost_reduction"] for zone in zones])
        CM_reduction   = np.array([results[zone]["CM_cost_reduction"] for zone in zones])

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(zones, spot_reduction,                                   alpha=1,         label="Spot Market")
        ax.barh(zones, CM_reduction, left=spot_reduction,                alpha=0.8,       label="Capacity Market")
        ax.barh(zones, AM_reduction, left=spot_reduction + CM_reduction, alpha=0.8,       label="Activation Market")

        fig.suptitle(f"CM Reservation rate: {int(actrate)}%\n AM Activation rate: {int(actrate)}%")
        ax.set_ylabel("Bidding Zones")
        ax.set_xlabel("Cost Reduction (%)")
        # ax.set_xticks(list(range(0,101,10)) + [120, 140])
        # ax.set_xticks(list(range(0,151,10)))
        ax.set_xlim([0,50+int(actrate)])
        ax.legend()
        ax.grid(axis='x', linestyle='-', alpha=0.6)
        ax.invert_yaxis()
        plt.tight_layout()

        # Save to PDF with same base name as JSON file
        base_name = os.path.splitext(json_file)[0]
        output_path = os.path.join(data_path, f"upper_earnings_estimate_cost_reduction_actrate{actrate}.pdf")
        plt.savefig(output_path)
        plt.close(fig)  # Don't display or hold plot in memory

        print(f"Saved plot to {output_path}")


def main3():

    this_path = os.path.abspath(os.path.dirname(__file__))

    geojson_dir = os.path.join(this_path, '../../data/bidding_zones_geodata/')

    geofiles = [
        'DK_1.geojson', 'DK_2.geojson', 'FI.geojson',
        'NO_1.geojson', 'NO_2.geojson', 'NO_3.geojson',
        'NO_4.geojson', 'NO_5.geojson',
        'SE_1.geojson', 'SE_2.geojson', 'SE_3.geojson', 'SE_4.geojson'
    ]


    json_file = "summary_results_actrate100.json"

    # Get the path to the current script directory
    data_path = os.path.abspath(os.path.dirname(__file__))

    # Style setup
    my_colors = ["slategray", "#77CDFF", "#F95454"]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=my_colors)
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams.update({'font.size': 12})

    filepath = os.path.join(data_path, json_file)

    # Load results
    with open(filepath, "r") as f:
        results = json.load(f)

    zones = list(results.keys())

    # Load and concatenate all GeoJSON files
    gdfs = []
    for file in geofiles:
        gdf = gpd.read_file(os.path.join(geojson_dir, file))
        # Add a new column for the zone name based on file name (remove extension and underscores)
        gdf["zone_name"] = file.replace(".geojson", "").replace("_", "")
        gdfs.append(gdf)

    full_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

    zone_costs = {}

    for zone in zones:
        zone_costs[zone] = results[zone]['spot_cost']
    

    # Add cost reduction to GeoDataFrame
    full_gdf["cost_reduction"] = full_gdf["zone_name"].map(zone_costs)

    # Plot heatmap
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    full_gdf.plot(
        column="cost_reduction",
        cmap="YlOrRd",
        linewidth=0.8,
        edgecolor='black',
        legend=True,
        ax=ax,
        vmin=0,         
        vmax=400000,
        legend_kwds={"shrink": 0.9}
    )

    offsets = {
        "DK1": (-3.1,   0.0     ),
        "DK2": (2.0,    -1.0    ),
        "FI":  (0.9,    0.0     ),
        "NO1": (0.0,    0.1     ),
        "NO2": (0.0,    0.0     ),
        "NO3": (0.0,    0.0     ),
        "NO4": (-0.7,   0.7     ),
        "NO5": (0.0,    -0.1    ),
        "SE1": (0.0,    0.0     ),
        "SE2": (0.0,    0.0     ),
        "SE3": (0.0,    0.0     ),
        "SE4": (0.0,    0.1     ),
    }

    for idx, row in full_gdf.iterrows():
        zone = row["zone_name"]
        centroid = row["geometry"].centroid
        dx, dy = offsets.get(zone, (0, 0))
        x_text, y_text = centroid.x + dx, centroid.y + dy

        # Add label
        ax.text(
            x_text,
            y_text,
            zone,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            # path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

        # # Add pointer line for DK zones
        # if zone in ["DK1", "DK2"]:
        #     ax.annotate(
        #         "",  # No text, just arrow
        #         xy=(centroid.x, centroid.y),
        #         xytext=(x_text, y_text),
        #         textcoords="data",
        #         arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
        #     )


    ax.set_title("Electricity Cost Across Nordic Bidding Zones")
    ax.axis("off")
    plt.tight_layout()

    output_path = os.path.join(data_path, f"nordic_cost_reduction_map.pdf")
    fig.savefig(output_path, bbox_inches='tight')

    
    # plt.savefig(output_path)
    # plt.close(fig)  # Don't display or hold plot in memory
    # plt.show()

    # Optionally save as PDF





if __name__ == "__main__":
    # main2()
    main3()

    