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

        data['Clearing Price Up'] = pd.to_numeric(data['Clearing Price Up'].str.replace(',', '.'), errors='coerce')
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

    all_files = [f for f in os.listdir(data_folder) if "Nordpool_BalanceMarket" in f and f.endswith(".csv")]
    all_data = []

    for file in all_files:
        filepath = os.path.join(data_folder, file)

        # Load the mFRR data
        data = pd.read_csv(filepath, delimiter=";", encoding="utf-8")

        data = data[['Delivery Start (CET)'] + [column for column in data.columns if bidding_zone in column]]

        data[['Start Time']] = data['Delivery Start (CET)'].str.extract(
            r'(\d{2}.\d{2}.\d{4} \d{2}:\d{2}:\d{2})'
        )
        data['Start Time'] = pd.to_datetime(data['Start Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

        up_price_col    = f"{bidding_zone} Up Price (EUR)"     
        down_price_col  = f"{bidding_zone} Down Price (EUR)" 

        data = data[['Start Time', up_price_col, down_price_col]].rename(
            columns={up_price_col: 'Clearing Price Up', down_price_col: 'Clearing Price Down'}
        )

        data['Clearing Price Up']   = pd.to_numeric(data['Clearing Price Up'],   errors='coerce')
        data['Clearing Price Down'] = pd.to_numeric(data['Clearing Price Down'], errors='coerce')
        
        # Append processed data to the list
        all_data.append(data)

    assert len(all_data) > 0, 'Expected non-empty list of data. Verify correctly specified import path.'

    # Merge all data and sort by 'Start Time'
    merged_data = pd.concat(all_data, ignore_index=True)
    merged_data.sort_values(by='Start Time', inplace=True)
    # merged_data.fillna(0, inplace=True)

    return merged_data


def load_nordpool_activation_data(data_folder, bidding_zone):
    """

    """

    all_files = [f for f in os.listdir(data_folder) if "Nordpool_BalanceMarket" in f and f.endswith(".csv")]
    all_data = []

    for file in all_files:
        filepath = os.path.join(data_folder, file)

        # Load the mFRR data
        data = pd.read_csv(filepath, delimiter=";", encoding="utf-8")

        data = data[['Delivery Start (CET)'] + [column for column in data.columns if bidding_zone in column]]

        data[['Start Time']] = data['Delivery Start (CET)'].str.extract(
            r'(\d{2}.\d{2}.\d{4} \d{2}:\d{2}:\d{2})'
        )
        data['Start Time'] = pd.to_datetime(data['Start Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

        # Convert 'Offered' and 'Activated' to numeric
        data[['Offered Up', 'Activated Up']] = data[[f"{bidding_zone} Accepted Up Volume (MW)" , f"{bidding_zone} Activated Up Volume (MW)" ]].apply(pd.to_numeric, errors='coerce')
        data[['Offered Down', 'Activated Down']] = data[[f"{bidding_zone} Accepted Down Volume (MW)" , f"{bidding_zone} Activated Down Volume (MW)" ]].apply(pd.to_numeric, errors='coerce')
        
        data = data[['Start Time', 'Offered Up', 'Activated Up', 'Offered Down', 'Activated Down']]

        # Append to the list
        all_data.append(data)
    
    # Concatenate all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by Start Time
    combined_df.sort_values(by='Start Time', inplace=True)
    
    return combined_df



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
    and store the data in a CSV file without duplicates.
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
                [th.text.strip().lstrip('MBA|') for th in bidding_zone_row.find_all("th")][2:], 2
            ))

            headers_row = table.find_all("tr")[2]
            headers_cols = [th.text.strip()[:-10].strip() for th in headers_row.find_all("th")]

            new_header_cols = ['Date', 'Hour'] + [
                f"{bidding_zone_cols[i]} {direction_prefix} {headers_cols[i]}" for i in range(len(headers_cols))
            ]

            zone_indices = {zone: i for i, zone in enumerate(new_header_cols) if 'NO' in zone}

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
        existing_df = pd.read_csv(target_file_path)
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
                f"&dataItems.values=PRICE&dataItems.values=VOLUME"
                f"&reserveType.values=A97&balancingTypes=TERTIARY&reserveSource.values=ALL&aFRRmFRRType.values=A47"
            )

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
                    existing_df = pd.read_csv(target_file_path)
                except FileNotFoundError:
                    existing_df = pd.DataFrame()

                # Remove old entries for the current date to avoid duplicates
                existing_df = existing_df[existing_df["Date"] != date_str]

                # Append new data and save
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_csv(target_file_path, index=False)

                # print(f"Data for {date_str} saved to {target_file_path}")

                # Sort file to ensure it's always chronological
                df = pd.read_csv(target_file_path)
                df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
                df = df.sort_values(by=["Date", "Hour"])
                df["Date"] = df["Date"].dt.strftime("%d.%m.%Y")
                df.to_csv(target_file_path, index=False)
            
            else:
                print(f"WARNING: Data for {date_str} was empty")


            time.sleep(np.random.uniform(0.1, 0.5))
            pbar.update(1)

    
    print(f"Successfully imported data from {start_date} to {end_date}")

