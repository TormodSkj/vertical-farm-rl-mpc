import os
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime, timedelta




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


# Example use:
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/"
target_file = root_path + 'nucs_data.csv'
fetch_CM_data_nucs(target_file, "12-02-2024", "14-03-2024")