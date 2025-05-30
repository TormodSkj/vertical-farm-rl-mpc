import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import os

# def main():
#     # Load data

#     data_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
#     filepath = os.path.join(data_path, "photosynthesis_data.csv")

#     df = pd.read_csv(filepath)
#     df['time'] = pd.to_datetime(df['time'], format="%H:%M")

#     # Plot
#     fig, ax = plt.subplots(figsize=(7, 4))

#     # Solid line: predicted
#     ax.plot(df['time'], df['A_predicted'], label='Predicted CO₂ uptake rate', color='green')

#     # Dashed line: actual with lag
#     ax.plot(df['time'], df['A_actual'], linestyle='--', label='Typical lags in response', color='green')

#     # Fill loss area
#     ax.fill_between(df['time'], df['A_predicted'], df['A_actual'], 
#                     where=(df['A_predicted'] > df['A_actual']),
#                     interpolate=True, color='green', alpha=0.3,
#                     label='Loss in photosynthetic efficiency during light fluctuations')

#     # Axis formatting
#     ax.set_ylabel("A (μmol m⁻² s⁻¹)")
#     ax.set_xlabel("Time (hours)")
#     ax.set_ylim(0, 35)
#     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Load the image

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    img_path = os.path.join(data_path, "original_plot.png")
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Define color thresholds
    dark_green_range = ((0, 70, 20), (74, 140, 100))   # Rough RGB range for dark green
    light_green_range = ((179, 204, 189), (179, 204, 189))  # Rough RGB range for light green

    height, width, _ = img_np.shape
    dark_green_y = []
    light_green_y = []

    x_range = range(60, 396)
    y_range = range(30, 230)
    y_offset = min(y_range)
    y_max = max(y_range)

    for x in x_range:
        column = img_np[:, x]

        column = column[min(y_range):max(y_range)]
        
        # Mask for dark green pixels
        dark_mask = np.all((column >= dark_green_range[0]) & (column <= dark_green_range[1]), axis=1)
        if np.any(dark_mask):
            y_vals = np.where(dark_mask)[0]
            dark_green_y.append(np.mean(y_vals))
        else:
            dark_green_y.append(np.nan)
        
        # Mask for light green pixels
        light_mask = np.all((column >= light_green_range[0]) & (column <= light_green_range[1]), axis=1)
        if np.any(light_mask):
            y_vals = np.where(light_mask)[0]
            light_green_y.append(np.max(y_vals))
        else:
            light_green_y.append(np.nan)

    dark_green_y = y_max - (np.array(dark_green_y) + y_offset)
    light_green_y = y_max - (np.array(light_green_y) + y_offset)

    # Display how the extracted lines look
    plt.figure(figsize=(12, 6))
    # plt.imshow(img_np)
    plt.plot(x_range, dark_green_y, color="#004616", label='Dark Green (mean y)', linewidth=1)
    plt.plot(x_range, light_green_y, color="#004616", label='Light Green (min y)', linewidth=1, linestyle='--')
    plt.fill_between(x_range, dark_green_y, light_green_y, color="#B3D6BC", label='Light Green (min y)', linewidth=1)
    plt.legend()
    plt.title("Extracted Data Points from Image")
    plt.show()


if __name__ == '__main__':
    main()