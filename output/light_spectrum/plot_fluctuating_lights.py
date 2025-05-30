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
import colorsys
def rgb_to_hsv_array(rgb_arr):
    """Convert an (H, W, 3) RGB image array to HSV using colorsys."""
    hsv_img = np.zeros_like(rgb_arr, dtype=float)
    for y in range(rgb_arr.shape[0]):
        for x in range(rgb_arr.shape[1]):
            r, g, b = rgb_arr[y, x] / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_img[y, x] = [h * 360, s * 100, v * 100]  # Convert to degrees and %
    return hsv_img

def is_in_hsv_range(hsv_pixel, h_min, h_max, s_min, s_max, v_min, v_max):
    h, s, v = hsv_pixel
    return (h_min <= h <= h_max) and (s_min <= s <= s_max) and (v_min <= v <= v_max)

def main():
    # Load the image
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    img_path = os.path.join(data_path, "spectrum.png")
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Convert RGB to HSV
    hsv_img = rgb_to_hsv_array(img_np)

    # Define HSV ranges (h: 0–360, s/v: 0–100)
    red_range    = (0, 20, 50, 100, 50, 100)    # Red
    green_range  = (80, 160, 30, 100, 30, 100)  # Green
    blue_range   = (200, 260, 30, 100, 30, 100) # Blue

    height, width, _ = img_np.shape
    red_y, green_y, blue_y = [], [], []

    for x in range(width):
        red_found = green_found = blue_found = False
        r_vals, g_vals, b_vals = [], [], []

        for y in range(min(0, height), min(250, height)):  # Limit y-range to 0–250
            hsv_pixel = hsv_img[y, x]

            if is_in_hsv_range(hsv_pixel, *red_range):
                r_vals.append(y)
            if is_in_hsv_range(hsv_pixel, *green_range):
                g_vals.append(y)
            if is_in_hsv_range(hsv_pixel, *blue_range):
                b_vals.append(y)

        red_y.append(np.mean(r_vals) if r_vals else np.nan)
        green_y.append(np.mean(g_vals) if g_vals else np.nan)
        blue_y.append(np.mean(b_vals) if b_vals else np.nan)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(img_np)
    plt.plot(range(width), red_y, color="red", label="Red", linewidth=1)
    plt.plot(range(width), green_y, color="green", label="Green", linewidth=1)
    plt.plot(range(width), blue_y, color="blue", label="Blue", linewidth=1)
    plt.legend()
    plt.title("Extracted Color Lines using HSV (Standard Libraries Only)")
    plt.show()
        

if __name__ == '__main__':
    main()