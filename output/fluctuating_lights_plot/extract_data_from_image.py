import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Create time vector with 1-second resolution
start_time = datetime.strptime("13:00", "%H:%M")
time_vec = [start_time + timedelta(seconds=i) for i in range(3600)]

# Define parameters
period = 600  # 10 minutes in seconds
high = 32
low = 5

predicted = []
actual = []

for i in range(3600):
    cycle_time = i % period
    is_high = cycle_time < 300  # Light on in the first half

    predicted.append(high if is_high else low)

    if is_high:
        time_in_light = cycle_time
        a = low + (high - low) * (1 - np.exp(-time_in_light / 60))
    else:
        time_in_dark = cycle_time - 300 if cycle_time >= 300 else 0
        a = high - (high - low) * (1 - np.exp(-time_in_dark / 60))
    actual.append(a)

# Save as CSV
df = pd.DataFrame({
    "time": [t.strftime("%H:%M:%S") for t in time_vec],
    "A_predicted": predicted,
    "A_actual": actual
})
df.to_csv("photosynthesis_highres_data.csv", index=False)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time_vec, predicted, label="Predicted CO₂ uptake rate", color="green")
plt.plot(time_vec, actual, '--', label="Typical lags in response", color="green")
plt.fill_between(time_vec, predicted, actual,
                 where=np.array(predicted) > np.array(actual),
                 interpolate=True, color="green", alpha=0.3,
                 label="Loss in photosynthetic efficiency")

plt.xlabel("Time (hours)")
plt.ylabel("A (μmol m⁻² s⁻¹)")
plt.ylim(0, 35)
plt.title("Recreation of CO₂ Uptake with Light Fluctuations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
