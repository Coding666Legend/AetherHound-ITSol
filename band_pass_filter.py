
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Step 1: Sampling parameters
fs = 1000
t = np.arange(0, 1, 1/fs)

# Step 2: Generate signals
signal_10 = np.sin(2 * np.pi * 10 * t)
signal_100 = np.sin(2 * np.pi * 100 * t)
signal_200 = np.sin(2 * np.pi * 200 * t)

# Step 3: Combine signals
signal = signal_10 + signal_100 + signal_200

# Step 4: Design Band-Pass Filter
low_cutoff = 50
high_cutoff = 150

b, a = butter(
    4,
    [low_cutoff, high_cutoff],
    btype='band',
    fs=fs
)

# Step 5: Apply filter
filtered_signal = filtfilt(b, a, signal)

# Step 6: Plot results
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal (10, 100, 200 Hz)")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal)
plt.title("Band-Pass Filtered Signal (50–150 Hz)")
plt.xlabel("Time (seconds)")
plt.grid()

plt.tight_layout()
plt.show()
