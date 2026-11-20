
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Step 1: Sampling parameters
fs = 1000
t = np.arange(0, 1, 1/fs)

# Step 2: Generate signals
low_freq = np.sin(2 * np.pi * 10 * t)
high_freq = 0.5 * np.sin(2 * np.pi * 100 * t)

signal = low_freq + high_freq

# Step 3: Design High-Pass Filter
cutoff = 30
b, a = butter(4, cutoff, btype='high', fs=fs)

# Step 4: Apply filter
filtered_signal = filtfilt(b, a, signal)

# Step 5: Plot results
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal (10 Hz + 100 Hz)")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal)
plt.title("Filtered Signal (High-Pass Filter)")
plt.xlabel("Time (seconds)")
plt.grid()

plt.tight_layout()
plt.show()
