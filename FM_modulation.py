
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
Ac = 5
fc = 5000
fm = 100
fs = 100000
duration = 0.04

# Step 2: Generate time array
t = np.arange(0, duration, 1/fs)

# Step 3: Generate message signal
message = np.cos(2 * np.pi * fm * t)

# Step 4: Define frequency deviation
delta_f = 500

# Step 5: Calculate modulation index
beta = delta_f / fm

# Step 6: Generate FM signal
fm_signal = Ac * np.cos(
    2 * np.pi * fc * t
    + beta * np.sin(2 * np.pi * fm * t)
)

# Step 7: Calculate instantaneous frequency
instantaneous_frequency = (
    fc + delta_f * np.cos(2 * np.pi * fm * t)
)

# Step 8: Plot the signals
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.plot(t, message)
plt.title("Original Message Signal")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, fm_signal)
plt.title("Frequency Modulated (FM) Signal")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, instantaneous_frequency)
plt.title("Instantaneous Frequency")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.grid()

plt.tight_layout()
plt.show()
