# AetherHound - Sine Wave Generation

import numpy as np
import matplotlib.pyplot as plt

# Signal parameters

frequency = 1000
amplitude = 10
sample_rate = 100000
duration = 0.01

# Generate time values

time = np.arange(0, duration, 1/sample_rate)

# Generate sine wave

signal = amplitude * np.sin(2 * np.pi * frequency * time)

# Display the signal

plt.plot(time, signal)

plt.title("AetherHound - RF Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.show()