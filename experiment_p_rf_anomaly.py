
import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
fs = 5000
f = 100
duration = 1
A = 1

# Generate time samples
t = np.arange(0, duration, 1/fs)

# Generate normal RF signal
normal_signal = A * np.sin(2 * np.pi * f * t)

# Add small background noise
normal_signal += np.random.normal(0, 0.05, len(t))

# Generate anomalous RF signal
anomalous_signal = normal_signal.copy()

# Introduce amplitude anomaly
anomalous_signal[2000:2500] *= 5

# Plot signals
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(t, normal_signal)
plt.title("Normal RF Signal")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, anomalous_signal)
plt.title("RF Signal with Amplitude Anomaly")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
