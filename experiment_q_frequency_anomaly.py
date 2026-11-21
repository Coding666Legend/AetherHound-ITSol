
import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
fs = 5000
f_normal = 100
f_anomaly = 300
duration = 1

# Generate time samples
t = np.arange(0, duration, 1/fs)

# Generate normal signal
normal_signal = np.sin(2 * np.pi * f_normal * t)

# Create frequency array
frequency = np.full(len(t), f_normal)

# Introduce frequency anomaly
frequency[2000:2500] = f_anomaly

# Generate continuous phase
phase = np.zeros(len(t))
phase[1:] = 2 * np.pi * np.cumsum(frequency[:-1]) / fs

# Generate anomalous signal
anomalous_signal = np.sin(phase)

# Plot signals
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(t, normal_signal)
plt.title("Normal RF Signal - 100 Hz")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, anomalous_signal)
plt.title("RF Signal with Frequency Anomaly")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
