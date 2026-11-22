
import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
fs = 5000
f = 100
duration = 1
A = 1

# Generate time samples
t = np.arange(0, duration, 1/fs)

# Generate clean signal
clean_signal = A * np.sin(2 * np.pi * f * t)

# Add normal background noise
normal_signal = clean_signal + np.random.normal(
    0, 0.05, len(t)
)

# Create anomalous signal
anomalous_signal = normal_signal.copy()

# Generate sudden noise burst
noise_burst = np.random.normal(
    0, 1.5, 500
)

# Introduce noise anomaly
anomalous_signal[2000:2500] += noise_burst

# Plot signals
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(t, normal_signal)
plt.title("Normal RF Signal")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, anomalous_signal)
plt.title("RF Signal with Noise Anomaly")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
