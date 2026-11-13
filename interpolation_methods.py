
import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
frequency = 1000
amplitude = 5
sample_rate = 4000
duration = 0.005

# Original signal
continuous_time = np.linspace(
    0, duration, 2000, endpoint=False
)

original_signal = amplitude * np.sin(
    2 * np.pi * frequency * continuous_time
)

# Sampled signal
sample_time = np.arange(
    0, duration, 1 / sample_rate
)

sampled_signal = amplitude * np.sin(
    2 * np.pi * frequency * sample_time
)

# Linear interpolation
linear_signal = np.interp(
    continuous_time,
    sample_time,
    sampled_signal
)

# Zero-order hold
indices = np.searchsorted(
    sample_time, continuous_time, side="right"
) - 1

zoh_signal = sampled_signal[indices]

# Display graph
plt.figure(figsize=(12, 6))

plt.plot(
    continuous_time,
    original_signal,
    label="Original Signal"
)

plt.step(
    continuous_time,
    zoh_signal,
    where="post",
    label="Zero Order Hold"
)

plt.plot(
    continuous_time,
    linear_signal,
    "--",
    label="Linear Interpolation"
)

plt.scatter(
    sample_time,
    sampled_signal,
    color="red",
    label="Sampled Points"
)

plt.title("AetherHound - Interpolation Methods")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.show()
