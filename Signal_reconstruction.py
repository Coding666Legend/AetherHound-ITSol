
import numpy as np
import matplotlib.pyplot as plt

# 1. Signal parameters
frequency = 1000
amplitude = 5
sample_rate = 8000
duration = 0.01

# 2. Generate original continuous signal
continuous_time = np.linspace(
    0, duration, 10000, endpoint=False
)

original_signal = amplitude * np.sin(
    2 * np.pi * frequency * continuous_time
)

# 3. Generate sampled signal
sample_time = np.arange(
    0, duration, 1 / sample_rate
)

sampled_signal = amplitude * np.sin(
    2 * np.pi * frequency * sample_time
)

# 4. Reconstruct using linear interpolation
reconstructed_signal = np.interp(
    continuous_time,
    sample_time,
    sampled_signal
)

# 5. Display results
print("AetherHound - Signal Reconstruction")
print("Original Frequency:", frequency, "Hz")
print("Sampling Rate:", sample_rate, "Hz")
print("Total Samples:", len(sampled_signal))

# 6. Plot all signals
plt.figure(figsize=(12, 6))

plt.plot(
    continuous_time,
    original_signal,
    label="Original Signal"
)

plt.scatter(
    sample_time,
    sampled_signal,
    color="red",
    label="Sampled Points"
)

plt.plot(
    continuous_time,
    reconstructed_signal,
    "--",
    label="Reconstructed Signal"
)

plt.title("AetherHound - Signal Reconstruction")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.show()
