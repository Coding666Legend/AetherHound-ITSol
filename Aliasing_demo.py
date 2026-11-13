
import numpy as np
import matplotlib.pyplot as plt

# AetherHound - Aliasing Demonstration

# Original signal parameters
frequency = 1200
amplitude = 5

# Sampling parameters
sample_rate = 5000
duration = 0.02

# Calculate Nyquist rate
nyquist_rate = 2 * frequency

# Calculate alias frequency
alias_frequency = abs(
    (frequency + sample_rate / 2) % sample_rate
    - sample_rate / 2
) 

# Generate original continuous-looking signal
continuous_time = np.linspace(0, duration, 2000)

original_signal = amplitude * np.sin(
    2 * np.pi * frequency * continuous_time
)

# Generate sampling time values
sample_time = np.arange(
    0, duration, 1 / sample_rate
)

# Generate sampled signal
sampled_signal = amplitude * np.sin(
    2 * np.pi * frequency * sample_time
)

# Generate apparent alias signal
alias_signal = amplitude * np.sin(
    2 * np.pi * alias_frequency * continuous_time
)

# Display parameters
print("AetherHound - Aliasing Analysis")
print("Original Frequency:", frequency, "Hz")
print("Sampling Rate:", sample_rate, "Hz")
print("Nyquist Rate:", nyquist_rate, "Hz")
print("Alias Frequency:", alias_frequency, "Hz")

# Plot graphs
plt.figure(figsize=(12, 8))

# Original signal and sampled points
plt.subplot(2, 1, 1)

plt.plot(
    continuous_time,
    original_signal,
    label="Original 1200 Hz Signal"
)

plt.stem(
    sample_time,
    sampled_signal,
    linefmt="r-",
    markerfmt="ro",
    basefmt="k-",
    label="Sampled Points"
)

plt.title("Original Signal and Sampled Points")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Compare original and alias signals
plt.subplot(2, 1, 2)

plt.plot(
    continuous_time,
    original_signal,
    label="Original 1200 Hz Signal"
)

plt.plot(
    continuous_time,
    alias_signal,
    "--",
    label=f"Apparent {alias_frequency:.0f} Hz Alias"
)

plt.scatter(
    sample_time,
    sampled_signal,
    color="red",
    label="Sampled Points"
)

plt.title("Aliasing Effect")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
