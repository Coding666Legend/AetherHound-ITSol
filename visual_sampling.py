
import numpy as np
import matplotlib.pyplot as plt

# AetherHound - Visual Sampling Analysis

# Signal parameters
frequency = 1000
amplitude = 5
duration = 0.005

# Generate original continuous-looking signal
continuous_time = np.linspace(0, duration, 2000)

continuous_signal = amplitude * np.sin(
    2 * np.pi * frequency * continuous_time + np.pi / 4
)

# Different sampling rates
sampling_rates = [1000, 2000, 5000]

# Create three graphs
plt.figure(figsize=(12, 9))

for i, sample_rate in enumerate(sampling_rates):

    # Generate sampling time values
    sample_time = np.arange(
        0, duration, 1 / sample_rate
    )

    # Generate sampled signal
    sampled_signal = amplitude * np.sin(
        2 * np.pi * frequency * sample_time + np.pi / 4
    )

    # Plot the graph
    plt.subplot(3, 1, i + 1)

    plt.plot(
        continuous_time,
        continuous_signal,
        label="Original Signal"
    )

    plt.stem(
        sample_time,
        sampled_signal,
        linefmt="r-",
        markerfmt="ro",
        basefmt="k-",
        label="Sampled Points"
    )

    plt.title(
        f"Sampling Rate = {sample_rate} Hz"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid(True)
    plt.legend()

plt.tight_layout()

plt.suptitle(
    "AetherHound - Visual Sampling Analysis",
    y=1.02
)

plt.show()
