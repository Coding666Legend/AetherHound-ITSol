
import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
carrier_frequency = 1000
message_frequency = 50
carrier_amplitude = 5
modulation_index = 0.5

sample_rate = 10000
duration = 0.05

# Generate time values
time = np.arange(
    0, duration, 1/sample_rate
)

# Generate message signal
message_signal = np.cos(
    2 * np.pi * message_frequency * time
)

# Generate carrier signal
carrier_signal = carrier_amplitude * np.cos(
    2 * np.pi * carrier_frequency * time
)

# Generate AM signal
am_signal = carrier_amplitude * (
    1 + modulation_index * message_signal
) * np.cos(
    2 * np.pi * carrier_frequency * time
)

# Display information
print("AetherHound - AM Modulation")
print("Carrier Frequency:", carrier_frequency, "Hz")
print("Message Frequency:", message_frequency, "Hz")
print("Modulation Index:", modulation_index)

# Plot signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, message_signal)
plt.title("Message Signal")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, carrier_signal)
plt.title("Carrier Signal")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, am_signal)
plt.title("Amplitude Modulated Signal")
plt.xlabel("Time (seconds)")
plt.grid()

plt.tight_layout()
plt.show()
