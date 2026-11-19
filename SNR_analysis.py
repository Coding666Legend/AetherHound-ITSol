
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate signal
fs = 10000
t = np.arange(0, 0.05, 1/fs)

signal = np.cos(2 * np.pi * 100 * t)

# Step 2: Generate noise
noise = np.random.normal(0, 0.5, len(t))

# Step 3: Add signal and noise
noisy_signal = signal + noise

# Step 4: Calculate power
signal_power = np.mean(signal ** 2)
noise_power = np.mean(noise ** 2)

# Step 5: Calculate SNR
snr = 10 * np.log10(signal_power / noise_power)

print("Signal Power:", signal_power)
print("Noise Power:", noise_power)
print("SNR:", snr, "dB")

# Step 6: Plot signals
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, noisy_signal)
plt.title("Signal with Noise")
plt.grid()

plt.tight_layout()
plt.show()
