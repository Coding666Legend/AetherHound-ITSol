
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
fs = 10000
duration = 0.05
fm = 100
t = np.arange(0, duration, 1/fs)

# Step 2: Generate original message
message = np.cos(2 * np.pi * fm * t)

# Step 3: Generate Gaussian noise
noise = np.random.normal(0, 0.5, len(t))

# Step 4: Add noise to message
noisy_signal = message + noise

# Step 5: Plot signals
plt.figure(figsize=(12, 7))

plt.subplot(3, 1, 1)
plt.plot(t, message)
plt.title("Original Message Signal")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, noise)
plt.title("Gaussian Noise")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, noisy_signal)
plt.title("Message Signal with Noise")
plt.xlabel("Time (seconds)")
plt.grid()

plt.tight_layout()
plt.show()
