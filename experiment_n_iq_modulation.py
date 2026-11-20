
import numpy as np
import matplotlib.pyplot as plt

# Parameters
A = 1
fc = 100
fs = 5000
duration = 0.05

# Time samples
t = np.arange(0, duration, 1/fs)

# I and Q signals
I = A * np.cos(2 * np.pi * 20 * t)
Q = A * np.sin(2 * np.pi * 20 * t)

# Carrier signals
carrier_I = np.cos(2 * np.pi * fc * t)
carrier_Q = np.sin(2 * np.pi * fc * t)

# I/Q modulation
S = I * carrier_I - Q * carrier_Q

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t, S, label="Modulated RF Signal")

plt.title("AetherHound - I/Q Modulation")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.show()
