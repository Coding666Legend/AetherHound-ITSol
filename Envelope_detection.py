
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define signal parameters
Ac = 5
fc = 5000
fm = 100
m = 0.5
fs = 100000
duration = 0.04

# Step 2: Generate time values
t = np.arange(0, duration, 1/fs)

# Step 3: Generate message signal
message = np.cos(2 * np.pi * fm * t)

# Step 4: Generate AM signal
carrier = Ac * np.cos(2 * np.pi * fc * t)
am_signal = Ac * (1 + m * message) * np.cos(2 * np.pi * fc * t)

# Step 5: Half-wave rectification (diode simulation)
rectified = np.maximum(am_signal, 0)

# Step 6: Capacitor and resistor simulation
R = 10000
C = 47e-9
dt = 1 / fs
alpha = np.exp(-dt / (R * C))

envelope = np.zeros(len(t))

for i in range(1, len(t)):
    if rectified[i] > envelope[i-1]:
        envelope[i] = rectified[i]
    else:
        envelope[i] = alpha * envelope[i-1]

# Step 7: Recover original message
recovered = (envelope - Ac) / (Ac * m)

# Step 8: Plot the signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, message)
plt.title("Original Message Signal")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, am_signal)
plt.title("AM Modulated Signal")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, recovered)
plt.title("Recovered Message Signal")
plt.xlabel("Time (seconds)")
plt.grid()

plt.tight_layout()
plt.show()
