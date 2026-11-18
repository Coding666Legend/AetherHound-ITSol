
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Step 1: Define parameters
Ac = 5
fc = 5000
fm = 100
fs = 100000
duration = 0.04
delta_f = 500

# Step 2: Generate time array
t = np.arange(0, duration, 1/fs)

# Step 3: Generate original message
message = np.cos(2 * np.pi * fm * t)

# Step 4: Generate FM signal
beta = delta_f / fm

fm_signal = Ac * np.cos(
    2 * np.pi * fc * t
    + beta * np.sin(2 * np.pi * fm * t)
)

# Step 5: Apply Hilbert transform
analytic_signal = hilbert(fm_signal)

# Step 6: Extract instantaneous phase
phase = np.unwrap(np.angle(analytic_signal))

# Step 7: Calculate instantaneous frequency
inst_freq = np.gradient(phase, t) / (2 * np.pi)

# Step 8: Recover original message
recovered = (inst_freq - fc) / delta_f

# Step 9: Plot signals
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.plot(t, message)
plt.title("Original Message Signal")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, fm_signal)
plt.title("Frequency Modulated Signal")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, recovered)
plt.title("Recovered Message Signal")
plt.xlabel("Time (seconds)")
plt.grid()

plt.tight_layout()
plt.show()
