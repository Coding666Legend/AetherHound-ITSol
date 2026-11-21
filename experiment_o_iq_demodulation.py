
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parameters
A = 1
fc = 100
fm = 20
fs = 5000
duration = 0.1

# Time samples
t = np.arange(0, duration, 1/fs)

# Original I and Q signals
I = A * np.cos(2 * np.pi * fm * t)
Q = A * np.sin(2 * np.pi * fm * t)

# Carrier signals
carrier_I = np.cos(2 * np.pi * fc * t)
carrier_Q = np.sin(2 * np.pi * fc * t)

# Modulated signal
S = I * carrier_I - Q * carrier_Q

# Demodulation
I_mixed = 2 * S * carrier_I
Q_mixed = -2 * S * carrier_Q

# Low-pass filter
b, a = butter(4, 50/(fs/2), btype='low')

I_recovered = filtfilt(b, a, I_mixed)
Q_recovered = filtfilt(b, a, Q_mixed)

# Plot signals
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(t, I, label="Original I")
plt.plot(t, I_recovered, '--', label="Recovered I")
plt.title("I Signal Recovery")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, Q, label="Original Q")
plt.plot(t, Q_recovered, '--', label="Recovered Q")
plt.title("Q Signal Recovery")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
