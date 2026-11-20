
import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
A = 1
f = 100
fs = 5000
duration = 0.05

# Generate time samples
t = np.arange(0, duration, 1/fs)

# Generate I and Q signals
I = A * np.cos(2 * np.pi * f * t)
Q = A * np.sin(2 * np.pi * f * t)

# Plot both signals
plt.figure(figsize=(10, 5))

plt.plot(t, I, label="I Signal (Cosine)")
plt.plot(t, Q, label="Q Signal (Sine)")

plt.title("AetherHound - I/Q Signal Generation")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.show()
