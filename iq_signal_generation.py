
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
frequency = 100
sampling_rate = 10000
duration = 0.05
amplitude = 1

# Step 2: Generate time array
t = np.arange(0, duration, 1/sampling_rate)

# Step 3: Generate I signal (Cosine)
I = amplitude * np.cos(2 * np.pi * frequency * t)

# Step 4: Generate Q signal (Sine)
Q = amplitude * np.sin(2 * np.pi * frequency * t)

# Step 5: Plot I and Q signals
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(t, I)
plt.title("I Signal - In Phase")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, Q)
plt.title("Q Signal - Quadrature")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
