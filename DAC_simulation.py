
import numpy as np
import matplotlib.pyplot as plt

# 1. Signal parameters
frequency = 1000
amplitude = 5
sample_rate = 8000
duration = 0.005

# 2. Generate original signal
time = np.linspace(
    0, duration, 2000, endpoint=False
)

original_signal = amplitude * np.sin(
    2 * np.pi * frequency * time
)

# 3. Generate digital samples
sample_time = np.arange(
    0, duration, 1 / sample_rate
)

digital_samples = amplitude * np.sin(
    2 * np.pi * frequency * sample_time
)

# 4. DAC quantization
bits = 8
levels = 2 ** bits

step_size = (2 * amplitude) / levels

digital_codes = np.floor(
    (digital_samples + amplitude) / step_size
)

digital_codes = np.clip(
    digital_codes, 0, levels - 1
)

# 5. Convert digital codes into voltage
dac_output = (
    -amplitude
    + digital_codes * step_size
    + step_size / 2
)

# 6. Display information
print("AetherHound - DAC Simulation")
print("DAC Resolution:", bits, "bits")
print("Total Levels:", levels)
print("Sampling Rate:", sample_rate, "Hz")
print("Quantization Step:", step_size, "Volts")

# 7. Plot results
plt.figure(figsize=(12, 6))

plt.plot(
    time,
    original_signal,
    label="Original Analog Signal"
)

plt.step(
    sample_time,
    dac_output,
    where="post",
    label="DAC Output"
)

plt.scatter(
    sample_time,
    dac_output,
    color="red",
    label="Digital Voltage Levels"
)

plt.title("AetherHound - DAC Simulation")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage")
plt.legend()
plt.grid(True)

plt.show()
