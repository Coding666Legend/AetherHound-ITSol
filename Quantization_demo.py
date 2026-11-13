
import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
frequency = 5
amplitude = 5
sample_rate = 1000
duration = 1

# Generate time values
time = np.arange(0, duration, 1/sample_rate)

# Generate original sine wave
signal = amplitude * np.sin(2 * np.pi * frequency * time)

# Quantization function
def quantize(signal, bits):

    levels = 2 ** bits

    minimum = -amplitude
    maximum = amplitude

    step = (maximum - minimum) / (levels - 1)

    quantized = np.round(
        (signal - minimum) / step
    ) * step + minimum

    return quantized


# Quantize at different resolutions
signal_2bit = quantize(signal, 2)
signal_4bit = quantize(signal, 4)
signal_8bit = quantize(signal, 8)

# Plot comparison
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, signal, label="Original Signal")
plt.step(time, signal_2bit, where="post", label="2-bit")
plt.title("2-bit Quantization")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, signal, label="Original Signal")
plt.step(time, signal_4bit, where="post", label="4-bit")
plt.title("4-bit Quantization")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, signal, label="Original Signal")
plt.step(time, signal_8bit, where="post", label="8-bit")
plt.title("8-bit Quantization")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print("AetherHound - Quantization Analysis")

for bits in [2, 4, 8]:
    print("Resolution:", bits, "bits")
    print("Levels:", 2**bits)
