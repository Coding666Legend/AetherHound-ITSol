# AetherHound - Sampling Analysis

frequency = 1000

# Different sampling rates

sample_rate = 10000

# Calculate Nyquist rate

nyquist_rate = 2 * frequency

# Calculate samples per cycle

samples_per_cycle = sample_rate / frequency

print("AetherHound Sampling Analysis")

print("Signal Frequency:", frequency, "Hz")

print("Sampling Rate:", sample_rate, "Hz")

print("Nyquist Rate:", nyquist_rate, "Hz")

print("Samples Per Cycle:", samples_per_cycle)

if sample_rate > nyquist_rate:
    print("Sampling condition: Satisfied")
else:
    print("Sampling condition: Not satisfied")