# AetherHound - RF Parameters

frequency = 1000
amplitude = 5
sample_rate = 10000
duration = 2

# Calculate total samples

total_samples = sample_rate * duration

# Display information

print("AetherHound RF Module")

print("Frequency:", frequency, "Hz")
print("Amplitude:", amplitude)
print("Sample Rate:", sample_rate, "Hz")
print("Duration:", duration, "seconds")

print("Total Samples:", total_samples)