
import numpy as np
import matplotlib.pyplot as plt

# DAC parameters
bits = 3
reference_voltage = 5

# Calculate total levels
levels = 2 ** bits

# Generate digital input values
digital_values = np.arange(levels)

# Convert digital values into analog voltage
analog_voltage = (
    digital_values / (levels - 1)
) * reference_voltage

# Display results
print("AetherHound - DAC Working Simulation")
print("DAC Resolution:", bits, "bits")
print("Total Levels:", levels)

for i in range(levels):
    print(
        "Digital:", format(i, "03b"),
        "Analog Voltage:", round(analog_voltage[i], 2), "V"
    )

# Plot DAC output
plt.figure(figsize=(10, 5))

plt.step(
    digital_values,
    analog_voltage,
    where="post",
    label="DAC Output"
)

plt.scatter(
    digital_values,
    analog_voltage,
    color="red",
    label="Voltage Levels"
)

plt.xlabel("Digital Input")
plt.ylabel("Analog Voltage (V)")
plt.title("AetherHound - DAC Working Simulation")
plt.grid()
plt.legend()
plt.show()
