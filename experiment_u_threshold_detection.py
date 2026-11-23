
import pandas as pd
import matplotlib.pyplot as plt

# Load RF dataset
df = pd.read_csv("atherhound_rf_dataset.csv")

# Define detection threshold
threshold = 2.0

# Detect suspicious signal values
df["detected"] = (
    df["signal"].abs() > threshold
).astype(int)

# Display detection results
print("Threshold-Based RF Anomaly Detection")

print("\nDetection Summary:")
print(
    df.groupby("anomaly_type")["detected"]
    .agg(["sum", "count"])
)

# Visualize amplitude anomaly
amplitude_data = df[
    df["anomaly_type"] == "amplitude"
]

plt.figure(figsize=(12, 5))

plt.plot(
    amplitude_data["time"],
    amplitude_data["signal"],
    label="RF Signal"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Upper Threshold"
)

plt.axhline(
    -threshold,
    linestyle="--",
    label="Lower Threshold"
)

plt.title("AetherHound - Threshold-Based Detection")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()
plt.grid()
plt.show()

# Save detection results
df.to_csv(
    "atherhound_detection_results.csv",
    index=False
)

print("\nDetection completed successfully!")
