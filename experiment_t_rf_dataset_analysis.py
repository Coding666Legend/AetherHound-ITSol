
import pandas as pd
import matplotlib.pyplot as plt

# Load RF dataset
df = pd.read_csv("atherhound_rf_dataset.csv")

# Display dataset information
print("Dataset Information:")
print(df.info())

# Statistical analysis
print("\nStatistical Summary:")

print(
    df.groupby("anomaly_type")["signal"].agg(
        ["mean", "std", "min", "max"]
    )
)

# Signal categories
categories = [
    "normal",
    "amplitude",
    "frequency",
    "noise"
]

# Visualize signals
plt.figure(figsize=(12, 9))

for i, category in enumerate(categories):
    signal_data = df[df["anomaly_type"] == category]

    plt.subplot(4, 1, i + 1)

    plt.plot(
        signal_data["time"],
        signal_data["signal"]
    )

    plt.title(category.capitalize() + " RF Signal")
    plt.ylabel("Amplitude")
    plt.grid()

plt.xlabel("Time (seconds)")
plt.tight_layout()
plt.show()

print("\nRF Dataset Analysis Completed!")
