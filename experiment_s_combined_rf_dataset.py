
import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Signal parameters
fs = 5000
f = 100
duration = 1
A = 1

# Time samples
t = np.arange(0, duration, 1/fs)
N = len(t)

# 1. Normal signal
normal = A * np.sin(2 * np.pi * f * t)
normal += np.random.normal(0, 0.05, N)

# 2. Amplitude anomaly
amplitude = normal.copy()
amplitude[2000:2500] *= 5

# 3. Frequency anomaly
frequency = np.full(N, f)
frequency[2000:2500] = 300

phase = np.zeros(N)
phase[1:] = 2 * np.pi * np.cumsum(frequency[:-1]) / fs

frequency_signal = np.sin(phase)
frequency_signal += np.random.normal(0, 0.05, N)

# 4. Noise anomaly
noise = normal.copy()
noise[2000:2500] += np.random.normal(0, 1.5, 500)

# Combine all signals
signals = [
    ("normal", normal, 0),
    ("amplitude", amplitude, 1),
    ("frequency", frequency_signal, 1),
    ("noise", noise, 1)
]

# Create dataset
data = []

for sample_id, (name, signal, label) in enumerate(signals):
    for i in range(N):
        data.append({
            "sample_id": sample_id,
            "time": t[i],
            "signal": signal[i],
            "label": label,
            "anomaly_type": name
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save dataset
df.to_csv("atherhound_rf_dataset.csv", index=False)

print("Dataset created successfully!")
print("Total rows:", len(df))
print("\nDataset preview:")
print(df.head())

print("\nAnomaly distribution:")
print(df.groupby("anomaly_type")["label"].first())
