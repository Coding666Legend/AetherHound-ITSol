
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURES = [
    "signal_power",
    "spectral_entropy",
    "peak_frequency",
    "bandwidth",
    "noise_level"
]


def generate_dataset():
    rng = np.random.default_rng(42)

    normal = np.column_stack([
        rng.normal(0.5, 0.08, 1000),
        rng.normal(0.7, 0.05, 1000),
        rng.normal(1000, 100, 1000),
        rng.normal(200, 30, 1000),
        rng.normal(0.1, 0.02, 1000)
    ])

    anomaly = np.column_stack([
        rng.normal(0.9, 0.08, 200),
        rng.normal(0.3, 0.07, 200),
        rng.normal(1800, 150, 200),
        rng.normal(500, 60, 200),
        rng.normal(0.4, 0.05, 200)
    ])

    X = np.vstack([normal, anomaly])
    y = np.concatenate([
        np.zeros(1000, dtype=int),
        np.ones(200, dtype=int)
    ])

    df = pd.DataFrame(X, columns=FEATURES)
    df["label"] = y

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / "synthetic_signals.csv"
    df.to_csv(output_path, index=False)

    print("Dataset generated successfully!")
    print("Normal samples: 1000")
    print("Anomaly samples: 200")
    print("Total samples:", len(df))
    print("Saved to:", output_path)


if __name__ == "__main__":
    generate_dataset()
