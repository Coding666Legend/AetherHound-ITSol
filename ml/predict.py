
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from ml.autoencoder import Autoencoder

ROOT = Path(__file__).resolve().parent.parent


def predict_signal(signal_features):
    with open(ROOT / "models" / "config.json") as f:
        config = json.load(f)

    scaler = joblib.load(ROOT / "models" / "scaler.pkl")

    model = Autoencoder(config["input_dim"])
    model.load_state_dict(
        torch.load(
            ROOT / "models" / "autoencoder.pt",
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    sample = pd.DataFrame(
        [signal_features],
        columns=config["features"]
    )

    scaled = scaler.transform(sample)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        error = torch.mean(
            (tensor - output) ** 2
        ).item()

    return {
        "anomaly_score": float(error),
        "threshold": config["threshold"],
        "is_anomaly": bool(error > config["threshold"])
    }


if __name__ == "__main__":
    example = {
        "signal_power": 0.95,
        "spectral_entropy": 0.25,
        "peak_frequency": 1900,
        "bandwidth": 550,
        "noise_level": 0.45
    }

    result = predict_signal(example)
    print(result)
