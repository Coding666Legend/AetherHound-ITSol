
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ml.autoencoder import Autoencoder

ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(ROOT / "models" / "config.json") as f:
        config = json.load(f)

    df = pd.read_csv(ROOT / "data" / "synthetic_signals.csv")

    X_train, X_temp, y_train, y_temp = train_test_split(
        df[config["features"]],
        df["label"],
        test_size=0.4,
        random_state=42,
        stratify=df["label"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    scaler = joblib.load(ROOT / "models" / "scaler.pkl")
    X_scaled = scaler.transform(X_test)

    model = Autoencoder(config["input_dim"])
    model.load_state_dict(
        torch.load(
            ROOT / "models" / "autoencoder.pt",
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean(
            (X_tensor - reconstructed) ** 2, dim=1
        ).numpy()

    predictions = (
        errors > config["threshold"]
    ).astype(int)

    print("AetherHound Autoencoder Evaluation")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
