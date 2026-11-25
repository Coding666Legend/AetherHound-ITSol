
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from ml.autoencoder import Autoencoder
from ml.dataset import FEATURES

ROOT = Path(__file__).resolve().parent.parent


def main():
    torch.manual_seed(42)

    df = pd.read_csv(ROOT / "data" / "synthetic_signals.csv")

    X_train, X_temp, y_train, y_temp = train_test_split(
        df[FEATURES],
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

    scaler = StandardScaler()

    train_normal = X_train[y_train == 0]
    train_scaled = scaler.fit_transform(train_normal)

    model = Autoencoder(input_dim=len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        output = model(train_tensor)
        loss = criterion(output, train_tensor)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/100 | Loss: {loss.item():.6f}")

    model.eval()

    val_normal = X_val[y_val == 0]
    val_scaled = scaler.transform(val_normal)
    val_tensor = torch.tensor(val_scaled, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(val_tensor)
        errors = torch.mean(
            (val_tensor - reconstructed) ** 2, dim=1
        ).numpy()

    threshold = float(np.percentile(errors, 95))

    (ROOT / "models").mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        ROOT / "models" / "autoencoder.pt"
    )

    joblib.dump(
        scaler,
        ROOT / "models" / "scaler.pkl"
    )

    config = {
        "features": FEATURES,
        "input_dim": len(FEATURES),
        "threshold": threshold
    }

    with open(ROOT / "models" / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nTraining completed!")
    print("Anomaly threshold:", threshold)


if __name__ == "__main__":
    main()
