
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from ml.autoencoder import Autoencoder

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

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
    X_temp, y_temp,
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

with torch.no_grad():
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    reconstruction = model(tensor)
    errors = torch.mean(
        (tensor - reconstruction) ** 2,
        dim=1
    ).numpy()

predictions = (errors > config["threshold"]).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
).plot(cmap="Blues")

plt.title("AetherHound - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT / "confusion_matrix.png", dpi=200)
plt.close()

# Reconstruction error distribution
plt.figure(figsize=(9, 5))

plt.hist(
    errors[y_test.to_numpy() == 0],
    bins=30,
    alpha=0.7,
    label="Normal"
)

plt.hist(
    errors[y_test.to_numpy() == 1],
    bins=30,
    alpha=0.7,
    label="Anomaly"
)

plt.axvline(
    config["threshold"],
    color="red",
    linestyle="--",
    label="Anomaly Threshold"
)

plt.xlabel("Reconstruction Error")
plt.ylabel("Sample Count")
plt.title("AetherHound - Reconstruction Error Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "reconstruction_errors.png", dpi=200)
plt.close()

print("Graphs generated successfully!")
print("Saved to:", OUTPUT)
