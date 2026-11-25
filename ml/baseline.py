
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "synthetic_signals.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    normal_train = X_train[y_train == 0]

    scaler = StandardScaler()
    scaler.fit(normal_train)

    X_train_scaled = scaler.transform(normal_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    model.fit(X_train_scaled)

    predictions = model.predict(X_test_scaled)
    predicted_labels = (predictions == -1).astype(int)

    print("\nAetherHound Baseline Results")
    print(classification_report(
        y_test,
        predicted_labels,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))


if __name__ == "__main__":
    main()
