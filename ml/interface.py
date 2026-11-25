
import math
from numbers import Real

from ml.dataset import FEATURES
from ml.predict import predict_signal


def detect_anomaly(features: dict) -> dict:
    """
    Validate signal features and predict anomalies.
    """

    # Check input type
    if not isinstance(features, dict):
        raise TypeError("Input must be a dictionary")

    # Check missing features
    missing = set(FEATURES) - set(features.keys())

    if missing:
        raise ValueError(
            f"Missing features: {sorted(missing)}"
        )

    # Check unexpected features
    extra = set(features.keys()) - set(FEATURES)

    if extra:
        raise ValueError(
            f"Unexpected features: {sorted(extra)}"
        )

    # Check numeric values and NaN/Infinity
    for name in FEATURES:
        value = features[name]

        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"{name} must be a numeric value"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

    # Call trained ML model
    return predict_signal(features)


if __name__ == "__main__":
    sample = {
        "signal_power": 0.95,
        "spectral_entropy": 0.25,
        "peak_frequency": 1900,
        "bandwidth": 550,
        "noise_level": 0.45,
    }

    result = detect_anomaly(sample)
    print(result)
