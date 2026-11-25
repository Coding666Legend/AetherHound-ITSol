
from ml.interface import detect_anomaly


samples = {
    "Normal Signal": {
        "signal_power": 0.5,
        "spectral_entropy": 0.7,
        "peak_frequency": 1000,
        "bandwidth": 200,
        "noise_level": 0.1,
    },
    "Suspicious Signal": {
        "signal_power": 0.95,
        "spectral_entropy": 0.25,
        "peak_frequency": 1900,
        "bandwidth": 550,
        "noise_level": 0.45,
    },
}


def main():
    print("\nAETHERHOUND ML DETECTION DEMO")
    print("=" * 40)

    for name, features in samples.items():
        result = detect_anomaly(features)

        print(f"\nSample: {name}")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print(
            "Detection:",
            "ANOMALY" if result["is_anomaly"] else "NORMAL"
        )


if __name__ == "__main__":
    main()
