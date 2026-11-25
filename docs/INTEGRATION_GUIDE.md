
# AetherHound ML Integration Guide

## Module Owner
Member 04 - ML Engineer

## Prediction Interface

Import:

from ml.interface import detect_anomaly

## Required Input

A Python dictionary containing:

{
    "signal_power": 0.5,
    "spectral_entropy": 0.7,
    "peak_frequency": 1000,
    "bandwidth": 200,
    "noise_level": 0.1
}

## Example Usage

from ml.interface import detect_anomaly

result = detect_anomaly(features)

print(result)

## Output

{
    "anomaly_score": float,
    "threshold": float,
    "is_anomaly": bool
}

## Model Dependencies

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Required Model Files

models/autoencoder.pt
models/scaler.pkl
models/config.json

## Integration Notes

1. The RF/DSP module extracts signal features.
2. The ML interface accepts a feature dictionary.
3. The model normalizes the input.
4. The Autoencoder calculates reconstruction error.
5. The module returns the anomaly score and decision.

Important:
The current model is trained on synthetic features.
Feature definitions, units, and ranges must be agreed
with the RF/DSP team before real integration.

Only load trusted model artifacts.
