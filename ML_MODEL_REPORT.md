
# AetherHound - ML Model Report

## Role
Member 04 - Machine Learning Engineer

## Objective
Develop an anomaly detection module to identify unusual
patterns in numerical RF/acoustic signal features.

## Dataset
Synthetic signal feature dataset:
- Normal samples: 1000
- Anomaly samples: 200
- Total samples: 1200

Features:
1. Signal Power
2. Spectral Entropy
3. Peak Frequency
4. Bandwidth
5. Noise Level

## Preprocessing
StandardScaler is fitted using normal training data.

## Baseline Model
Algorithm: Isolation Forest

Accuracy on synthetic test data: 93%

## Deep Learning Model
Algorithm: PyTorch Autoencoder

Architecture:
Input (5) -> 16 -> 8 -> 3 -> 8 -> 16 -> Output (5)

Training:
- Optimizer: Adam
- Loss: Mean Squared Error
- Epochs: 100

Final training loss: 0.880297

## Anomaly Detection
Reconstruction error is calculated using MSE.

Threshold: 1.969702124595642

If reconstruction error exceeds the threshold,
the signal is classified as anomalous.

## Evaluation Results
Autoencoder accuracy: 95%
Anomaly precision: 0.75
Anomaly recall: 1.00
Anomaly F1-score: 0.86

Confusion Matrix:
[[187, 13],
 [0, 40]]

False positives: 13
False negatives: 0

## Limitations
Results are from an illustrative synthetic dataset.
Performance on real RF/acoustic covert channels has
not been established.

## Next Integration Requirement
RF/DSP modules must provide the five features
with consistent definitions and units.
