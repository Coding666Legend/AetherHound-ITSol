
import numpy as np
import torch

from ml.autoencoder import Autoencoder
from ml.predict import predict_signal


def test_autoencoder_output_shape():
    model = Autoencoder(input_dim=5)
    sample = torch.randn(4, 5)

    output = model(sample)

    assert output.shape == sample.shape


def test_anomaly_prediction():
    sample = {
        "signal_power": 0.95,
        "spectral_entropy": 0.25,
        "peak_frequency": 1900,
        "bandwidth": 550,
        "noise_level": 0.45,
    }

    result = predict_signal(sample)

    assert isinstance(result["is_anomaly"], bool)
    assert np.isfinite(result["anomaly_score"])
    assert result["anomaly_score"] >= 0
    assert result["threshold"] > 0
