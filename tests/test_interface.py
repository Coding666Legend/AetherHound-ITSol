
import pytest

from ml.interface import detect_anomaly


@pytest.fixture
def normal_features():
    return {
        "signal_power": 0.5,
        "spectral_entropy": 0.7,
        "peak_frequency": 1000,
        "bandwidth": 200,
        "noise_level": 0.1,
    }


def test_valid_input(normal_features):
    result = detect_anomaly(normal_features)

    assert isinstance(result["is_anomaly"], bool)
    assert result["anomaly_score"] >= 0
    assert result["threshold"] > 0


def test_missing_feature(normal_features):
    del normal_features["bandwidth"]

    with pytest.raises(ValueError):
        detect_anomaly(normal_features)


def test_extra_feature(normal_features):
    normal_features["unknown_feature"] = 123

    with pytest.raises(ValueError):
        detect_anomaly(normal_features)


def test_invalid_type(normal_features):
    normal_features["signal_power"] = "high"

    with pytest.raises(TypeError):
        detect_anomaly(normal_features)


def test_nan_value(normal_features):
    normal_features["signal_power"] = float("nan")

    with pytest.raises(ValueError):
        detect_anomaly(normal_features)


def test_infinite_value(normal_features):
    normal_features["bandwidth"] = float("inf")

    with pytest.raises(ValueError):
        detect_anomaly(normal_features)
