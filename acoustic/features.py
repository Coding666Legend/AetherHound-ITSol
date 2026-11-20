"""Acoustic feature extraction pipeline for AetherHound.

Extracts numerical descriptors (RMS energy, spectral centroid, spectral bandwidth,
zero-crossing rate, dominant frequency, spectral rolloff) from acoustic signals
and formats normalized FeatureFrames for machine learning anomaly detection.

Owner: Member 03 – DSP & Acoustic Engineer
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from acoustic.fft_analysis import compute_fft


@dataclass
class FeatureFrame:
    """Standardized feature frame contract for integration with Member 01 and Member 04."""

    modality: str = "acoustic"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_id: str = "acoustic_default"
    features: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    feature_names: List[str] = field(default_factory=list)


def compute_rms_energy(data: np.ndarray) -> float:
    """Compute the Root Mean Square (RMS) energy of a time-domain signal."""
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data**2)))


def compute_zero_crossing_rate(data: np.ndarray) -> float:
    """Compute the Zero-Crossing Rate (ZCR) of a time-domain signal."""
    if len(data) < 2:
        return 0.0
    # Sign differences between adjacent samples
    signs = np.sign(data)
    # Treat 0 as non-crossing by zeroing
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs))) / (2.0 * (len(data) - 1))
    return float(crossings)


def compute_spectral_centroid(frequencies: np.ndarray, magnitudes: np.ndarray) -> float:
    """Compute the Spectral Centroid (center of spectral mass) in Hz."""
    sum_mag = np.sum(magnitudes)
    if sum_mag == 0.0:
        return 0.0
    centroid = np.sum(frequencies * magnitudes) / sum_mag
    return float(centroid)


def compute_spectral_bandwidth(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    centroid: Optional[float] = None,
) -> float:
    """Compute the Spectral Bandwidth (spectral spread / variance around centroid) in Hz."""
    sum_mag = np.sum(magnitudes)
    if sum_mag == 0.0:
        return 0.0
    if centroid is None:
        centroid = compute_spectral_centroid(frequencies, magnitudes)
    variance = np.sum(((frequencies - centroid) ** 2) * magnitudes) / sum_mag
    return float(np.sqrt(variance))


def compute_dominant_frequency(frequencies: np.ndarray, magnitudes: np.ndarray) -> float:
    """Determine the Dominant Frequency (frequency bin with peak spectral magnitude) in Hz."""
    if len(magnitudes) == 0 or np.max(magnitudes) == 0.0:
        return 0.0
    peak_idx = int(np.argmax(magnitudes))
    return float(frequencies[peak_idx])


def compute_spectral_rolloff(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    roll_percent: float = 0.85,
) -> float:
    """Compute the Spectral Rolloff frequency in Hz below which roll_percent energy lies."""
    sum_mag = np.sum(magnitudes)
    if sum_mag == 0.0:
        return 0.0
    threshold = roll_percent * sum_mag
    cumulative_energy = np.cumsum(magnitudes)
    idx = int(np.searchsorted(cumulative_energy, threshold))
    idx = min(idx, len(frequencies) - 1)
    return float(frequencies[idx])


def extract_acoustic_features(
    data: np.ndarray,
    sample_rate: int = 44100,
) -> Dict[str, float]:
    """Extract a complete set of numerical acoustic features from audio data.

    Parameters
    ----------
    data : np.ndarray
        1D array of audio samples.
    sample_rate : int, optional
        Sampling frequency in Hz. Default 44100.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature name to numerical feature value.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    ValueError
        If data is empty or contains non-finite values.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a NumPy ndarray, got {type(data).__name__}.")

    if data.ndim != 1:
        raise ValueError(f"data must be a 1D array, got {data.ndim}D.")

    if data.size == 0:
        raise ValueError("data array must not be empty.")

    if not np.all(np.isfinite(data)):
        raise ValueError("data array contains non-finite values (NaN or Inf).")

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError(f"sample_rate must be an integer, got {type(sample_rate).__name__}.")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate} Hz.")

    # Time-domain features
    rms = compute_rms_energy(data)
    zcr = compute_zero_crossing_rate(data)

    # Frequency-domain features
    freqs, mags = compute_fft(data, sample_rate=sample_rate, window="hann")
    centroid = compute_spectral_centroid(freqs, mags)
    bandwidth = compute_spectral_bandwidth(freqs, mags, centroid=centroid)
    dominant_f = compute_dominant_frequency(freqs, mags)
    rolloff = compute_spectral_rolloff(freqs, mags, roll_percent=0.85)

    return {
        "rms_energy": rms,
        "zero_crossing_rate": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "dominant_frequency": dominant_f,
        "spectral_rolloff": rolloff,
    }


def create_feature_frame(
    data: np.ndarray,
    sample_rate: int = 44100,
    source_id: str = "acoustic_default",
    timestamp: Optional[datetime] = None,
) -> FeatureFrame:
    """Generate a standardized FeatureFrame ready for consumption by Member 04's ML detector.

    Parameters
    ----------
    data : np.ndarray
        1D array of audio samples.
    sample_rate : int, optional
        Sampling frequency in Hz. Default 44100.
    source_id : str, optional
        Stable identifier for source capture device or simulation model.
    timestamp : Optional[datetime], optional
        Timestamp for capture; defaults to current UTC time.

    Returns
    -------
    FeatureFrame
        Standardized container with modality 'acoustic', 1D feature array, and feature names.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    feature_dict = extract_acoustic_features(data, sample_rate=sample_rate)

    feature_names = sorted(feature_dict.keys())
    feature_values = np.array([feature_dict[name] for name in feature_names], dtype=np.float64)

    return FeatureFrame(
        modality="acoustic",
        timestamp=timestamp,
        source_id=source_id,
        features=feature_values,
        feature_names=feature_names,
    )
