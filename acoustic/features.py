"""Acoustic feature extraction pipeline for AetherHound.

Extracts numerical descriptors (RMS energy, spectral centroid, spectral bandwidth,
zero-crossing rate, dominant frequency, spectral rolloff) from acoustic signals
and formats normalized FeatureFrames for machine learning anomaly detection.

Owner: Member 03 – DSP & Acoustic Engineer
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple, Union
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize FeatureFrame to a dictionary for JSON serialization and cross-module transport."""
        return {
            "modality": self.modality,
            "timestamp": self.timestamp.isoformat(),
            "source_id": self.source_id,
            "features": self.features.tolist(),
            "feature_names": list(self.feature_names),
        }

    def get_feature_map(self) -> Dict[str, float]:
        """Return a mapping of feature names to their corresponding scalar values."""
        return {name: float(val) for name, val in zip(self.feature_names, self.features)}


def _validate_spectral_vectors(frequencies: np.ndarray, magnitudes: np.ndarray) -> None:
    """Validate spectral frequencies and magnitude arrays."""
    if not isinstance(frequencies, np.ndarray) or not isinstance(magnitudes, np.ndarray):
        raise TypeError("frequencies and magnitudes must be NumPy ndarrays.")

    if frequencies.ndim != 1 or magnitudes.ndim != 1:
        raise ValueError("frequencies and magnitudes must be 1D arrays.")

    if len(frequencies) != len(magnitudes):
        raise ValueError(
            f"frequencies and magnitudes must have matching lengths, got {len(frequencies)} and {len(magnitudes)}."
        )

    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(magnitudes)):
        raise ValueError("frequencies and magnitudes must contain only finite numbers.")

    if np.any(frequencies < 0.0):
        raise ValueError("frequencies array must not contain negative frequency values.")

    if np.any(magnitudes < 0.0):
        raise ValueError("magnitudes array must not contain negative magnitude values.")


def compute_rms_energy(data: np.ndarray) -> float:
    """Compute the Root Mean Square (RMS) energy of a time-domain signal."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a NumPy ndarray, got {type(data).__name__}.")
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError(f"data must have a floating-point dtype, got {data.dtype}.")
    if data.ndim != 1:
        raise ValueError(f"data must be a 1D array, got {data.ndim}D.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data contains non-finite values.")
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data**2)))


def compute_zero_crossing_rate(data: np.ndarray) -> float:
    """Compute the Zero-Crossing Rate (ZCR) of a time-domain signal."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a NumPy ndarray, got {type(data).__name__}.")
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError(f"data must have a floating-point dtype, got {data.dtype}.")
    if data.ndim != 1:
        raise ValueError(f"data must be a 1D array, got {data.ndim}D.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data contains non-finite values.")
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
    _validate_spectral_vectors(frequencies, magnitudes)
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
    _validate_spectral_vectors(frequencies, magnitudes)
    if centroid is not None:
        if isinstance(centroid, bool) or not isinstance(centroid, (int, float)):
            raise TypeError(f"centroid must be numeric, got {type(centroid).__name__}.")
        if not math.isfinite(centroid):
            raise ValueError(f"centroid must be a finite number, got {centroid}.")
        if centroid < 0.0:
            raise ValueError(f"centroid must be non-negative, got {centroid} Hz.")

    sum_mag = np.sum(magnitudes)
    if sum_mag == 0.0:
        return 0.0
    if centroid is None:
        centroid = compute_spectral_centroid(frequencies, magnitudes)
    variance = np.sum(((frequencies - centroid) ** 2) * magnitudes) / sum_mag
    return float(np.sqrt(variance))


def compute_dominant_frequency(frequencies: np.ndarray, magnitudes: np.ndarray) -> float:
    """Determine the Dominant Frequency (frequency bin with peak spectral magnitude) in Hz."""
    _validate_spectral_vectors(frequencies, magnitudes)
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
    _validate_spectral_vectors(frequencies, magnitudes)

    if isinstance(roll_percent, bool) or not isinstance(roll_percent, (int, float)):
        raise TypeError(f"roll_percent must be numeric, got {type(roll_percent).__name__}.")

    if not math.isfinite(roll_percent) or roll_percent <= 0.0 or roll_percent > 1.0:
        raise ValueError(f"roll_percent must be in range (0.0, 1.0], got {roll_percent}.")

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
        1D floating-point array of audio samples.
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
        If data is empty, non-finite, not floating-point, or sample_rate <= 0.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a NumPy ndarray, got {type(data).__name__}.")

    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError(f"data must have a floating-point dtype, got {data.dtype}.")

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
        1D floating-point array of audio samples.
    sample_rate : int, optional
        Sampling frequency in Hz. Default 44100.
    source_id : str, optional
        Stable non-empty identifier for source capture device or simulation model.
    timestamp : Optional[datetime], optional
        Timestamp for capture; defaults to current UTC time. Must be timezone-aware if passed.

    Returns
    -------
    FeatureFrame
        Standardized container with modality 'acoustic', 1D feature array, and feature names.
    """
    if not isinstance(source_id, str) or len(source_id.strip()) == 0:
        raise ValueError(f"source_id must be a non-empty string, got '{source_id}'.")

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    else:
        if not isinstance(timestamp, datetime):
            raise TypeError(f"timestamp must be a datetime object, got {type(timestamp).__name__}.")
        if timestamp.tzinfo is None:
            # Attach UTC timezone if naive datetime is passed
            timestamp = timestamp.replace(tzinfo=timezone.utc)

    feature_dict = extract_acoustic_features(data, sample_rate=sample_rate)

    feature_names = sorted(feature_dict.keys())
    feature_values = np.array([feature_dict[name] for name in feature_names], dtype=np.float64)

    return FeatureFrame(
        modality="acoustic",
        timestamp=timestamp,
        source_id=source_id.strip(),
        features=feature_values,
        feature_names=feature_names,
    )
