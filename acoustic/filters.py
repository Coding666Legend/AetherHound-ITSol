"""Digital audio filtering module for AetherHound.

Provides Butterworth low-pass, high-pass, and band-pass filters with zero-phase
distortion (via forward-backward filtering) using SciPy for isolating acoustic
target bands and attenuating server-room ambient noise.

Owner: Member 03 – DSP & Acoustic Engineer
"""

import math
from typing import Tuple, Union
import numpy as np
from scipy import signal


def _validate_filter_inputs(
    data: np.ndarray,
    sample_rate: int,
    order: int,
) -> Tuple[np.ndarray, int, int, float]:
    """Validate shared filter arguments and return sanitized parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a NumPy ndarray, got {type(data).__name__}.")

    if data.size == 0:
        raise ValueError("data array must not be empty.")

    if not np.all(np.isfinite(data)):
        raise ValueError("data array contains non-finite values (NaN or Inf).")

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError(f"sample_rate must be an integer, got {type(sample_rate).__name__}.")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate} Hz.")

    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise TypeError(f"order must be an integer, got {type(order).__name__}.")

    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}.")

    nyquist = float(sample_rate) / 2.0
    return data, int(sample_rate), int(order), nyquist


def butter_lowpass_filter(
    data: np.ndarray,
    cutoff_freq: float,
    sample_rate: int = 44100,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter to audio data.

    Parameters
    ----------
    data : np.ndarray
        Input audio array (1D or 2D).
    cutoff_freq : float
        Cutoff frequency in Hz. Must be strictly between 0 and Nyquist (sample_rate / 2).
    sample_rate : int, optional
        Sampling frequency in Hz (default 44100).
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered audio array matching the input shape and dtype.
    """
    data, sample_rate, order, nyquist = _validate_filter_inputs(data, sample_rate, order)

    if isinstance(cutoff_freq, bool) or not isinstance(cutoff_freq, (int, float)):
        raise TypeError(f"cutoff_freq must be numeric, got {type(cutoff_freq).__name__}.")
    if not math.isfinite(cutoff_freq):
        raise ValueError("cutoff_freq must be a finite number.")

    if cutoff_freq <= 0.0 or cutoff_freq >= nyquist:
        raise ValueError(
            f"cutoff_freq ({cutoff_freq} Hz) must be strictly between 0 and Nyquist ({nyquist} Hz)."
        )

    # Minimum length required by scipy.signal.filtfilt
    min_len = 3 * (2 * order + 1)
    if data.shape[0] <= min_len:
        raise ValueError(
            f"Data length ({data.shape[0]}) is too short for filter order {order}. "
            f"Requires at least {min_len + 1} samples."
        )

    norm_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, norm_cutoff, btype="lowpass", analog=False)
    filtered = signal.filtfilt(b, a, data, axis=0)

    return filtered.astype(data.dtype)


def butter_highpass_filter(
    data: np.ndarray,
    cutoff_freq: float,
    sample_rate: int = 44100,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter to audio data.

    Parameters
    ----------
    data : np.ndarray
        Input audio array (1D or 2D).
    cutoff_freq : float
        Cutoff frequency in Hz. Must be strictly between 0 and Nyquist (sample_rate / 2).
    sample_rate : int, optional
        Sampling frequency in Hz (default 44100).
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered audio array matching the input shape and dtype.
    """
    data, sample_rate, order, nyquist = _validate_filter_inputs(data, sample_rate, order)

    if isinstance(cutoff_freq, bool) or not isinstance(cutoff_freq, (int, float)):
        raise TypeError(f"cutoff_freq must be numeric, got {type(cutoff_freq).__name__}.")
    if not math.isfinite(cutoff_freq):
        raise ValueError("cutoff_freq must be a finite number.")

    if cutoff_freq <= 0.0 or cutoff_freq >= nyquist:
        raise ValueError(
            f"cutoff_freq ({cutoff_freq} Hz) must be strictly between 0 and Nyquist ({nyquist} Hz)."
        )

    min_len = 3 * (2 * order + 1)
    if data.shape[0] <= min_len:
        raise ValueError(
            f"Data length ({data.shape[0]}) is too short for filter order {order}. "
            f"Requires at least {min_len + 1} samples."
        )

    norm_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, norm_cutoff, btype="highpass", analog=False)
    filtered = signal.filtfilt(b, a, data, axis=0)

    return filtered.astype(data.dtype)


def butter_bandpass_filter(
    data: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sample_rate: int = 44100,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter to audio data.

    Parameters
    ----------
    data : np.ndarray
        Input audio array (1D or 2D).
    low_cutoff : float
        Lower cutoff frequency in Hz. Must be > 0 and < high_cutoff.
    high_cutoff : float
        Upper cutoff frequency in Hz. Must be > low_cutoff and < Nyquist (sample_rate / 2).
    sample_rate : int, optional
        Sampling frequency in Hz (default 44100).
    order : int, optional
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered audio array matching the input shape and dtype.
    """
    data, sample_rate, order, nyquist = _validate_filter_inputs(data, sample_rate, order)

    for name, val in [("low_cutoff", low_cutoff), ("high_cutoff", high_cutoff)]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(val).__name__}.")
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number.")

    if low_cutoff <= 0.0:
        raise ValueError(f"low_cutoff must be positive, got {low_cutoff} Hz.")

    if high_cutoff >= nyquist:
        raise ValueError(
            f"high_cutoff ({high_cutoff} Hz) must be strictly below Nyquist ({nyquist} Hz)."
        )

    if low_cutoff >= high_cutoff:
        raise ValueError(
            f"low_cutoff ({low_cutoff} Hz) must be strictly less than high_cutoff ({high_cutoff} Hz)."
        )

    min_len = 3 * (2 * order + 1)
    if data.shape[0] <= min_len:
        raise ValueError(
            f"Data length ({data.shape[0]}) is too short for filter order {order}. "
            f"Requires at least {min_len + 1} samples."
        )

    norm_low = low_cutoff / nyquist
    norm_high = high_cutoff / nyquist
    b, a = signal.butter(order, [norm_low, norm_high], btype="bandpass", analog=False)
    filtered = signal.filtfilt(b, a, data, axis=0)

    return filtered.astype(data.dtype)
