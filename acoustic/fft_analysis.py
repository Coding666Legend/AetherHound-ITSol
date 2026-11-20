"""Frequency-domain spectral analysis module for AetherHound.

Provides Fast Fourier Transform (FFT) computation, spectral peak detection,
and harmonic identification using NumPy and SciPy to analyze acoustic frequency
characteristics and detect periodic covert patterns.

Owner: Member 03 – DSP & Acoustic Engineer
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal


def compute_fft(
    data: np.ndarray,
    sample_rate: int = 44100,
    window: Optional[str] = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided discrete Fourier transform (magnitude spectrum).

    Parameters
    ----------
    data : np.ndarray
        1D array of audio samples.
    sample_rate : int, optional
        Sampling frequency in Hz. Default is 44100.
    window : Optional[str], optional
        Windowing function to taper signal edges and reduce spectral leakage:
        'hann', 'hamming', 'blackman', 'boxcar' (rectangular), or None.
        Default is 'hann'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (frequencies, magnitudes) where frequencies is a 1D array of frequency
        bins from 0 to Nyquist, and magnitudes is the normalized magnitude spectrum.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If data is empty, non-finite, or sample_rate <= 0.
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

    n = len(data)

    # Apply window function
    if window is not None:
        valid_windows = ("hann", "hamming", "blackman", "boxcar")
        if window not in valid_windows:
            raise ValueError(f"Unsupported window '{window}'. Must be one of {valid_windows} or None.")
        win_array = signal.get_window(window, n)
        windowed_data = data * win_array
        # Window amplitude correction factor
        scale_factor = 2.0 / np.sum(win_array) if np.sum(win_array) > 0 else 1.0 / n
    else:
        windowed_data = data
        scale_factor = 2.0 / n

    # Real FFT for efficient computation of positive frequency half-spectrum
    fft_complex = np.fft.rfft(windowed_data)
    magnitudes = np.abs(fft_complex) * scale_factor
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # DC component scaling adjustment
    if len(magnitudes) > 0:
        magnitudes[0] /= 2.0

    return frequencies, magnitudes


def detect_spectral_peaks(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    threshold_rel: float = 0.05,
    min_distance_hz: float = 10.0,
    max_peaks: int = 10,
) -> List[Tuple[float, float]]:
    """Detect dominant spectral peaks in the frequency magnitude spectrum.

    Parameters
    ----------
    frequencies : np.ndarray
        1D array of frequency bin centers in Hz.
    magnitudes : np.ndarray
        1D array of corresponding spectral magnitudes.
    threshold_rel : float, optional
        Minimum peak magnitude relative to max magnitude in range [0.0, 1.0].
        Default is 0.05.
    min_distance_hz : float, optional
        Minimum separation between adjacent detected peaks in Hz. Default is 10.0.
    max_peaks : int, optional
        Maximum number of prominent peaks to return. Default is 10.

    Returns
    -------
    List[Tuple[float, float]]
        List of (frequency_hz, magnitude) tuples sorted by magnitude descending.
    """
    if not isinstance(frequencies, np.ndarray) or not isinstance(magnitudes, np.ndarray):
        raise TypeError("frequencies and magnitudes must be NumPy arrays.")

    if len(frequencies) != len(magnitudes):
        raise ValueError("frequencies and magnitudes arrays must have identical length.")

    if len(magnitudes) == 0:
        return []

    if isinstance(threshold_rel, bool) or not isinstance(threshold_rel, (int, float)):
        raise TypeError(f"threshold_rel must be numeric, got {type(threshold_rel).__name__}.")

    if not math.isfinite(threshold_rel) or threshold_rel < 0.0:
        raise ValueError("threshold_rel must be a non-negative finite number.")

    max_mag = np.max(magnitudes)
    if max_mag == 0.0:
        return []

    abs_threshold = threshold_rel * max_mag

    # Calculate bin spacing in Hz
    freq_step = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    distance_bins = max(1, int(round(min_distance_hz / freq_step)))

    peak_indices, _ = signal.find_peaks(magnitudes, height=abs_threshold, distance=distance_bins)

    # Collect peaks and sort by magnitude descending
    peaks = [(float(frequencies[idx]), float(magnitudes[idx])) for idx in peak_indices]
    peaks.sort(key=lambda item: item[1], reverse=True)

    return peaks[:max_peaks]


def analyze_harmonics(
    base_frequency: float,
    peak_frequencies: List[float],
    tolerance_hz: float = 5.0,
    max_harmonics: int = 5,
) -> Dict[int, Optional[float]]:
    """Determine whether detected spectral peaks correspond to harmonic overtones.

    Parameters
    ----------
    base_frequency : float
        Target fundamental frequency in Hz (f0).
    peak_frequencies : List[float]
        List of detected spectral peak frequencies in Hz.
    tolerance_hz : float, optional
        Acceptable frequency deviation in Hz. Default is 5.0.
    max_harmonics : int, optional
        Number of harmonic orders to evaluate [1, max_harmonics]. Default is 5.

    Returns
    -------
    Dict[int, Optional[float]]
        Dictionary mapping harmonic order k -> matched frequency or None if not found.
    """
    if base_frequency <= 0:
        raise ValueError(f"base_frequency must be positive, got {base_frequency} Hz.")

    results: Dict[int, Optional[float]] = {}
    for order in range(1, max_harmonics + 1):
        target_f = base_frequency * order
        best_match: Optional[float] = None
        min_dev = float("inf")

        for f in peak_frequencies:
            dev = abs(f - target_f)
            if dev <= tolerance_hz and dev < min_dev:
                min_dev = dev
                best_match = f

        results[order] = best_match

    return results
