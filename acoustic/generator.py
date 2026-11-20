"""Synthetic acoustic signal generator module for AetherHound.

Provides functions to generate synthetic acoustic waveforms (such as sinusoidal
tones, server-room fan hums, and periodic signals) for offline testing, digital
signal processing, and covert acoustic channel simulation without physical hardware.

Owner: Member 03 – DSP & Acoustic Engineer
"""

import math
from typing import Any, Optional, Sequence
import numpy as np


def generate_sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 1.0,
    phase: float = 0.0,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Generate a synthetic continuous sine wave audio signal.

    Parameters
    ----------
    frequency : float
        Frequency of the sine wave in Hertz (Hz). Must be strictly positive,
        finite, and below the Nyquist frequency (sample_rate / 2).
    duration : float
        Duration of the generated signal in seconds. Must be strictly positive,
        finite, and long enough to yield at least one audio sample.
    sample_rate : int, optional
        Sampling frequency in samples per second (Hz). Must be a positive,
        finite integer. Default is 44100 Hz (standard audio rate).
    amplitude : float, optional
        Peak amplitude scaling factor. Must be a finite value in the range [0.0, 1.0]
        for normalized digital audio. Default is 1.0 (full scale).
    phase : float, optional
        Initial phase offset in radians. Must be a finite numeric value.
        Default is 0.0.
    dtype : Any, optional
        NumPy floating-point data type for the output array (e.g., np.float64,
        np.float32). Default is np.float64.

    Returns
    -------
    np.ndarray
        1D NumPy array containing the generated audio samples.

    Raises
    ------
    TypeError
        If frequency, duration, sample_rate, amplitude, or phase are booleans or
        not numeric, or if dtype cannot be interpreted as a NumPy dtype.
    ValueError
        If frequency, duration, sample_rate, amplitude, or phase are NaN or infinite.
        If frequency <= 0 or frequency >= sample_rate / 2 (Nyquist limit).
        If duration <= 0 or results in zero samples.
        If sample_rate <= 0.
        If amplitude < 0.0 or amplitude > 1.0.
        If dtype is not a floating-point data type.
    """
    # 1. Type validation and boolean rejection for all numeric parameters
    numeric_params = [
        ("Frequency", frequency),
        ("Duration", duration),
        ("Sample rate", sample_rate),
        ("Amplitude", amplitude),
        ("Phase", phase),
    ]
    for param_name, param_val in numeric_params:
        if isinstance(param_val, bool) or not isinstance(param_val, (int, float)):
            raise TypeError(
                f"{param_name} must be a real numeric value, got {type(param_val).__name__}."
            )

    # 2. Rejection of NaN and infinity for all numeric parameters
    for param_name, param_val in numeric_params:
        if not math.isfinite(param_val):
            raise ValueError(f"{param_name} must be a finite number, got {param_val}.")

    # 3. Dtype validation (must be valid NumPy floating-point type)
    try:
        target_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Invalid NumPy dtype specification: {dtype}.") from err

    if not np.issubdtype(target_dtype, np.floating):
        raise ValueError(
            f"Output dtype must be a floating-point type (e.g., np.float64, np.float32), "
            f"got {target_dtype}."
        )

    # 4. Parameter range and physical constraint validation
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate} Hz.")
    sample_rate_int = int(sample_rate)

    if frequency <= 0:
        raise ValueError(f"Frequency must be positive, got {frequency} Hz.")

    nyquist = sample_rate_int / 2.0
    if frequency >= nyquist:
        raise ValueError(
            f"Frequency ({frequency} Hz) must be strictly below the Nyquist limit "
            f"({nyquist} Hz) for sample rate {sample_rate_int} Hz."
        )

    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration} s.")

    if amplitude < 0.0 or amplitude > 1.0:
        raise ValueError(f"Amplitude must be in range [0.0, 1.0], got {amplitude}.")

    # 5. Safe sample count calculation (guarding against extremely small positive durations)
    num_samples = int(round(duration * sample_rate_int))
    if num_samples <= 0:
        min_duration = 0.5 / sample_rate_int
        raise ValueError(
            f"Duration ({duration} s) at sample rate {sample_rate_int} Hz produces 0 samples. "
            f"Duration must be at least ~{min_duration:.2e} s to yield at least 1 sample."
        )

    # 6. Time vector computation: t = [0, 1/fs, 2/fs, ..., (N-1)/fs]
    t = np.arange(num_samples, dtype=np.float64) / sample_rate_int

    # 7. Generate continuous sine wave: y(t) = A * sin(2 * pi * f * t + phase)
    signal = amplitude * np.sin(2.0 * np.pi * frequency * t + phase)

    return signal.astype(target_dtype)


def generate_fan_noise(
    base_frequency: float,
    duration: float,
    sample_rate: int = 44100,
    harmonics: Optional[Sequence[float]] = (1.0, 0.5, 0.25, 0.125),
    amplitude: float = 0.8,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Generate synthetic server-room cooling fan acoustic noise.

    Simulates the acoustic profile of computer or server cooling fans, comprising
    a fundamental rotational blade-pass tone, multiple harmonic overtones, and
    broadband aerodynamic turbulence noise.

    Parameters
    ----------
    base_frequency : float
        Fundamental rotational frequency / Blade Pass Frequency (Hz).
        Must be strictly positive and below the Nyquist limit.
    duration : float
        Duration of the generated signal in seconds. Must be strictly positive.
    sample_rate : int, optional
        Sampling rate in Hz. Default is 44100 Hz.
    harmonics : Sequence[float], optional
        Relative amplitude weights for the harmonics starting from the fundamental
        (1st harmonic, 2nd harmonic, etc.). Default is (1.0, 0.5, 0.25, 0.125).
    amplitude : float, optional
        Peak normalized output amplitude in range [0.0, 1.0]. Default is 0.8.
    noise_level : float, optional
        Relative level of broadband turbulent airflow noise in range [0.0, 1.0].
        Default is 0.1.
    seed : Optional[int], optional
        Random seed for reproducible broadband noise generation. Default is None.
    dtype : Any, optional
        NumPy floating-point data type. Default is np.float64.

    Returns
    -------
    np.ndarray
        1D NumPy array containing the synthesized fan noise audio samples.

    Raises
    ------
    TypeError
        If inputs have invalid types, if seed is not an integer, or if dtype is invalid.
    ValueError
        If parameters are out of range, non-finite, or if harmonic frequencies
        exceed the Nyquist limit.
    """
    # 1. Validate numeric parameters
    params = [
        ("Base frequency", base_frequency),
        ("Duration", duration),
        ("Sample rate", sample_rate),
        ("Amplitude", amplitude),
        ("Noise level", noise_level),
    ]
    for param_name, param_val in params:
        if isinstance(param_val, bool) or not isinstance(param_val, (int, float)):
            raise TypeError(
                f"{param_name} must be a real numeric value, got {type(param_val).__name__}."
            )
        if not math.isfinite(param_val):
            raise ValueError(f"{param_name} must be a finite number, got {param_val}.")

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"Seed must be an integer, got {type(seed).__name__}.")
        if seed < 0:
            raise ValueError(f"Seed must be non-negative, got {seed}.")

    # 2. Dtype validation
    try:
        target_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Invalid NumPy dtype specification: {dtype}.") from err

    if not np.issubdtype(target_dtype, np.floating):
        raise ValueError(f"Output dtype must be floating-point, got {target_dtype}.")

    # 3. Range and constraint checks
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate} Hz.")
    sample_rate_int = int(sample_rate)

    if base_frequency <= 0:
        raise ValueError(f"Base frequency must be positive, got {base_frequency} Hz.")

    nyquist = sample_rate_int / 2.0
    if base_frequency >= nyquist:
        raise ValueError(
            f"Base frequency ({base_frequency} Hz) must be strictly below Nyquist limit ({nyquist} Hz)."
        )

    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration} s.")

    if amplitude < 0.0 or amplitude > 1.0:
        raise ValueError(f"Amplitude must be in range [0.0, 1.0], got {amplitude}.")

    if noise_level < 0.0 or noise_level > 1.0:
        raise ValueError(f"Noise level must be in range [0.0, 1.0], got {noise_level}.")

    # 4. Harmonics validation
    if harmonics is None or len(harmonics) == 0:
        harm_weights = [1.0]
    else:
        harm_weights = []
        for i, h in enumerate(harmonics):
            if isinstance(h, bool) or not isinstance(h, (int, float)):
                raise TypeError(f"Harmonic weight at index {i} must be numeric, got {type(h).__name__}.")
            if not math.isfinite(h) or h < 0.0:
                raise ValueError(f"Harmonic weight at index {i} must be a non-negative finite number, got {h}.")
            harm_weights.append(float(h))

    # Verify highest harmonic frequency
    max_harmonic_idx = len(harm_weights)
    highest_freq = base_frequency * max_harmonic_idx
    if highest_freq >= nyquist:
        raise ValueError(
            f"Highest harmonic frequency ({highest_freq} Hz) for harmonic order {max_harmonic_idx} "
            f"reaches or exceeds the Nyquist limit ({nyquist} Hz)."
        )

    num_samples = int(round(duration * sample_rate_int))
    if num_samples <= 0:
        raise ValueError(f"Duration {duration}s produces 0 samples at {sample_rate_int}Hz.")

    # 5. Synthesize harmonic tones
    t = np.arange(num_samples, dtype=np.float64) / sample_rate_int
    tonal_signal = np.zeros(num_samples, dtype=np.float64)

    for order, weight in enumerate(harm_weights, start=1):
        if weight > 0.0:
            freq = base_frequency * order
            tonal_signal += weight * np.sin(2.0 * np.pi * freq * t)

    # 6. Synthesize broadband turbulent noise
    if noise_level > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, num_samples)
        # Apply smoothing / blend noise
        combined = tonal_signal + (noise_level * noise)
    else:
        combined = tonal_signal

    # 7. Normalize peak amplitude to requested target amplitude
    peak = np.max(np.abs(combined))
    if peak > 0.0 and amplitude > 0.0:
        normalized = combined * (amplitude / peak)
    elif amplitude == 0.0 or peak == 0.0:
        normalized = np.zeros_like(combined)
    else:
        normalized = combined

    return normalized.astype(target_dtype)
