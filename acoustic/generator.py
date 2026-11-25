"""Synthetic acoustic signal generator module for AetherHound.

Provides functions to generate synthetic acoustic waveforms (such as sinusoidal
tones, server-room fan hums, and covert acoustic channel anomalies) for offline
testing, digital signal processing, and anomaly detection without physical hardware.

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
    """Generate a synthetic continuous sine wave audio signal."""
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
        if not math.isfinite(param_val):
            raise ValueError(f"{param_name} must be a finite number, got {param_val}.")

    try:
        target_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Invalid NumPy dtype specification: {dtype}.") from err

    if not np.issubdtype(target_dtype, np.floating):
        raise ValueError(
            f"Output dtype must be a floating-point type (e.g., np.float64, np.float32), "
            f"got {target_dtype}."
        )

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

    num_samples = int(round(duration * sample_rate_int))
    if num_samples <= 0:
        min_duration = 0.5 / sample_rate_int
        raise ValueError(
            f"Duration ({duration} s) at sample rate {sample_rate_int} Hz produces 0 samples. "
            f"Duration must be at least ~{min_duration:.2e} s to yield at least 1 sample."
        )

    t = np.arange(num_samples, dtype=np.float64) / sample_rate_int
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
    """Generate synthetic server-room cooling fan acoustic noise."""
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

    try:
        target_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Invalid NumPy dtype specification: {dtype}.") from err

    if not np.issubdtype(target_dtype, np.floating):
        raise ValueError(
            f"Output dtype must be a floating-point type (e.g., np.float64, np.float32), "
            f"got {target_dtype}."
        )

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

    max_harmonic_idx = len(harm_weights)
    highest_freq = base_frequency * max_harmonic_idx
    if highest_freq >= nyquist:
        raise ValueError(
            f"Highest harmonic frequency ({highest_freq} Hz) for harmonic order {max_harmonic_idx} "
            f"reaches or exceeds the Nyquist limit ({nyquist} Hz)."
        )

    num_samples = int(round(duration * sample_rate_int))
    if num_samples <= 0:
        min_duration = 0.5 / sample_rate_int
        raise ValueError(
            f"Duration ({duration} s) at sample rate {sample_rate_int} Hz produces 0 samples. "
            f"Duration must be at least ~{min_duration:.2e} s to yield at least 1 sample."
        )

    t = np.arange(num_samples, dtype=np.float64) / sample_rate_int
    tonal_signal = np.zeros(num_samples, dtype=np.float64)

    for order, weight in enumerate(harm_weights, start=1):
        if weight > 0.0:
            freq = base_frequency * order
            tonal_signal += weight * np.sin(2.0 * np.pi * freq * t)

    if noise_level > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, num_samples)
        combined = tonal_signal + (noise_level * noise)
    else:
        combined = tonal_signal

    peak = np.max(np.abs(combined))
    if peak > 0.0 and amplitude > 0.0:
        normalized = combined * (amplitude / peak)
    elif amplitude == 0.0 or peak == 0.0:
        normalized = np.zeros_like(combined)
    else:
        normalized = combined

    return normalized.astype(target_dtype)


def generate_acoustic_anomaly(
    base_frequency: float,
    duration: float,
    sample_rate: int = 44100,
    anomaly_type: str = "frequency_shift",
    shift_hz: float = 40.0,
    pulse_duration: float = 0.5,
    pulse_interval: float = 1.0,
    harmonics: Optional[Sequence[float]] = (1.0, 0.5, 0.25),
    amplitude: float = 0.8,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Generate synthetic acoustic signals containing covert communication anomalies.

    Simulates compromised computers modulating CPU cooling fans to transmit covert
    signals via physical acoustic emissions (e.g., Fansmitter-style covert channels).

    Supported anomaly types:
    - 'frequency_shift': Fan rotational frequency shifts by shift_hz during active pulses (FSK).
    - 'pulsed_burst': Periodic burst pulses of anomalous harmonic audio (OOK / amplitude bursts).
    - 'frequency_modulation': Continuous sinusoidal frequency oscillation around the base frequency.

    Parameters
    ----------
    base_frequency : float
        Normal baseline fan rotational frequency (Hz).
    duration : float
        Total duration in seconds.
    sample_rate : int, optional
        Sampling rate in Hz (default 44100).
    anomaly_type : str, optional
        Type of covert anomaly ('frequency_shift', 'pulsed_burst', 'frequency_modulation').
    shift_hz : float, optional
        Frequency deviation in Hz during anomaly (default 40.0).
    pulse_duration : float, optional
        Duration of each anomalous pulse in seconds (default 0.5).
    pulse_interval : float, optional
        Repetition interval between pulse start times in seconds (default 1.0).
    harmonics : Sequence[float], optional
        Relative harmonic overtone weights (default (1.0, 0.5, 0.25)).
    amplitude : float, optional
        Peak normalized output amplitude (default 0.8).
    noise_level : float, optional
        Broadband airflow turbulence noise level (default 0.05).
    seed : Optional[int], optional
        Random seed for noise reproducibility.
    dtype : Any, optional
        NumPy floating-point data type (default np.float64).

    Returns
    -------
    np.ndarray
        1D NumPy array containing the anomalous acoustic signal.
    """
    valid_types = ("frequency_shift", "pulsed_burst", "frequency_modulation")
    if not isinstance(anomaly_type, str) or anomaly_type not in valid_types:
        raise ValueError(f"anomaly_type must be one of {valid_types}, got '{anomaly_type}'.")

    numeric_params = [
        ("Base frequency", base_frequency),
        ("Duration", duration),
        ("Sample rate", sample_rate),
        ("Shift Hz", shift_hz),
        ("Pulse duration", pulse_duration),
        ("Pulse interval", pulse_interval),
        ("Amplitude", amplitude),
        ("Noise level", noise_level),
    ]
    for param_name, param_val in numeric_params:
        if isinstance(param_val, bool) or not isinstance(param_val, (int, float)):
            raise TypeError(f"{param_name} must be a real numeric value, got {type(param_val).__name__}.")
        if not math.isfinite(param_val):
            raise ValueError(f"{param_name} must be a finite number, got {param_val}.")

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"Seed must be an integer, got {type(seed).__name__}.")
        if seed < 0:
            raise ValueError(f"Seed must be non-negative, got {seed}.")

    try:
        target_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Invalid NumPy dtype specification: {dtype}.") from err

    if not np.issubdtype(target_dtype, np.floating):
        raise ValueError(
            f"Output dtype must be a floating-point type (e.g., np.float64, np.float32), "
            f"got {target_dtype}."
        )

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate} Hz.")
    sample_rate_int = int(sample_rate)
    nyquist = sample_rate_int / 2.0

    if base_frequency <= 0:
        raise ValueError(f"Base frequency must be positive, got {base_frequency} Hz.")

    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration} s.")

    if pulse_duration <= 0:
        raise ValueError(f"Pulse duration must be positive, got {pulse_duration} s.")

    if pulse_interval <= 0:
        raise ValueError(f"Pulse interval must be positive, got {pulse_interval} s.")

    if anomaly_type in ("frequency_shift", "pulsed_burst") and pulse_duration > pulse_interval:
        raise ValueError(
            f"Pulse duration ({pulse_duration}s) cannot exceed pulse interval ({pulse_interval}s)."
        )

    if amplitude < 0.0 or amplitude > 1.0:
        raise ValueError(f"Amplitude must be in range [0.0, 1.0], got {amplitude}.")

    if noise_level < 0.0 or noise_level > 1.0:
        raise ValueError(f"Noise level must be in range [0.0, 1.0], got {noise_level}.")

    # Parse harmonics
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

    # Check max frequency against Nyquist
    max_mult = len(harm_weights)
    max_possible_f = (base_frequency + abs(shift_hz)) * max_mult
    if max_possible_f >= nyquist:
        raise ValueError(
            f"Maximum anomalous harmonic frequency ({max_possible_f:.1f} Hz) reaches or exceeds Nyquist ({nyquist} Hz)."
        )

    num_samples = int(round(duration * sample_rate_int))
    if num_samples <= 0:
        min_duration = 0.5 / sample_rate_int
        raise ValueError(
            f"Duration ({duration} s) at sample rate {sample_rate_int} Hz produces 0 samples. "
            f"Duration must be at least ~{min_duration:.2e} s to yield at least 1 sample."
        )

    t = np.arange(num_samples, dtype=np.float64) / sample_rate_int
    signal = np.zeros(num_samples, dtype=np.float64)

    # Construct instantaneous frequency trajectory
    if anomaly_type == "frequency_shift":
        # Periodic pulse square wave gate
        time_in_cycle = np.fmod(t, pulse_interval)
        pulse_active = time_in_cycle < pulse_duration
        inst_f = base_frequency + (shift_hz * pulse_active.astype(np.float64))
        # Integrate instantaneous frequency to get phase
        phase_accum = 2.0 * np.pi * np.cumsum(inst_f) / sample_rate_int

        for order, weight in enumerate(harm_weights, start=1):
            if weight > 0.0:
                signal += weight * np.sin(order * phase_accum)

    elif anomaly_type == "pulsed_burst":
        time_in_cycle = np.fmod(t, pulse_interval)
        pulse_active = (time_in_cycle < pulse_duration).astype(np.float64)

        # Baseline fan noise is always present; pulsed burst adds extra anomalous tone
        for order, weight in enumerate(harm_weights, start=1):
            if weight > 0.0:
                # Normal baseline fan
                signal += weight * np.sin(2.0 * np.pi * (base_frequency * order) * t)
                # Bursts at shifted frequency
                burst_freq = (base_frequency + shift_hz) * order
                signal += (weight * 1.5 * pulse_active) * np.sin(2.0 * np.pi * burst_freq * t)

    elif anomaly_type == "frequency_modulation":
        # Continuous sinusoidal oscillation modeling RPM modulation: f(t) = f0 + shift * sin(2*pi*f_mod*t)
        f_mod = 1.0 / pulse_interval
        inst_f = base_frequency + shift_hz * np.sin(2.0 * np.pi * f_mod * t)
        phase_accum = 2.0 * np.pi * np.cumsum(inst_f) / sample_rate_int

        for order, weight in enumerate(harm_weights, start=1):
            if weight > 0.0:
                signal += weight * np.sin(order * phase_accum)

    # Add turbulent background noise
    if noise_level > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, num_samples)
        combined = signal + (noise_level * noise)
    else:
        combined = signal

    peak = np.max(np.abs(combined))
    if peak > 0.0 and amplitude > 0.0:
        normalized = combined * (amplitude / peak)
    elif amplitude == 0.0 or peak == 0.0:
        normalized = np.zeros_like(combined)
    else:
        normalized = combined

    return normalized.astype(target_dtype)
