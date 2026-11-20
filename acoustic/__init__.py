"""AetherHound Acoustic Digital Signal Processing (DSP) Module.

This package provides audio signal generation, digital filtering,
frequency-domain analysis (FFT), spectrogram generation, and acoustic feature
extraction for detecting covert channels in air-gapped environments.

Ownership: Member 03 – DSP & Acoustic Engineer
"""

from acoustic.audio_io import read_wav, write_wav
from acoustic.features import (
    FeatureFrame,
    compute_dominant_frequency,
    compute_rms_energy,
    compute_spectral_bandwidth,
    compute_spectral_centroid,
    compute_spectral_rolloff,
    compute_zero_crossing_rate,
    create_feature_frame,
    extract_acoustic_features,
)
from acoustic.fft_analysis import (
    analyze_harmonics,
    compute_fft,
    detect_spectral_peaks,
)
from acoustic.filters import (
    butter_bandpass_filter,
    butter_highpass_filter,
    butter_lowpass_filter,
)
from acoustic.generator import (
    generate_acoustic_anomaly,
    generate_fan_noise,
    generate_sine_wave,
)
from acoustic.spectrogram import compute_spectrogram, export_spectrogram_plot

__version__ = "1.0.0"
__author__ = "Member 03 – DSP & Acoustic Engineer"

__all__ = [
    "FeatureFrame",
    "analyze_harmonics",
    "butter_bandpass_filter",
    "butter_highpass_filter",
    "butter_lowpass_filter",
    "compute_dominant_frequency",
    "compute_fft",
    "compute_rms_energy",
    "compute_spectral_bandwidth",
    "compute_spectral_centroid",
    "compute_spectral_rolloff",
    "compute_spectrogram",
    "compute_zero_crossing_rate",
    "create_feature_frame",
    "detect_spectral_peaks",
    "export_spectrogram_plot",
    "extract_acoustic_features",
    "generate_acoustic_anomaly",
    "generate_fan_noise",
    "generate_sine_wave",
    "read_wav",
    "write_wav",
]
