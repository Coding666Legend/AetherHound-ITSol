"""AetherHound Acoustic Digital Signal Processing (DSP) Module.

This package provides audio signal generation, digital filtering,
frequency-domain analysis (FFT), spectrogram generation, and acoustic feature
extraction for detecting covert channels in air-gapped environments.

Ownership: Member 03 – DSP & Acoustic Engineer
"""

from acoustic.generator import generate_fan_noise, generate_sine_wave

__version__ = "0.2.0"
__author__ = "Member 03 – DSP & Acoustic Engineer"

__all__ = [
    "generate_fan_noise",
    "generate_sine_wave",
]
