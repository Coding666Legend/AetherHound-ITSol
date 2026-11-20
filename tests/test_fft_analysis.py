"""Unit tests for the FFT frequency analysis module in AetherHound."""

import unittest
import numpy as np

from acoustic.fft_analysis import (
    analyze_harmonics,
    compute_fft,
    detect_spectral_peaks,
)
from acoustic.generator import generate_fan_noise, generate_sine_wave


class TestFFTAnalysis(unittest.TestCase):
    """Test suite for FFT calculation, peak detection, and harmonic analysis."""

    def setUp(self):
        self.sr = 44100
        self.duration = 1.0

    def test_single_tone_fft_accuracy(self):
        """Verify FFT accurately determines frequency and magnitude of a pure tone."""
        target_f = 440.0
        target_amp = 0.8
        sig = generate_sine_wave(
            target_f, self.duration, self.sr, amplitude=target_amp, phase=0.0
        )

        freqs, mags = compute_fft(sig, sample_rate=self.sr, window=None)
        peak_idx = np.argmax(mags)
        detected_f = freqs[peak_idx]
        detected_amp = mags[peak_idx]

        self.assertAlmostEqual(detected_f, target_f, delta=1.0)
        self.assertAlmostEqual(detected_amp, target_amp, delta=0.05)

    def test_fft_windowing_options(self):
        """Verify windowing options (hann, hamming, blackman, boxcar)."""
        sig = generate_sine_wave(500.0, 0.5, self.sr)
        for win in ["hann", "hamming", "blackman", "boxcar", None]:
            with self.subTest(window=win):
                freqs, mags = compute_fft(sig, sample_rate=self.sr, window=win)
                self.assertEqual(len(freqs), len(mags))
                self.assertEqual(freqs[0], 0.0)
                self.assertAlmostEqual(freqs[-1], self.sr / 2.0, delta=1.0)

    def test_detect_spectral_peaks(self):
        """Verify detection of prominent harmonic peaks from synthetic fan audio."""
        f0 = 120.0
        sig = generate_fan_noise(
            base_frequency=f0,
            duration=1.0,
            sample_rate=self.sr,
            harmonics=(1.0, 0.5, 0.25),
            noise_level=0.0,
            amplitude=1.0,
        )

        freqs, mags = compute_fft(sig, sample_rate=self.sr, window="hann")
        peaks = detect_spectral_peaks(freqs, mags, threshold_rel=0.1, min_distance_hz=20.0, max_peaks=5)

        # Expected harmonic peaks at 120, 240, 360 Hz
        peak_frequencies = [p[0] for p in peaks]
        self.assertGreaterEqual(len(peak_frequencies), 3)

        for expected_f in [120.0, 240.0, 360.0]:
            found = any(abs(pf - expected_f) <= 2.0 for pf in peak_frequencies)
            self.assertTrue(found, f"Expected peak near {expected_f} Hz not found in {peak_frequencies}")

    def test_analyze_harmonics(self):
        """Verify harmonic matching logic."""
        base_f = 100.0
        detected_peaks = [100.2, 199.8, 301.0, 750.0]  # harmonics 1, 2, 3 and unrelated 750

        harm_map = analyze_harmonics(base_f, detected_peaks, tolerance_hz=3.0, max_harmonics=4)
        self.assertAlmostEqual(harm_map[1], 100.2, delta=0.5)
        self.assertAlmostEqual(harm_map[2], 199.8, delta=0.5)
        self.assertAlmostEqual(harm_map[3], 301.0, delta=0.5)
        self.assertIsNone(harm_map[4])  # 4th harmonic (400 Hz) absent

    def test_invalid_parameters(self):
        """Verify error handling on invalid inputs."""
        with self.assertRaises(ValueError):
            compute_fft(np.array([]), sample_rate=self.sr)
        with self.assertRaises(ValueError):
            compute_fft(np.array([[1.0], [2.0]]), sample_rate=self.sr)
        with self.assertRaises(ValueError):
            compute_fft(np.array([1.0, 2.0]), sample_rate=-44100)
        with self.assertRaises(ValueError):
            compute_fft(np.array([1.0, 2.0]), window="invalid_window")


if __name__ == "__main__":
    unittest.main()
