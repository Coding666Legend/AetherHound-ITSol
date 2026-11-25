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

    def test_dc_constant_signal_spectrum(self):
        """Verify constant DC signal energy is concentrated in DC bin (0 Hz)."""
        dc_signal = np.full(1000, 0.5, dtype=np.float64)
        freqs, mags = compute_fft(dc_signal, sample_rate=1000, window=None)
        peak_idx = np.argmax(mags)
        self.assertEqual(freqs[peak_idx], 0.0)
        self.assertAlmostEqual(mags[0], 0.5, delta=0.01)

    def test_non_floating_dtype_rejection(self):
        """Verify integer and boolean arrays are rejected in compute_fft."""
        with self.assertRaises(ValueError):
            compute_fft(np.array([1, 2, 3, 4], dtype=np.int32), sample_rate=self.sr)
        with self.assertRaises(ValueError):
            compute_fft(np.array([True, False, True], dtype=bool), sample_rate=self.sr)

    def test_detect_spectral_peaks_validation(self):
        """Verify boundary and type validation in detect_spectral_peaks."""
        freqs = np.linspace(0, 1000, 100)
        mags = np.sin(freqs) ** 2

        # threshold_rel out of range [0, 1]
        with self.assertRaises(ValueError):
            detect_spectral_peaks(freqs, mags, threshold_rel=-0.1)
        with self.assertRaises(ValueError):
            detect_spectral_peaks(freqs, mags, threshold_rel=1.5)

        # min_distance_hz <= 0
        with self.assertRaises(ValueError):
            detect_spectral_peaks(freqs, mags, min_distance_hz=0.0)

        # max_peaks < 1
        with self.assertRaises(ValueError):
            detect_spectral_peaks(freqs, mags, max_peaks=0)

        # Mismatched array lengths
        with self.assertRaises(ValueError):
            detect_spectral_peaks(freqs[:50], mags)

        # 2D arrays
        with self.assertRaises(ValueError):
            detect_spectral_peaks(freqs.reshape(10, 10), mags.reshape(10, 10))

    def test_analyze_harmonics_validation(self):
        """Verify boundary and type validation in analyze_harmonics."""
        peaks = [100.0, 200.0]

        # base_frequency <= 0
        with self.assertRaises(ValueError):
            analyze_harmonics(0.0, peaks)
        with self.assertRaises(ValueError):
            analyze_harmonics(-100.0, peaks)

        # tolerance_hz <= 0
        with self.assertRaises(ValueError):
            analyze_harmonics(100.0, peaks, tolerance_hz=0.0)

        # max_harmonics < 1
        with self.assertRaises(ValueError):
            analyze_harmonics(100.0, peaks, max_harmonics=0)

        # Boolean parameters
        with self.assertRaises(TypeError):
            analyze_harmonics(True, peaks)
        with self.assertRaises(TypeError):
            analyze_harmonics(100.0, peaks, tolerance_hz=True)
        with self.assertRaises(TypeError):
            analyze_harmonics(100.0, peaks, max_harmonics=True)

    def test_invalid_parameters(self):
        """Verify error handling on invalid inputs."""
        with self.assertRaises(ValueError):
            compute_fft(np.array([], dtype=np.float64), sample_rate=self.sr)
        with self.assertRaises(ValueError):
            compute_fft(np.array([[1.0], [2.0]]), sample_rate=self.sr)
        with self.assertRaises(ValueError):
            compute_fft(np.array([1.0, 2.0]), sample_rate=-44100)
        with self.assertRaises(ValueError):
            compute_fft(np.array([1.0, 2.0]), window="invalid_window")

    def test_detect_spectral_peaks_negative_values_regression(self):
        """Verify negative frequency bins or negative magnitudes raise ValueError in peak detection."""
        neg_freqs = np.array([-5.0, 0.0, 5.0], dtype=np.float64)
        mags = np.array([0.1, 0.5, 0.1], dtype=np.float64)
        with self.assertRaises(ValueError):
            detect_spectral_peaks(neg_freqs, mags)

        valid_freqs = np.array([0.0, 5.0, 10.0], dtype=np.float64)
        neg_mags = np.array([0.1, -0.5, 0.1], dtype=np.float64)
        with self.assertRaises(ValueError):
            detect_spectral_peaks(valid_freqs, neg_mags)


if __name__ == "__main__":
    unittest.main()
