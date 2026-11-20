"""Unit tests for the acoustic feature extraction module in AetherHound."""

from datetime import datetime, timezone
import unittest
import numpy as np

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
from acoustic.generator import generate_sine_wave


class TestFeatures(unittest.TestCase):
    """Test suite for feature extraction and FeatureFrame creation."""

    def setUp(self):
        self.sr = 44100
        self.duration = 1.0
        # 1000 Hz pure sine wave of amplitude 1.0
        self.tone_1000 = generate_sine_wave(
            1000.0, self.duration, self.sr, amplitude=1.0, phase=0.0
        )

    def test_rms_energy(self):
        """Verify RMS energy of a sine wave equals peak / sqrt(2)."""
        rms = compute_rms_energy(self.tone_1000)
        expected_rms = 1.0 / np.sqrt(2.0)
        self.assertAlmostEqual(rms, expected_rms, delta=0.01)

    def test_zero_crossing_rate(self):
        """Verify zero-crossing rate of a 1000 Hz wave at 44.1 kHz."""
        # 1000 cycles per second = 2000 zero crossings per second
        # ZCR = 2000 / 44100 approx 0.04535
        zcr = compute_zero_crossing_rate(self.tone_1000)
        expected_zcr = 2.0 * 1000.0 / self.sr
        self.assertAlmostEqual(zcr, expected_zcr, delta=0.005)

    def test_spectral_features_on_pure_tone(self):
        """Verify spectral centroid, bandwidth, and dominant frequency on pure tone."""
        freqs = np.fft.rfftfreq(len(self.tone_1000), d=1.0 / self.sr)
        mags = np.abs(np.fft.rfft(self.tone_1000))

        dom_f = compute_dominant_frequency(freqs, mags)
        self.assertAlmostEqual(dom_f, 1000.0, delta=1.0)

        centroid = compute_spectral_centroid(freqs, mags)
        self.assertAlmostEqual(centroid, 1000.0, delta=5.0)

        bandwidth = compute_spectral_bandwidth(freqs, mags, centroid=centroid)
        self.assertLess(bandwidth, 50.0)  # Pure tone has very narrow bandwidth

        rolloff = compute_spectral_rolloff(freqs, mags, roll_percent=0.85)
        self.assertAlmostEqual(rolloff, 1000.0, delta=5.0)

    def test_zero_silence_handling(self):
        """Verify silence (all zeros) returns 0.0 without division by zero errors."""
        silent = np.zeros(1000, dtype=np.float64)
        features = extract_acoustic_features(silent, sample_rate=self.sr)

        for name, val in features.items():
            self.assertEqual(val, 0.0, f"Feature {name} should be 0.0 for silence")

    def test_extract_acoustic_features_keys_and_values(self):
        """Verify extract_acoustic_features returns all expected keys with finite floats."""
        features = extract_acoustic_features(self.tone_1000, sample_rate=self.sr)
        expected_keys = {
            "rms_energy",
            "zero_crossing_rate",
            "spectral_centroid",
            "spectral_bandwidth",
            "dominant_frequency",
            "spectral_rolloff",
        }
        self.assertEqual(set(features.keys()), expected_keys)
        for k, v in features.items():
            self.assertIsInstance(v, float)
            self.assertTrue(np.isfinite(v))

    def test_create_feature_frame(self):
        """Verify FeatureFrame container matches architectural expectations."""
        now = datetime.now(timezone.utc)
        frame = create_feature_frame(
            self.tone_1000,
            sample_rate=self.sr,
            source_id="mic_sensor_alpha",
            timestamp=now,
        )

        self.assertIsInstance(frame, FeatureFrame)
        self.assertEqual(frame.modality, "acoustic")
        self.assertEqual(frame.source_id, "mic_sensor_alpha")
        self.assertEqual(frame.timestamp, now)
        self.assertIsInstance(frame.features, np.ndarray)
        self.assertEqual(len(frame.features), len(frame.feature_names))
        self.assertTrue(np.all(np.isfinite(frame.features)))

    def test_invalid_parameters(self):
        """Verify error checking on invalid parameters."""
        with self.assertRaises(ValueError):
            extract_acoustic_features(np.array([]))
        with self.assertRaises(ValueError):
            extract_acoustic_features(np.array([1.0, np.nan]))
        with self.assertRaises(ValueError):
            extract_acoustic_features(np.array([1.0, 2.0]), sample_rate=-44100)


if __name__ == "__main__":
    unittest.main()
